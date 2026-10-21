function pc = classify_spike_provenance(times_struct, spikes_struct, subset_times, tol)
% CLASSIFY_SPIKE_PROVENANCE  Shared core for compute_spike_provenance_breakdown.m
% (counts) and compute_spike_waveform_groups.m (actual spike times per
% category) -- ONE place that decides is_forced/is_rescued/which-mask-failed
% per spike, so the two can't silently disagree.
%
% Uses TOLERANCE-based matching (nearest_match, same convention as
% membership_sets/nearest_match everywhere else in this pipeline) instead of
% exact ismember for every cross-file spike-time lookup (forced_sts,
% rescued_sts, and the index_all lookup for mask attribution). forced/
% rescue_mask/index_all live in different files (times_*.mat vs
% *_spikes.mat) that reach these times through different code paths --
% exact equality risked silently mis-classifying a genuinely rescued/forced
% spike as "original" on a float precision mismatch, exactly the failure
% mode is_rescued is supposed to prevent.
%
% Only meaningful for RESCUE ARMS (wave_clus + rescue_spikes metadata) --
% GT/hand-selected data has none of these fields, so don't call this on it.
%
% VARIABLES
%   times_struct  - input: the arm's loaded times_*.mat (needs cluster_class;
%                   forced/cluster_class_pre_rescue/rescue_mask used if present)
%   spikes_struct - input: the arm's loaded *_spikes.mat (needs index_all;
%                   mask_nonart/mask_non_quarantine/quarantine_properties/
%                   mask_non_refract used if present). Pass [] if unavailable --
%                   is_forced still works from times_struct alone; is_rescued
%                   and the mask columns come back all-false/empty.
%   subset_times  - input: absolute spike times (ms) to classify -- must be a
%                   subset of times_struct.cluster_class(:,2)
%   tol           - input: match tolerance (ms), same convention as elsewhere
%   pc            - output: struct with
%                   .is_forced/.is_rescued  - logical, aligned to subset_times,
%                       NOT mutually exclusive (a spike can be both)
%                   .mask_labels  - cellstr, only masks actually present
%                   .mask_fail    - logical, numel(subset_times) x numel(mask_labels):
%                       does THIS subset spike fail that check, regardless of
%                       whether it's rescued (caller restricts to is_rescued
%                       rows for the "why was it rescued" breakdown)
%   forced_sts/rescued_sts - every forced/rescued spike's time in the WHOLE
%                   times/spikes file (not just this subset)
%                   .idx_all      - per-subset-spike index into
%                       spikes_struct.index_all (0 = no match within tol) --
%                       exposed so callers can pull ANY other spikes_struct
%                       field (e.g. quarantine_properties.width) for the same
%                       spikes without re-deriving this lookup themselves
%                       (see extract_quarantine_properties_table.m)
    subset_times = subset_times(:);
    n = numel(subset_times);
    pc = struct('is_forced', false(n,1), 'is_rescued', false(n,1), ...
                'mask_labels', {{}}, 'mask_fail', false(n,0), 'idx_all', zeros(n,1));
    if n == 0
        return;
    end

    forced_sts = [];
    if isfield(times_struct, 'forced') && ~isempty(times_struct.forced)
        f_vec = logical(times_struct.forced(:));
        if isfield(times_struct, 'cluster_class_pre_rescue') && numel(f_vec) == size(times_struct.cluster_class_pre_rescue, 1)
            forced_sts = times_struct.cluster_class_pre_rescue(f_vec, 2);
        elseif numel(f_vec) == size(times_struct.cluster_class, 1)
            forced_sts = times_struct.cluster_class(f_vec, 2);
        end
    end
    if ~isempty(forced_sts)
        pc.is_forced = nearest_match(subset_times, forced_sts, tol) > 0;
    end

    rescued_sts = [];
    if isfield(times_struct, 'rescue_mask') && ~isempty(times_struct.rescue_mask) ...
            && ~isempty(spikes_struct) && isfield(spikes_struct, 'index_all')
        rescued_sts = spikes_struct.index_all(logical(times_struct.rescue_mask(:)));
    end
    if ~isempty(rescued_sts)
        pc.is_rescued = nearest_match(subset_times, rescued_sts, tol) > 0;
    end

    if isempty(spikes_struct) || ~isfield(spikes_struct, 'index_all')
        return;
    end
    idx_all = nearest_match(subset_times, spikes_struct.index_all(:), tol);   % 0 = no match within tol
    pc.idx_all = idx_all;

    labels = {}; fail_cols = {};

    if isfield(spikes_struct, 'mask_nonart')
        mv = logical(spikes_struct.mask_nonart(:));
        labels{end+1} = 'bundle';                              % cross-channel/bundle artifact
        fail_cols{end+1} = lookup_fail(~mv, idx_all);
    end

    % Waveform-QC family: fan out into the per-flag fail_* columns when the
    % spikes file has them (MATLAB within_channel.m does); otherwise fall back
    % to the single mask_non_quarantine vector. quarantine_properties can exist
    % WITHOUT any fail_* columns (Python mcws saves only the raw metrics), and
    % keying on its mere presence dropped waveform QC from the breakdown entirely.
    wc_fields = {'fail_structural', 'fail_amp', 'fail_width', 'fail_prom_ratio', 'fail_multipeak'};
    wc_labels = {'structural', 'low amplitude', 'bad width', 'bad prominence ratio', 'multipeak'};
    have = false(size(wc_fields));
    if isfield(spikes_struct, 'quarantine_properties')
        qp = spikes_struct.quarantine_properties;
        have = cellfun(@(f) isfield(qp, f), wc_fields);
    end
    if any(have)
        for i = find(have)
            labels{end+1} = wc_labels{i};
            fail_cols{end+1} = lookup_fail(logical(qp.(wc_fields{i})(:)), idx_all);
        end
    elseif isfield(spikes_struct, 'mask_non_quarantine')
        mv = logical(spikes_struct.mask_non_quarantine(:));
        labels{end+1} = 'waveform QC';
        fail_cols{end+1} = lookup_fail(~mv, idx_all);
    end

    if isfield(spikes_struct, 'mask_non_refract')
        mv = logical(spikes_struct.mask_non_refract(:));
        labels{end+1} = 'refractory';
        fail_cols{end+1} = lookup_fail(~mv, idx_all);
    end

    pc.mask_labels = labels;
    if ~isempty(fail_cols)
        pc.mask_fail = cell2mat(fail_cols);
    end
end

function col = lookup_fail(fail_vec, idx_all)
% col(i) = fail_vec(idx_all(i)) when idx_all(i)>0 (a match was found within
% tol), else false (no match -> can't say it failed this check).
    col = false(numel(idx_all), 1);
    hit = idx_all > 0;
    col(hit) = fail_vec(idx_all(hit));
end
