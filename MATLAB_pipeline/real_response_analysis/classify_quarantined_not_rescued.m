function groups = classify_quarantined_not_rescued(times_struct, spikes_struct)
% CLASSIFY_QUARANTINED_NOT_RESCUED  Splits a rescue arm's RAW (pre-
% clustering) spike population into per-quarantine-mask groups, restricted
% to the spikes that failed that mask and were NEVER rescued back in.
%
% Deliberately does NOT use classify_spike_provenance.m /
% compute_spike_waveform_groups.m: both of those require their subset_times
% to already be a member of times_struct.cluster_class(:,2) -- the FINAL,
% accepted population. A quarantined-and-never-rescued spike is, by
% definition, NOT in cluster_class at all (rescue is what would have put it
% there); it only exists in spikes_struct's raw index_all/spikes arrays.
% This function walks that raw population directly instead, using the same
% "quarantined" definition rescue_mask_breakdown.m uses: any spike where one
% of the masks in times_struct.quarantine_masks_used reads false.
%
% VARIABLES
%   times_struct   - input: the arm's loaded times_*.mat (needs rescue_mask,
%                    quarantine_masks_used -- same fields rescue_mask_
%                    breakdown.m reads)
%   spikes_struct  - input: the arm's loaded *_spikes.mat (needs index_all,
%                    spikes_all, and whichever of mask_nonart/quarantine_
%                    properties/mask_non_refract quarantine_masks_used lists).
%                    Deliberately spikes_all, NOT spikes: spikes_all is the
%                    full waveform matrix aligned 1:1 with index_all/the
%                    mask_* vectors (all n_all rows); spikes only holds the
%                    accepted-subset waveforms (far fewer rows), which is
%                    why indexing it with an index_all-length mask blew past
%                    its bounds.
%   groups         - output: struct array, one row per mask actually in
%                    play, with .name (mask label), .times (index_all times,
%                    ms, of that mask's quarantined-not-rescued spikes),
%                    .waveforms (their raw waveform rows from spikes_struct.
%                    spikes_all -- pulled directly by position, no
%                    tolerance-match needed since these ARE spikes_struct's
%                    own rows), .pct_of_total (this group's spike count as a
%                    % of total_qnr -- see below). Non-disjoint across groups
%                    (a spike failing 2 masks appears in both, so
%                    pct_of_total across groups sums to >100%, same
%                    "involvement" convention as compute_spike_provenance_
%                    breakdown.m's mask_counts).
%   total_qnr      - output: the shared denominator every group's
%                    pct_of_total is relative to -- the UNION of spikes that
%                    failed at least one of the 3 top-level masks
%                    (mask_nonart/mask_non_quarantine/mask_non_refract) and
%                    were never rescued (same "quarantined" definition
%                    rescue_mask_breakdown.m uses). The finer quarantine_
%                    properties sub-groups (structural/amp/width/...) are
%                    percentaged against this SAME total, not a smaller
%                    "waveform QC only" one, so every panel on the page is
%                    directly comparable.
    groups = struct('name', {}, 'times', {}, 'waveforms', {}, 'pct_of_total', {});
    total_qnr = 0;
    if isempty(spikes_struct) || ~isfield(spikes_struct, 'index_all') || ~isfield(spikes_struct, 'spikes_all')
        return;
    end
    n_all = numel(spikes_struct.index_all);
    if isfield(times_struct, 'rescue_mask') && ~isempty(times_struct.rescue_mask) ...
            && numel(times_struct.rescue_mask) == n_all
        rescue_mask = logical(reshape(times_struct.rescue_mask, 1, []));
    else
        rescue_mask = false(1, n_all);
    end

    valid_masks = {'mask_nonart', 'mask_non_quarantine', 'mask_non_refract'};
    valid_labels = {'bundle', 'waveform QC', 'refractory'};
    if isfield(times_struct, 'quarantine_masks_used') && ~isempty(times_struct.quarantine_masks_used)
        masks_used = cellstr(times_struct.quarantine_masks_used);
        masks_used = masks_used(ismember(masks_used, valid_masks));
    else
        masks_used = valid_masks(isfield(spikes_struct, valid_masks));
    end

    % Union across the 3 top-level masks -> total_qnr, the shared % denominator.
    fail_any = false(1, n_all);
    for i = 1:numel(valid_masks)
        mask_name = valid_masks{i};
        if ~any(strcmp(masks_used, mask_name)) || ~isfield(spikes_struct, mask_name)
            continue;
        end
        passed = logical(reshape(spikes_struct.(mask_name), 1, []));
        if numel(passed) ~= n_all, continue; end
        fail_any = fail_any | ~passed;
    end
    total_qnr = nnz(fail_any & ~rescue_mask);

    index_all = reshape(spikes_struct.index_all, 1, []);
    for i = 1:numel(valid_masks)
        mask_name = valid_masks{i};
        if ~any(strcmp(masks_used, mask_name)) || ~isfield(spikes_struct, mask_name)
            continue;
        end
        passed = logical(reshape(spikes_struct.(mask_name), 1, []));
        if numel(passed) ~= n_all
            continue;
        end
        sel = ~passed & ~rescue_mask;   % failed this mask AND never rescued
        groups(end+1) = struct('name', valid_labels{i}, 'times', index_all(sel)', ...
            'waveforms', spikes_struct.spikes_all(sel, :), ...
            'pct_of_total', 100 * nnz(sel) / max(total_qnr, 1)); %#ok<AGROW>
    end

    % quarantine_properties (the wave_clus-side structural/amp/width/prom/
    % multipeak flags) uses per-flag columns, not a single mask_* vector --
    % same fan-out compute_spike_provenance_breakdown.m/classify_spike_
    % provenance.m already do for the RESCUED side.
    if any(strcmp(masks_used, 'mask_non_quarantine')) && isfield(spikes_struct, 'quarantine_properties')
        qp = spikes_struct.quarantine_properties;
        wc_fields = {'fail_structural', 'fail_amp', 'fail_width', 'fail_prom_ratio', 'fail_multipeak'};
        wc_labels = {'structural', 'low amplitude', 'bad width', 'bad prominence ratio', 'multipeak'};
        for i = 1:numel(wc_fields)
            if ~isfield(qp, wc_fields{i}), continue; end
            failed = logical(reshape(qp.(wc_fields{i}), 1, []));
            if numel(failed) ~= n_all, continue; end
            sel = failed & ~rescue_mask;
            groups(end+1) = struct('name', wc_labels{i}, 'times', index_all(sel)', ...
                'waveforms', spikes_struct.spikes_all(sel, :), ...
                'pct_of_total', 100 * nnz(sel) / max(total_qnr, 1)); %#ok<AGROW>
        end
    end
end
