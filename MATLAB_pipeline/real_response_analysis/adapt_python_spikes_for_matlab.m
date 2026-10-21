function spikes_out = adapt_python_spikes_for_matlab(times_py)
% ADAPT_PYTHON_SPIKES_FOR_MATLAB  Builds a MATLAB *_spikes.mat-compatible
% struct from a Python-pipeline times_*.mat. Python bundles the raw spike
% population (index/spikes/quarantine) into the SAME file MATLAB splits into
% times_*.mat (accepted population) + *_spikes.mat (raw population +
% quarantine masks) -- this is the *_spikes.mat half of that split, derived
% from the same times_py struct adapt_python_times_for_matlab.m adapts for
% the times_*.mat half. Pass this wherever this pipeline's code expects a
% loaded *_spikes.mat struct (classify_spike_provenance.m, rescue_mask_
% breakdown.m, classify_quarantined_not_rescued.m, ...).
%
% MATLAB's mask_nonart/mask_non_quarantine/mask_non_refract are PASS
% vectors (true = spike passed that check, see classify_spike_
% provenance.m's `~mv` = failed usage); Python's quarantine.<rejector>
% fields are FAIL vectors (true = flagged/quarantined) -- inverted here.
% Same rejector-name pairing as adapt_python_times_for_matlab.m.
%
% index_all/spikes_all (full raw population) vs index/spikes (PRE-rescue
% accepted-only subset) is a real, separate distinction in MATLAB's own
% *_spikes.mat files, confirmed against a real rescue_sd1 file: index_all/
% spikes_all/mask_* all have the same 29770 rows; index/spikes have only
% 7257, EXACTLY matching that same file's times_*.mat cluster_class_
% pre_rescue row count. do_structure_mu_BCM_online3.m reads index/spikes
% (not index_all/spikes_all) to seed the base grapes/grapes_blanks.mat
% before do_structure_sorted/do_structure_rescue run -- so it needs the
% pre-rescue-accepted subset, same population adapt_python_times_for_
% matlab.m's cluster_class_pre_rescue reconstructs (real class, not yet
% rescued).
%
% VARIABLES
%   times_py    - input: loaded Python times_*.mat struct (needs index,
%                 spikes, cluster_class, quarantine.bundle_collision/
%                 within_channel/refractory_period/rescued [rescued optional
%                 -- absent means nothing was rescued]; quarantine.
%                 refractory_period_full_waveform, if present, is a nested
%                 metadata struct and is deliberately ignored -- see below;
%                 quarantine_properties used as-is if present)
%   spikes_out  - output: struct with index_all, spikes_all, index, spikes,
%                 mask_nonart, mask_non_quarantine, mask_non_refract,
%                 quarantine_properties (flattened -- see below)
    spikes_out = struct();
    spikes_out.index_all = reshape(double(times_py.index), 1, []);   % 1xN row, like MATLAB's
    spikes_out.spikes_all = times_py.spikes;
    n = numel(spikes_out.index_all);

    cc = times_py.cluster_class;
    real_class = cc(:,1) ~= 0;
    if isfield(times_py, 'quarantine') && isfield(times_py.quarantine, 'rescued')
        rescued = logical(times_py.quarantine.rescued(:));
    else
        rescued = false(n, 1);
    end
    pre_rescue_accepted = real_class & ~rescued;
    spikes_out.index = times_py.index(pre_rescue_accepted);
    spikes_out.spikes = times_py.spikes(pre_rescue_accepted, :);

    q = struct();
    if isfield(times_py, 'quarantine')
        q = times_py.quarantine;
    end

    if isfield(q, 'bundle_collision')
        spikes_out.mask_nonart = reshape(~logical(q.bundle_collision), 1, []);
    end
    if isfield(q, 'within_channel')
        spikes_out.mask_non_quarantine = reshape(~logical(q.within_channel), 1, []);
    end
    if isfield(q, 'refractory_period')
        % refractory_period_full_waveform is deliberately NOT folded in here:
        % it's a nested metadata struct (waveforms/start_time_ms/n_samples/
        % spike_indices), not a per-spike boolean -- confirmed against a real
        % Python times file, and Python's own quarantine.all is itself just
        % OR(refractory_period, bundle_collision, within_channel), so
        % excluding it here matches Python's own quarantine semantics.
        spikes_out.mask_non_refract = reshape(~logical(q.refractory_period), 1, []);
    end
    if isfield(times_py, 'quarantine_properties')
        qp = flatten_quarantine_properties(times_py.quarantine_properties);
        qc = struct();
        if isfield(times_py, 'par') && isfield(times_py.par, 'qc_params')
            qc = times_py.par.qc_params;
        end
        spikes_out.quarantine_properties = derive_waveform_qc_flags(qp, qc);
    end
end

function qp = derive_waveform_qc_flags(qp, qc)
% Python saves only the combined within_channel mask plus the raw per-spike
% metrics; MATLAB's within_channel.m also saves one fail_* flag per check,
% which is what the per-reason rescue breakdown fans out on. Rebuild them
% from Python's metrics with within_channel.m's exact definitions. Verified
% on ch328: the same formulas on MATLAB's own metrics reproduce MATLAB's
% fail_* with 0 mismatches, and OR(structural,amp,width,prom_ratio) on
% Python's metrics reproduces Python's within_channel on all 28539 spikes.
    need = {'peak_sample_20', 'width', 'prominence_ratio', 'low_low_amp_spike'};
    if any(isfield(qp, {'fail_structural','fail_amp','fail_width','fail_prom_ratio'})) || ~all(isfield(qp, need)) ...
            || ~all(isfield(qc, {'min_width_idx','max_width_idx','final_prominence_ratio_pass'}))
        return;
    end
    p20 = double(qp.peak_sample_20(:));
    w   = double(qp.width(:));
    pr  = double(qp.prominence_ratio(:));
    st  = isnan(p20);                                   % no peak at sample 20 (or no peaks)
    qp.fail_structural = st;
    qp.fail_amp        = logical(qp.low_low_amp_spike(:));
    qp.fail_width      = ~st & (isnan(w) | w < qc.min_width_idx | w > qc.max_width_idx);
    qp.fail_prom_ratio = ~st & ~isnan(pr) & pr <= qc.final_prominence_ratio_pass;
    if all(isfield(qp, {'left_peak_ratio','right_peak_ratio'}))
        thr = 0.5;                                      % within_channel.m's default
        if isfield(qc, 'multipeak_ratio_threshold'), thr = qc.multipeak_ratio_threshold; end
        worst = max([double(qp.left_peak_ratio(:)), double(qp.right_peak_ratio(:))], [], 2);
        qp.fail_multipeak = worst > thr & isfinite(worst);
    end
    % opposite-peak rule (python, 2026-10-19 on; part of within_channel there): quarantined when the
    % opposite-polarity ratio on BOTH sides of the main peak is above qc.opposite_peak_ratio_both_sides.
    % Only for runs that had the rule (the threshold is in qc_params); python's own flag when saved.
    if isfield(qc, 'opposite_peak_ratio_both_sides')
        if isfield(qp, 'opposite_both_sides')
            v = double(qp.opposite_both_sides(:)); qp.fail_opposite_peak = v > 0 & isfinite(v);
        elseif all(isfield(qp, {'left_ratio_used', 'right_ratio_used'}))
            thr = double(qc.opposite_peak_ratio_both_sides);
            qp.fail_opposite_peak = double(qp.left_ratio_used(:)) > thr & double(qp.right_ratio_used(:)) > thr;
        end
    end
end

function qp_out = flatten_quarantine_properties(qp_raw)
% Python (mcws) saves quarantine_properties as an Nx1/1xN STRUCT ARRAY --
% one struct per spike, each field a scalar. MATLAB's own convention (and
% every consumer: extract_quarantine_properties_table.m, classify_
% quarantined_not_rescued.m) expects the transpose: a single 1x1 struct-of-
% arrays, each field an N-length column. Confirmed against a real MATLAB
% *_spikes.mat (shape (1,1), each field (29770,1)) vs a real Python
% times_*.mat (shape (1,28539), each field (1,1)) -- exactly this mismatch,
% and exactly what produced "Index exceeds the number of array elements.
% Index must not exceed 1." in extract_quarantine_properties_table.m: MATLAB
% silently took qp(1).<field> as a scalar via comma-separated-list
% truncation, then tried to index that scalar at a position > 1.
    % Some runs save it as a 1xN CELL of per-spike structs (same fields)
    % instead of a struct array -- join those into the struct-array form.
    if iscell(qp_raw)
        qp_raw = [qp_raw{:}];
    end
    n = numel(qp_raw);
    if n <= 1
        qp_out = qp_raw;
        return;
    end
    fns = fieldnames(qp_raw);
    qp_out = struct();
    for i = 1:numel(fns)
        fn = fns{i};
        vals = arrayfun(@(s) s.(fn), qp_raw(:), 'UniformOutput', false);
        try
            qp_out.(fn) = cell2mat(vals);
        catch
            qp_out.(fn) = vals;   % non-numeric field (e.g. a reason string) -- keep as column cellstr
        end
    end
end
