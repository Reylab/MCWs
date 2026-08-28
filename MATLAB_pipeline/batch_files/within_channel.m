function within_channel(channels, varargin)
    % Function: within_channel
    % Description: Filters spikes based on waveform characteristics (amplitude, width,
    %              prominence ratio, and opposite-polarity multi-peak structure) using a
    %              robust, polarity-aware method. The resulting mask is combined with an
    %              existing collision mask for cumulative filtering.
    %
    % Channels: The list of microelectrode channels (channel IDs) to process.
    %
    % Optional name-value args:
    %   'use_parallel' (default false) - run channels with parfor instead of for.
    %   'detections'   (default 'all') - which per-spike checks contribute to the
    %                    quarantine mask. Either the string 'all' (the standard checks:
    %                    structural + amp + width + prom_ratio) or a cell/string array of
    %                    check names, any of:
    %                       'structural' - no usable peak at sample 20 / too-short waveform
    %                       'amp'        - peak amplitude below the low percentile cutoff
    %                       'width'      - half-height width outside [min_width_idx, max_width_idx]
    %                       'prom_ratio' - main/secondary prominence ratio below threshold
    %                       'multipeak'  - strong opposite-polarity peak on either side of
    %                                      the main peak (bilateral search around sample 20)
    %                    'multipeak' is OPT-IN: it is NOT part of 'all' and only runs when
    %                    named explicitly, e.g. {'amp','width','multipeak','prom_ratio'}.
    %   'filter_mode'  (default 'all') - LEGACY alias, honoured only when 'detections' is
    %                    NOT supplied. 'all' -> the standard checks (no multipeak);
    %                    'amp_only' -> {'amp'} (behaves exactly like the old
    %                    amplitude-percentile-only mode).
    %
    % Notes:
    %   * The 12 opposite-polarity descriptors (left_peak_/right_peak_/max_left_/max_right_
    %     with _amp,_loc,_ratio suffixes) are ALWAYS computed and stored in
    %     quarantine_properties, regardless of whether 'multipeak' is an active detection.
    %     The 'multipeak' toggle only controls whether they feed the quarantine mask.
    %   * quarantine_properties also carries a per-check boolean mask for every detection
    %     (fail_structural, fail_amp, fail_width, fail_prom_ratio, fail_multipeak) so the
    %     mask can be re-derived downstream for any subset of checks.

    valid_detections = {'structural', 'amp', 'width', 'prom_ratio', 'multipeak'};
    % 'all' (and the legacy filter_mode 'all') expands to these. 'multipeak' is
    % deliberately excluded -> it only runs when named explicitly in 'detections'.
    default_detections = {'structural', 'amp', 'width', 'prom_ratio'};

    p = inputParser;
    addParameter(p, 'use_parallel', false, @islogical);
    addParameter(p, 'detections', 'all', @(x) ischar(x) || iscellstr(x) || isstring(x));
    addParameter(p, 'filter_mode', 'all', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});

    use_parallel = p.Results.use_parallel;

    % Resolve the active detection set.
    if ismember('detections', p.UsingDefaults)
        % No explicit 'detections' -> honour the legacy 'filter_mode' alias.
        switch lower(char(p.Results.filter_mode))
            case 'all'
                active_detections = default_detections;
            case 'amp_only'
                active_detections = {'amp'};
            otherwise
                error('within_channel:BadFilterMode', ...
                    'Unknown filter_mode ''%s'' (expected ''all'' or ''amp_only'').', ...
                    char(p.Results.filter_mode));
        end
    else
        active_detections = normalize_detections(p.Results.detections, valid_detections, default_detections);
    end
    filter_mode = char(p.Results.filter_mode);
    detections_str = strjoin(active_detections, '+');

    within_channel_tic = tic;

    % Define Quality Control (QC) parameters for spike shape analysis
    qc_params = struct();
    qc_params.min_amplitude_percentile = 5; % Spikes below this P2P amplitude percentile are quarantined
    qc_params.min_width_idx = 3;            % Min width of main feature (in samples)
    qc_params.max_width_idx = 15;           % Max width of main feature (in samples)
    qc_params.prominence_ratio_threshold = 0.01; % Secondary feature prominence must be > 1% of main peak amp
    qc_params.final_prominence_ratio_pass = 2; % Main feature Prominence/Amplitude threshold for complex spikes
    qc_params.multipeak_ratio_threshold = 0.5; % 'multipeak' fails when the larger of left/right opposite-polarity
                                               % peak ratio (|amp| / peak-to-peak) exceeds this. 0.5 ~ the
                                               % opposite-polarity peak rivals the main peak. TUNE PER DATASET.

    load('NSx','NSx');
    % Filter NSx structure to include only the specified channels
    NSx_proc = NSx(ismember(cell2mat({NSx.chan_ID}),channels));
    num_channels_proc = length(NSx_proc);
    fprintf('Starting robust waveform quality check on %d channels (Detections: %s, Parallel: %d)...\n', ...
        num_channels_proc, detections_str, use_parallel);

    dates = dir(fullfile(pwd, 'spikes*'));
    dates = dates([dates.isdir]);
    if isempty(dates), error('No spikes folders found.'); end
    [~, idx] = max([dates.datenum]);
    active_spikes_dir = fullfile(pwd, dates(idx).name);

    chan_lbls = {NSx_proc.output_name};

    % Toggle between parfor and for loops
    if use_parallel
        parfor k = 1:num_channels_proc
            process_within_channel(chan_lbls{k}, active_spikes_dir, qc_params, active_detections, filter_mode, k, num_channels_proc);
        end
    else
        for k = 1:num_channels_proc
            process_within_channel(chan_lbls{k}, active_spikes_dir, qc_params, active_detections, filter_mode, k, num_channels_proc);
        end
    end

    within_channel_toc = toc(within_channel_tic);
    fprintf("within_channel DONE in %s seconds.\n", num2str(within_channel_toc, '%2.2f'));
end


function det = normalize_detections(raw, valid_detections, default_detections)
    % Turn the user 'detections' arg into a validated, de-duplicated cellstr.
    % The 'all' keyword expands to default_detections (multipeak stays opt-in).
    if (ischar(raw) && strcmpi(strtrim(raw), 'all')) || ...
       (isstring(raw) && isscalar(raw) && strcmpi(strtrim(raw), "all"))
        det = default_detections;
        return;
    end
    if ischar(raw)
        raw = {raw};
    elseif isstring(raw)
        raw = cellstr(raw);
    end
    raw = lower(strtrim(raw(:)'));
    raw = raw(~cellfun(@isempty, raw));
    bad = raw(~ismember(raw, valid_detections));
    if ~isempty(bad)
        error('within_channel:BadDetection', ...
            'Unknown detection(s): %s. Valid: %s.', ...
            strjoin(bad, ', '), strjoin(valid_detections, ', '));
    end
    if isempty(raw)
        error('within_channel:NoDetections', ...
            'No valid detections requested. Valid: %s.', strjoin(valid_detections, ', '));
    end
    det = unique(raw, 'stable');
end


% --- LOCAL HELPER FUNCTION FOR LOOP BODY ---
function process_within_channel(ch_lbl, active_spikes_dir, qc_params, active_detections, filter_mode, k, num_channels_proc)
    % Target the file inside our locked directory
    spike_file = fullfile(active_spikes_dir, sprintf('%s_spikes.mat', ch_lbl));

    try
        % fprintf('ch.%d/%d %s: loading %s\n', k, num_channels_proc, ch_lbl, spike_file);
        SPK = load(spike_file);

        % Load full spike set
        if isfield(SPK,'spikes_all')
            spikes_all = SPK.spikes_all;
            index_all  = SPK.index_all;
        else
            spikes_all = SPK.spikes;
            index_all = SPK.index;
        end
        % Load existing collision mask. mask_non_collision is TRUE for spikes that passed the initial filtering.
        if isfield(SPK,'mask_nonart')
            mask_used = 2;
            mask_non_collision = SPK.mask_nonart;
        else
            mask_used = 1;
            mask_non_collision = true(size(index_all));
            warning('ArtifactRemoval:NoCollisionMask', 'No collision mask found for %s. Assuming all spikes are non-collision.', ch_lbl);
        end

        if isfield(SPK, 'mask_taskspks')
            mask_taskspks = SPK.mask_taskspks;
        else
            mask_taskspks = true(size(index_all));
        end

        % Normalize orientation to avoid implicit expansion (Nx1 & 1xN -> NxN).
        index_all = index_all(:);
        mask_non_collision = logical(mask_non_collision(:));
        mask_taskspks = logical(mask_taskspks(:));

        % map the loaded mask back to the original variable name for saving
        mask_nonart = mask_non_collision;

        % Compute every per-spike metric and per-check fail mask (all detections).
        % fprintf('ch.%d/%d %s: running waveform QC\n', k, num_channels_proc, ch_lbl);
        [~, quarantine_properties] = analyze_spike_waveforms(spikes_all, qc_params);

        % DECISION MASK LOGIC
        % Assemble the quarantine mask from ONLY the active detections. Each detection
        % contributes an independent boolean fail mask; a spike is quarantined if it
        % fails any active check.
        n_spikes = size(spikes_all, 1);
        fail_total = false(n_spikes, 1);
        if ismember('structural', active_detections)
            fail_total = fail_total | quarantine_properties.fail_structural(:);
        end
        if ismember('amp', active_detections)
            fail_total = fail_total | quarantine_properties.fail_amp(:);
        end
        if ismember('width', active_detections)
            fail_total = fail_total | quarantine_properties.fail_width(:);
        end
        if ismember('prom_ratio', active_detections)
            fail_total = fail_total | quarantine_properties.fail_prom_ratio(:);
        end
        if ismember('multipeak', active_detections)
            fail_total = fail_total | quarantine_properties.fail_multipeak(:);
        end
        mask_non_quarantine = ~fail_total;

        % Re-derive 'reason' so it reflects only the active detections
        % (priority: structural > amp > prom_ratio > width > multipeak).
        quarantine_properties.reason = compose_reasons(quarantine_properties, active_detections);

        if numel(mask_non_collision) ~= numel(mask_non_quarantine) || numel(mask_taskspks) ~= numel(mask_non_quarantine)
            error('ArtifactRemoval:MaskLengthMismatch', ...
                'Mask lengths differ for %s (collision=%d, quarantine=%d, task=%d).', ...
                ch_lbl, numel(mask_non_collision), numel(mask_non_quarantine), numel(mask_taskspks));
        end

        % Combine Masks: Spike must pass collision check AND quarantine check
        mask_total_pass = mask_non_collision & mask_non_quarantine & mask_taskspks;

        % Final cleaned indices
        index = index_all(mask_total_pass);
        % Final cleaned waveforms (overwriting 'spikes_coll_only' to hold the fully filtered set)
        spikes = spikes_all(mask_total_pass, :);

        % Update the main 'par' structure with the new QC parameters
        par = SPK.par;
        par.qc_params = qc_params;
        par.filter_mode = filter_mode;              % legacy field, kept for logging
        par.detections = active_detections;         % the checks actually applied

        % fprintf('ch.%d/%d %s: saving filtered results\n', k, num_channels_proc, ch_lbl);
        % fprintf('  -> Quarantined: %d\n', nnz(~mask_non_quarantine));
        % Remove -append to fully overwrite file, ensuring old unfiltered spikes don't persist

        index = reshape(index, 1, []);
        index_all = reshape(index_all, 1, []);
        mask_nonart = reshape(mask_nonart, 1, []);
        mask_non_quarantine = reshape(mask_non_quarantine,1,[]);

        save(spike_file, ...
             'index', 'spikes', 'index_all', 'spikes_all', 'par', 'mask_nonart', ...
             'mask_non_quarantine', 'quarantine_properties','-append');

        num_removed_this_step = sum(mask_non_collision) - sum(mask_total_pass);
        num_total_spikes = numel(index_all);

        fprintf('ch.%d of %d: %s. Masks used (%d): quarantined %d spikes. Remaining: %d/%d (%.2f%%)\n', ...
            k, num_channels_proc, ch_lbl, mask_used, num_removed_this_step, sum(mask_total_pass), num_total_spikes, sum(mask_total_pass)/num_total_spikes*100);

    catch ME
        fprintf('  -> FAILED to process channel %s: %s\n', ch_lbl, ME.message);
    end
end


function reasons = compose_reasons(qp, active_detections)
    % First-failing reason per spike among the ACTIVE detections only.
    % Priority: structural > amp > prom_ratio > width > multipeak. 'pass' if none.
    n = numel(qp.fail_structural);
    reasons = repmat({'pass'}, n, 1);

    if ismember('structural', active_detections)
        idx = qp.fail_structural(:) & strcmp(reasons, 'pass');
        reasons(idx) = qp.structural_reason(idx);
    end
    if ismember('amp', active_detections)
        idx = qp.fail_amp(:) & strcmp(reasons, 'pass');
        reasons(idx) = {'low_amplitude'};
    end
    if ismember('prom_ratio', active_detections)
        idx = qp.fail_prom_ratio(:) & strcmp(reasons, 'pass');
        reasons(idx) = {'bad_prominence_ratio'};
    end
    if ismember('width', active_detections)
        idx = qp.fail_width(:) & strcmp(reasons, 'pass');
        reasons(idx) = {'bad_width'};
    end
    if ismember('multipeak', active_detections)
        idx = qp.fail_multipeak(:) & strcmp(reasons, 'pass');
        reasons(idx) = {'multipeak_opposite_peak'};
    end
end


function [quarantine_mask, quarantine_properties] = analyze_spike_waveforms(spikes, par)
    % Computes every per-spike waveform metric plus an independent boolean fail mask
    % for each detection: structural, amplitude, width, prominence ratio, and the
    % bilateral opposite-polarity multi-peak check. The caller chooses which masks to
    % combine. quarantine_mask (first output) is the union of ALL checks.
    %
    % Detection <-> metric mapping:
    %   structural  -> fail_structural (no peaks / no peak at sample 20 / short waveform)
    %   amp         -> fail_amp        (low_low_amp_spike: |peak_sample_20| below percentile)
    %   width       -> fail_width      (half-height width NaN or outside [min,max])
    %   prom_ratio  -> fail_prom_ratio (multi-peak with significant secondary, ratio <= pass)
    %   multipeak   -> fail_multipeak  (max(left_peak_ratio, right_peak_ratio) > threshold)

    num_spikes = size(spikes, 1);
    sample_20_idx = 20;

    % Multi-peak threshold with a defensive default (older 'par' structs may lack it).
    if isfield(par, 'multipeak_ratio_threshold')
        multipeak_ratio_threshold = par.multipeak_ratio_threshold;
    else
        multipeak_ratio_threshold = 0.5;
    end

    % Per-check fail masks
    fail_structural = false(num_spikes, 1);
    fail_amp        = false(num_spikes, 1);
    fail_width      = false(num_spikes, 1);
    fail_prom_ratio = false(num_spikes, 1);
    fail_multipeak  = false(num_spikes, 1);

    % Per-spike metrics for quarantine_properties
    prominence_ratio = nan(num_spikes, 1);
    num_peaks_arr = zeros(num_spikes, 1);
    prominence_sample_20 = nan(num_spikes, 1);
    other_prominence = nan(num_spikes, 1);
    width = nan(num_spikes, 1);
    peak_sample_20 = nan(num_spikes, 1);
    other_peak_loc = nan(num_spikes, 1);
    peak_pos_max = nan(num_spikes, 1);
    prominence_pos_max = nan(num_spikes, 1);
    low_low_amp_spike = false(num_spikes, 1);

    % Bilateral opposite-polarity descriptors (see opposite_polarity_descriptors).
    % Always computed; _loc uses -1 as the "undefined" sentinel.
    left_peak_amp   = nan(num_spikes, 1);
    left_peak_loc   = -ones(num_spikes, 1);
    left_peak_ratio = nan(num_spikes, 1);
    right_peak_amp   = nan(num_spikes, 1);
    right_peak_loc   = -ones(num_spikes, 1);
    right_peak_ratio = nan(num_spikes, 1);
    max_left_amp   = nan(num_spikes, 1);
    max_left_loc   = -ones(num_spikes, 1);
    max_left_ratio = nan(num_spikes, 1);
    max_right_amp   = nan(num_spikes, 1);
    max_right_loc   = -ones(num_spikes, 1);
    max_right_ratio = nan(num_spikes, 1);

    % Sub-reason for structural failures ('pass' otherwise).
    structural_reason = repmat({'pass'}, num_spikes, 1);

    % Store peak info for each spike to avoid recalculating findpeaks
    peak_info = cell(num_spikes, 1);

    if size(spikes, 2) < sample_20_idx
        warning('ArtifactRemoval:ShortWaveform', ...
            'Waveform length (%d) is shorter than sample_20_idx (%d). Quarantining all spikes.', ...
            size(spikes, 2), sample_20_idx);
        fail_structural(:) = true;
        structural_reason(:) = {'short_waveform'};
        quarantine_mask = true(num_spikes, 1);
        quarantine_properties = assemble_properties( ...
            prominence_ratio, num_peaks_arr, prominence_sample_20, other_prominence, width, ...
            peak_sample_20, other_peak_loc, peak_pos_max, prominence_pos_max, low_low_amp_spike, ...
            left_peak_amp, left_peak_loc, left_peak_ratio, right_peak_amp, right_peak_loc, right_peak_ratio, ...
            max_left_amp, max_left_loc, max_left_ratio, max_right_amp, max_right_loc, max_right_ratio, ...
            fail_structural, fail_amp, fail_width, fail_prom_ratio, fail_multipeak, structural_reason);
        return;
    end

    % SINGLE PASS: Extract peaks and metrics for all spikes
    for i = 1:num_spikes

        waveform = spikes(i, :);
        sample_20_value = waveform(sample_20_idx);

        % Invert if trough at sample 20
        if sample_20_value < 0
            signal_for_analysis = -waveform;
        else
            signal_for_analysis = waveform;
        end

        [pks, locs, w, p] = findpeaks(signal_for_analysis);

        % Also compute positive peaks for properties (on original waveform)
        [pos_peaks, pos_locs, ~, pos_prominences] = findpeaks(waveform);
        % Negative-going peaks (troughs) of the original waveform, used only by the
        % bilateral opposite-polarity search below (matches Python find_peaks(-w)).
        [~, neg_locs] = findpeaks(-waveform);

        % Bilateral opposite-polarity descriptors around the main peak. Computed for
        % every spike with a finite, non-zero value at sample 20 (does not require a
        % local peak there), matching the Python _peak_metrics_per_spike behaviour.
        dd = opposite_polarity_descriptors(waveform, neg_locs(:).', pos_locs(:).', sample_20_idx);
        left_peak_amp(i)   = dd.left_peak_amp;
        left_peak_loc(i)   = dd.left_peak_loc;
        left_peak_ratio(i) = dd.left_peak_ratio;
        right_peak_amp(i)   = dd.right_peak_amp;
        right_peak_loc(i)   = dd.right_peak_loc;
        right_peak_ratio(i) = dd.right_peak_ratio;
        max_left_amp(i)   = dd.max_left_amp;
        max_left_loc(i)   = dd.max_left_loc;
        max_left_ratio(i) = dd.max_left_ratio;
        max_right_amp(i)   = dd.max_right_amp;
        max_right_loc(i)   = dd.max_right_loc;
        max_right_ratio(i) = dd.max_right_ratio;

        % Store peak info and basic metrics
        peak_info{i} = struct('pks', pks, 'locs', locs, 'w', w, 'p', p, ...
                              'pos_peaks', pos_peaks, 'pos_prominences', pos_prominences, ...
                              'has_peak_at_20', any(locs == sample_20_idx), ...
                              'peak_sample_20_raw', sample_20_value);

        num_peaks_arr(i) = length(pks);

        % Pre-identify spikes with no peaks
        if isempty(pks)
            fail_structural(i) = true;
            structural_reason{i} = 'no_peaks_found';
            continue;
        end

        % Extract peak at sample 20 if it exists
        if peak_info{i}.has_peak_at_20
            peak_sample_20(i) = sample_20_value;
        end
    end

    % Calculate amplitude threshold from 5th percentile
    abs_peak_amps = abs(peak_sample_20);
    valid_peak_amps = abs_peak_amps(isfinite(abs_peak_amps));

    if isempty(valid_peak_amps)
        amp_threshold = inf;
    else
        amp_threshold = prctile(valid_peak_amps, par.min_amplitude_percentile);
    end

    % DECISION TREE: Apply all decisions (minimal reprocessing)
    for i = 1:num_spikes
        % Retrieve the original waveform for this spike
        waveform = spikes(i, :);

        info = peak_info{i};
        pks = info.pks;
        locs = info.locs;
        w = info.w;
        p = info.p;
        pos_peaks = info.pos_peaks;
        pos_prominences = info.pos_prominences;

        % No peak at sample 20: structural failure. Properties tied to the main
        % feature cannot be calculated, so skip the rest.
        if ~info.has_peak_at_20
            fail_structural(i) = true;
            structural_reason{i} = 'no_peak_at_sample20';
            continue;
        end

        % Get main peak index
        is_main_feature = (locs == sample_20_idx);
        main_peak_idx = find(is_main_feature, 1, 'first');

        % Extract main peak metrics
        main_pk_amp = pks(main_peak_idx);
        main_pk_width = calc_baseline_width(waveform, sample_20_idx);
        main_pk_prominence = p(main_peak_idx);
        width(i) = main_pk_width;
        prominence_sample_20(i) = main_pk_prominence;

        % Calculate positive peaks for properties (already computed in first pass)
        if ~isempty(pos_peaks)
            [peak_pos_max(i), max_pos_amp_idx] = max(pos_peaks);
            prominence_pos_max(i) = pos_prominences(max_pos_amp_idx);
        end

        % DECISION 2 & 3: Peak count and secondary prominence
        if length(pks) == 1
            % Single peak: proceed to width check

        elseif length(pks) > 1
            % Multi-peak case
            other_prominences_arr = p(~is_main_feature);
            other_peaks_arr = locs(~is_main_feature);

            if ~isempty(other_prominences_arr)
                [max_other_prominence, max_prom_idx_local] = max(other_prominences_arr);
                other_prominence(i) = max_other_prominence;
                other_peak_loc(i) = other_peaks_arr(max_prom_idx_local);

                % DECISION 3a: Check if secondary < 1% of main peak amplitude
                if max_other_prominence >= (par.prominence_ratio_threshold * main_pk_amp)
                    % Secondary is significant: check prominence ratio
                    if max_other_prominence > 0
                        prominence_ratio(i) = main_pk_prominence / max_other_prominence;
                    else
                        prominence_ratio(i) = inf;
                    end

                    % DECISION 3b: Prominence ratio must be > 2 to proceed
                    % Allow NaN ratios (exceptions: single-peak, weak secondary) and computed ratios > 2
                    if isnan(prominence_ratio(i)) || prominence_ratio(i) > par.final_prominence_ratio_pass
                        % Ratio is good (or NaN exception), continue to width check
                    else
                        % Ratio is bad
                        fail_prom_ratio(i) = true;
                    end
                end
            end
        end

        % DECISION 4: Width check
        if isnan(width(i)) || width(i) < par.min_width_idx || width(i) > par.max_width_idx
            fail_width(i) = true;
        end
    end

    % RETROACTIVE: mark ALL spikes whose main-peak amplitude is below the low
    % percentile, regardless of any other failure. This is the 'amp' detection.
    abs_peak_amps = abs(peak_sample_20);
    valid_peak_amps = abs_peak_amps(isfinite(abs_peak_amps));
    if ~isempty(valid_peak_amps)
        low_amp_threshold = prctile(valid_peak_amps, par.min_amplitude_percentile);
        low_low_amp_spike = abs(peak_sample_20) < low_amp_threshold;
        low_low_amp_spike(~isfinite(low_low_amp_spike)) = false;
    end
    fail_amp = logical(low_low_amp_spike(:));

    % MULTI-PEAK detection: fail when the larger of the left/right opposite-polarity
    % LOCAL-PEAK ratios exceeds the threshold (single-side reject). NaN ratios (no
    % opposite-polarity peak on that side) are treated as "no evidence" -> pass.
    worst_side_ratio = max([left_peak_ratio(:), right_peak_ratio(:)], [], 2);
    fail_multipeak = worst_side_ratio > multipeak_ratio_threshold;
    fail_multipeak(~isfinite(worst_side_ratio)) = false;
    fail_multipeak = logical(fail_multipeak(:));

    quarantine_mask = fail_structural | fail_amp | fail_width | fail_prom_ratio | fail_multipeak;

    % DEBUG TRACKING
    n_single_peak = sum(num_peaks_arr == 1);
    n_multi_peak = sum(num_peaks_arr > 1);
    n_nan_ratio = sum(isnan(prominence_ratio));
    n_inf_ratio = sum(isinf(prominence_ratio));
    n_good_ratio = sum(prominence_ratio > par.final_prominence_ratio_pass & isfinite(prominence_ratio));
    n_bad_ratio = sum(prominence_ratio <= par.final_prominence_ratio_pass & isfinite(prominence_ratio));
    n_nan_width = sum(isnan(width));
    n_good_width = sum(width >= par.min_width_idx & width <= par.max_width_idx);
    n_bad_width = sum((width < par.min_width_idx | width > par.max_width_idx) & ~isnan(width));

    % fprintf('\n[DEBUG] Decision tree stats:\n');
    % fprintf('  Single-peak spikes: %d\n', n_single_peak);
    % fprintf('  Multi-peak spikes: %d\n', n_multi_peak);
    % fprintf('  Ratios: NaN=%d, Inf=%d, Good(>%.1f)=%d, Bad(<=%.1f)=%d\n', ...
    %     n_nan_ratio, n_inf_ratio, par.final_prominence_ratio_pass, n_good_ratio, par.final_prominence_ratio_pass, n_bad_ratio);
    % fprintf('  Width: NaN=%d, Good=[%.1f-%.1f]=%d, Bad=%d\n', ...
    %     n_nan_width, par.min_width_idx, par.max_width_idx, n_good_width, n_bad_width);
    % fprintf('  Total quarantined: %d / %d\n\n', sum(quarantine_mask), num_spikes);

    quarantine_properties = assemble_properties( ...
        prominence_ratio, num_peaks_arr, prominence_sample_20, other_prominence, width, ...
        peak_sample_20, other_peak_loc, peak_pos_max, prominence_pos_max, low_low_amp_spike, ...
        left_peak_amp, left_peak_loc, left_peak_ratio, right_peak_amp, right_peak_loc, right_peak_ratio, ...
        max_left_amp, max_left_loc, max_left_ratio, max_right_amp, max_right_loc, max_right_ratio, ...
        fail_structural, fail_amp, fail_width, fail_prom_ratio, fail_multipeak, structural_reason);
end


function qp = assemble_properties( ...
        prominence_ratio, num_peaks_arr, prominence_sample_20, other_prominence, width, ...
        peak_sample_20, other_peak_loc, peak_pos_max, prominence_pos_max, low_low_amp_spike, ...
        left_peak_amp, left_peak_loc, left_peak_ratio, right_peak_amp, right_peak_loc, right_peak_ratio, ...
        max_left_amp, max_left_loc, max_left_ratio, max_right_amp, max_right_loc, max_right_ratio, ...
        fail_structural, fail_amp, fail_width, fail_prom_ratio, fail_multipeak, structural_reason)

    qp = struct();

    % --- legacy per-spike metrics (unchanged) ---
    qp.prominence_ratio = prominence_ratio(:);
    qp.num_peaks = num_peaks_arr(:);
    qp.prominence_sample_20 = prominence_sample_20(:);
    qp.other_prominence = other_prominence(:);
    qp.width = width(:);
    qp.peak_sample_20 = peak_sample_20(:);
    qp.other_peak_loc = other_peak_loc(:);
    qp.peak_pos_max = peak_pos_max(:);
    qp.prominence_pos_max = prominence_pos_max(:);
    qp.low_low_amp_spike = low_low_amp_spike(:);

    % --- bilateral opposite-polarity descriptors (new) ---
    % left_peak_*  : largest opposite-polarity LOCAL PEAK strictly left of sample 20
    % right_peak_* : same, strictly right of sample 20
    % max_left_*   : most-opposite-polarity SAMPLE (any sample) left of sample 20
    % max_right_*  : same, right of sample 20
    % _amp = signed waveform value; _loc = 1-based sample index (-1 = undefined);
    % _ratio = |amp| / (max(waveform) - min(waveform)).
    qp.left_peak_amp = left_peak_amp(:);
    qp.left_peak_loc = left_peak_loc(:);
    qp.left_peak_ratio = left_peak_ratio(:);
    qp.right_peak_amp = right_peak_amp(:);
    qp.right_peak_loc = right_peak_loc(:);
    qp.right_peak_ratio = right_peak_ratio(:);
    qp.max_left_amp = max_left_amp(:);
    qp.max_left_loc = max_left_loc(:);
    qp.max_left_ratio = max_left_ratio(:);
    qp.max_right_amp = max_right_amp(:);
    qp.max_right_loc = max_right_loc(:);
    qp.max_right_ratio = max_right_ratio(:);

    % --- per-check fail masks (new) ---
    qp.fail_structural = logical(fail_structural(:));
    qp.fail_amp = logical(fail_amp(:));
    qp.fail_width = logical(fail_width(:));
    qp.fail_prom_ratio = logical(fail_prom_ratio(:));
    qp.fail_multipeak = logical(fail_multipeak(:));
    qp.structural_reason = structural_reason(:);

    % 'reason' is filled in by the caller (compose_reasons) so it reflects only the
    % active detection set; default here to the all-checks first-failing reason.
    qp.reason = compose_reasons(qp, {'structural', 'amp', 'prom_ratio', 'width', 'multipeak'});
end


function d = opposite_polarity_descriptors(w, neg_locs, pos_locs, target)
    % Port of Python _peak_metrics_per_spike: four directional descriptors of
    % opposite-polarity activity around the main peak at sample `target`.
    %
    % w        : 1xN waveform.
    % neg_locs : 1-based locations of findpeaks(-w)  (negative-going local peaks).
    % pos_locs : 1-based locations of findpeaks(w)   (positive-going local peaks).
    % target   : 1-based index of the main peak (sample 20).
    %
    % For a canonical negative-going spike (w(target) < 0) "opposite polarity" is
    % positive-going; for an inverted spike (w(target) > 0) it is negative-going.
    %
    % Returns struct with fields {left_peak,right_peak,max_left,max_right} x
    % {_amp,_loc,_ratio}. _loc uses -1 as the undefined sentinel; _amp/_ratio use NaN.

    d = struct('left_peak_amp', NaN, 'left_peak_loc', -1, 'left_peak_ratio', NaN, ...
               'right_peak_amp', NaN, 'right_peak_loc', -1, 'right_peak_ratio', NaN, ...
               'max_left_amp', NaN, 'max_left_loc', -1, 'max_left_ratio', NaN, ...
               'max_right_amp', NaN, 'max_right_loc', -1, 'max_right_ratio', NaN);

    n = numel(w);
    if n < target
        return;
    end
    main_amp = w(target);
    if ~isfinite(main_amp) || main_amp == 0
        return;
    end
    main_negative = main_amp < 0;

    % Opposite-polarity LOCAL-PEAK set for left_peak / right_peak.
    if main_negative
        opp = pos_locs(w(pos_locs) > 0);
    else
        cands = neg_locs(neg_locs ~= target);
        if isempty(cands)
            opp = cands;
        else
            opp = cands(w(cands) < 0);
        end
    end
    opp = opp(:).';

    left_locs = opp(opp < target);
    right_locs = opp(opp > target);

    [d.left_peak_loc, d.left_peak_amp] = best_abs_peak(w, left_locs);
    [d.right_peak_loc, d.right_peak_amp] = best_abs_peak(w, right_locs);

    % Most-opposite-polarity SAMPLE on each side (any sample, not just local peaks).
    [d.max_left_loc, d.max_left_amp] = best_opposite_sample(w(1:target-1), 1, main_negative);
    [d.max_right_loc, d.max_right_amp] = best_opposite_sample(w(target+1:end), target+1, main_negative);

    p2p = max(w) - min(w);
    d.left_peak_ratio = amp_ratio(d.left_peak_amp, p2p);
    d.right_peak_ratio = amp_ratio(d.right_peak_amp, p2p);
    d.max_left_ratio = amp_ratio(d.max_left_amp, p2p);
    d.max_right_ratio = amp_ratio(d.max_right_amp, p2p);
end


function [loc, amp] = best_abs_peak(w, locs)
    % Largest-|amplitude| entry among `locs` (1-based indices into w).
    loc = -1;
    amp = NaN;
    if isempty(locs)
        return;
    end
    amps = w(locs);
    [~, j] = max(abs(amps));
    loc = locs(j);
    amp = amps(j);
end


function [loc, amp] = best_opposite_sample(slice_, abs_start, neg_main)
    % Most-opposite-polarity sample within `slice_`. abs_start is the 1-based
    % absolute index of slice_(1). Returns (-1, NaN) when no opposite-polarity
    % sample exists on that side.
    loc = -1;
    amp = NaN;
    if isempty(slice_)
        return;
    end
    if neg_main
        if ~any(slice_ > 0)
            return;
        end
        [v, j] = max(slice_);
        if v <= 0
            return;
        end
    else
        if ~any(slice_ < 0)
            return;
        end
        [v, j] = min(slice_);
        if v >= 0
            return;
        end
    end
    loc = j + abs_start - 1;
    amp = v;
end


function r = amp_ratio(amp, p2p)
    if p2p > 0 && isfinite(amp)
        r = abs(amp) / p2p;
    else
        r = NaN;
    end
end


function width_val = calc_baseline_width(waveform, peak_idx)
    % Measure width using a baseline-to-peak half-height rule.
    width_val = nan;

    if numel(waveform) < peak_idx
        return;
    end


    n = numel(waveform);
    peak_voltage = waveform(peak_idx);

    % baseline computed from first/last up-to-5 samples (matches Python behavior)
    h = min(5, n);
    first_seg = waveform(1:h);
    last_seg = waveform(max(1, n-h+1):n);
    baseline = mean([first_seg, last_seg]);

    % The crossing search below only works correctly when the main feature is a
    % trough (peak_voltage < baseline): waveform(peak_idx) fails the '>' test,
    % so the search correctly walks outward to the true half-height crossings.
    % For a genuine positive-going peak, waveform(peak_idx) trivially satisfies
    % '>', which collapses the search onto the peak's own sample and returns
    % NaN. Normalize to trough-shape here so the same logic works either way.
    if peak_voltage > baseline
        waveform = -waveform;
        peak_voltage = -peak_voltage;
        baseline = -baseline;
    end

    half_amplitude = baseline + (peak_voltage - baseline) / 2;

    peak_idx_0based = peak_idx - 1;  % Convert 1-based peak_idx to 0-based

    % LEFT crossing: find last sample > half before the peak
    left_idx = nan;
    left_candidates = find(waveform(1:peak_idx) > half_amplitude);
    if ~isempty(left_candidates)
        ci_1based = left_candidates(end);  % 1-based MATLAB index
        ci_0based = ci_1based - 1;         % Convert to 0-based


        if (ci_0based < peak_idx_0based) && (ci_1based + 1 <= peak_idx)
            y0 = waveform(ci_1based + 1);
            y1 = waveform(ci_1based);
            x0 = ci_0based + 1;
            x1 = ci_0based;
            left_idx = interp1([y0, y1], [x0, x1], half_amplitude, 'linear', 'extrap');
        end
    end

    % RIGHT crossing: find first sample > half at/after the peak
    right_idx = nan;
    seg = waveform(peak_idx:end);
    right_candidates = find(seg > half_amplitude);
    if ~isempty(right_candidates)
        ci_local_1based = right_candidates(1);  % 1-based index within waveform(peak_idx:end)
        ci_local_0based = ci_local_1based - 1;
        ci_global_0based = peak_idx_0based + ci_local_0based;  % 0-based global index
        ci_global_1based = ci_global_0based + 1;

        if (ci_global_0based - 1 >= 0) && (ci_global_1based <= n)
            y_vals = waveform(ci_global_1based - 1 : ci_global_1based);
            x_vals = [ci_global_0based - 1, ci_global_0based];
            right_idx = interp1(y_vals, x_vals, half_amplitude, 'linear', 'extrap');
        end
    end

    if ~isnan(left_idx) && ~isnan(right_idx)
        width_val = right_idx - left_idx;
    end
end
