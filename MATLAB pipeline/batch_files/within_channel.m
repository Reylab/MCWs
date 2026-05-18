function within_channel(channels)
    % Function: within_channel
    % Description: Filters spikes based on waveform characteristics (amplitude, width, multi-peak structure)
    %              using a robust, polarity-aware method. The resulting mask is combined with an existing
    %              collision mask for cumulative filtering.
    % Channels: The list of microelectrode channels (channel IDs) to process.
    

    within_channel_tic = tic;
    
    % Define Quality Control (QC) parameters for spike shape analysis
    par.qc_params = struct();
    par.qc_params.min_amplitude_percentile = 5; % Spikes below this P2P amplitude percentile are quarantined
    par.qc_params.min_width_idx = 3;            % Min width of main feature (in samples)
    par.qc_params.max_width_idx = 15;           % Max width of main feature (in samples)
    par.qc_params.prominence_ratio_threshold = 0.01; % Secondary feature prominence must be > 1% of main peak amp
    par.qc_params.final_prominence_ratio_pass = 2; % Main feature Prominence/Amplitude threshold for complex spikes
    
    load('NSx','NSx');
    % Filter NSx structure to include only the specified channels
    NSx_proc = NSx(ismember(cell2mat({NSx.chan_ID}),channels));
    
    num_channels_proc = length(NSx_proc);
    fprintf('Starting robust waveform quality check on %d channels...\n', num_channels_proc);
    
    for k = 1:num_channels_proc
        ch_info = NSx_proc(k);
        ch_lbl = ch_info.output_name;
        spike_file = sprintf('%s_spikes.mat', ch_lbl);
                
        try
            fprintf('ch.%d/%d %s: loading %s\n', k, num_channels_proc, ch_lbl, spike_file);
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

            % mask_quarantine_local is TRUE for spikes that FAIL the shape/amplitude QC test
            fprintf('ch.%d/%d %s: running waveform QC\n', k, num_channels_proc, ch_lbl);
            [mask_quarantine_local, quarantine_properties] = analyze_spike_waveforms(spikes_all, par.qc_params);
            mask_quarantine_local = logical(mask_quarantine_local(:));
            
            % mask_non_quarantine is TRUE for spikes that PASS the shape/amplitude QC test
            mask_non_quarantine = ~mask_quarantine_local;

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

            fprintf('ch.%d/%d %s: saving filtered results\n', k, num_channels_proc, ch_lbl);
            fprintf('  -> Quarantined: %d\n', nnz(~mask_non_quarantine));
            % Remove -append to fully overwrite file, ensuring old unfiltered spikes don't persist
            save(spike_file, ...
                 'index', 'spikes', 'index_all', 'spikes_all', 'par', 'mask_nonart', ...
                 'mask_non_quarantine', 'quarantine_properties','-append');
            
            num_removed_this_step = sum(mask_non_collision) - sum(mask_total_pass);
            num_total_spikes = numel(index_all);
            
            fprintf('ch.%d of %d: %s. Masks used (%d): quarantined %d spikes. Remaining: %d/%d (%.2f%%)\n', ...
                k, num_channels_proc, ch_lbl, mask_used, num_removed_this_step, sum(mask_total_pass), num_total_spikes, sum(mask_total_pass)/num_total_spikes*100);

            % Release channel-local data before next channel.
            clear SPK spikes_all index_all index spikes mask_non_collision mask_nonart mask_non_quarantine mask_taskspks mask_total_pass quarantine_properties mask_quarantine_local;

        catch ME
            fprintf('  -> FAILED to process channel %s: %s\n', ch_lbl, ME.message);
        end
    end
    
    within_channel_toc = toc(within_channel_tic);
    fprintf("within_channel DONE in %s seconds.\n", num2str(within_channel_toc, '%2.2f'));
end


function [quarantine_mask, quarantine_properties] = analyze_spike_waveforms(spikes, par)
    % Analyzes spikes using decision tree: amplitude -> peak count -> prominence -> width.
    % Single-pass vectorized processing for efficiency with large spike counts.
    
    num_spikes = size(spikes, 1);
    sample_20_idx = 20;

    quarantine_mask = false(num_spikes, 1);

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
    
    % Store peak info for each spike to avoid recalculating findpeaks
    peak_info = cell(num_spikes, 1);

    if size(spikes, 2) < sample_20_idx
        warning('ArtifactRemoval:ShortWaveform', ...
            'Waveform length (%d) is shorter than sample_20_idx (%d). Quarantining all spikes.', ...
            size(spikes, 2), sample_20_idx);
        quarantine_mask(:) = true;
        quarantine_properties = struct();
        quarantine_properties.prominence_ratio = prominence_ratio(:);
        quarantine_properties.num_peaks = num_peaks_arr(:);
        quarantine_properties.prominence_sample_20 = prominence_sample_20(:);
        quarantine_properties.other_prominence = other_prominence(:);
        quarantine_properties.width = width(:);
        quarantine_properties.peak_sample_20 = peak_sample_20(:);
        quarantine_properties.other_peak_loc = other_peak_loc(:);
        quarantine_properties.peak_pos_max = peak_pos_max(:);
        quarantine_properties.prominence_pos_max = prominence_pos_max(:);
        quarantine_properties.low_low_amp_spike = low_low_amp_spike(:);
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
        [pos_peaks, ~, ~, pos_prominences] = findpeaks(waveform);
        
        % Store peak info and basic metrics
        peak_info{i} = struct('pks', pks, 'locs', locs, 'w', w, 'p', p, ...
                              'pos_peaks', pos_peaks, 'pos_prominences', pos_prominences, ...
                              'has_peak_at_20', any(locs == sample_20_idx), ...
                              'peak_sample_20_raw', sample_20_value);
        
        num_peaks_arr(i) = length(pks);
        
        % Pre-identify spikes with no peaks
        if isempty(pks)
            quarantine_mask(i) = true;
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
        quarantine_mask(:) = true;
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
        
        % No peak at sample 20: quarantine
        if ~info.has_peak_at_20
            % We skip entirely here because properties tied to the main feature cannot be calculated
            quarantine_mask(i) = true;
            continue;
        end
        
        % DECISION 1: Amplitude threshold check
        if abs(peak_sample_20(i)) < amp_threshold
            quarantine_mask(i) = true;
            % Do NOT continue here - we still want to calculate and save the properties
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
                        % Ratio is bad, quarantine
                        quarantine_mask(i) = true;
                        % Do NOT continue here - we still want to finish width checks
                    end
                end
            end
        end
        
        % DECISION 4: Width check (final test)
        % Preserve if width is valid (NaN ratio from exceptions already passed here)
        if isnan(width(i)) || width(i) < par.min_width_idx || width(i) > par.max_width_idx
            quarantine_mask(i) = true;
        end
    end
    
    % Handle NaN values
    low_low_amp_spike(~isfinite(low_low_amp_spike)) = false;
    
    % RETROACTIVE: Mark ALL spikes that are low-amplitude, regardless of quarantine reason
    % This matches Python's behavior of flagging amplitude separately from quarantine reason
    abs_peak_amps = abs(peak_sample_20);
    valid_peak_amps = abs_peak_amps(isfinite(abs_peak_amps));
    if ~isempty(valid_peak_amps)
        low_amp_threshold = prctile(valid_peak_amps, par.min_amplitude_percentile);
        low_low_amp_spike = abs(peak_sample_20) < low_amp_threshold;
        low_low_amp_spike(~isfinite(low_low_amp_spike)) = false;
    end

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
    
    fprintf('\n[DEBUG] Decision tree stats:\n');
    fprintf('  Single-peak spikes: %d\n', n_single_peak);
    fprintf('  Multi-peak spikes: %d\n', n_multi_peak);
    fprintf('  Ratios: NaN=%d, Inf=%d, Good(>%.1f)=%d, Bad(<=%.1f)=%d\n', ...
        n_nan_ratio, n_inf_ratio, par.final_prominence_ratio_pass, n_good_ratio, par.final_prominence_ratio_pass, n_bad_ratio);
    fprintf('  Width: NaN=%d, Good=[%.1f-%.1f]=%d, Bad=%d\n', ...
        n_nan_width, par.min_width_idx, par.max_width_idx, n_good_width, n_bad_width);
    fprintf('  Total quarantined: %d / %d\n\n', sum(quarantine_mask), num_spikes);

    quarantine_properties = struct();
    quarantine_properties.prominence_ratio = prominence_ratio(:);
    quarantine_properties.num_peaks = num_peaks_arr(:);
    quarantine_properties.prominence_sample_20 = prominence_sample_20(:);
    quarantine_properties.other_prominence = other_prominence(:);
    quarantine_properties.width = width(:);
    quarantine_properties.peak_sample_20 = peak_sample_20(:);
    quarantine_properties.other_peak_loc = other_peak_loc(:);
    quarantine_properties.peak_pos_max = peak_pos_max(:);
    quarantine_properties.prominence_pos_max = prominence_pos_max(:);
    quarantine_properties.low_low_amp_spike = low_low_amp_spike(:);
end


function width_val = calc_baseline_width(waveform, peak_idx)
    % Measure width using a baseline-to-peak half-height rule.
    % This mirrors the Python helper instead of using findpeaks' built-in width.
    width_val = nan;

    if numel(waveform) < peak_idx
        return;
    end

    % Use the same 0-based interpolation approach as the Python implementation
    n = numel(waveform);
    peak_voltage = waveform(peak_idx);

    % baseline computed from first/last up-to-5 samples (matches Python behavior)
    h = min(5, n);
    first_seg = waveform(1:h);
    last_seg = waveform(max(1, n-h+1):n);
    baseline = mean([first_seg, last_seg]);

    half_amplitude = baseline + (peak_voltage - baseline) / 2;

    % Match Python's width calculation exactly
    peak_idx_0based = peak_idx - 1;  % Convert 1-based peak_idx to 0-based
    
    % LEFT crossing: find last sample > half before the peak
    left_idx = nan;
    left_candidates = find(waveform(1:peak_idx) > half_amplitude);
    if ~isempty(left_candidates)
        ci_1based = left_candidates(end);  % 1-based MATLAB index
        ci_0based = ci_1based - 1;         % Convert to 0-based
        % Python: y0=w[ci+1], y1=w[ci]; x0=ci+1, x1=ci
        % MATLAB: y0=waveform(ci_1based+1), y1=waveform(ci_1based); x0=ci_0based+1, x1=ci_0based
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
        ci_global_1based = ci_global_0based + 1;  % Convert to 1-based for MATLAB access
        % Python: y_vals=w[ci_global-1:ci_global+1]; x_vals=[ci_global-1, ci_global]
        % MATLAB: waveform(ci_global_1based-1:ci_global_1based) gets 0-based positions [ci_global_0based-1, ci_global_0based]
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