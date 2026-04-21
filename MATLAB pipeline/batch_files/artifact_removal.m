function artifact_removal(channels)
    % Function: artifact_removal
    % Description: Filters spikes based on waveform characteristics (amplitude, width, multi-peak structure)
    %              using a robust, polarity-aware method. The resulting mask is combined with an existing
    %              collision mask for cumulative filtering.
    % Channels: The list of microelectrode channels (channel IDs) to process.
    
    artifact_removal_tic = tic;
    
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
                
        try
            SPK = load(sprintf('%s_spikes.mat', ch_lbl));
            
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

            % map the loaded mask back to the original variable name for saving
            mask_nonart = mask_non_collision;

            % mask_quarantine_local is TRUE for spikes that FAIL the shape/amplitude QC test
            [mask_quarantine_local, quarantine_properties] = analyze_spike_waveforms(spikes_all, par.qc_params);
            
            % mask_non_quarantine is TRUE for spikes that PASS the shape/amplitude QC test
            mask_non_quarantine = ~mask_quarantine_local;
            
            % Combine Masks: Spike must pass collision check AND quarantine check
            mask_total_pass = mask_non_collision & mask_non_quarantine & mask_taskspks;

            % Final cleaned indices
            index = index_all(mask_total_pass);
            % Final cleaned waveforms (overwriting 'spikes_coll_only' to hold the fully filtered set)
            spikes = spikes_all(mask_total_pass, :);
            
            % Update the main 'par' structure with the new QC parameters
            par = SPK.par;

            % Save updated masks and quarantine properties
            save(sprintf('%s_spikes.mat', ch_lbl), ...
                 "index", "spikes", "index_all", "spikes_all", "par", "mask_nonart", ...
                 "mask_non_quarantine", "quarantine_properties", "-append") 
            
            num_removed_this_step = sum(mask_non_collision) - sum(mask_total_pass);
            num_total_spikes = numel(index_all);
            
            fprintf('ch.%d of %d: %s. Masks used (%d): quarantined %d spikes. Remaining: %d/%d (%.2f%%)\n', ...
                k, num_channels_proc, ch_lbl, mask_used, num_removed_this_step, sum(mask_total_pass), num_total_spikes, sum(mask_total_pass)/num_total_spikes*100);

        catch ME
            fprintf('  -> FAILED to process channel %s: %s\n', ch_lbl, ME.message);
        end
    end
    
    artifact_removal_toc = toc(artifact_removal_tic);
    fprintf("artifact_removal DONE in %s seconds.\n", num2str(artifact_removal_toc, '%2.2f'));
end


function [quarantine_mask, quarantine_properties] = analyze_spike_waveforms(spikes, par)
    % Analyzes a matrix of spikes using polarity-aware prominence and width.
    % Returns a logical mask (quarantine_mask) where TRUE means the spike is an artifact.
    
    num_spikes = size(spikes, 1);
    sample_20_idx = 20;

    quarantine_mask = false(1,num_spikes);

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

    amplitudes = max(spikes, [], 2) - min(spikes, [], 2);
    
    % Test 1: Amplitude Threshold (Quarantine if P2P amplitude is too low)
    amp_threshold = prctile(amplitudes, par.min_amplitude_percentile);
    is_low_amplitude = amplitudes < amp_threshold;
    quarantine_mask(is_low_amplitude) = true;
    
    spikes_to_check = find(~quarantine_mask);
    
    for i = 1:numel(spikes_to_check)
        idx = spikes_to_check(i);
        waveform = spikes(idx, :);

        % Determine polarity from sample 20
        sample_20_value = waveform(sample_20_idx);

        if sample_20_value < 0
            % Trough at sample 20: invert so trough is analyzed as a peak.
            signal_for_analysis = -waveform;
        else
            % Peak at sample 20: analyze waveform directly.
            signal_for_analysis = waveform;
        end

        % Extract peaks, locations, widths, and prominences
        [pks, locs, w, p] = findpeaks(signal_for_analysis); 
        
        if isempty(pks)
            quarantine_mask(idx) = true; 
            continue;
        end
        
        % Find the primary peak within the peaks list
        is_main_feature = (locs == sample_20_idx);
        main_peak_idx = find(is_main_feature, 1, 'first');
        target_peak_found = any(is_main_feature);
        num_peaks_arr(idx) = length(pks);

        % Do not re-anchor to any other peak if sample 20 is not the main feature.
        if ~target_peak_found
            quarantine_mask(idx) = true;
            [other_prominence(idx), max_prom_idx_local] = max(p);
            other_peak_loc(idx) = locs(max_prom_idx_local);
            continue;
        end

        main_pk_amp = pks(main_peak_idx);
        main_pk_width = w(main_peak_idx);
        main_pk_prominence = p(main_peak_idx);
        width(idx) = main_pk_width;

        peak_sample_20(idx) = waveform(sample_20_idx);
        prominence_sample_20(idx) = main_pk_prominence;
        if num_peaks_arr(idx) > 1
            other_prominences_arr = p(~is_main_feature);
            other_peaks_arr = locs(~is_main_feature);

            if ~isempty(other_prominences_arr)
                [max_other_prominence, max_prom_idx_local] = max(other_prominences_arr);
                other_prominence(idx) = max_other_prominence;
                other_peak_loc(idx) = other_peaks_arr(max_prom_idx_local);

                if isnan(max_other_prominence)
                    prominence_ratio(idx) = nan;
                elseif max_other_prominence > 0
                    prominence_ratio(idx) = main_pk_prominence / max_other_prominence;
                else
                    prominence_ratio(idx) = inf;
                end
            end
        end

        % Calculate positive peaks and prominences for quarantine properties
        [pos_peaks, ~, ~, pos_prominences] = findpeaks(waveform);
        if ~isempty(pos_peaks)
            [peak_pos_max(idx), max_pos_amp_idx] = max(pos_peaks);
            prominence_pos_max(idx) = pos_prominences(max_pos_amp_idx);
        end

        % Test 2: Single Peak vs. Multi-Peak Analysis
        if length(pks) == 1
            % Single Peak: Quarantine if width is outside the desired range
            if main_pk_width < par.min_width_idx || main_pk_width > par.max_width_idx
                quarantine_mask(idx) = true;
            end
            
        elseif length(pks) > 1 % Multi-Peak Case (length(pks) > 1)

            % Secondary-peak logic:
            % 1) If largest secondary prominence is < 1% of main peak amplitude -> pass
            % 2) Else if main prominence / largest secondary prominence > 2 -> pass
            % 3) Otherwise -> quarantine
            secondary_peaks_p = p;
            secondary_peaks_p(main_peak_idx) = 0; % Ignore the main feature's prominence

            max_secondary_prominence = max(secondary_peaks_p);
            secondary_is_tiny = max_secondary_prominence < (par.prominence_ratio_threshold * main_pk_amp);

            if secondary_is_tiny
                quarantine_mask(idx) = false;
            else
                prominence_ratio_pass = (main_pk_prominence / max_secondary_prominence) > par.final_prominence_ratio_pass;
                if ~prominence_ratio_pass
                    quarantine_mask(idx) = true;
                end
            end
        end
    end

    valid_peaks = peak_sample_20(isfinite(peak_sample_20));
    if ~isempty(valid_peaks)
        low_amp_threshold = prctile(abs(valid_peaks), par.min_amplitude_percentile);
        low_low_amp_spike = abs(peak_sample_20) < low_amp_threshold;
    else
        low_low_amp_spike = false(num_spikes, 1);
    end
    low_low_amp_spike(~isfinite(low_low_amp_spike)) = false;

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