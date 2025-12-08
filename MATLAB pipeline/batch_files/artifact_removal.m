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
    par.qc_params.final_prominence_ratio_pass = 0.8; % Main feature Prominence/Amplitude threshold for complex spikes
    
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
            
            % map the loaded mask back to the original variable name for saving
            mask_nonart = mask_non_collision;

            % mask_quarantine_local is TRUE for spikes that FAIL the shape/amplitude QC test
            mask_quarantine_local = analyze_spike_waveforms(spikes_all, par.qc_params);
            
            % mask_non_quarantine is TRUE for spikes that PASS the shape/amplitude QC test
            mask_non_quarantine = ~mask_quarantine_local;
            
            % Combine Masks: Spike must pass collision check AND quarantine check
            mask_total_pass = mask_non_collision & mask_non_quarantine;

            % Final cleaned indices
            index = index_all(mask_total_pass);
            % Final cleaned waveforms (overwriting 'spikes_coll_only' to hold the fully filtered set)
            spikes = spikes_all(mask_total_pass, :);
            
            % Update the main 'par' structure with the new QC parameters
            par = SPK.par;

            par.qc_params = struct();
            par.qc_params.min_amplitude_percentile = 5; % Spikes below this P2P amplitude percentile are quarantined
            par.qc_params.min_width_idx = 3;            % Min width of main feature (in samples)
            par.qc_params.max_width_idx = 15;           % Max width of main feature (in samples)
            par.qc_params.prominence_ratio_threshold = 0.01; % Secondary feature prominence must be > 1% of main peak amp
            par.qc_params.final_prominence_ratio_pass = 0.8; % Main feature Prominence/Amplitude threshold for complex spikes
            
            % Save the data, matching the required output variables plus the new masks
            save(sprintf('%s_spikes.mat', ch_lbl), ...
                 "index", "spikes", "index_all", "spikes_all", "par", "mask_nonart", ...
                 "mask_non_quarantine", "-append") 
            
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


function [quarantine_mask] = analyze_spike_waveforms(spikes, par)
    % Analyzes a matrix of spikes using polarity-aware prominence and width.
    % Returns a logical mask (quarantine_mask) where TRUE means the spike is an artifact.
    
    num_spikes = size(spikes, 1);
    quarantine_mask = false(1,num_spikes);
    amplitudes = max(spikes, [], 2) - min(spikes, [], 2);
    
    % Test 1: Amplitude Threshold (Quarantine if P2P amplitude is too low)
    amp_threshold = prctile(amplitudes, par.min_amplitude_percentile);
    is_low_amplitude = amplitudes < amp_threshold;
    quarantine_mask(is_low_amplitude) = true;
    
    spikes_to_check = find(~quarantine_mask);
    
    for i = 1:length(spikes_to_check)
        idx = spikes_to_check(i);
        waveform = spikes(idx, :);
        
        % 1. Determine Polarity (Identify the main extremum: max magnitude point)
        [min_val, min_idx] = min(waveform);
        [max_val, max_idx] = max(waveform);
        
        if abs(min_val) >= abs(max_val)
            % Primary feature is a trough (negative deflection). Analyze -waveform to treat trough as a positive peak.
            signal_for_analysis = -waveform;
            main_extremum_index = min_idx;
        else
            % Primary feature is a peak (positive deflection). Analyze +waveform.
            signal_for_analysis = waveform;
            main_extremum_index = max_idx;
        end
        
        % 2. Get Features using findpeaks on the signed signal
        [pks, locs, w, p] = findpeaks(signal_for_analysis); 
        
        if isempty(pks)
            quarantine_mask(idx) = true; 
            continue;
        end
        
        % Find the primary peak within the peaks list
        is_main_feature = (locs == main_extremum_index);
        main_peak_idx = find(is_main_feature, 1, 'first');
        
        % Fallback for cases where the main extremum is not detected as a peak
        if isempty(main_peak_idx)
             [~, main_peak_idx] = max(pks);
        end
        
        main_pk_amp = pks(main_peak_idx);
        main_pk_width = w(main_peak_idx);
        main_pk_prominence = p(main_peak_idx);
        
        
        % Test 2: Single Peak vs. Multi-Peak Analysis
        if length(pks) == 1
            % Single Peak: Quarantine if width is outside the desired range
            if main_pk_width < par.min_width_idx || main_pk_width > par.max_width_idx
                quarantine_mask(idx) = true;
            end
            
        elseif length(pks) > 1 % Multi-Peak Case (length(pks) > 1)
            
            % Test 3 (Sub-test): Check if any secondary peak is negligible (prominence < 1% of main amp)
            secondary_peaks_p = p;
            secondary_peaks_p(main_peak_idx) = 0; % Ignore the main feature's prominence
            prominence_ratio_check = secondary_peaks_p ./ main_pk_amp;
            
            % If secondary peak prominence is low, treat the spike as single-peaked and run width test
            if any(prominence_ratio_check < par.prominence_ratio_threshold)
                if main_pk_width < par.min_width_idx || main_pk_width > par.max_width_idx
                    quarantine_mask(idx) = true;
                end
            else
                % Test 4: Final Prominence Ratio and Width Test (for truly complex, multi-peaked shapes)
                
                has_desired_width = main_pk_width >= par.min_width_idx && main_pk_width <= par.max_width_idx;
                
                % Check if width is sufficient AND Prominence/Amplitude is high enough to pass
                final_prominence_ratio_pass = (main_pk_prominence / main_pk_amp) > par.final_prominence_ratio_pass;
                
                if ~(has_desired_width && final_prominence_ratio_pass)
                    quarantine_mask(idx) = true; % Fails width or final prominence test
                end
            end
        end
    end
end