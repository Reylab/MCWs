function refract_viol(channels)
    % Function: refract_viol
    % Description: Filters spikes that violate the refractory period.
    %              The refractory parameters (ref_ms or ref) mimic those used
    %              in amp_detect.m. The resulting mask is combined with existing
    %              masks for cumulative filtering.
    % Channels: The list of microelectrode channels (channel IDs) to process.
    
    refract_viol_tic = tic;
    
    load('NSx','NSx');
    % Filter NSx structure to include only the specified channels
    NSx_proc = NSx(ismember(cell2mat({NSx.chan_ID}),channels));
    
    num_channels_proc = length(NSx_proc);
    fprintf('Starting refractory violation check on %d channels...\n', num_channels_proc);
    
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
                index_all  = SPK.index;
            end
            
            par = SPK.par;
            
            % Determine refractory period in milliseconds (mimicking amp_detect_old.m logic)
            if isfield(par, 'ref_ms')
                ref_val = par.ref_ms;
            elseif isfield(par, 'ref')
                % par.ref was historically in samples, convert to ms
                ref_val = par.ref / (par.sr / 1000);
            else
                ref_val = 1.5; % Default fallback to 1.5ms
            end
            
            % Identify violations mirroring the chronological check in amp_detect_old.m
            mask_refract = false(size(index_all));
            last_accepted = -inf;
            
            for i = 1:length(index_all)
                % Check if spike is within the refractory period of the last accepted spike
                if index_all(i) < last_accepted + ref_val
                    mask_refract(i) = true;
                else
                    last_accepted = index_all(i);
                end
            end
            
            mask_non_refract = ~mask_refract;
            
            % Load existing masks to create a cumulative mask
            if isfield(SPK, 'mask_nonart')
                mask_nonart = SPK.mask_nonart;
            else
                mask_nonart = true(size(index_all));
            end
            
            if isfield(SPK, 'mask_non_quarantine')
                mask_non_quarantine = SPK.mask_non_quarantine;
            else
                mask_non_quarantine = true(size(index_all));
            end
            
            if isfield(SPK, 'mask_taskspks')
                mask_taskspks = SPK.mask_taskspks;
            else
                mask_taskspks = true(size(index_all));
            end
            
            % Apply previous and new mask
            mask_tot = mask_nonart & mask_non_quarantine & mask_taskspks & mask_non_refract;
            
            spikes = spikes_all(mask_tot, :);
            index = index_all(mask_tot);
            
            save(spike_file, ...
                 "index", "spikes", "index_all", "spikes_all", "par", ...
                 "mask_non_refract", "-append");
            
            fprintf('  -> %d/%d spikes flagged as refractory violations (%.2f%%)\n', ...
                sum(mask_refract), length(index_all), (sum(mask_refract)/length(index_all))*100);
                
        catch ME
            fprintf('  -> FAILED to process channel %s: %s\n', ch_lbl, ME.message);
        end
    end
    
    refract_viol_toc = toc(refract_viol_tic);
    fprintf("refract_viol DONE in %s seconds.\n", num2str(refract_viol_toc, '%2.2f'));
end