function rescue_spikes(channels, varargin)
% rescue_spikes - Attempts to reclassify quarantined spikes using template matching
% Inputs:
%   channels - vector of channel IDs
%   Optional: 'parallel', true/false, ...
% Usage:
%   rescue_spikes(channels, 'parallel', true)

% Parse optional arguments
parallel = false;
for v = 1:2:length(varargin)
    if strcmp(varargin{v}, 'parallel')
        parallel = varargin{v+1};
    end
end

fprintf('Starting rescue_spikes on %d channels...\n', length(channels));

if parallel
    parfor kk = 1:length(channels)
        process_channel_rescue(channels(kk));
    end
else
    for kk = 1:length(channels)
        process_channel_rescue(channels(kk));
    end
end

fprintf('rescue_spikes DONE.\n');
end

function process_channel_rescue(ch)
    try
        ch_lbl = get_channel_label(ch); % Helper to get output_name or label
        fname_spk = sprintf('%s_spikes.mat', ch_lbl);
        fname_times = sprintf('times_%s.mat', ch_lbl);
        SPK = load(fname_spk);
        spikes_all = SPK.spikes_all;
        index_all = SPK.index_all;


        if isfield(SPK,'mask_non_quarantine')
            mask_non_quarantine = SPK.mask_non_quarantine;
        else
            mask_non_quarantine = false(size(index_all));
        end

        if isfield(SPK,'mask_nonart')
            mask_non_collision = SPK.mask_nonart;
        else
            mask_non_collision = true(size(index_all));
        end
        
        if isfield(SPK,'mask_taskspks')
            mask_task = SPK.mask_taskspks;
        else
            mask_task = true(size(index_all));
        end
        
        
        par = SPK.par;
        index = SPK.index;

        % Find quarantined spikes: excluded by artifact, not by collision
        mask_quar = ~mask_non_quarantine & mask_non_collision & mask_task;
        if ~any(mask_quar)
            fprintf('  Channel %s: No quarantined spikes to rescue.\n', ch_lbl);
            return;
        end
        % Features for quarantined spikes
        spikes_quar = spikes_all(mask_quar, :);
        index_quar = index_all(mask_quar);
        % Load times file and get clustering info
        if exist(fname_times, 'file')
            S = load(fname_times);
            
            % SAVE PRE-RESCUE BACKUP (only if not already saved)
            if ~isfield(S, 'spikes_pre_rescue')
                spikes_pre_rescue = S.spikes;
                index_pre_rescue = index;
                cluster_class_pre_rescue = S.cluster_class;
                save(fname_times, 'spikes_pre_rescue', 'index_pre_rescue', ...
                     'cluster_class_pre_rescue', '-append');
            end
            
            cluster_class = S.cluster_class;
            spikes = S.spikes;
            coeff = S.coeff;
            inspk_good = S.inspk;
            class_good_mask = cluster_class(:,1) ~= 0;
            class_good = cluster_class(class_good_mask, 1);
        else %if no times file, means unclustered/assume multiunit
            spikes = SPK.spikes;
            coeff = 1:64; % default to first 64 coeffs
            inspk_good = local_wavelet_decomp(spikes,par.scales);
            class_good = ones(size(spikes,1),1);
        end
        % Use class_good as those with cluster_class ~= 0

          % Use force_membership_wc to assign clusters to quarantined spikes
        class_quar = force_membership_wc(spikes, class_good, spikes_quar, par);
        rescued_idx = find(class_quar ~= 0);
        if isempty(rescued_idx)
            fprintf('  Channel %s: No spikes rescued.\n', ch_lbl);
        else
            fprintf('  Channel %s: Rescued %d spikes.\n', ch_lbl, numel(rescued_idx));
            for i = 1:numel(rescued_idx)
               % fprintf('    Rescued spike at index %d (global index %d) assigned to cluster %d\n', ...
               %     rescued_idx(i), index_quar(rescued_idx(i)), class_quar(rescued_idx(i)));
            end
        end
        % Merge rescued spikes with original clustered spikes
        spikes_rescued = spikes_quar(rescued_idx, :);
        index_rescued = index_quar(rescued_idx);
        class_rescued = class_quar(rescued_idx)';
        inspk_all_coeff = local_wavelet_decomp(spikes_rescued,par.scales);
        inspk_rescued = inspk_all_coeff(:,coeff);
        % Combine
        index_combined = [index; index_rescued];
        spikes_combined = [spikes; spikes_rescued];
        class_combined = [cluster_class(:,1); class_rescued];
        inspk_combined = [inspk_good; inspk_rescued];
        cluster_class_combined = [class_combined, index_combined];
        % Sort by index
        [index_sorted, sort_idx] = sort(index_combined);
        spikes_sorted = spikes_combined(sort_idx, :);
        cluster_class_sorted = cluster_class_combined(sort_idx, :);
        inspk_sorted = inspk_combined(sort_idx, :);
        % Save in times format (same as do_clustering)
        % Save as apikes, index, inspk, cluster_class for consistency
        spikes = spikes_sorted;
        index = index_sorted;
        inspk = inspk_sorted;
        cluster_class = cluster_class_sorted;
        % Save rescued info and quarantined spikes/indices/classes
        spikes_quarantined = spikes_quar;
        index_quarantined = index_quar;
        class_quarantined = class_quar;
        save(fname_times, 'spikes', 'index', 'inspk', 'cluster_class', 'spikes_quarantined', 'index_quarantined', 'class_quarantined', '-append');
        % Also save rescue info in spikes file
        save(fname_times, 'class_quar', 'index_quar', 'rescued_idx', '-append');
    catch ME
        fprintf('  Channel %d: Error - %s\n', ch, ME.message);
    end
end

function ch_lbl = get_channel_label(ch)
     % Accepts either an integer channel number or a string filename
     if isnumeric(ch)
         files = dir(sprintf('*_%d_spikes.mat', ch));
         if ~isempty(files)
             [~, name, ~] = fileparts(files(1).name);
            % strip trailing '_spikes' if present to avoid duplicate suffix later
            if endsWith(name, '_spikes')
                ch_lbl = name(1:end-length('_spikes'));
            else
                ch_lbl = name;
            end
         else
             ch_lbl = num2str(ch); % fallback
         end
     elseif ischar(ch) || isstring(ch)
         % If input is a string, use it directly (strip _spikes.mat if present)
         [~, name, ~] = fileparts(char(ch));
        if endsWith(name, '_spikes')
            ch_lbl = name(1:end-length('_spikes'));
        else
            ch_lbl = name;
        end
     else
         error('Channel input must be integer or string filename');
     end
 end


function inspk = local_wavelet_decomp(spikes, scales)
    % Computes Haar wavelet coefficients for each spike
    nspk = size(spikes,1);
  
    cc=zeros(nspk,ls);
    ls = size(spikes,2);

    try
        spikes_l = reshape(spikes',numel(spikes),1);
        if exist('wavedec')
            [c_l,l_wc] = wavedec(spikes_l,scales,'haar');
        else
            [c_l,l_wc] = fix_wavedec(spikes_l,scales);
        end
        wv_c = [0;l_wc(1:end-1)];
        nc = wv_c/nspk;
        wccum = cumsum(wv_c);
        nccum = cumsum(nc);
        for cf = 2:length(nc)
            cc(:,nccum(cf-1)+1:nccum(cf)) = reshape(c_l(wccum(cf-1)+1:wccum(cf)),nc(cf),nspk)';
        end
    catch
        if exist('wavedec')                             % Looks for Wavelets Toolbox
            for i=1:nspk                                % Wavelet decomposition
                [c,l] = wavedec(spikes(i,:),scales,'haar');
                cc(i,1:ls) = c(1:ls);
            end
        else
            for i=1:nspk                                % Replaces Wavelets Toolbox, if not available
                [c,l] = fix_wavedec(spikes(i,:),scales);
                cc(i,1:ls) = c(1:ls);
            end
        end
        
    end
    inspk = cc;
end