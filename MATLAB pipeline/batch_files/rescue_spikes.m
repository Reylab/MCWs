function rescue_spikes(varargin)
% rescue_spikes - Attempts to reclassify quarantined spikes using template matching
% Inputs:
%   channels - vector of channel IDs
%   Optional: 'parallel', true/false, ...
% Usage:
%   rescue_spikes(channels, 'parallel', true)

% Parse optional arguments
p = inputParser;

addParameter(p,'channels',[],@isnumeric);
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'restore', false, @islogical);

parse(p, varargin{:});

channels = p.Results.channels;
parallel = p.Results.parallel;
restore = p.Results.restore;

if restore
    fprintf('Starting rescue_spikes RESTORE on %d channels...\n', length(channels));
else
    fprintf('Starting rescue_spikes on %d channels...\n', length(channels));
end

if parallel
    parfor kk = 1:length(channels)
        process_channel_rescue(channels(kk), restore);
    end
else
    for kk = 1:length(channels)
        process_channel_rescue(channels(kk), restore);
    end
end

fprintf('rescue_spikes DONE.\n');
end

function process_channel_rescue(ch, restore)
    try
        ch_lbl = get_channel_label(ch); % Helper to get output_name or label
        fname_spk = sprintf('%s_spikes.mat', ch_lbl);
        fname_times = sprintf('times_%s.mat', ch_lbl);
        
        if restore
            if exist(fname_spk, 'file')
                vars_spk = load(fname_spk);
                
                % Check if rescue_mask exists and has rescued spikes
                has_rescued_spikes = isfield(vars_spk, 'rescue_mask') && ~isempty(vars_spk.rescue_mask) && any(vars_spk.rescue_mask);
                
                if has_rescued_spikes
                    % 1. Restore Spikes File - remove rescued timestamps from index
                    rescued_timestamps = vars_spk.index_all(vars_spk.rescue_mask);
                    to_remove_spk = ismember(vars_spk.index, rescued_timestamps);
                    
                    if any(to_remove_spk)
                        vars_spk.spikes(to_remove_spk, :) = [];
                        vars_spk.index(to_remove_spk) = [];
                    end
                    vars_spk.rescue_mask = false(size(vars_spk.index_all));
                    save(fname_spk, '-struct', 'vars_spk');
                    fprintf('  Channel %s: Restored spikes file (removed %d rescued spikes).\n', ch_lbl, sum(to_remove_spk));
                end
                
                % 2. Restore times file from backup
                if exist(fname_times, 'file')
                    vars_times = load(fname_times);
                    
                    % Restore from backup if it exists
                    if isfield(vars_times, 'spikes_pre_rescue')
                        vars_times.spikes = vars_times.spikes_pre_rescue;
                        vars_times.cluster_class = vars_times.cluster_class_pre_rescue;
                        if isfield(vars_times, 'index_pre_rescue')
                            vars_times.index = vars_times.index_pre_rescue;
                        end
                        if isfield(vars_times, 'inspk') && isfield(vars_times, 'spikes_pre_rescue')
                            % Recalculate inspk size to match restored spikes
                            vars_times.inspk = vars_times.inspk(1:size(vars_times.spikes_pre_rescue,1), :);
                        end
                        fprintf('  Channel %s: Restored times file from backup.\n', ch_lbl);
                    end
                    
                    % Clean up rescue artifacts
                    fields_to_remove = {'spikes_quarantined', 'index_quarantined', 'class_quarantined', ...
                        'class_quar', 'index_quar', 'rescued_idx', 'spikes_pre_rescue', 'index_pre_rescue', ...
                        'cluster_class_pre_rescue', 'rescue_mask'};
                    for f = 1:length(fields_to_remove)
                        if isfield(vars_times, fields_to_remove{f})
                            vars_times = rmfield(vars_times, fields_to_remove{f});
                        end
                    end
                    
                    save(fname_times, '-struct', 'vars_times');
                end
                
                if ~has_rescued_spikes
                    if exist(fname_times, 'file')
                        fprintf('  Channel %s: No rescue mask found, but cleaned up times file.\n', ch_lbl);
                    else
                        fprintf('  Channel %s: No rescue mask found to restore.\n', ch_lbl);
                    end
                end
            else
                fprintf('  Channel %s: Spikes file not found.\n', ch_lbl);
            end
            return;
        end

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
        spikes = SPK.spikes;

       

        % Build rescue mask - select which spikes to attempt rescue on
       
        % SINGLE MASK OPTIONS (all spikes from one category):
        % mask_quar = ~mask_non_quarantine;     % Only quarantine
       % mask_quar = ~mask_non_collision;      % Only collision
         %mask_quar = ~mask_task;               % Only task-excluded
        
        % COMBINED MASK OPTIONS (use OR to combine - no duplicates):
      %   mask_quar = ~mask_non_quarantine | ~mask_non_collision;                % Quarantine + collision
       %  mask_quar = ~mask_non_quarantine | ~mask_task;                         % Quarantine + task-excluded
       %  mask_quar = ~mask_non_collision | ~mask_task;                          % Collision + task-excluded
        mask_quar = ~mask_non_quarantine | ~mask_non_collision | ~mask_task;     % All three combined

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
            
            if ~isfield(S, 'spikes_pre_rescue')
                spikes_pre_rescue = S.spikes;
                index_pre_rescue = index;
                cluster_class_pre_rescue = S.cluster_class;
                save(fname_times, 'spikes_pre_rescue', 'index_pre_rescue', ...
                     'cluster_class_pre_rescue', '-append');
            else
                fprintf("spikes already rescued on Channel %d\n",ch);
                return
            end
            
            cluster_class = S.cluster_class;
            if isfield(S, 'coeff')
                coeff = S.coeff;
            else
                coeff = 1:64; % Fallback if coeff missing
            end
            inspk_good = S.inspk;

        else %if no times file, means unclustered/assume multiunit
            % Uncomment the following lines to skip rescue for unclustered channels:
            % fprintf('  Channel %s: Skipping - no times file (unclustered).\n', ch_lbl);
            % return
            
            spikes = SPK.spikes;
            coeff = 1:64; % default to first 64 coeffs
            inspk_good = local_wavelet_decomp(spikes);
            class_good = ones(size(spikes,1),1);
            cluster_class = [class_good, index(:)];
            mask_quar = ~mask_non_quarantine | ~mask_non_collision;
        end
        % Use class_good as those with cluster_class ~= 0
        class_good_mask = cluster_class(:,1) ~= 0;
        class_good = cluster_class(class_good_mask, 1);
        inspk_good_classified = inspk_good(class_good_mask, :);
        spikes_good_classified = spikes(class_good_mask, :);
        % Calculate Haar wavelet features for quarantined spikes
        inspk_quar_full = local_wavelet_decomp(spikes_quar);
        inspk_quar = inspk_quar_full(:, coeff); % Use same coeffs as clustering
        % Use force_membership_wc to assign clusters to quarantined spikes
        class_quar = force_membership_wc(spikes_good_classified, class_good, spikes_quar, par);
        rescued_idx = find(class_quar ~= 0);
        if isempty(rescued_idx)
            fprintf('  Channel %s: No spikes rescued.\n', ch_lbl);
        else
            fprintf('  Channel %s: Rescued %d spikes.\n', ch_lbl, numel(rescued_idx));
        end
        % Merge rescued spikes with original clustered spikes
        spikes_rescued = spikes_quar(rescued_idx, :);
        index_rescued = index_quar(rescued_idx);
        class_rescued = class_quar(rescued_idx)';
        inspk_rescued = inspk_quar(rescued_idx, :);
        % Combine (force column vectors for index since some files save as row)
        index_combined = [index(:); index_rescued(:)];
        spikes_combined = [spikes; spikes_rescued];
        class_combined = [cluster_class(:,1); class_rescued(:)];
        inspk_combined = [inspk_good; inspk_rescued];

        cluster_class_combined = zeros(length(class_combined),2);
        cluster_class_combined(:,1) = class_combined;
        cluster_class_combined(:,2) = index_combined;
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

        rescue_mask = false(size(index_all));
        quar_indices_all = find(mask_quar);
        rescue_mask(quar_indices_all(rescued_idx)) = true;
        
        if exist(fname_times, 'file')
            save(fname_times, 'spikes', 'inspk', 'cluster_class','rescue_mask', '-append');
        else
            save(fname_times, 'spikes', 'inspk', 'cluster_class', 'rescue_mask','par');
        end
        % Also save rescue info in spikes file
        save(fname_times, 'class_quar', 'index_quar', 'rescued_idx', '-append');
        
        % Create and save rescue_mask to spikes file

        save(fname_spk, 'spikes','index','rescue_mask', '-append');
        
    catch ME
        fprintf('  Channel %d: Error - %s\n', ch, ME.message);
        % Print full stack trace to help debug index/bounds errors
        try
            report = getReport(ME, 'extended');
            fprintf('%s\n', report);
        catch
            % Fallback if getReport fails for some reason
            fprintf('  (Could not get full report)\n');
        end
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


function inspk = local_wavelet_decomp(spikes)
    % Computes Haar wavelet coefficients for each spike
    % Do it like wave_features: use wavedec if available, else fix_wavedec
    nspk = size(spikes,1);
    L = size(spikes,2);
    scales = 4; % match typical default used elsewhere
    cc = zeros(nspk, L);
    try
        spikes_l = reshape(spikes', numel(spikes), 1);
        if exist('wavedec', 'file')
            [c_l, l_wc] = wavedec(spikes_l, scales, 'haar');
        else
            [c_l, l_wc] = fix_wavedec(spikes_l, scales);
        end
        wv_c = [0; l_wc(1:end-1)];
        nc = wv_c / nspk;
        wccum = cumsum(wv_c);
        nccum = cumsum(nc);
        for cf = 2:length(nc)
            cc(:, nccum(cf-1)+1:nccum(cf)) = reshape(c_l(wccum(cf-1)+1:wccum(cf)), nc(cf), nspk)';
        end
    catch
        if exist('wavedec', 'file')
            for i = 1:nspk
                [c, ~] = wavedec(spikes(i,:), scales, 'haar');
                cc(i, 1:L) = c(1:L);
            end
        else
            for i = 1:nspk
                [c, ~] = fix_wavedec(spikes(i,:), scales);
                cc(i, 1:L) = c(1:L);
            end
        end
    end
    inspk = cc;
end