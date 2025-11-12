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
        mask_non_quarantine = SPK.mask_non_quarantine;
        mask_non_collision = SPK.mask_nonart;
        par = SPK.par;
        index = SPK.index;

        % Find quarantined spikes: excluded by artifact, not by collision
        mask_quar = ~mask_non_quarantine & mask_non_collision;
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
            cluster_class = S.cluster_class;
            spikes = S.spikes;
            coeff = S.coeff;
            inspk_good = S.inspk;
        else
            error('No times file for channel %s', ch_lbl);
        end
        % Use class_good as those with cluster_class ~= 0
        class_good_mask = cluster_class(:,1) ~= 0;
        class_good = cluster_class(class_good_mask, 1);
        % Calculate Haar wavelet features for quarantined spikes
        inspk_quar_full = local_wavelet_decomp(spikes_quar);
        inspk_quar = inspk_quar_full(:, coeff); % Use same coeffs as clustering
        % Use force_membership_wc to assign clusters to quarantined spikes
        class_quar = force_membership_wc(spikes, class_good, spikes_quar, par);
        rescued_idx = find(class_quar ~= 0);
        if isempty(rescued_idx)
            fprintf('  Channel %s: No spikes rescued.\n', ch_lbl);
        else
            fprintf('  Channel %s: Rescued %d spikes.\n', ch_lbl, numel(rescued_idx));
            for i = 1:numel(rescued_idx)
                fprintf('    Rescued spike at index %d (global index %d) assigned to cluster %d\n', ...
                    rescued_idx(i), index_quar(rescued_idx(i)), class_quar(rescued_idx(i)));
            end
        end
        % Merge rescued spikes with original clustered spikes
        spikes_rescued = spikes_quar(rescued_idx, :);
        index_rescued = index_quar(rescued_idx);
        class_rescued = class_quar(rescued_idx)';
        inspk_rescued = inspk_quar(rescued_idx, :);
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

% --- Local Haar wavelet decomposition function (copied from gmm_1channel/wave_features) ---
function inspk = local_wavelet_decomp(spikes)
    % Computes Haar wavelet coefficients for each spike
    nspk = size(spikes,1);
    L = size(spikes,2);
    inspk = zeros(nspk, L);
    for i = 1:nspk
        % Use MATLAB's wavedec for Haar decomposition
        [C,~] = wavedec(spikes(i,:), 4, 'haar');
        inspk(i,1:length(C)) = C;
    end
end