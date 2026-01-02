function plot_rescue_clusters(channel_id, varargin)
% PLOT_RESCUE_CLUSTERS Visualize cluster rescue process showing pre-rescue, rescued, and post-rescue states
%
% Inputs:
%   channel_id  - Channel identifier or channel label
%   varargin    - Optional name-value pairs:
%                 'MaxSpikes' - Maximum number of spikes to plot per cluster (default: 2000)
%                 'SamplingRate' - Sampling rate in Hz (for time axis in ms)
%                 'SaveFigure' - If true, saves figure to file (default: false)
%
% Output:
%   Creates a figure with 3 rows x N columns showing:
%   Row 1: All clusters (pre-rescue, excluding quarantined spikes)
%   Row 2: Rescued spikes added to each cluster (or not-rescued for cluster 0)
%   Row 3: Final clusters (post-rescue, with rescued spikes integrated)
%   Cluster 0 (noise) is plotted on the rightmost column
%
% Example:
%   plot_rescue_clusters(1)
%   plot_rescue_clusters(1, 'MaxSpikes', 1500, 'SamplingRate', 30000)

% Parse optional inputs
p = inputParser;
addParameter(p, 'MaxSpikes', 2000, @isnumeric);
addParameter(p, 'SamplingRate', [], @isnumeric);
addParameter(p, 'SaveFigure', false, @islogical);
parse(p, varargin{:});

max_spikes = p.Results.MaxSpikes;
fs = p.Results.SamplingRate;
save_fig = p.Results.SaveFigure;

% Get channel label
ch_lbl = get_channel_label(channel_id);

% Load data
fname_times = sprintf('times_%s.mat', ch_lbl);
fname_spk = sprintf('%s_spikes.mat', ch_lbl);

if ~exist(fname_times, 'file')
    error('Times file not found for channel %s. Channel may not be clustered.', ch_lbl);
end

% Load clustering data
S_times = load(fname_times);
S_spk = load(fname_spk);

% Check if rescue has been performed
if ~isfield(S_times, 'rescue_mask') || isempty(S_times.rescue_mask)
    error('Rescue has not been performed on channel %s.', ch_lbl);
end

% Extract data
cluster_class = S_times.cluster_class;  % [cluster_id, timestamp]
spikes_post = S_times.spikes;
rescue_mask = S_times.rescue_mask;
index_all = S_spk.index_all;
spikes_all = S_spk.spikes_all;

% Get pre-rescue data
if isfield(S_times, 'cluster_class_pre_rescue')
    cluster_class_pre = S_times.cluster_class_pre_rescue;
    spikes_pre = S_times.spikes_pre_rescue;
else
    warning('Pre-rescue data not found. Using current data as baseline.');
    cluster_class_pre = cluster_class;
    spikes_pre = spikes_post;
end

% Get unique clusters and sort (cluster 0 should be on the right)
unique_clusters = unique(cluster_class_pre(:,1));
cluster_ids = unique_clusters(unique_clusters ~= 0);  % Non-noise clusters
cluster_ids = sort(cluster_ids);
if ismember(0, unique_clusters)
    cluster_ids = [cluster_ids; 0];  % Add cluster 0 at the end
end

num_clusters = length(cluster_ids);

% Create time vector if not provided
if ~isempty(fs)
    time_vec = (0:size(spikes_all, 2)-1) / fs * 1000;
    x_label = 'Time (ms)';
else
    time_vec = 1:size(spikes_all, 2);
    x_label = 'Sample';
end

% Create figure
fig = figure('Name', sprintf('Channel %s Cluster Rescue Visualization', ch_lbl), ...
             'NumberTitle', 'off', ...
             'Position', [50, 50, 300*num_clusters, 900]);

% Color settings
color_pre = [0, 0, 1, 0.3];        % Blue with transparency
color_rescued = [0, 1, 0, 0.3];    % Green with transparency  
color_not_rescued = [1, 0, 0, 0.3]; % Red with transparency
color_post = [0, 0.5, 0.5, 0.3];   % Teal with transparency

% Process each cluster
for ic = 1:num_clusters
    clust_id = cluster_ids(ic);
    
    % Row 1: Pre-rescue spikes (all spikes originally in this cluster)
    subplot(3, num_clusters, ic);
    mask_pre = cluster_class_pre(:,1) == clust_id;
    spikes_clust_pre = spikes_pre(mask_pre, :);
    
    % Subsample if needed
    if size(spikes_clust_pre, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_clust_pre, 1), max_spikes));
        spikes_clust_pre = spikes_clust_pre(idx_subsample, :);
    end
    
    if ~isempty(spikes_clust_pre)
        hold on;
        for i = 1:size(spikes_clust_pre, 1)
            plot(time_vec, spikes_clust_pre(i, :), 'Color', color_pre, 'LineWidth', 0.5);
        end
        hold off;
    end
    
    if clust_id == 0
        title(sprintf('Cluster %d (Noise)\nPre-rescue: n=%d', clust_id, sum(mask_pre)), ...
              'FontWeight', 'bold', 'Color', [0.5, 0.5, 0.5]);
    else
        title(sprintf('Cluster %d\nPre-rescue: n=%d', clust_id, sum(mask_pre)), ...
              'FontWeight', 'bold');
    end
    ylabel('Amplitude');
    grid on;
    box on;
    
    % Row 2: Rescued spikes for this cluster (or not-rescued for cluster 0)
    subplot(3, num_clusters, ic + num_clusters);
    
    if clust_id == 0
        % For cluster 0, show spikes that were NOT rescued
        rescued_timestamps = index_all(rescue_mask);
        mask_quar = ~ismember(index_all, cluster_class_pre(:,2)) & ~ismember(index_all, rescued_timestamps);
        spikes_not_rescued = spikes_all(mask_quar, :);
        
        % Subsample if needed
        if size(spikes_not_rescued, 1) > max_spikes
            idx_subsample = round(linspace(1, size(spikes_not_rescued, 1), max_spikes));
            spikes_not_rescued = spikes_not_rescued(idx_subsample, :);
        end
        
        if ~isempty(spikes_not_rescued)
            hold on;
            for i = 1:size(spikes_not_rescued, 1)
                plot(time_vec, spikes_not_rescued(i, :), 'Color', color_not_rescued, 'LineWidth', 0.5);
            end
            hold off;
        end
        title(sprintf('Not Rescued\n(Added to noise): n=%d', sum(mask_quar)), ...
              'FontWeight', 'bold', 'Color', 'r');
    else
        % For other clusters, show rescued spikes
        mask_post = cluster_class(:,1) == clust_id;
        timestamps_post = cluster_class(mask_post, 2);
        
        % Find which of these are rescued spikes
        rescued_timestamps = index_all(rescue_mask);
        mask_rescued = ismember(timestamps_post, rescued_timestamps);
        
        % Get the actual spike waveforms
        spikes_clust_post = spikes_post(mask_post, :);
        spikes_rescued = spikes_clust_post(mask_rescued, :);
        
        % Subsample if needed
        if size(spikes_rescued, 1) > max_spikes
            idx_subsample = round(linspace(1, size(spikes_rescued, 1), max_spikes));
            spikes_rescued = spikes_rescued(idx_subsample, :);
        end
        
        if ~isempty(spikes_rescued)
            hold on;
            for i = 1:size(spikes_rescued, 1)
                plot(time_vec, spikes_rescued(i, :), 'Color', color_rescued, 'LineWidth', 0.5);
            end
            hold off;
        end
        title(sprintf('Rescued Spikes: n=%d', sum(mask_rescued)), ...
              'FontWeight', 'bold', 'Color', [0, 0.6, 0]);
    end
    ylabel('Amplitude');
    grid on;
    box on;
    
    % Row 3: Post-rescue spikes (final cluster state)
    subplot(3, num_clusters, ic + 2*num_clusters);
    mask_post = cluster_class(:,1) == clust_id;
    spikes_clust_post = spikes_post(mask_post, :);
    
    % Subsample if needed
    if size(spikes_clust_post, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_clust_post, 1), max_spikes));
        spikes_clust_post = spikes_clust_post(idx_subsample, :);
    end
    
    if ~isempty(spikes_clust_post)
        hold on;
        for i = 1:size(spikes_clust_post, 1)
            plot(time_vec, spikes_clust_post(i, :), 'Color', color_post, 'LineWidth', 0.5);
        end
        hold off;
    end
    
    if clust_id == 0
        title(sprintf('Post-rescue\n(New cluster 0): n=%d', sum(mask_post)), ...
              'FontWeight', 'bold', 'Color', [0.5, 0.5, 0.5]);
    else
        title(sprintf('Post-rescue: n=%d', sum(mask_post)), ...
              'FontWeight', 'bold');
    end
    xlabel(x_label);
    ylabel('Amplitude');
    grid on;
    box on;
end

% Add overall title
sgtitle(sprintf('Channel %s - Cluster Rescue Visualization', ch_lbl), ...
        'FontSize', 16, 'FontWeight', 'bold');

% Save figure if requested
if save_fig
    saveas(fig, sprintf('%s_rescue_clusters.png', ch_lbl));
    fprintf('Figure saved: %s_rescue_clusters.png\n', ch_lbl);
end

end

function ch_lbl = get_channel_label(channel_id)
% Helper function to get channel label
% Tries to find channel in NSx structure, otherwise uses channel_id directly

if isnumeric(channel_id)
    % Try to load NSx and find the channel
    if exist('NSx', 'var') || exist('NSx.mat', 'file')
        load('NSx', 'NSx');
        idx = find([NSx.chan_ID] == channel_id, 1);
        if ~isempty(idx)
            ch_lbl = NSx(idx).output_name;
        else
            ch_lbl = sprintf('ch%d', channel_id);
        end
    else
        ch_lbl = sprintf('ch%d', channel_id);
    end
else
    % Already a string label
    ch_lbl = channel_id;
end

end
