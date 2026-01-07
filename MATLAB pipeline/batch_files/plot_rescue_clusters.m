function plot_rescue_clusters(varargin)
% PLOT_RESCUE_CLUSTERS Visualize cluster rescue process showing pre-rescue, rescued, and post-rescue states
%
% Name-Value Inputs:
%   'channels'     - Channel number(s) (e.g., 1 or [1, 2, 5] or 1:10) [REQUIRED]
%   'MaxSpikes'    - Maximum number of spikes to plot per cluster (default: 2000)
%   'SamplingRate' - Sampling rate in Hz (for time axis in ms)
%   'SaveFigure'   - If true, saves figure to file (default: false)
%   'ShowFigure'   - If true, displays figure (default: true)
%   'OutputSuffix' - String to append to folder name (default: '')
%                    Saves to: rescue_clusters<suffix>/
%
% Output:
%   Creates a figure per channel with 3 rows x N columns showing:
%   Row 1: All clusters (pre-rescue, excluding quarantined spikes)
%   Row 2: Rescued spikes added to each cluster (or not-rescued for cluster 0)
%   Row 3: Final clusters (post-rescue, with rescued spikes integrated)
%   Cluster 0 (noise) is plotted on the rightmost column
%
% Example:
%   plot_rescue_clusters('channels', 1)
%   plot_rescue_clusters('channels', 1:10, 'SaveFigure', true, 'ShowFigure', false)
%   plot_rescue_clusters('channels', [1 2 5], 'SaveFigure', true, 'OutputSuffix', '_v2')

% Parse optional inputs
p = inputParser;
addParameter(p, 'channels', [], @isnumeric);
addParameter(p, 'MaxSpikes', 2000, @isnumeric);
addParameter(p, 'SamplingRate', [], @isnumeric);
addParameter(p, 'SaveFigure', false, @islogical);
addParameter(p, 'ShowFigure', true, @islogical);
addParameter(p, 'OutputSuffix', '', @ischar);
parse(p, varargin{:});

channel_nums = p.Results.channels;
max_spikes = p.Results.MaxSpikes;
fs = p.Results.SamplingRate;
save_fig = p.Results.SaveFigure;
show_fig = p.Results.ShowFigure;
output_suffix = p.Results.OutputSuffix;

% Validate required inputs
if isempty(channel_nums)
    error('Must specify ''channels'' parameter');
end

% Set visibility
if show_fig
    vis_str = 'on';
else
    vis_str = 'off';
end

% Process each channel
for ch_idx = 1:length(channel_nums)
    channel_id = channel_nums(ch_idx);
    process_single_rescue_plot(channel_id, max_spikes, fs, save_fig, show_fig, vis_str, output_suffix);
end

fprintf('Done processing %d channels.\n', length(channel_nums));

end % end main function

%% Helper function for single channel processing
function process_single_rescue_plot(channel_id, max_spikes, fs, save_fig, show_fig, vis_str, output_suffix)

% Get channel label
ch_lbl = get_channel_label(channel_id);

% Load data
fname_times = sprintf('times_%s.mat', ch_lbl);
fname_spk = sprintf('%s_spikes.mat', ch_lbl);

if ~exist(fname_times, 'file')
    warning('Times file not found for channel %s. Skipping...', ch_lbl);
    return;
end

% Load clustering data
S_times = load(fname_times);
S_spk = load(fname_spk);

% Check if rescue has been performed
if ~isfield(S_times, 'rescue_mask') || isempty(S_times.rescue_mask)
    warning('Rescue has not been performed on channel %s. Skipping...', ch_lbl);
    return;
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

% Get unique clusters and sort (cluster 0 should be on the right of page 1)
unique_clusters = unique(cluster_class_pre(:,1));
non_zero_clusters = unique_clusters(unique_clusters ~= 0);  % Non-noise clusters
non_zero_clusters = sort(non_zero_clusters);
has_cluster_0 = ismember(0, unique_clusters);

num_clusters = length(non_zero_clusters) + has_cluster_0;

% Create time vector if not provided
if ~isempty(fs)
    time_vec = (0:size(spikes_all, 2)-1) / fs * 1000;
    x_label = 'Time (ms)';
else
    time_vec = 1:size(spikes_all, 2);
    x_label = 'Sample';
end

% Cluster colors (from make_cluster_report.m) - index 1 = cluster 0 (black)
leicolors = [0 0 0; 0 0 1; 1 0 0; 0 0.5 0; 0.62 0 0; 0.42 0 0.76; ...
             0.97 0.52 0.03; 0.52 0.25 0; 1 0.10 0.72; 0.55 0.55 0.55; ...
             0.59 0.83 0.31; 0.97 0.62 0.86; 0.62 0.76 1.0];

% Pagination: max 6 clusters per page
% Page 1: up to 5 non-zero clusters + cluster 0 (if exists)
% Page 2+: remaining non-zero clusters (6 per page)
max_nonzero_page1 = 5;  % Reserve 1 spot for cluster 0 on page 1
max_clusters_later = 6;

num_nonzero = length(non_zero_clusters);
if num_nonzero <= max_nonzero_page1
    num_pages = 1;
else
    remaining = num_nonzero - max_nonzero_page1;
    num_pages = 1 + ceil(remaining / max_clusters_later);
end

for page = 1:num_pages
    % Get clusters for this page
    if page == 1
        % Page 1: first 5 non-zero clusters + cluster 0
        idx_end = min(max_nonzero_page1, num_nonzero);
        page_cluster_ids = non_zero_clusters(1:idx_end);
        if has_cluster_0
            page_cluster_ids = [page_cluster_ids; 0];  % Add cluster 0 at end
        end
    else
        % Pages 2+: next batch of non-zero clusters
        idx_start = max_nonzero_page1 + (page - 2) * max_clusters_later + 1;
        idx_end = min(idx_start + max_clusters_later - 1, num_nonzero);
        page_cluster_ids = non_zero_clusters(idx_start:idx_end);
    end
    num_clusters_page = length(page_cluster_ids);
    
    % Create figure (white background)
    if num_pages > 1
        fig_title = sprintf('Channel %s Cluster Rescue Visualization (Page %d/%d)', ch_lbl, page, num_pages);
    else
        fig_title = sprintf('Channel %s Cluster Rescue Visualization', ch_lbl);
    end
    % Minimum width to avoid clipping title
    fig_width = max(300*num_clusters_page, 600);
    fig = figure('Name', fig_title, ...
                 'NumberTitle', 'off', ...
                 'Visible', vis_str, ...
                 'Color', 'w', ...
                 'Position', [50, 50, fig_width, 900]);
    set(fig, 'InvertHardcopy', 'off');  % Preserve colors when saving

    % Process each cluster on this page
    for ic = 1:num_clusters_page
        clust_id = page_cluster_ids(ic);
    
    % Get color for this cluster (cluster 0 -> index 1, cluster 1 -> index 2, etc.)
    color_idx = mod(clust_id, size(leicolors, 1)) + 1;
    cluster_color = leicolors(color_idx, :);
    cluster_color_alpha = [cluster_color, 0.3];  % Add transparency
    
    % Row 1: Pre-rescue spikes (all spikes originally in this cluster)
    subplot(3, num_clusters_page, ic);
    set(gca, 'Color', 'w');  % White plot background
    mask_pre = cluster_class_pre(:,1) == clust_id;
    spikes_clust_pre = spikes_pre(mask_pre, :);
    
    % Subsample if needed
    if size(spikes_clust_pre, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_clust_pre, 1), max_spikes));
        spikes_clust_pre = spikes_clust_pre(idx_subsample, :);
    end
    
    if ~isempty(spikes_clust_pre)
        hold on;
        % Plot all waveforms at once (transpose so each column is a waveform)
        h = plot(time_vec, spikes_clust_pre', 'LineWidth', 0.5);
        set(h, 'Color', cluster_color_alpha);
        plot(time_vec, mean(spikes_clust_pre, 1), 'Color', cluster_color, 'LineWidth', 2.4);
        hold off;
    end
    
    title(sprintf('Cluster %d\nPre-rescue: n=%d', clust_id, sum(mask_pre)), ...
          'FontWeight', 'bold');
    ylabel('Amplitude');
    grid on;
    box on;
    
    % Row 2: Rescued spikes for this cluster (or not-rescued for cluster 0)
    subplot(3, num_clusters_page, ic + num_clusters_page);
    set(gca, 'Color', 'w');  % White plot background
    
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
            % Plot all waveforms at once
            h = plot(time_vec, spikes_not_rescued', 'LineWidth', 0.5);
            set(h, 'Color', cluster_color_alpha);
            if size(spikes_not_rescued, 1) > 0
                plot(time_vec, mean(spikes_not_rescued, 1), 'Color', cluster_color, 'LineWidth', 2.4);
            end
            hold off;
        end
        title(sprintf('Not Rescued\n(Stayed out): n=%d', sum(mask_quar)), ...
              'FontWeight', 'bold');
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
            % Plot all waveforms at once
            h = plot(time_vec, spikes_rescued', 'LineWidth', 0.5);
            set(h, 'Color', cluster_color_alpha);
            plot(time_vec, mean(spikes_rescued, 1), 'Color', cluster_color, 'LineWidth', 2.4);
            hold off;
        end
        title(sprintf('Rescued Spikes: n=%d', sum(mask_rescued)), ...
              'FontWeight', 'bold');
    end
    ylabel('Amplitude');
    grid on;
    box on;
    
    % Row 3: Post-rescue spikes (final cluster state)
    subplot(3, num_clusters_page, ic + 2*num_clusters_page);
    set(gca, 'Color', 'w');  % White plot background
    mask_post = cluster_class(:,1) == clust_id;
    spikes_clust_post = spikes_post(mask_post, :);
    
    % Subsample if needed
    if size(spikes_clust_post, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_clust_post, 1), max_spikes));
        spikes_clust_post = spikes_clust_post(idx_subsample, :);
    end
    
    if ~isempty(spikes_clust_post)
        hold on;
        % Plot all waveforms at once
        h = plot(time_vec, spikes_clust_post', 'LineWidth', 0.5);
        set(h, 'Color', cluster_color_alpha);
        plot(time_vec, mean(spikes_clust_post, 1), 'Color', cluster_color, 'LineWidth', 2.4);
        hold off;
    end
    
    title(sprintf('Post-rescue: n=%d', sum(mask_post)), ...
          'FontWeight', 'bold');
    xlabel(x_label);
    ylabel('Amplitude');
    grid on;
    box on;
    end  % end cluster loop

    % Add overall title
    if num_pages > 1
        sgtitle(sprintf('Channel %s - Cluster Rescue (Page %d/%d)', ch_lbl, page, num_pages), ...
                'FontSize', 16, 'FontWeight', 'bold');
    else
        sgtitle(sprintf('Channel %s - Cluster Rescue Visualization', ch_lbl), ...
                'FontSize', 16, 'FontWeight', 'bold');
    end

    % Save figure if requested
    if save_fig
        % Create output folder
        save_folder = sprintf('rescue_clusters%s', output_suffix);
        if ~exist(save_folder, 'dir')
            mkdir(save_folder);
        end
        if num_pages > 1
            save_name = fullfile(save_folder, sprintf('%s_page%d.png', ch_lbl, page));
        else
            save_name = fullfile(save_folder, sprintf('%s.png', ch_lbl));
        end
        
        % Force render and save
        drawnow;
        exportgraphics(fig, save_name, 'Resolution', 150);
        fprintf('Figure saved: %s\n', save_name);
    end

    % Close if not showing
    if ~show_fig
        close(fig);
    end
end  % end page loop

end % end helper function

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
