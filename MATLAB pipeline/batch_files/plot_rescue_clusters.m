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
addParameter(p, 'IncludeTask', false, @islogical);  % Set to true to include task spikes in rescue
parse(p, varargin{:});

channel_nums = p.Results.channels;
max_spikes = p.Results.MaxSpikes;
fs = p.Results.SamplingRate;
save_fig = p.Results.SaveFigure;
show_fig = p.Results.ShowFigure;
output_suffix = p.Results.OutputSuffix;
include_task = p.Results.IncludeTask;

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
    process_single_rescue_plot(channel_id, max_spikes, fs, save_fig, show_fig, vis_str, output_suffix, include_task);
end

fprintf('Done processing %d channels.\n', length(channel_nums));

end % end main function

%% Helper function for single channel processing
function process_single_rescue_plot(channel_id, max_spikes, fs, save_fig, show_fig, vis_str, output_suffix, include_task)

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

% All masks and rescue_mask index into spikes_all/index_all
index_all = S_spk.index_all;
spikes_all = S_spk.spikes_all;
rescue_mask = S_times.rescue_mask;  % indexes into spikes_all

% Load masks from spikes file
if isfield(S_spk, 'mask_non_quarantine')
    mask_quarantine = ~S_spk.mask_non_quarantine;
else
    mask_quarantine = false(size(index_all));
end

if isfield(S_spk, 'mask_nonart')
    mask_collision = ~S_spk.mask_nonart;
else
    mask_collision = false(size(index_all));
end

if isfield(S_spk, 'mask_taskspks')
    mask_task_excluded = ~S_spk.mask_taskspks;
else
    mask_task_excluded = false(size(index_all));
end

% Mask counts
n_quar = sum(mask_quarantine);
n_coll = sum(mask_collision);
if include_task
    n_task = sum(mask_task_excluded);
else
    n_task = 0;  % Don't count task spikes in rescue pool
end

% Current times file has combined data (original good + rescued)
cluster_class = S_times.cluster_class;  % [cluster_id, timestamp]
spikes_combined = S_times.spikes;

% Timestamps of rescued spikes
rescued_timestamps = index_all(rescue_mask);

% Figure out which rows in cluster_class are rescued vs original
% cluster_class(:,2) contains the timestamp for each spike
mask_is_rescued = ismember(cluster_class(:,2), rescued_timestamps);

% Counts
num_total = size(cluster_class, 1);
num_rescued = sum(mask_is_rescued);
num_original = sum(~mask_is_rescued);

fprintf('  Channel %s: Original=%d, Rescued=%d, Total=%d\n', ...
    ch_lbl, num_original, num_rescued, num_total);

% Get unique clusters (cluster 0 on the right)
unique_clusters = unique(cluster_class(:,1));
non_zero_clusters = sort(unique_clusters(unique_clusters ~= 0));
has_cluster_0 = ismember(0, unique_clusters);

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
    
    % Mask for this cluster in the combined data
    mask_this_clust = cluster_class(:,1) == clust_id;
    
    % Within this cluster, separate original vs rescued using mask_is_rescued
    mask_original_clust = mask_this_clust & ~mask_is_rescued;
    mask_rescued_clust = mask_this_clust & mask_is_rescued;
    
    n_original_clust = sum(mask_original_clust);
    n_rescued_clust = sum(mask_rescued_clust);
    n_total_clust = sum(mask_this_clust);
    
    % Row 1: Original spikes in this cluster (before rescue was added)
    subplot(3, num_clusters_page, ic);
    set(gca, 'Color', 'w');
    spikes_original_clust = spikes_combined(mask_original_clust, :);
    
    % Subsample if needed
    if size(spikes_original_clust, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_original_clust, 1), max_spikes));
        spikes_original_clust = spikes_original_clust(idx_subsample, :);
    end
    
    if ~isempty(spikes_original_clust)
        hold on;
        h = plot(time_vec, spikes_original_clust', 'LineWidth', 0.5);
        set(h, 'Color', cluster_color_alpha);
        plot(time_vec, mean(spikes_original_clust, 1), 'Color', cluster_color, 'LineWidth', 2.4);
        hold off;
    end
    
    pct_of_clust = 100 * n_original_clust / max(n_total_clust, 1);
    title(sprintf('Cluster %d\nOriginal: n=%d (%.0f%% of clust)', clust_id, n_original_clust, pct_of_clust), ...
          'FontWeight', 'bold');
    ylabel('Amplitude');
    grid on;
    box on;
    
    % Row 2: Rescued spikes assigned to this cluster
    subplot(3, num_clusters_page, ic + num_clusters_page);
    set(gca, 'Color', 'w');
    
    if clust_id == 0
        % For cluster 0, show quarantined spikes that were NOT rescued
        if isfield(S_times, 'class_quar')
            class_quar = S_times.class_quar;
            n_not_rescued = sum(class_quar == 0);
            if isfield(S_times, 'index_quar')
                index_quar = S_times.index_quar;
                not_rescued_timestamps = index_quar(class_quar == 0);
                mask_not_rescued_all = ismember(index_all, not_rescued_timestamps);
                spikes_not_rescued = spikes_all(mask_not_rescued_all, :);
            else
                spikes_not_rescued = [];
            end
        else
            n_not_rescued = 0;
            spikes_not_rescued = [];
        end
        
        % Subsample if needed
        if size(spikes_not_rescued, 1) > max_spikes
            idx_subsample = round(linspace(1, size(spikes_not_rescued, 1), max_spikes));
            spikes_not_rescued = spikes_not_rescued(idx_subsample, :);
        end
        
        if ~isempty(spikes_not_rescued)
            hold on;
            h = plot(time_vec, spikes_not_rescued', 'LineWidth', 0.5);
            set(h, 'Color', cluster_color_alpha);
            plot(time_vec, mean(spikes_not_rescued, 1), 'Color', cluster_color, 'LineWidth', 2.4);
            hold off;
        end
        total_quar = n_not_rescued + num_rescued;
        pct_not_rescued = 100 * n_not_rescued / max(total_quar, 1);
        title(sprintf('Not Rescued\nn=%d (%.0f%% of quar)', n_not_rescued, pct_not_rescued), 'FontWeight', 'bold');
    else
        % Show rescued spikes assigned to this cluster
        spikes_rescued_clust = spikes_combined(mask_rescued_clust, :);
        
        % Subsample if needed
        if size(spikes_rescued_clust, 1) > max_spikes
            idx_subsample = round(linspace(1, size(spikes_rescued_clust, 1), max_spikes));
            spikes_rescued_clust = spikes_rescued_clust(idx_subsample, :);
        end
        
        if ~isempty(spikes_rescued_clust)
            hold on;
            h = plot(time_vec, spikes_rescued_clust', 'LineWidth', 0.5);
            set(h, 'Color', cluster_color_alpha);
            plot(time_vec, mean(spikes_rescued_clust, 1), 'Color', cluster_color, 'LineWidth', 2.4);
            hold off;
        end
        pct_of_rescued = 100 * n_rescued_clust / max(num_rescued, 1);
        title(sprintf('Rescued: n=%d\n(%.0f%% of %d rescued)', n_rescued_clust, pct_of_rescued, num_rescued), ...
              'FontWeight', 'bold');
    end
    ylabel('Amplitude');
    grid on;
    box on;
    
    % Row 3: Combined (original + rescued)
    subplot(3, num_clusters_page, ic + 2*num_clusters_page);
    set(gca, 'Color', 'w');
    spikes_total_clust = spikes_combined(mask_this_clust, :);
    
    % Subsample if needed
    if size(spikes_total_clust, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_total_clust, 1), max_spikes));
        spikes_total_clust = spikes_total_clust(idx_subsample, :);
    end
    
    if ~isempty(spikes_total_clust)
        hold on;
        h = plot(time_vec, spikes_total_clust', 'LineWidth', 0.5);
        set(h, 'Color', cluster_color_alpha);
        plot(time_vec, mean(spikes_total_clust, 1), 'Color', cluster_color, 'LineWidth', 2.4);
        hold off;
    end
    
    pct_of_total = 100 * n_total_clust / max(num_total, 1);
    pct_rescued_in_clust = 100 * n_rescued_clust / max(n_total_clust, 1);
    title(sprintf('Combined: n=%d (%.0f%% of total)\n%d+%d (%.0f%% rescued)', ...
          n_total_clust, pct_of_total, n_original_clust, n_rescued_clust, pct_rescued_in_clust), ...
          'FontWeight', 'bold');
    xlabel(x_label);
    ylabel('Amplitude');
    grid on;
    box on;
    end  % end cluster loop

    % Build rescue pool description (which masks are being rescued)
    rescue_masks_used = {};
    if n_quar > 0, rescue_masks_used{end+1} = 'QC'; end
    if n_coll > 0, rescue_masks_used{end+1} = 'Coll'; end
    if n_task > 0, rescue_masks_used{end+1} = 'Task'; end
    if isempty(rescue_masks_used)
        rescue_str = 'None';
    else
        rescue_str = strjoin(rescue_masks_used, '+');
    end

    % Add overall title with mask info (three lines, no interpreter for underscores)
    if num_pages > 1
        title_line1 = sprintf('Channel %s (Page %d/%d)', ch_lbl, page, num_pages);
    else
        title_line1 = sprintf('Channel %s', ch_lbl);
    end
    title_line2 = sprintf('Excluded: QC=%d, Coll=%d, Task=%d', n_quar, n_coll, n_task);
    title_line3 = sprintf('Rescuing: %s', rescue_str);
    sgtitle({title_line1, title_line2, title_line3}, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

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
