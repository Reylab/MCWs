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

% Helper function for single channel processing
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
n_total_spikes = size(spikes_all, 1);

fprintf('  Channel %s: Original=%d, Rescued=%d, Total=%d\n', ...
    ch_lbl, num_original, num_rescued, num_total);

% Get unique clusters (cluster 0 on the right)
unique_clusters = unique(cluster_class(:,1));
non_zero_clusters = sort(unique_clusters(unique_clusters ~= 0));
has_cluster_0 = ismember(0, unique_clusters);

% Force include cluster 0 if there are quarantined spikes
if n_quar > 0 && ~has_cluster_0
    unique_clusters = [unique_clusters; 0];
    has_cluster_0 = true;
end

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

% Pagination: max 3 clusters per page (with density, so 6 columns)
% Page 1: up to 2 non-zero clusters + cluster 0 (if exists)
% Page 2+: remaining non-zero clusters (3 per page)
max_nonzero_page1 = 2;  % Reserve 1 spot for cluster 0 on page 1
max_clusters_later = 3;

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
    fig_width = max(300*2*num_clusters_page, 600);
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
    
    % Collect all spike data for this cluster to determine common y-limits
    spikes_original_clust = spikes_combined(mask_original_clust, :);
    spikes_total_clust = spikes_combined(mask_this_clust, :);
    
    % Subsample if needed for y-limits calculation
    if size(spikes_original_clust, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_original_clust, 1), max_spikes));
        spikes_original_clust_sub = spikes_original_clust(idx_subsample, :);
    else
        spikes_original_clust_sub = spikes_original_clust;
    end
    
    if size(spikes_total_clust, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_total_clust, 1), max_spikes));
        spikes_total_clust_sub = spikes_total_clust(idx_subsample, :);
    else
        spikes_total_clust_sub = spikes_total_clust;
    end
    
    if clust_id == 0
        % For cluster 0, collect not rescued spikes
        if isfield(S_times, 'class_quar')
            class_quar = S_times.class_quar;
            if isfield(S_times, 'index_quar')
                index_quar = S_times.index_quar;
                not_rescued_timestamps = index_quar(class_quar == 0);
                mask_not_rescued_all = ismember(index_all, not_rescued_timestamps);
                spikes_rescued_clust = spikes_all(mask_not_rescued_all, :);
            else
                spikes_rescued_clust = [];
            end
        else
            spikes_rescued_clust = [];
        end
    else
        spikes_rescued_clust = spikes_combined(mask_rescued_clust, :);
    end
    
    % Subsample spikes_rescued_clust if needed for y-limits
    if size(spikes_rescued_clust, 1) > max_spikes
        idx_subsample = round(linspace(1, size(spikes_rescued_clust, 1), max_spikes));
        spikes_rescued_clust_sub = spikes_rescued_clust(idx_subsample, :);
    else
        spikes_rescued_clust_sub = spikes_rescued_clust;
    end
    
    % Determine common y-limits for this cluster
    all_spikes = [spikes_original_clust_sub; spikes_rescued_clust_sub; spikes_total_clust_sub];
    if ~isempty(all_spikes)
        y_min = min(all_spikes(:));
        y_max = max(all_spikes(:));
        y_range = y_max - y_min;
        if y_range == 0
            y_range = 1e-6;  % Avoid zero range
        end
        y_limits = [y_min - 0.05*y_range, y_max + 0.05*y_range];
    else
        y_limits = [-1, 1];  % Default if no spikes
    end
    
    % Row 1: Original spikes in this cluster (before rescue was added)
    waveform_pos = (ic-1)*2 + 1;
    density_pos = (ic-1)*2 + 2;
    subplot(3, 2*num_clusters_page, waveform_pos);
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
    title(sprintf('Cluster %d\nOriginal: n=%d', clust_id, n_original_clust), ...
          'FontWeight', 'bold');
    ylabel('Amplitude');
    ylim(y_limits);
    grid on;
    box on;
    
    % Density for row 1
    subplot(3, 2*num_clusters_page, density_pos);
    set(gca, 'Color', 'w');
    density_image_matlab(spikes_original_clust, gca, fs);
    ylabel('Amplitude');
    ylim(y_limits);
    grid on;
    box on;
    
    % Row 2: Rescued spikes assigned to this cluster
    waveform_pos = 2*num_clusters_page + (ic-1)*2 + 1;
    density_pos = 2*num_clusters_page + (ic-1)*2 + 2;
    subplot(3, 2*num_clusters_page, waveform_pos);
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
        title(sprintf('Not Rescued\nn=%d', n_not_rescued), 'FontWeight', 'bold');
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
        title(sprintf('Rescued: n=%d', n_rescued_clust), ...
              'FontWeight', 'bold');
    end
    ylabel('Amplitude');
    ylim(y_limits);
    grid on;
    box on;
    
    % Density for row 2
    subplot(3, 2*num_clusters_page, density_pos);
    set(gca, 'Color', 'w');
    if clust_id == 0
        density_image_matlab(spikes_not_rescued, gca, fs);
    else
        density_image_matlab(spikes_rescued_clust, gca, fs);
    end
    ylabel('Amplitude');
    ylim(y_limits);
    grid on;
    box on;
    
    % Row 3: Combined (original + rescued)
    waveform_pos = 4*num_clusters_page + (ic-1)*2 + 1;
    density_pos = 4*num_clusters_page + (ic-1)*2 + 2;
    subplot(3, 2*num_clusters_page, waveform_pos);
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
    title(sprintf('Combined: n=%d', ...
          n_total_clust), ...
          'FontWeight', 'bold');
    xlabel(x_label);
    ylabel('Amplitude');
    ylim(y_limits);
    grid on;
    box on;
    
    % Density for row 3
    subplot(3, 2*num_clusters_page, density_pos);
    set(gca, 'Color', 'w');
    density_image_matlab(spikes_total_clust, gca, fs);
    xlabel(x_label);
    ylabel('Amplitude');
    ylim(y_limits);
    grid on;
    box on;
    end  % end cluster loop

    % Build rescue pool description (which masks are being rescued)
    rescue_masks_used = {};
    if n_quar > 0, rescue_masks_used{end+1} = 'Quar'; end
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
    title_line2 = sprintf('Excluded: Quar=%d, Coll=%d, Task=%d', n_quar, n_coll, n_task);
    title_line3 = sprintf('Rescuing: %s', rescue_str);
    pct_orig = num_original / n_total_spikes * 100;
    pct_rescued = num_rescued / n_total_spikes * 100;
    pct_remain = (n_total_spikes - num_total) / n_total_spikes * 100;
    title_line4 = sprintf('%.1f%% originally clustered, %.1f%% rescued, %.1f%% remaining', pct_orig, pct_rescued, pct_remain);
    sgtitle({title_line1, title_line2, title_line3, title_line4}, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

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

% Define cluster colors for metrics
leicolors = [0 0 0; 0 0 1; 1 0 0; 0 0.5 0; 0.62 0 0; 0.42 0 0.76; ...
             0.97 0.52 0.03; 0.52 0.25 0; 1 0.10 0.72; 0.55 0.55 0.55; ...
             0.59 0.83 0.31; 0.97 0.62 0.86; 0.62 0.76 1.0];

% Create metrics pages

% Compute mask breakdown for rescued and total
rescued_indices = find(rescue_mask);
mask_types = {'Quar', 'Coll', 'Task', 'Quar+Coll', 'Quar+Task', 'Coll+Task', 'Quar+Coll+Task'};
mask_counts = zeros(1, length(mask_types));
for i = 1:length(rescued_indices)
    idx = rescued_indices(i);
    has_quar = mask_quarantine(idx);
    has_coll = mask_collision(idx);
    has_task = mask_task_excluded(idx);
    if has_quar && has_coll && has_task
        mask_counts(7) = mask_counts(7) + 1;
    elseif has_quar && has_coll
        mask_counts(4) = mask_counts(4) + 1;
    elseif has_quar && has_task
        mask_counts(5) = mask_counts(5) + 1;
    elseif has_coll && has_task
        mask_counts(6) = mask_counts(6) + 1;
    elseif has_quar
        mask_counts(1) = mask_counts(1) + 1;
    elseif has_coll
        mask_counts(2) = mask_counts(2) + 1;
    elseif has_task
        mask_counts(3) = mask_counts(3) + 1;
    end
end
total_mask_counts = zeros(1, length(mask_types));
for idx = 1:length(index_all)
    has_quar = mask_quarantine(idx);
    has_coll = mask_collision(idx);
    has_task = mask_task_excluded(idx);
    if has_quar && has_coll && has_task
        total_mask_counts(7) = total_mask_counts(7) + 1;
    elseif has_quar && has_coll
        total_mask_counts(4) = total_mask_counts(4) + 1;
    elseif has_quar && has_task
        total_mask_counts(5) = total_mask_counts(5) + 1;
    elseif has_coll && has_task
        total_mask_counts(6) = total_mask_counts(6) + 1;
    elseif has_quar
        total_mask_counts(1) = total_mask_counts(1) + 1;
    elseif has_coll
        total_mask_counts(2) = total_mask_counts(2) + 1;
    elseif has_task
        total_mask_counts(3) = total_mask_counts(3) + 1;
    end
end

% Page 1: Rescued per cluster
fig_metrics1 = figure('Name', sprintf('Channel %s Rescue Metrics - Per Cluster', ch_lbl), ...
                     'NumberTitle', 'off', ...
                     'Visible', vis_str, ...
                     'Color', 'w', ...
                     'Position', [100, 100, 800, 600]);
subplot(1,1,1);
cluster_ids = unique(cluster_class(:,1));
rescued_per_cluster = zeros(size(cluster_ids));
for i = 1:length(cluster_ids)
    clust_id = cluster_ids(i);
    rescued_per_cluster(i) = sum(cluster_class(:,1) == clust_id & mask_is_rescued);
end
colors = zeros(length(cluster_ids), 3);
for i = 1:length(cluster_ids)
    clust_id = cluster_ids(i);
    color_idx = mod(clust_id, size(leicolors, 1)) + 1;
    colors(i, :) = leicolors(color_idx, :);
end
h = bar(cluster_ids, rescued_per_cluster, 'FaceColor', 'flat', 'CData', colors);
xlabel('Cluster ID');
ylabel('Number of Rescued Spikes');
title(sprintf('Rescued Spikes per Cluster (Total Rescued: %d)', num_rescued));
grid on;
if save_fig
    save_name = fullfile(save_folder, sprintf('%s_metrics_clusters.png', ch_lbl));
    exportgraphics(fig_metrics1, save_name, 'Resolution', 150);
    fprintf('Metrics figure saved: %s\n', save_name);
end
if ~show_fig
    close(fig_metrics1);
end

% Page 3: Clusters and Masks
fig_metrics3 = figure('Name', sprintf('Channel %s Rescue Metrics - Clusters and Masks', ch_lbl), ...
                     'NumberTitle', 'off', ...
                     'Visible', vis_str, ...
                     'Color', 'w', ...
                     'Position', [100, 100, 1200, 800]);
% For each cluster, count rescued by mask type
num_clusters = length(cluster_ids);
mask_matrix = zeros(num_clusters, 7);
for c = 1:num_clusters
    clust_id = cluster_ids(c);
    clust_rescued_indices = find(cluster_class(:,1) == clust_id & mask_is_rescued);
    for i = 1:length(clust_rescued_indices)
        row = clust_rescued_indices(i);
        timestamp = cluster_class(row,2);
        idx = find(index_all == timestamp);
        has_quar = mask_quarantine(idx);
        has_coll = mask_collision(idx);
        has_task = mask_task_excluded(idx);
        if has_quar && has_coll && has_task
            mask_matrix(c,7) = mask_matrix(c,7) + 1;
        elseif has_quar && has_coll
            mask_matrix(c,4) = mask_matrix(c,4) + 1;
        elseif has_quar && has_task
            mask_matrix(c,5) = mask_matrix(c,5) + 1;
        elseif has_coll && has_task
            mask_matrix(c,6) = mask_matrix(c,6) + 1;
        elseif has_quar
            mask_matrix(c,1) = mask_matrix(c,1) + 1;
        elseif has_coll
            mask_matrix(c,2) = mask_matrix(c,2) + 1;
        elseif has_task
            mask_matrix(c,3) = mask_matrix(c,3) + 1;
        end
    end
end
subplot(1,1,1);
x = 1:length(cluster_ids);
h = bar(x, mask_matrix, 'grouped');
set(gca, 'XTick', x, 'XTickLabel', arrayfun(@num2str, cluster_ids, 'UniformOutput', false));
legend(mask_types, 'Location', 'eastoutside');
xlabel('Cluster ID');
ylabel('Number of Rescued Spikes');
title(sprintf('Rescued Spikes by Cluster and Mask Combination (Total Rescued: %d)', num_rescued));
grid on;
if save_fig
    save_name = fullfile(save_folder, sprintf('%s_metrics_clusters_masks.png', ch_lbl));
    exportgraphics(fig_metrics3, save_name, 'Resolution', 150);
    fprintf('Metrics figure saved: %s\n', save_name);
end
if ~show_fig
    close(fig_metrics3);
end

% Page 4: Rescue Efficiency per Mask
fig_metrics4 = figure('Name', sprintf('Channel %s Rescue Metrics - Efficiency per Mask', ch_lbl), ...
                     'NumberTitle', 'off', ...
                     'Visible', vis_str, ...
                     'Color', 'w', ...
                     'Position', [100, 100, 1000, 600]);
subplot(1,1,1);
Y = [total_mask_counts; mask_counts]';  % 7 x 2
h = bar(1:7, Y, 'grouped');
set(gca, 'XTick', 1:7, 'XTickLabel', mask_types, 'ColorOrder', [0.7 0.7 0.7; 0 0 0]);
legend({'Total Quarantined', 'Rescued'}, 'Location', 'northeast');
% Add percentage labels below x-axis: rescued / total for each mask
pct = mask_counts ./ max(total_mask_counts, 1) * 100;  % avoid divide by zero
for i = 1:7
    text(i, -max(Y(:))*0.05, sprintf('%.1f%%', pct(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 8);
end
ylabel('Number of Spikes');
title(sprintf('Rescue Efficiency per Mask Combination (Total Quarantined: %d, Rescued: %d)', sum(total_mask_counts), sum(mask_counts)));
grid on;
if save_fig
    save_name = fullfile(save_folder, sprintf('%s_metrics_efficiency.png', ch_lbl));
    exportgraphics(fig_metrics4, save_name, 'Resolution', 150);
    fprintf('Metrics figure saved: %s\n', save_name);
end
if ~show_fig
    close(fig_metrics4);
end

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
% Helper: 2D waveform density image (copied from make_cluster_report)
function density_image_matlab(W, ax, samplerate_hz, varargin)
    p = inputParser;
    addParameter(p, 'w', 400, @isscalar);
    addParameter(p, 'h', 240, @isscalar);
    addParameter(p, 'cmap', 'inferno', @ischar);
    parse(p, varargin{:});

    if isempty(W)
        axis(ax,'off'); return;
    end

    [n, T] = size(W);
    
    % Add max waveforms limit
    max_waveforms = 5000;
    if n > max_waveforms
        rng(42, 'twister');
        subsample_idx = randperm(n, max_waveforms);
        W = W(subsample_idx, :);
        n = max_waveforms;
    end
    
    x_min = 0; x_max = T-1;
    y_min = min(W(:)); y_max = max(W(:));
    
    pad = 1e-6 * (y_max - y_min);
    if pad == 0, pad = 1e-6; end
    y_min = y_min - pad; y_max = y_max + pad;

    x_edges = linspace(x_min, x_max, p.Results.w+1);
    y_edges = linspace(y_min, y_max, p.Results.h+1);

    % Reduce interpolation factor
    interpolation_factor = 10;
    Ti = max(2, min(T * interpolation_factor, p.Results.w));
    
    x_orig = 0:(T-1);
    x_hi = linspace(x_min, x_max, Ti);
    hi = zeros(n, Ti);
    for i = 1:n
        hi(i,:) = interp1(x_orig, W(i,:), x_hi, 'linear');
    end

    t_flat = repmat(x_hi, 1, n);
    a_flat = hi';
    a_flat = a_flat(:);
    t_flat = t_flat(:);

    H = histcounts2(t_flat, a_flat, x_edges, y_edges);
    
    D = H' ./ (numel(t_flat) * mean(diff(x_edges)) * mean(diff(y_edges)));
    
    D_log = log10(D + eps);

    D_pos = D(D>0);
    if isempty(D_pos), D_pos = 1; end
    vmin_log = log10(max(min(D_pos), 1e-12));
    vmax_log = log10(max(D(:)));
    if vmax_log <= vmin_log, vmax_log = vmin_log + 1; end 
    
    imagesc(ax, [x_min x_max], [y_min y_max], D_log);
    set(ax, 'YDir', 'normal');

    try colormap(ax, p.Results.cmap); catch, colormap(ax, 'hot'); end
    caxis(ax, [vmin_log, vmax_log]);

    set(ax, 'Color', 'white', 'XColor', [0 0 0], 'YColor', [0 0 0], 'Box', 'off');

    nx = min(5, max(2, ceil((x_max - x_min)/max(1, round((T)/10)))));
    xt_pos = round(linspace(x_min, x_max, nx));
    xt_pos = unique(max(0, min(T-1, xt_pos)));
    if ~isempty(xt_pos)
        set(ax, 'XTick', xt_pos);
        if ~isempty(samplerate_hz)
            xtlbls = arrayfun(@(x) sprintf('%.0f', x/samplerate_hz*1e3), xt_pos, 'UniformOutput', false);
            set(ax, 'XTickLabel', xtlbls, 'XColor', [0 0 0]);
            xlabel(ax, 'Time (ms)', 'Color', [0 0 0]);
        else
            xtlbls = arrayfun(@(x) sprintf('%d', x), xt_pos, 'UniformOutput', false);
            set(ax, 'XTickLabel', xtlbls, 'XColor', [0 0 0]);
            xlabel(ax, 'Sample', 'Color', [0 0 0]);
        end
    end

    axis(ax, 'tight');
    set(ax, 'YColor', [0 0 0], 'XColor', [0 0 0], 'Visible', 'on');
end
