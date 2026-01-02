function plot_channel_spikes_with_mask(varargin)
% PLOT_CHANNEL_SPIKES_WITH_MASK Visualize all spikes vs kept vs removed
%
% Name-Value Inputs:
%   'channels'     - Channel number(s) (e.g., 1 or [1, 2, 5] or 1:10) [REQUIRED]
%   'mask'         - Name of mask: 'mask_nonart', 'mask_non_quarantine', 'mask_taskspks' [REQUIRED]
%   'spike_dir'    - Directory containing the *_spikes.mat files (default = pwd)
%   'show_figures' - true/false to show figures (default = true)
%   'parallel'     - true/false to use parallel processing (default = false)
%   'max_spikes'   - Maximum number of spikes to plot per panel (default = 2000)
%
% The function looks for files matching: *<channel_num>_spikes.mat
% These files should contain: spikes_all and the specified mask variable
%
% Figures are saved to: <spike_dir>/<channel_id>_<mask_name>.png
%
% Creates a figure per channel with 3 subplots (stacked vertically):
%   1. All spikes (gray)
%   2. Removed spikes (red) - where mask = false
%   3. Kept spikes (blue) - where mask = true
%
% Example:
%   plot_channel_spikes_with_mask('channels', 1, 'mask', 'mask_nonart')
%   plot_channel_spikes_with_mask('channels', [1 2 5], 'mask', 'mask_non_quarantine', 'show_figures', false)
%   plot_channel_spikes_with_mask('channels', 1:10, 'mask', 'mask_taskspks', 'parallel', true, 'max_spikes', 1000)

% Parse inputs
p = inputParser;
addParameter(p, 'channels', [], @isnumeric);
addParameter(p, 'mask', '', @ischar);
addParameter(p, 'spike_dir', pwd, @ischar);
addParameter(p, 'show_figures', true, @islogical);
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'max_spikes', 2000, @isnumeric);
parse(p, varargin{:});

channel_nums = p.Results.channels;
mask_name = p.Results.mask;
spike_dir = p.Results.spike_dir;
show_figures = p.Results.show_figures;
use_parallel = p.Results.parallel;
max_spikes = p.Results.max_spikes;

% Validate required inputs
if isempty(channel_nums)
    error('Must specify ''channels'' parameter');
end
if isempty(mask_name)
    error('Must specify ''mask'' parameter');
end

% Set visibility string
if show_figures
    vis_str = 'on';
else
    vis_str = 'off';
end

% Number of channels
num_channels = length(channel_nums);

% Process channels (parallel or serial)
if use_parallel
    % Check for parallel pool
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('Starting parallel pool...\n');
        parpool;
    end
    fprintf('Processing %d channels in parallel...\n', num_channels);
    parfor ch_idx = 1:num_channels
        process_single_channel(channel_nums(ch_idx), mask_name, spike_dir, vis_str, show_figures, max_spikes);
    end
else
    for ch_idx = 1:num_channels
        process_single_channel(channel_nums(ch_idx), mask_name, spike_dir, vis_str, show_figures, max_spikes);
    end
end

fprintf('Done processing %d channels.\n', num_channels);

end % end main function

%% Helper function for single channel processing
function process_single_channel(channel_num, mask_name, spike_dir, vis_str, show_figures, max_spikes)
    
    % Find the spike file for this channel
    file_pattern = fullfile(spike_dir, sprintf('*%d_spikes.mat', channel_num));
    files = dir(file_pattern);
    
    if isempty(files)
        warning('No spike file found for channel %d, skipping...', channel_num);
        return;
    elseif length(files) > 1
        warning('Multiple files found for channel %d, using: %s', channel_num, files(1).name);
    end
    
    spike_file = fullfile(spike_dir, files(1).name);
    fprintf('Loading: %s\n', spike_file);
    
    % Load the spike file
    data = load(spike_file);
    
    % Get spikes_all
    if isfield(data, 'spikes_all')
        spikes = data.spikes_all;
    elseif isfield(data, 'spikes')
        spikes = data.spikes;
    else
        warning('Channel %d: spike file must contain spikes_all or spikes, skipping...', channel_num);
        return;
    end
    
    % Get the requested mask
    if ~isfield(data, mask_name)
        available = fieldnames(data);
        mask_fields = available(contains(available, 'mask'));
        warning('Channel %d: Mask "%s" not found. Available: %s. Skipping...', ...
                channel_num, mask_name, strjoin(mask_fields, ', '));
        return;
    end
    mask = data.(mask_name);
    
    % Validate
    if size(spikes, 1) ~= length(mask)
        warning('Channel %d: spikes (%d) and mask (%d) size mismatch, skipping...', ...
                channel_num, size(spikes,1), length(mask));
        return;
    end
    
    % Get channel name from filename for title
    [~, fname, ~] = fileparts(files(1).name);
    channel_id = fname;
    
    % Convert mask to logical
    mask = logical(mask);
    
    % Time vector = sample indices (like make_cluster_report)
    tvec = 1:size(spikes, 2);
    
    % Counts
    num_total = size(spikes, 1);
    num_removed = sum(~mask);
    num_kept = sum(mask);
    
    % Create figure (vertical layout: taller than wide, dark figure background)
    fig = figure('Name', sprintf('Channel %s - Mask Visualization', num2str(channel_id)), ...
           'NumberTitle', 'off', ...
           'Visible', vis_str, ...
           'Color', [0.15 0.15 0.15], ...
           'Position', [100, 100, 600, 900]);
    
    % Colors with transparency (like make_cluster_report)
    color_all = [0.5, 0.5, 0.5, 0.15];     % Gray
    color_removed = [1, 0, 0, 0.15];        % Red  
    color_kept = [0, 0, 1, 0.15];           % Blue
    
    % --- Subplot 1: All spikes ---
    ax1 = subplot(3, 1, 1);
    set(ax1, 'Color', 'w');  % White plot background
    hold(ax1, 'on');
    % Subsample if needed
    if num_total > max_spikes
        idx_all = randperm(num_total, max_spikes);
        plot(ax1, tvec, spikes(idx_all, :)', 'Color', color_all, 'LineWidth', 0.8);
    else
        plot(ax1, tvec, spikes', 'Color', color_all, 'LineWidth', 0.8);
    end
    plot(ax1, tvec, mean(spikes, 1), 'k', 'LineWidth', 2.4);
    hold(ax1, 'off');

    title(ax1, sprintf('All Spikes (n=%d)', num_total), 'FontWeight', 'bold');

    xlabel(ax1, 'Samples'); ylabel(ax1, 'Amplitude');
    grid(ax1, 'on'); box(ax1, 'off');
    set(ax1, 'GridAlpha', 0.25, 'LineWidth', 0.8);
    
    % --- Subplot 2: Removed spikes (mask = false) ---
    ax2 = subplot(3, 1, 2);
    set(ax2, 'Color', 'w');  % White plot background
    if num_removed > 0
        removed_spikes = spikes(~mask, :);
        hold(ax2, 'on');
        % Subsample if needed
        if num_removed > max_spikes
            idx_rem = randperm(num_removed, max_spikes);
            plot(ax2, tvec, removed_spikes(idx_rem, :)', 'Color', color_removed, 'LineWidth', 0.8);
        else
            plot(ax2, tvec, removed_spikes', 'Color', color_removed, 'LineWidth', 0.8);
        end
        plot(ax2, tvec, mean(removed_spikes, 1), 'k', 'LineWidth', 2.4);
        hold(ax2, 'off');
        title(ax2, sprintf('Removed (n=%d)', num_removed), 'FontWeight', 'bold', 'Color', 'r');

    else
        title(ax2, 'Removed (n=0)', 'FontWeight', 'bold', 'Color', 'r');
        text(ax2, 0.5, 0.5, 'None removed', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    xlabel(ax2, 'Samples'); ylabel(ax2, 'Amplitude');
    grid(ax2, 'on'); box(ax2, 'off');
    set(ax2, 'GridAlpha', 0.25, 'LineWidth', 0.8);
    
    % --- Subplot 3: Kept spikes (mask = true) ---
    ax3 = subplot(3, 1, 3);
    set(ax3, 'Color', 'w');  % White plot background
    if num_kept > 0
        kept_spikes = spikes(mask, :);
        hold(ax3, 'on');
        % Subsample if needed
        if num_kept > max_spikes
            idx_kept = randperm(num_kept, max_spikes);
            plot(ax3, tvec, kept_spikes(idx_kept, :)', 'Color', color_kept, 'LineWidth', 0.8);
        else
            plot(ax3, tvec, kept_spikes', 'Color', color_kept, 'LineWidth', 0.8);
        end
        plot(ax3, tvec, mean(kept_spikes, 1), 'k', 'LineWidth', 2.4);
        hold(ax3, 'off');
        title(ax3, sprintf('Kept (n=%d)', num_kept), 'FontWeight', 'bold', 'Color', 'b');

    else
        title(ax3, 'Kept (n=0)', 'FontWeight', 'bold', 'Color', 'b');
        text(ax3, 0.5, 0.5, 'None kept', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    xlabel(ax3, 'Samples'); ylabel(ax3, 'Amplitude');
    grid(ax3, 'on'); box(ax3, 'off');
    set(ax3, 'GridAlpha', 0.25, 'LineWidth', 0.8);
    
    % Overall title (white text for dark background)
    sgtitle(sprintf('%s - %s', channel_id, mask_name), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
    
    % Force figure to render before saving
    drawnow;
    
    % Save figure to folder named after mask
    save_folder = fullfile(spike_dir, mask_name);
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    save_name = fullfile(save_folder, sprintf('%s.png', channel_id));
    
    % Use exportgraphics (works better with invisible figures and parallel)
    exportgraphics(fig, save_name, 'Resolution', 150);
    fprintf('Saved: %s\n', save_name);
    
    % Close if not showing
    if ~show_figures
        close(fig);
    end

end % end helper function
