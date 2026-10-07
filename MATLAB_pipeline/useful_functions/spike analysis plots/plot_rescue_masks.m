function plot_rescue_masks(varargin)
% PLOT_RESCUE_MASKS Visualize spikes with rescue data for a given mask
%
% Name-Value Inputs:
%   'channels'     - Channel number(s) (e.g., 1 or [1, 2, 5] or 1:10) [REQUIRED]
%   'mask'         - Name of mask: 'mask_nonart', 'mask_non_quarantine', 'mask_taskspks' [REQUIRED]
%   'spike_dir'    - Directory containing the *_spikes.mat files (default = pwd)
%   'show_figures' - true/false to show figures (default = true)
%   'parallel'     - true/false to use parallel processing (default = false)
%   'max_spikes'   - Maximum number of spikes to plot per panel (default = 5000)
%
% The function looks for files matching: *<channel_num>_spikes.mat
% These files should contain: spikes_all, the specified mask variable, and rescue_mask
%
% Figures are saved to: <spike_dir>/<channel_id>_<mask_name>_rescue.png
%
% Creates a figure per channel with subplots (stacked vertically):
%   1. All spikes (gray) - sampled
%   2. Removed spikes (red) - where mask = false and not rescued
%   3. Rescued spikes (green) - where rescue_mask = true and mask = false
%   4. Kept spikes (blue) - where mask = true
%
% Example:
%   plot_rescue_masks('channels', 1, 'mask', 'mask_nonart')

% Parse inputs
p = inputParser;
addParameter(p, 'channels', [], @isnumeric);
addParameter(p, 'mask', '', @ischar);
addParameter(p, 'spike_dir', pwd, @ischar);
addParameter(p, 'show_figures', false, @islogical);
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'max_spikes', 5000, @isnumeric);
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
    
    % Get rescue_mask
    if ~isfield(data, 'rescue_mask')
        warning('Channel %d: rescue_mask not found, skipping...', channel_num);
        return;
    end
    rescue_mask = data.rescue_mask;
    
    % Get channel name from filename for title
    [~, fname, ~] = fileparts(files(1).name);
    channel_id = fname;
    
    % Time vector = sample indices
    tvec = 1:size(spikes, 2);
    
    % Counts
    num_total = size(spikes, 1);

    % Convert masks to logical column vectors and validate shape strictly
    mask = logical(mask);
    if ~isvector(mask) || numel(mask) ~= num_total
        error('plot_rescue_masks:BadMask', 'Mask ''%s'' must be a logical vector with length equal to number of spikes (%d).', mask_name, num_total);
    end
    mask = mask(:);
    
    rescue_mask = logical(rescue_mask);
    if ~isvector(rescue_mask) || numel(rescue_mask) ~= num_total
        error('plot_rescue_masks:BadRescueMask', 'rescue_mask must be a logical vector with length equal to number of spikes (%d).', num_total);
    end
    rescue_mask = rescue_mask(:);

    % Define categories
    removed_not_rescued = ~mask & ~rescue_mask;
    rescued = rescue_mask & ~mask;
    kept = mask;
    
    num_removed_not_rescued = sum(removed_not_rescued);
    num_rescued = sum(rescued);
    num_kept = sum(kept);
    
    num_rows = 4;  % All, Removed (not rescued), Rescued, Kept
    
    % Define y-limits from spike data: min/max across all spikes, plus 10 units margin
    y_min = min(spikes(:));
    y_max = max(spikes(:));
    margin = 10;
    y_limits = [y_min - margin, y_max + margin];

    % Create figure
    fig = figure('Name', sprintf('Channel %s - Rescue Visualization', num2str(channel_id)), ...
        'NumberTitle', 'off', ...
        'Visible', vis_str, ...
        'Color', 'w', ...
        'Position', [100, 100, 1400, 1200]);  % Taller for 4 rows
    
    % Colors with transparency
    color_all = [0.5, 0.5, 0.5, 0.15];     % Gray
    color_removed = [1, 0, 0, 0.15];        % Red  
    color_rescued = [0, 1, 0, 0.15];        % Green
    color_kept = [0, 0, 1, 0.15];           % Blue
    
    % Use a tiled layout (4 rows x 3 columns): waveform | density | text
    t = tiledlayout(fig, num_rows, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Sampling: stratified across categories
    n_all = min(max_spikes, num_total);
    % Quotas for each category
    total_removed = num_removed_not_rescued + num_rescued;
    if total_removed > 0
        removed_quota = max(1, round(n_all * num_removed_not_rescued / num_total));
        rescued_quota = max(1, round(n_all * num_rescued / num_total));
    else
        removed_quota = 0;
        rescued_quota = 0;
    end
    kept_quota = max(1, round(n_all * num_kept / num_total));
    
    % Adjust to fit n_all
    total_quota = removed_quota + rescued_quota + kept_quota;
    if total_quota > n_all
        scale = n_all / total_quota;
        removed_quota = round(removed_quota * scale);
        rescued_quota = round(rescued_quota * scale);
        kept_quota = n_all - removed_quota - rescued_quota;
    elseif total_quota < n_all
        extra = n_all - total_quota;
        kept_quota = kept_quota + extra;  % Add extra to kept
    end
    
    % Sample indices
    idx_removed_pool = find(removed_not_rescued);
    idx_rescued_pool = find(rescued);
    idx_kept_pool = find(kept);
    
    if removed_quota > 0 && ~isempty(idx_removed_pool)
        idx_removed = idx_removed_pool(randperm(length(idx_removed_pool), min(removed_quota, length(idx_removed_pool))));
    else
        idx_removed = [];
    end
    if rescued_quota > 0 && ~isempty(idx_rescued_pool)
        idx_rescued = idx_rescued_pool(randperm(length(idx_rescued_pool), min(rescued_quota, length(idx_rescued_pool))));
    else
        idx_rescued = [];
    end
    if kept_quota > 0 && ~isempty(idx_kept_pool)
        idx_kept = idx_kept_pool(randperm(length(idx_kept_pool), min(kept_quota, length(idx_kept_pool))));
    else
        idx_kept = [];
    end
    
    idx_all = [idx_removed; idx_rescued; idx_kept];

    % --- Row 1: All spikes ---
    ax1 = nexttile(t, 1);
    set(ax1, 'Color', 'w'); hold(ax1, 'on');
    if ~isempty(idx_all)
        plot(ax1, tvec, spikes(idx_all, :), 'Color', color_all, 'LineWidth', 0.8);
        plot(ax1, tvec, mean(spikes, 1), 'k', 'LineWidth', 2.4);
    end
    hold(ax1, 'off');
    title(ax1, sprintf('All Spikes: %d total', num_total), 'FontWeight', 'bold');
    ylim(ax1, y_limits);
    xlabel(ax1, 'Samples'); ylabel(ax1, 'Amplitude');
    grid(ax1, 'on'); box(ax1, 'off');
    set(ax1, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    ax1_den = nexttile(t, 2);
    if ~isempty(idx_all)
        W_all = spikes(idx_all, :);
    else
        W_all = spikes;
    end
    density_image_matlab(W_all, ax1_den, [], 'cmap', 'inferno');
    set(ax1_den, 'YLim', y_limits);

    ax1_txt = nexttile(t, 3);
    axis(ax1_txt, 'off');

    % --- Row 2: Removed (not rescued) ---
    ax2 = nexttile(t, 4);
    set(ax2, 'Color', 'w'); hold(ax2, 'on');
    if ~isempty(idx_removed)
        plot(ax2, tvec, spikes(idx_removed, :), 'Color', color_removed, 'LineWidth', 0.8);
        plot(ax2, tvec, mean(spikes(idx_removed, :), 1), 'k', 'LineWidth', 2.4);
    end
    hold(ax2, 'off');
    title(ax2, sprintf('Removed (not rescued): %d (%.1f%%)', num_removed_not_rescued, 100*num_removed_not_rescued/num_total), 'FontWeight', 'bold', 'Color', 'r');
    ylim(ax2, y_limits);
    xlabel(ax2, 'Samples'); ylabel(ax2, 'Amplitude');
    grid(ax2, 'on'); box(ax2, 'off');
    set(ax2, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    ax2_den = nexttile(t, 5);
    if ~isempty(idx_removed)
        W_removed = spikes(idx_removed, :);
    else
        W_removed = [];
    end
    density_image_matlab(W_removed, ax2_den, [], 'cmap', 'inferno');
    set(ax2_den, 'YLim', y_limits);

    ax2_txt = nexttile(t, 6);
    axis(ax2_txt, 'off');

    % --- Row 3: Rescued ---
    ax3 = nexttile(t, 7);
    set(ax3, 'Color', 'w'); hold(ax3, 'on');
    if ~isempty(idx_rescued)
        plot(ax3, tvec, spikes(idx_rescued, :), 'Color', color_rescued, 'LineWidth', 0.8);
        plot(ax3, tvec, mean(spikes(idx_rescued, :), 1), 'k', 'LineWidth', 2.4);
    end
    hold(ax3, 'off');
    title(ax3, sprintf('Rescued: %d (%.1f%%)', num_rescued, 100*num_rescued/num_total), 'FontWeight', 'bold', 'Color', 'g');
    ylim(ax3, y_limits);
    xlabel(ax3, 'Samples'); ylabel(ax3, 'Amplitude');
    grid(ax3, 'on'); box(ax3, 'off');
    set(ax3, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    ax3_den = nexttile(t, 8);
    if ~isempty(idx_rescued)
        W_rescued = spikes(idx_rescued, :);
    else
        W_rescued = [];
    end
    density_image_matlab(W_rescued, ax3_den, [], 'cmap', 'inferno');
    set(ax3_den, 'YLim', y_limits);

    ax3_txt = nexttile(t, 9);
    axis(ax3_txt, 'off');

    % --- Row 4: Kept ---
    ax4 = nexttile(t, 10);
    set(ax4, 'Color', 'w'); hold(ax4, 'on');
    if ~isempty(idx_kept)
        plot(ax4, tvec, spikes(idx_kept, :), 'Color', color_kept, 'LineWidth', 0.8);
        plot(ax4, tvec, mean(spikes(idx_kept, :), 1), 'k', 'LineWidth', 2.4);
    end
    hold(ax4, 'off');
    title(ax4, sprintf('Kept: %d (%.1f%%)', num_kept, 100*num_kept/num_total), 'FontWeight', 'bold', 'Color', 'b');
    ylim(ax4, y_limits);
    xlabel(ax4, 'Samples'); ylabel(ax4, 'Amplitude');
    grid(ax4, 'on'); box(ax4, 'off');
    set(ax4, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    ax4_den = nexttile(t, 11);
    if ~isempty(idx_kept)
        W_kept = spikes(idx_kept, :);
    else
        W_kept = [];
    end
    density_image_matlab(W_kept, ax4_den, [], 'cmap', 'inferno');
    set(ax4_den, 'YLim', y_limits);

    ax4_txt = nexttile(t, 12);
    axis(ax4_txt, 'off');

    % Overall title
    sgtitle(sprintf('%s - %s Rescue', channel_id, mask_name), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');

    % Force figure to render before saving
    drawnow;
    
    % Save figure
    save_folder = fullfile(spike_dir, sprintf('%s_rescue', mask_name));
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    save_name = fullfile(save_folder, sprintf('%s.png', channel_id));
    
    exportgraphics(fig, save_name, 'Resolution', 300);
    fprintf('Saved: %s\n', save_name);
    
    % Close if not showing
    if ~show_figures
        close(fig);
    end

end % end helper function

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