function plot_channel_spikes_with_mask(varargin)
% PLOT_CHANNEL_SPIKES_WITH_MASK Visualize all spikes vs kept vs removed
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
%   plot_channel_spikes_with_mask('channels', 1, 'mask', 'mask_nonart', 'rescue', true)

% Parse inputs
p = inputParser;
addParameter(p, 'channels', [], @isnumeric);
addParameter(p, 'mask', '', @ischar);
addParameter(p, 'spike_dir', pwd, @ischar);
addParameter(p, 'show_figures', false, @islogical);
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'max_spikes', 5000, @isnumeric);
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
    
    % (mask validation moved below after num_total is known)
    
    % Get channel name from filename for title
    [~, fname, ~] = fileparts(files(1).name);
    channel_id = fname;
    
    % convert mask later after validating shape
    
    % Time vector = sample indices (like make_cluster_report)
    tvec = 1:size(spikes, 2);
    
    % Counts
    num_total = size(spikes, 1);

    % Convert mask to logical column vector and validate shape strictly
    mask = logical(mask);
    if ~isvector(mask) || numel(mask) ~= num_total
        error('plot_channel_spikes_with_mask:BadMask', 'Mask ''%s'' must be a logical vector with length equal to number of spikes (%d).', mask_name, num_total);
    end
    mask = mask(:);

    num_removed = sum(~mask);
    num_kept = sum(mask);
    num_rows = 3;
    
    % Define y-limits from spike data: min/max across all spikes, plus 10 units margin
    y_min = min(spikes(:));
    y_max = max(spikes(:));
    margin = 10;
    y_limits = [y_min - margin, y_max + margin];

    % --- Compute task/time/spike ratios for annotations ---
    % Robustly extract spike timestamps (index_all or index), handling numeric or cell formats
    idx_ts = [];
    if isfield(data, 'index_all') && ~isempty(data.index_all)
        idx_field = data.index_all;
    elseif isfield(data, 'index') && ~isempty(data.index)
        idx_field = data.index;
    else
        idx_field = [];
    end
    if ~isempty(idx_field)
        if isnumeric(idx_field)
            idx_ts = double(idx_field(:));
        elseif iscell(idx_field)
            try
                idx_ts = double(vertcat(idx_field{:}));
            catch
                try
                    idx_ts = double(cell2mat(idx_field));
                catch
                    idx_ts = [];
                end
            end
        end
    end

    % Sequence (task) periods (if present) - accept numeric matrix or cell of rows
    seq_mat = [];
    if isfield(data, 'seq_beg_end_mat') && ~isempty(data.seq_beg_end_mat)
        s = data.seq_beg_end_mat;
        if isnumeric(s)
            seq_mat = double(s);
        elseif iscell(s)
            try
                seq_mat = double(vertcat(s{:}));
            catch
                try
                    seq_mat = double(cell2mat(s));
                catch
                    seq_mat = [];
                end
            end
        end
    end

    % Task mask saved per-spike (if present) - handle numeric/cell/logical
    mask_task_saved = [];
    if isfield(data, 'mask_taskspks') && ~isempty(data.mask_taskspks)
        m = data.mask_taskspks;
        if islogical(m) || isnumeric(m)
            mask_task_saved = logical(m(:));
        elseif iscell(m)
            try
                mask_task_saved = logical(vertcat(m{:}));
            catch
                try
                    mask_task_saved = logical(cell2mat(m));
                catch
                    mask_task_saved = [];
                end
            end
        end
    end

    % Compute per-spike task membership based on timestamps.
    % Best practice: derive membership from `seq_beg_end_mat` + `index`/`index_all` whenever possible
    % because all masks relate to the same `spikes_all` timestamps. Only fall back to a saved
    % `mask_taskspks` if time information is not available.
    mask_task_time = [];
    if ~isempty(seq_mat) && ~isempty(idx_ts) && numel(idx_ts) == num_total
        mask_task_time = false(num_total,1);
        for r = 1:size(seq_mat,1)
            mask_task_time = mask_task_time | (idx_ts >= seq_mat(r,1) & idx_ts <= seq_mat(r,2));
        end
    elseif ~isempty(mask_task_saved) && numel(mask_task_saved) == num_total
        mask_task_time = mask_task_saved;
    end

    % Ensure mask_task_time is a logical column vector matching spikes
    if ~isempty(mask_task_time)
        mask_task_time = logical(mask_task_time(:));
        if numel(mask_task_time) ~= num_total
            error('plot_channel_spikes_with_mask:BadMaskShape', 'Per-spike task mask must be a vector matching number of spikes (%d).', num_total);
        end
    end

    % Enforce presence of time-based task membership for meaningful percentages
    if isempty(mask_task_time)
        error('plot_channel_spikes_with_mask:MissingTaskInfo', ['Missing per-spike task membership information.\n' ...
            'Provide ''seq_beg_end_mat'' with matching ''index''/''index_all'' timestamps,\n' ...
            'or supply ''mask_taskspks'' as a logical vector matching spikes.']);
    end

    % Time-based task ratio (if seq_mat available)
    time_task_ratio = NaN;
    if ~isempty(seq_mat)
        % compute total time covered by timestamps if available, otherwise use seq_mat span
        if ~isempty(idx_ts)
            total_time = double(max(idx_ts) - min(idx_ts));
        else
            total_time = double(max(seq_mat(:)) - min(seq_mat(:)));
        end
        if total_time > 0
            task_time = double(sum(seq_mat(:,2) - seq_mat(:,1)));
            time_task_ratio = task_time / total_time;
        end
    end

    % Spike removal ratios (based on the active mask)
    removed_idx = ~mask;
    num_removed = sum(removed_idx);
    num_kept = sum(mask);

    % Task-related spike counts based on time-membership (mask_task_time)
    task_total = NaN; task_removed = NaN; task_kept = NaN; frac_removed_are_task = NaN;
    if ~isempty(mask_task_time)
        task_total = sum(mask_task_time);
        task_removed = sum(mask_task_time & removed_idx);
        task_kept = sum(mask_task_time & mask);
        if num_removed > 0
            frac_removed_are_task = task_removed / num_removed;
        end
    end

    % (compute values above; annotation will be concise and placed later)
    
    % Create figure (vertical layout: taller than wide, white figure background)
        fig = figure('Name', sprintf('Channel %s - Mask Visualization', num2str(channel_id)), ...
            'NumberTitle', 'off', ...
            'Visible', vis_str, ...
            'Color', 'w', ...
            'Position', [100, 100, 1400, 900]);
    
    % Colors with transparency (like make_cluster_report)
    color_all = [0.5, 0.5, 0.5, 0.15];     % Gray
    color_removed = [1, 0, 0, 0.15];        % Red  
    color_kept = [0, 0, 1, 0.15];           % Blue
    
    % --- Subplot 1: All spikes (sampled, stratified) ---
    ax1 = subplot(3, 1, 1);
    set(ax1, 'Color', 'w');
    hold(ax1, 'on');

    n_all = min(5000, num_total);
    % Stratified sampling: keep ratio of kept/masked
    masked_quota = min(num_removed, max(1, round(n_all * num_removed / num_total)));
    kept_quota = min(num_kept, n_all - masked_quota);
    if kept_quota + masked_quota < n_all
        extra = n_all - kept_quota - masked_quota;
        masked_quota = min(num_removed, masked_quota + extra);
    end

    idx_masked_pool = find(~mask);
    idx_kept_pool = find(mask);
    if masked_quota > 0
        idx_masked = idx_masked_pool(randperm(num_removed, masked_quota));
    else
        idx_masked = [];
    end
    if kept_quota > 0
        idx_kept = idx_kept_pool(randperm(num_kept, kept_quota));
    else
        idx_kept = [];
    end
    idx_all = [idx_kept idx_masked];
    % Ensure column vectors and concatenate vertically to form a single index vector
    idx_masked = idx_masked(:);
    idx_kept = idx_kept(:);
    idx_all = [idx_kept; idx_masked];

    plot(ax1, tvec, spikes(idx_all, :), 'Color', color_all, 'LineWidth', 0.8);
    plot(ax1, tvec, mean(spikes, 1), 'k', 'LineWidth', 2.4);
    hold(ax1, 'off');


    title(ax1, sprintf('All Spikes: %d total', num_total), 'FontWeight', 'bold');
    ylim(ax1, y_limits);
    xlabel(ax1, 'Samples'); ylabel(ax1, 'Amplitude');
    grid(ax1, 'on'); box(ax1, 'off');
    set(ax1, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    % --- Subplot 2: Masked spikes from sampled set ---
    ax2 = subplot(3, 1, 2);
    set(ax2, 'Color', 'w');
    idx_masked_in_sample = idx_all(ismember(idx_all, idx_masked_pool));
    n_masked_in_sample = length(idx_masked_in_sample);
    if n_masked_in_sample > 0
        removed_spikes = spikes(idx_masked_in_sample, :);
        hold(ax2, 'on');
        plot(ax2, tvec, removed_spikes', 'Color', color_removed, 'LineWidth', 0.8);
        plot(ax2, tvec, mean(removed_spikes, 1), 'k', 'LineWidth', 2.4);
        hold(ax2, 'off');
        title(ax2, sprintf('Masked in Sample: %d (%.1f%% of sample, %.1f%% of all masked)', n_masked_in_sample, 100*n_masked_in_sample/length(idx_all), 100*n_masked_in_sample/max(num_removed,1)), 'FontWeight', 'bold', 'Color', 'r');
    else
        title(ax2, sprintf('Masked in Sample: 0 (0.0%%)'), 'FontWeight', 'bold', 'Color', 'r');
        text(ax2, 0.5, 0.5, 'None removed in sample', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    ylim(ax2, y_limits);
    xlabel(ax2, 'Samples'); ylabel(ax2, 'Amplitude');
    grid(ax2, 'on'); box(ax2, 'off');
    set(ax2, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    % --- Subplot 3: Kept spikes from sampled set ---
    ax3 = subplot(3, 1, 3);
    set(ax3, 'Color', 'w');
    idx_kept_in_sample = idx_all(ismember(idx_all, idx_kept_pool));
    n_kept_in_sample = length(idx_kept_in_sample);
    if n_kept_in_sample > 0
        kept_spikes = spikes(idx_kept_in_sample, :);
        hold(ax3, 'on');
        plot(ax3, tvec, kept_spikes', 'Color', color_kept, 'LineWidth', 0.8);
        plot(ax3, tvec, mean(kept_spikes, 1), 'k', 'LineWidth', 2.4);
        hold(ax3, 'off');
        title(ax3, sprintf('Kept in Sample: %d (%.1f%% of sample, %.1f%% of all kept)', n_kept_in_sample, 100*n_kept_in_sample/length(idx_all), 100*n_kept_in_sample/max(num_kept,1)), 'FontWeight', 'bold', 'Color', 'b');
    else
        title(ax3, sprintf('Kept in Sample: 0 (0.0%%)'), 'FontWeight', 'bold', 'Color', 'b');
        text(ax3, 0.5, 0.5, 'None kept in sample', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    ylim(ax3, y_limits);
    xlabel(ax3, 'Samples'); ylabel(ax3, 'Amplitude');
    grid(ax3, 'on'); box(ax3, 'off');
    set(ax3, 'GridAlpha', 0.25, 'LineWidth', 0.8);

    ax3_den = nexttile(t, 8);
    if exist('idx_kept_in_sample','var') && ~isempty(idx_kept_in_sample)
        W_kept = spikes(idx_kept_in_sample, :);
    else
        W_kept = [];
    end
    density_image_matlab(W_kept, ax3_den, [], 'cmap', 'inferno');
    set(ax3_den, 'YLim', y_limits);

    % Text axis for row 3 (tile 9)
    ax3_txt = nexttile(t, 9);
    axis(ax3_txt, 'off');
    
    % (Removed duplicate density-axes creation — density images are
    % already created using the tiled layout above.)

    % Overall title
    sgtitle(sprintf('%s - %s', channel_id, mask_name), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');

    % Build concise side-summary
    try
        % Time ratio (show once in top text tile)
        if ~isnan(time_task_ratio)
            time_task_pct = round(100 * time_task_ratio);
            time_nontask_pct = 100 - time_task_pct;
            L_time = sprintf('Time: %d%% task | %d%% non-task', time_task_pct, time_nontask_pct);
        else
            L_time = 'Time: N/A';
        end

        % Debug: print internal counts used to compute percentages
        % fprintf('DEBUG mask_name=%s, num_total=%d, num_removed=%d, num_kept=%d\n', mask_name, num_total, num_removed, num_kept);
        % if ~isempty(mask_task_time)
        %     fprintf('DEBUG mask_task_time: length=%d, sum(task)=%d\n', numel(mask_task_time), sum(mask_task_time));
        % else
        %     fprintf('DEBUG mask_task_time: <empty>\n');
        % end
        % if ~isempty(idx_ts)
        %     fprintf('DEBUG idx_ts: length=%d, min=%g, max=%g\n', numel(idx_ts), min(idx_ts), max(idx_ts));
        % else
        %     fprintf('DEBUG idx_ts: <empty>\n');
        % end
        % if ~isempty(seq_mat)
        %     fprintf('DEBUG seq_mat: size=%dx%d, first=[%g %g]\n', size(seq_mat,1), size(seq_mat,2), seq_mat(1,1), seq_mat(1,2));
        % else
        %     fprintf('DEBUG seq_mat: <empty>\n');
        % end

            % If a saved mask exists, show its stats and overlap with time-derived mask
            if ~isempty(mask_task_saved)
                % fprintf('DEBUG mask_task_saved: length=%d, sum=%d\n', numel(mask_task_saved), sum(mask_task_saved));
                if ~isempty(mask_task_time)
                    common = sum(mask_task_saved & mask_task_time);
                    only_saved = sum(mask_task_saved & ~mask_task_time);
                    only_time = sum(~mask_task_saved & mask_task_time);
                    % fprintf('DEBUG overlap saved vs time: common=%d, only_saved=%d, only_time=%d\n', common, only_saved, only_time);
                end
            else
                % fprintf('DEBUG mask_task_saved: <empty>\n');
            end

        % Overall spikes (all): percent in task vs non-task (based on time-membership)
        if ~isempty(mask_task_time)
            spikes_task_pct_all = round(100 * double(sum(mask_task_time)) / max(1, double(num_total)));
            spikes_nontask_pct_all = 100 - spikes_task_pct_all;
            L_spikes_all = sprintf('spikes_all: %d%% task | %d%% non-task', spikes_task_pct_all, spikes_nontask_pct_all);
        else
            L_spikes_all = 'spikes_all: N/A';
        end

        % Removed row: percent of removed spikes that are task vs non-task (use time-based membership)
        if ~isempty(mask_task_time) && num_removed > 0
            removed_task_pct = round(100 * double(task_removed) / max(1, double(num_removed)));
            removed_nontask_pct = 100 - removed_task_pct;
            L_removed = sprintf('Removed: %d%% task | %d%% non-task', removed_task_pct, removed_nontask_pct);
        elseif num_removed == 0
            L_removed = 'Removed: 0 spikes';
        else
            L_removed = 'Removed: N/A';
        end

        % Kept row: percent of kept spikes that are task vs non-task (use time-based membership)
        if ~isempty(mask_task_time) && num_kept > 0
            kept_task_pct = round(100 * double(task_kept) / max(1, double(num_kept)));
            kept_nontask_pct = 100 - kept_task_pct;
            L_kept = sprintf('Kept: %d%% task | %d%% non-task', kept_task_pct, kept_nontask_pct);
        elseif num_kept == 0
            L_kept = 'Kept: 0 spikes';
        else
            L_kept = 'Kept: N/A';
        end

        % Ensure char and truncate excessively long strings
        L_time = char(L_time);
        L_removed = char(L_removed);
        L_kept = char(L_kept);
        max_len = 200;
        if numel(L_time) > max_len, L_time = [L_time(1:max_len-3) '...']; end
        if numel(L_removed) > max_len, L_removed = [L_removed(1:max_len-3) '...']; end
        if numel(L_kept) > max_len, L_kept = [L_kept(1:max_len-3) '...']; end

        % Print debug values to console to help diagnose display issues
        % fprintf('Side-text TOP: %s\n', L_time);
        % fprintf('Side-text REMOVED: %s\n', L_removed);
        % fprintf('Side-text KEPT: %s\n', L_kept);

        % Place summaries into the right-hand text axes (top shows time + spikes_all)
        try
            cla(ax1_txt);
            text(ax1_txt, 0.5, 0.62, {L_time}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
            text(ax1_txt, 0.5, 0.38, {L_spikes_all}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'Interpreter', 'none');
        catch ME
            warning('Failed to write top text: %s (len=%d)', ME.message, numel(L_time));
            cla(ax1_txt);
            text(ax1_txt, 0.5, 0.5, {'Time: N/A'}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Interpreter', 'none');
        end

        try
            cla(ax2_txt);
            text(ax2_txt, 0.5, 0.5, {L_removed}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'Interpreter', 'none');
        catch ME
            warning('Failed to write removed text: %s (len=%d)', ME.message, numel(L_removed));
            cla(ax2_txt);
            text(ax2_txt, 0.5, 0.5, {'Removed: N/A'}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Interpreter', 'none');
        end

        try
            cla(ax3_txt);
            text(ax3_txt, 0.5, 0.5, {L_kept}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'Interpreter', 'none');
        catch ME
            warning('Failed to write kept text: %s (len=%d)', ME.message, numel(L_kept));
            cla(ax3_txt);
            text(ax3_txt, 0.5, 0.5, {'Kept: N/A'}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Interpreter', 'none');
        end

        if num_rows > 3
            try
                cla(ax4_txt);
                text(ax4_txt, 0.5, 0.5, {'Rescued from this mask'}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'Interpreter', 'none');
            catch ME
                warning('Failed to write rescued text: %s', ME.message);
                cla(ax4_txt);
                text(ax4_txt, 0.5, 0.5, {'Rescued: N/A'}, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Interpreter', 'none');
            end
        end
    catch
        % ignore annotation errors
    end
    
    % Force figure to render before saving
    drawnow;
    
    % Save figure to folder named after mask
    save_folder = fullfile(spike_dir, mask_name);
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    save_name = fullfile(save_folder, sprintf('%s.png', channel_id));
    
    % Use exportgraphics (works better with invisible figures and parallel)
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

