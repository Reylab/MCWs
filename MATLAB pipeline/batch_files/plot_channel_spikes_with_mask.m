function plot_channel_spikes_with_mask(spikes, mask, channel_id, varargin)
% PLOT_CHANNEL_SPIKES_WITH_MASK Visualize all spikes, filtered spikes, and remaining spikes
%
% Inputs:
%   spikes      - Matrix of spike waveforms (num_spikes x num_samples)
%   mask        - Logical/binary vector (num_spikes x 1) indicating which spikes to keep
%                 true/1 = keep, false/0 = remove
%   channel_id  - Channel identifier (for plot title)
%   varargin    - Optional name-value pairs:
%                 'FigureHandle' - Existing figure handle to use
%                 'TimeVector'   - Time vector for x-axis (default: sample indices)
%                 'SamplingRate' - Sampling rate in Hz (for time axis in ms)
%                 'Index'        - Index/spike times vector (num_spikes x 1) for indexing
%
% Output:
%   Creates a figure with 3 subplots showing:
%   1. All spikes overlayed
%   2. Filtered (removed) spikes
%   3. Remaining (kept) spikes
%
% Example:
%   plot_channel_spikes_with_mask(spikes_all, mask_nonart, 1)
%   plot_channel_spikes_with_mask(spikes_all, mask_nonart, 1, 'SamplingRate', 30000)
%   plot_channel_spikes_with_mask(spikes_all, mask_nonart, 1, 'Index', index_all)

% Parse optional inputs
p = inputParser;
addParameter(p, 'FigureHandle', [], @(x) isempty(x) || ishandle(x));
addParameter(p, 'TimeVector', [], @isnumeric);
addParameter(p, 'SamplingRate', [], @isnumeric);
addParameter(p, 'Index', [], @isnumeric);
parse(p, varargin{:});

fig_handle = p.Results.FigureHandle;
time_vec = p.Results.TimeVector;
fs = p.Results.SamplingRate;
index_vec = p.Results.Index;

% Validate inputs
if size(spikes, 1) ~= length(mask)
    error('Number of spikes must match length of mask');
end

% Validate index if provided
if ~isempty(index_vec) && length(index_vec) ~= length(mask)
    error('Length of index must match length of mask');
end

% Convert mask to logical if needed
mask = logical(mask);

% Create time vector if not provided
if isempty(time_vec)
    if ~isempty(fs)
        % Convert to milliseconds
        time_vec = (0:size(spikes, 2)-1) / fs * 1000;
        x_label = 'Time (ms)';
    else
        time_vec = 1:size(spikes, 2);
        x_label = 'Sample';
    end
else
    x_label = 'Time (ms)';
end

% Get index subsets if provided
if ~isempty(index_vec)
    index_removed = index_vec(~mask);
    index_kept = index_vec(mask);
end

% Get spike counts
num_total = size(spikes, 1);
num_removed = sum(~mask);
num_remaining = sum(mask);

% Create or use existing figure
if isempty(fig_handle)
    fig_handle = figure('Name', sprintf('Channel %s Spike Visualization', num2str(channel_id)), ...
                       'NumberTitle', 'off', ...
                       'Position', [100, 100, 1200, 400]);
else
    figure(fig_handle);
    clf;
end

% Color settings
color_all = [0.5, 0.5, 0.5, 0.3];  % Gray with transparency
color_removed = [1, 0, 0, 0.3];    % Red with transparency
color_kept = [0, 0, 1, 0.3];       % Blue with transparency

% Subplot 1: All spikes
subplot(1, 3, 1);
hold on;
for i = 1:num_total
    plot(time_vec, spikes(i, :), 'Color', color_all, 'LineWidth', 0.5);
end
hold off;
title(sprintf('All Spikes (n=%d)', num_total), 'FontWeight', 'bold');
xlabel(x_label);
ylabel('Amplitude');
grid on;
box if ~isempty(index_vec)
        title(sprintf('Filtered Spikes (n=%d)\nIndex range: [%d - %d]', ...
              num_removed, min(index_removed), max(index_removed)), ...
              'FontWeight', 'bold', 'Color', 'r');
    else
        title(sprintf('Filtered Spikes (n=%d)', num_removed), 'FontWeight', 'bold', 'Color', 'r');
    end

% Subplot 2: Filtered (removed) spikes
subplot(1, 3, 2);
if num_removed > 0
    hold on;
    removed_spikes = spikes(~mask, :);
    for i = 1:num_removed
        plot(time_vec, removed_spikes(i, :), 'Color', color_removed, 'LineWidth', 0.5);
    end
    hold off;
    title(sprintf('Filtered Spikes (n=%d)', num_removed), 'FontWeight', 'bold', 'Color', 'r');
else
    title('Filtered Spikes (n=0)', 'FontWeight', 'bold', 'Color', 'r');
    text(0.5, 0.5, 'No spikes filtered', 'Units', 'normalized', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
end
xlabel(x_label);
ylabel('Amplitude');
gridif ~isempty(index_vec)
        title(sprintf('Remaining Spikes (n=%d)\nIndex range: [%d - %d]', ...
              num_remaining, min(index_kept), max(index_kept)), ...
              'FontWeight', 'bold', 'Color', 'b');
    else
        title(sprintf('Remaining Spikes (n=%d)', num_remaining), 'FontWeight', 'bold', 'Color', 'b');
    end
box on;

% Subplot 3: Remaining (kept) spikes
subplot(1, 3, 3);
if num_remaining > 0
    hold on;
    kept_spikes = spikes(mask, :);
    for i = 1:num_remaining
        plot(time_vec, kept_spikes(i, :), 'Color', color_kept, 'LineWidth', 0.5);
    end
    hold off;
    title(sprintf('Remaining Spikes (n=%d)', num_remaining), 'FontWeight', 'bold', 'Color', 'b');
else
    title('Remaining Spikes (n=0)', 'FontWeight', 'bold', 'Color', 'b');
    text(0.5, 0.5, 'No spikes remaining', 'Units', 'normalized', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
end
xlabel(x_label);
ylabel('Amplitude');
grid on;
box on;

% Add overall title
sgtitle(sprintf('Channel %s - Spike Mask Visualization', num2str(channel_id)), ...
        'FontSize', 14, 'FontWeight', 'bold');

end
