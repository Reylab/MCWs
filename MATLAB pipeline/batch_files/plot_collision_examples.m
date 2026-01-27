function plot_collision_examples(bundle_name, varargin)
% PLOT_COLLISION_EXAMPLES Plot up to 5 random collision events from saved spike files
% Shows all bundles involved in each collision with different colors per bundle
%
% Inputs:
%   bundle_name - Name of bundle to analyze (e.g., 'arrayA1')
%
% Name-Value Inputs:
%   'num_examples' - Number of collision examples to plot (default = 5)
%   'spike_dir'    - Directory containing spike files (default = pwd)
%   'show_figure'  - true/false to show figure (default = true)
%
% Example:
%   plot_collision_examples('arrayA1')
%   plot_collision_examples('arrayA1', 'num_examples', 3, 'show_figure', false)

% Parse inputs
p = inputParser;
addRequired(p, 'bundle_name', @ischar);
addParameter(p, 'num_examples', 5, @isnumeric);
addParameter(p, 'spike_dir', pwd, @ischar);
addParameter(p, 'show_figure', true, @islogical);
parse(p, bundle_name, varargin{:});

num_examples = p.Results.num_examples;
spike_dir = p.Results.spike_dir;
show_figure = p.Results.show_figure;

% Load NSx structure
load('NSx', 'NSx');

% Find channels for this bundle
pos_chans = find(arrayfun(@(x) strcmp(x.bundle, bundle_name), NSx));

if isempty(pos_chans)
    error('Bundle "%s" not found in NSx', bundle_name);
end

fprintf('Loading spike data for bundle: %s (%d channels)\n', bundle_name, length(pos_chans));

% Reconstruct all_spktimes, which_chan, and waveforms from saved files
all_spktimes = [];
which_chan = [];
all_waveforms = [];
chan_spike_map = struct(); % Map to store channel's spike data

for k = 1:length(pos_chans)
    spike_file = fullfile(spike_dir, sprintf('%s_spikes.mat', NSx(pos_chans(k)).output_name));
    if ~exist(spike_file, 'file')
        warning('Spike file not found: %s', spike_file);
        continue;
    end
    
    SPK = load(spike_file);
    
    % Get all spike times and waveforms (not just the kept ones)
    if isfield(SPK, 'index_all')
        spike_times = SPK.index_all;
        waveforms = SPK.spikes_all;
    else
        spike_times = SPK.index;
        waveforms = SPK.spikes;
    end
    
    % Store mapping for this channel
    chan_id = NSx(pos_chans(k)).chan_ID;
    chan_spike_map(k).chan_id = chan_id;
    chan_spike_map(k).spike_times = spike_times;
    chan_spike_map(k).waveforms = waveforms;
    
    all_spktimes = [all_spktimes spike_times];
    which_chan = [which_chan chan_id * ones(size(spike_times))];
end

% Sort them
[all_spktimes, II] = sort(all_spktimes);
which_chan = which_chan(II);

fprintf('Total spikes loaded: %d\n', length(all_spktimes));

% Find artifact indices from mask_nonart (false = artifact)
artifact_idxs = [];
for k = 1:length(pos_chans)
    spike_file = fullfile(spike_dir, sprintf('%s_spikes.mat', NSx(pos_chans(k)).output_name));
    if ~exist(spike_file, 'file')
        continue;
    end
    
    SPK = load(spike_file);
    if isfield(SPK, 'mask_nonart') && isfield(SPK, 'index_all')
        % Find which of this channel's spikes are artifacts
        art_times = SPK.index_all(~SPK.mask_nonart);
        % Find their positions in all_spktimes
        artifact_idxs = [artifact_idxs find(ismember(all_spktimes, art_times))];
    end
end
artifact_idxs = unique(artifact_idxs);

fprintf('Total collision spikes: %d (%.1f%%)\n', length(artifact_idxs), ...
        100 * length(artifact_idxs) / length(all_spktimes));

if isempty(artifact_idxs)
    fprintf('No collisions detected to plot.\n');
    return;
end

% Load parameters from first available spike fil
for k = 1:length(pos_chans)
    spike_file = fullfile(spike_dir, sprintf('%s_spikes.mat', NSx(pos_chans(k)).output_name));
    if exist(spike_file, 'file')
        SPK = load(spike_file);
        if isfield(SPK, 'par')
            t_win = SPK.par.t_win;
            bundle_min_art = SPK.par.bundle_min_art;
            params_found = true;
            fprintf('Loaded parameters from: %s\n', NSx(pos_chans(k)).output_name);
            break;
        end
    end
end

if ~params_found
    warning('Parameters not found in any spike file, using defaults: t_win=0.5, bundle_min_art=6');
end

% Find collision events (groups of consecutive artifact indices)
collision_events = {};
current_event = artifact_idxs(1);

for i = 2:length(artifact_idxs)
    if all_spktimes(artifact_idxs(i)) < all_spktimes(current_event(1)) + t_win
        current_event = [current_event artifact_idxs(i)];
    else
        collision_events{end+1} = current_event;
        current_event = artifact_idxs(i);
    end
end
collision_events{end+1} = current_event; % Add last event

fprintf('Total collision events: %d\n', length(collision_events));

% Select up to num_examples random collision events
num_events = min(num_examples, length(collision_events));
if length(collision_events) > num_examples
    selected = randperm(length(collision_events), num_examples);
else
    selected = 1:num_events;
end

% Create channel ID to bundle lookup map
chan_to_bundle = containers.Map('KeyType', 'double', 'ValueType', 'any');
for n = 1:length(NSx)
    chan_to_bundle(NSx(n).chan_ID) = NSx(n).bundle;
end

% Define colors for different channels
unique_chans = unique(which_chan);
colors = lines(length(unique_chans)); % Generate distinct colors
chan_colors = containers.Map('KeyType', 'double', 'ValueType', 'any');
for c = 1:length(unique_chans)
    chan_colors(unique_chans(c)) = colors(c, :);
end

% Create figure
if show_figure
    vis_str = 'on';
else
    vis_str = 'off';
end

% Plot in batches of 5 per figure
max_per_page = 5;
num_pages = ceil(num_events / max_per_page);

% TIMING PLOTS - Show when spikes occurred
for page = 1:num_pages
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'w', 'Visible', vis_str);
    
    % Determine range for this page
    start_idx = (page - 1) * max_per_page + 1;
    end_idx = min(page * max_per_page, num_events);
    events_this_page = end_idx - start_idx + 1;
    
    for ev_idx = 1:events_this_page
        ev = start_idx + ev_idx - 1;
        subplot(events_this_page, 1, ev_idx);
        hold on;
        
        event_idxs = collision_events{selected(ev)};
        event_times = all_spktimes(event_idxs);
        event_chans = which_chan(event_idxs);
        
        % Plot each spike as vertical line colored by channel
        for i = 1:length(event_times)
            ch = event_chans(i);
            color = chan_colors(ch);
            plot([event_times(i) event_times(i)], [ch-0.4 ch+0.4], ...
                 'Color', color, 'LineWidth', 3);
        end
        
        % Format plot
        ylabel('Channel ID');
        xlabel('Time (samples)');
        title(sprintf('Collision %d: %d channels, %d spikes at time ~%.0f', ...
                      selected(ev), length(unique(event_chans)), length(event_times), min(event_times)), ...
              'Interpreter', 'none');
        grid on;
        ylim([min(event_chans)-1, max(event_chans)+1]);
        xlim([min(event_times)-t_win*0.1, max(event_times)+t_win*0.1]);
        
        % Add legend for channels
        channels_in_collision = unique(event_chans);
        legend_entries = cell(1, length(channels_in_collision));
        legend_handles = [];
        for c = 1:length(channels_in_collision)
            ch = channels_in_collision(c);
            color = chan_colors(ch);
            h = plot(NaN, NaN, 'Color', color, 'LineWidth', 3);
            legend_handles = [legend_handles h];
            legend_entries{c} = sprintf('Ch %d', ch);
        end
        legend(legend_handles, legend_entries, 'Location', 'best');
        
        hold off;
    end
    
    if num_pages > 1
        sgtitle(sprintf('Collision Timing - %s (Page %d/%d)', bundle_name, page, num_pages), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    else
        sgtitle(sprintf('Collision Timing - %s', bundle_name), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    end
    
    % Save timing figure
    save_folder = fullfile(spike_dir, 'collision_pics');
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    
    if num_pages > 1
        save_name = fullfile(save_folder, sprintf('%s_collision_timing_p%d.png', bundle_name, page));
    else
        save_name = fullfile(save_folder, sprintf('%s_collision_timing.png', bundle_name));
    end
    
    exportgraphics(fig, save_name, 'Resolution', 150);
    fprintf('Saved collision timing to: %s\n', save_name);
    
    if ~show_figure
        close(fig);
    end
end

% WAVEFORM PLOTS - Show overlaid waveforms
for page = 1:num_pages
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'w', 'Visible', vis_str);
    
    % Determine range for this page
    start_idx = (page - 1) * max_per_page + 1;
    end_idx = min(page * max_per_page, num_events);
    events_this_page = end_idx - start_idx + 1;
    
    for ev_idx = 1:events_this_page
        ev = start_idx + ev_idx - 1;
        subplot(events_this_page, 1, ev_idx);
        hold on;
        
        event_idxs = collision_events{selected(ev)};
        event_times = all_spktimes(event_idxs);
        event_chans = which_chan(event_idxs);
        
        % Get waveforms for each spike in this collision
        for i = 1:length(event_times)
            t = event_times(i);
            ch = event_chans(i);
            
            % Find this spike's waveform
            for k = 1:length(chan_spike_map)
                if chan_spike_map(k).chan_id == ch
                    spike_idx = find(chan_spike_map(k).spike_times == t, 1);
                    if ~isempty(spike_idx)
                        waveform = chan_spike_map(k).waveforms(spike_idx, :);
                        tvec = 1:length(waveform);
                        
                        % Plot waveform with channel-specific color
                        color = chan_colors(ch);
                        plot(tvec, waveform, 'Color', color, 'LineWidth', 1.5);
                    end
                    break;
                end
            end
        end
        
        % Format plot
        ylabel('Amplitude');
        xlabel('Samples');
        title(sprintf('Collision %d: %d channels, %d spikes at time ~%.0f', ...
                      selected(ev), length(unique(event_chans)), length(event_times), min(event_times)), ...
              'Interpreter', 'none');
        grid on;
        
        % Add legend for channels
        channels_in_collision = unique(event_chans);
        legend_entries = cell(1, length(channels_in_collision));
        legend_handles = [];
        for c = 1:length(channels_in_collision)
            ch = channels_in_collision(c);
            color = chan_colors(ch);
            h = plot(NaN, NaN, 'Color', color, 'LineWidth', 1.5);
            legend_handles = [legend_handles h];
            legend_entries{c} = sprintf('Ch %d', ch);
        end
        legend(legend_handles, legend_entries, 'Location', 'best');
        
        hold off;
    end
    
    if num_pages > 1
        sgtitle(sprintf('Collision Waveforms - %s (Page %d/%d)', bundle_name, page, num_pages), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    else
        sgtitle(sprintf('Collision Waveforms - %s', bundle_name), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    end
    
    % Save waveform figure to collision_pics folder
    save_folder = fullfile(spike_dir, 'collision_pics');
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    
    if num_pages > 1
        save_name = fullfile(save_folder, sprintf('%s_collision_waveforms_p%d.png', bundle_name, page));
    else
        save_name = fullfile(save_folder, sprintf('%s_collision_waveforms.png', bundle_name));
    end
    
    exportgraphics(fig, save_name, 'Resolution', 150);
    fprintf('Saved collision waveforms to: %s\n', save_name);
    
    if ~show_figure
        close(fig);
    end
end

end
