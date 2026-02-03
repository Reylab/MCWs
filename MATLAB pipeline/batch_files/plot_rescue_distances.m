function plot_rescue_distances(ch, num_spikes)
% plot_rescue_distances - Plot normalized winning distances for randomly selected quarantined spikes
% Inputs:
%   ch - channel ID or label
%   num_spikes - number of random quarantined spikes to plot (default 100)

if nargin < 2
    num_spikes = 100;
end

ch_lbl = get_channel_label(ch);
fname_spk = sprintf('%s_spikes.mat', ch_lbl);
fname_times = sprintf('times_%s.mat', ch_lbl);

if ~exist(fname_spk, 'file') || ~exist(fname_times, 'file')
    error('Required files not found for channel %s', ch_lbl);
end

% Load data
SPK = load(fname_spk);
par = SPK.par;

S = load(fname_times);

if ~isfield(S, 'spikes_quarantined') || isempty(S.spikes_quarantined)
    error('No quarantined spikes found for channel %s', ch_lbl);
end

spikes_quar = S.spikes_quarantined;
class_quar = S.class_quarantined;

% Load good spikes and build templates
cluster_class = S.cluster_class;
spikes_good = S.spikes;
class_good_mask = cluster_class(:,1) ~= 0;
class_good = cluster_class(class_good_mask, 1);
spikes_good_classified = spikes_good(class_good_mask, :);

[centers, maxdist, ~] = build_templates(class_good, spikes_good_classified);

% Randomly select spikes
n_quar = size(spikes_quar, 1);
if n_quar > num_spikes
    rand_idx = randperm(n_quar, num_spikes);
else
    rand_idx = 1:n_quar;
    num_spikes = n_quar;
end

selected_spikes = spikes_quar(rand_idx, :);
selected_classes = class_quar(rand_idx);

% Load masks
mask_non_quarantine = SPK.mask_non_quarantine;
mask_non_collision = SPK.mask_nonart;
mask_task = SPK.mask_taskspks;

selected_indices = S.index_quarantined(rand_idx);

% Determine mask and status for each spike
mask_all = zeros(1, num_spikes);
status_all = zeros(1, num_spikes);

for i = 1:num_spikes
    idx = selected_indices(i);
    is_quar = ~mask_non_quarantine(idx);
    is_coll = ~mask_non_collision(idx);
    is_task_excl = ~mask_task(idx);
    
    if is_quar && ~is_coll && ~is_task_excl
        mask_id = 1; % only quarantine
    elseif ~is_quar && is_coll && ~is_task_excl
        mask_id = 2; % only collision
    elseif ~is_quar && ~is_coll && is_task_excl
        mask_id = 3; % only task-excluded
    elseif is_quar && is_coll && ~is_task_excl
        mask_id = 4; % quarantine & collision
    elseif is_quar && ~is_coll && is_task_excl
        mask_id = 5; % quarantine & task-excluded
    elseif ~is_quar && is_coll && is_task_excl
        mask_id = 6; % collision & task-excluded
    elseif is_quar && is_coll && is_task_excl
        mask_id = 7; % all three
    else
        mask_id = 1; % fallback
    end
    mask_all(i) = mask_id;
    
    rescued_w = norm_dist_weighted(i) < par.template_sdnum;
    rescued_uw = norm_dist_unweighted(i) < par.template_sdnum;
    
    if rescued_w && rescued_uw
        status_id = 1; % rescued in both
    elseif rescued_w && ~rescued_uw
        status_id = 2; % rescued only with weight
    elseif ~rescued_w && rescued_uw
        status_id = 3; % rescued only without weight
    else
        status_id = 4; % not rescued
    end
    status_all(i) = status_id;
end

% Define colors and markers
colors = [0 0 1; 0 1 0; 1 0 0; 0 1 1; 1 0 1; 1 1 0; 0.5 0.5 0.5]; % 7 colors
markers = {'o', '^', 'v', 'x'}; % 4 markers

% Plot
figure;

% Weighted plot
subplot(1,2,1);
hold on;
for status = 1:4
    idx_status = find(status_all == status);
    if ~isempty(idx_status)
        scatter(1:length(idx_status), norm_dist_weighted(idx_status), 50, colors(mask_all(idx_status), :), markers{status}, 'filled');
    end
end
plot([1, num_spikes], [par.template_sdnum, par.template_sdnum], 'k--', 'LineWidth', 2);
xlabel('Spike Index');
ylabel('Normalized Winning Distance');
title(sprintf('Weighted Winning Distances - Channel %s', ch_lbl));
grid on;
legend({'Both', 'Weight Enabled', 'Weight Prevented', 'Neither'}, 'Location', 'best');

% Unweighted plot
subplot(1,2,2);
hold on;
for status = 1:4
    idx_status = find(status_all == status);
    if ~isempty(idx_status)
        scatter(1:length(idx_status), norm_dist_unweighted(idx_status), 50, colors(mask_all(idx_status), :), markers{status}, 'filled');
    end
end
plot([1, num_spikes], [par.template_sdnum, par.template_sdnum], 'k--', 'LineWidth', 2);
xlabel('Spike Index');
ylabel('Normalized Winning Distance');
title(sprintf('Unweighted Winning Distances - Channel %s', ch_lbl));
grid on;
legend({'Both', 'Weight Enabled', 'Weight Prevented', 'Neither'}, 'Location', 'best');

sgtitle('Winning Distances for Quarantined Spikes');

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

function [normConst, weights] = get_weight_vector(spike_x, pk_weight, amp_dir)
    if strcmp(amp_dir, 'neg')
        wav = -spike_x;
    else
        wav = spike_x;
    end
    [pks, locs, w, p] = findpeaks(wav);
    w_vect = ones(size(spike_x, 2), 1);
    
    if ~isempty(pks)
        [pk, peak_loc] = max(pks);
        width_max = w(peak_loc);
        peak_loc = locs(peak_loc);
        
        left_width = round(peak_loc - width_max/2);
        if left_width <= 0
            left_width = 1;
        end
        right_width = round(peak_loc + width_max/2);
        if right_width > size(spike_x, 2)
            right_width = size(spike_x, 2);
        end
        
        w_vect(left_width:right_width) = pk_weight;
    end
    
    weights = w_vect';
    normConst = sqrt(length(weights)) / sqrt(sum(weights));
end