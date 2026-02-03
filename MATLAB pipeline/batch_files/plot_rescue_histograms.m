function plot_rescue_histograms(ch, num_bins)
% plot_rescue_histograms - Plot histograms of normalized winning distances for quarantined spikes
% Inputs:
%   ch - channel ID or label
%   num_bins - number of bins for histograms (default 20)

if nargin < 2
    num_bins = 20;
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

% Load good spikes and build templates
cluster_class = S.cluster_class;
spikes_good = S.spikes;
class_good_mask = cluster_class(:,1) ~= 0;
class_good = cluster_class(class_good_mask, 1);
spikes_good_classified = spikes_good(class_good_mask, :);

[centers, maxdist, ~] = build_templates(class_good, spikes_good_classified);

% Compute winning distances for all quarantined spikes
n_quar = size(spikes_quar, 1);
norm_dist_weighted = zeros(1, n_quar);

for i = 1:n_quar
    spike = spikes_quar(i, :);
    
    % Compute distances to all centers
    dist_w = zeros(1, size(centers, 1));
    
    for cls = 1:size(centers, 1)
        center = centers(cls, :);
        
        % Weighted
        [normConst_w, w] = get_weight_vector(spike, par.pk_weight, par.amp_dir);
        dist_w(cls) = normConst_w * sqrt(sum(w .* (spike - center).^2));
    end
    
    % Find winning class (min distance)
    [~, win_cls_w] = min(dist_w);
    
    % Normalized by winning class's maxdist
    norm_dist_weighted(i) = dist_w(win_cls_w) / maxdist(win_cls_w);
end

% Determine rescued vs non-rescued
rescued_mask = norm_dist_weighted < par.template_sdnum;
dist_all = norm_dist_weighted;
dist_rescued = norm_dist_weighted(rescued_mask);
dist_non_rescued = norm_dist_weighted(~rescued_mask);

% Plot histograms
figure;

% All spikes
subplot(1,3,1);
histogram(dist_all, num_bins, 'FaceColor', 'b', 'EdgeColor', 'k');
hold on;
xline(par.template_sdnum, 'r--', 'LineWidth', 2);
xlabel('Normalized Winning Distance');
ylabel('Count');
title('All Quarantined Spikes');
grid on;

% Rescued spikes
subplot(1,3,2);
if ~isempty(dist_rescued)
    histogram(dist_rescued, num_bins, 'FaceColor', 'g', 'EdgeColor', 'k');
end
hold on;
xline(par.template_sdnum, 'r--', 'LineWidth', 2);
xlabel('Normalized Winning Distance');
ylabel('Count');
title('Rescued Spikes');
grid on;

% Non-rescued spikes
subplot(1,3,3);
if ~isempty(dist_non_rescued)
    histogram(dist_non_rescued, num_bins, 'FaceColor', 'r', 'EdgeColor', 'k');
end
hold on;
xline(par.template_sdnum, 'r--', 'LineWidth', 2);
xlabel('Normalized Winning Distance');
ylabel('Count');
title('Non-Rescued Spikes');
grid on;

sgtitle(sprintf('Histogram of Normalized Winning Distances - Channel %s', ch_lbl));

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