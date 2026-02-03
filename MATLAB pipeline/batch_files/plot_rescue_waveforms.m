function plot_rescue_waveforms(ch, max_waveforms)
% plot_rescue_waveforms - Plot mean waveforms for different spike categories
% Inputs:
%   ch - channel ID or label
%   max_waveforms - maximum number of waveforms to sample per category (default 2000)

if nargin < 2
    max_waveforms = 2000;
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

% Load good spikes (original)
cluster_class = S.cluster_class;
spikes_good = S.spikes;
class_good_mask = cluster_class(:,1) ~= 0;
spikes_original = spikes_good(class_good_mask, :);

% Load good spikes and build templates
class_good = cluster_class(class_good_mask, 1);
spikes_good_classified = spikes_good(class_good_mask, :);

[centers, maxdist, ~] = build_templates(class_good, spikes_good_classified);

% Compute winning distances and status for all quarantined spikes
n_quar = size(spikes_quar, 1);
norm_dist_weighted = zeros(1, n_quar);
norm_dist_unweighted = zeros(1, n_quar);

for i = 1:n_quar
    spike = spikes_quar(i, :);
    
    % Compute distances to all centers
    dist_w = zeros(1, size(centers, 1));
    dist_uw = zeros(1, size(centers, 1));
    
    for cls = 1:size(centers, 1)
        center = centers(cls, :);
        
        % Weighted
        [normConst_w, w] = get_weight_vector(spike, par.pk_weight, par.amp_dir);
        dist_w(cls) = normConst_w * sqrt(sum(w .* (spike - center).^2));
        
        % Unweighted
        w_unw = ones(size(spike));
        normConst_unw = sqrt(length(w_unw)) / sqrt(sum(w_unw));
        dist_uw(cls) = normConst_unw * sqrt(sum(w_unw .* (spike - center).^2));
    end
    
    % Find winning class (min distance)
    [~, win_cls_w] = min(dist_w);
    [~, win_cls_uw] = min(dist_uw);
    
    % Normalized by winning class's maxdist
    norm_dist_weighted(i) = dist_w(win_cls_w) / maxdist(win_cls_w);
    norm_dist_unweighted(i) = dist_uw(win_cls_uw) / maxdist(win_cls_uw);
end

% Determine status
rescued_w = norm_dist_weighted < par.template_sdnum;
rescued_uw = norm_dist_unweighted < par.template_sdnum;

% Categorize waveforms
waveforms_original = sample_waveforms(spikes_original, max_waveforms);
waveforms_both = sample_waveforms(spikes_quar(rescued_w & rescued_uw, :), max_waveforms);
waveforms_weighted_only = sample_waveforms(spikes_quar(rescued_w & ~rescued_uw, :), max_waveforms);
waveforms_unweighted_only = sample_waveforms(spikes_quar(~rescued_w & rescued_uw, :), max_waveforms);
waveforms_none = sample_waveforms(spikes_quar(~rescued_w & ~rescued_uw, :), max_waveforms);

% Plot
figure;

categories = {'Original', 'Rescued Both', 'Rescued Weighted Only', 'Rescued Unweighted Only', 'Not Rescued'};
waveform_data = {waveforms_original, waveforms_both, waveforms_weighted_only, waveforms_unweighted_only, waveforms_none};

for i = 1:5
    subplot(1,5,i);
    if ~isempty(waveform_data{i})
        mean_wave = mean(waveform_data{i}, 1);
        std_wave = std(waveform_data{i}, 0, 1);
        x = 1:length(mean_wave);
        plot(mean_wave, 'k', 'LineWidth', 2);
        hold on;
        fill([x, fliplr(x)], [mean_wave + std_wave, fliplr(mean_wave - std_wave)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    xlabel('Time');
    ylabel('Amplitude');
    title(sprintf('%s (N=%d)', categories{i}, size(waveform_data{i}, 1)));
    grid on;
end

sgtitle(sprintf('Mean Waveforms - Channel %s', ch_lbl));

end

function sampled = sample_waveforms(waveforms, max_n)
    n = size(waveforms, 1);
    if n > max_n
        idx = randperm(n, max_n);
        sampled = waveforms(idx, :);
    else
        sampled = waveforms;
    end
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