function plot_rescue_histograms(ch, varargin)
    % plot_rescue_histograms - Plot histograms of normalized winning distances for quarantined spikes
    % Inputs:
    %   ch - channel ID or label
    %   num_bins - number of bins for histograms (default 20)

    p = inputParser;
    addParameter(p, 'num_bins', 20, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'include_task', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'weight',1, @(x) isnumeric(x) && isscalar(x) && x > 0);

    parse(p, varargin{:});
    num_bins = p.Results.num_bins;
    include_task = p.Results.include_task;
    weight = p.Results.weight;
    
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
    spikes_all = SPK.spikes_all;
    index_all = SPK.index_all;

    % Load masks from SPK file
    if isfield(SPK, 'mask_non_quarantine')
        mask_non_quarantine = SPK.mask_non_quarantine;
    else
        mask_non_quarantine = false(size(index_all));
    end

    if isfield(SPK, 'mask_nonart')
        mask_non_collision = SPK.mask_nonart;
    else
        mask_non_collision = true(size(index_all));
    end

    if isfield(SPK, 'mask_taskspks')
        mask_task = SPK.mask_taskspks;
    else
        mask_task = true(size(index_all));
    end

    % Build quarantine mask - combine masks (spikes excluded by any mask)
    % mask_quar = ~mask_non_quarantine;                              % Only quarantine
    % mask_quar = ~mask_non_collision;                               % Only collision
    % mask_quar = ~mask_task;
    %                                         % Only task-excluded
   
   if ~include_task
        mask_quar = ~mask_non_quarantine | ~mask_non_collision;      % Quarantine + collision
    else
        mask_quar = ~mask_non_quarantine | ~mask_non_collision | ~mask_task;  % All three
    end
    % mask_quar = ~mask_non_quarantine | ~mask_non_collision | ~mask_task;  % All three

    if ~any(mask_quar)
        error('No quarantined spikes found for channel %s', ch_lbl);
    end

    spikes_quar = spikes_all(mask_quar, :);

    S = load(fname_times);

    % Load good spikes and build templates
    cluster_class = S.cluster_class;
    spikes_good = S.spikes;
    class_good_mask = cluster_class(:,1) ~= 0;
    class_good = cluster_class(class_good_mask, 1);
    spikes_good_classified = spikes_good(class_good_mask, :);

    % Exclude rescued spikes from template building to match rescue process
    if isfield(SPK, 'rescue_mask') && ~isempty(SPK.rescue_mask) && any(SPK.rescue_mask)
        % Get timestamps of rescued spikes from index_all
        rescued_timestamps = SPK.index_all(SPK.rescue_mask);
        
        % Get timestamps of good classified spikes from SPK.index
        good_timestamps = SPK.index(class_good_mask);
        
        % Find which good spikes are rescued by matching timestamps
        original_good_mask = ~ismember(good_timestamps, rescued_timestamps);
    else
        original_good_mask = true(size(class_good));
    end
    
    [centers, maxdist, ~] = build_templates(class_good(original_good_mask), spikes_good_classified(original_good_mask, :));
    
    % Set up parameters for distance computation
    par.pk_weight = weight;
    par.amp_dir = 'neg';
    
    % Initialize distance arrays
    n_quar = size(spikes_quar, 1);
    norm_dist_weighted = zeros(1, n_quar);
    raw_dist_weighted = zeros(1, n_quar);

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
        
        % Raw distance (winning class)
        raw_dist_weighted(i) = dist_w(win_cls_w);
        
        % Normalized by winning class's maxdist
        norm_dist_weighted(i) = dist_w(win_cls_w) / maxdist(win_cls_w);
    end

    % Determine rescued vs non-rescued (based on normalized)
    rescued_mask = norm_dist_weighted < par.template_sdnum;
    dist_all = norm_dist_weighted;
    dist_rescued = norm_dist_weighted(rescued_mask);
    dist_non_rescued = norm_dist_weighted(~rescued_mask);
    
    % Raw distances split by rescue status
    raw_dist_all = raw_dist_weighted;
    raw_dist_rescued = raw_dist_weighted(rescued_mask);
    raw_dist_non_rescued = raw_dist_weighted(~rescued_mask);

    fig = figure;

    % All spikes
    subplot(1,3,1);
    histogram(dist_all, num_bins, 'FaceColor', 'b', 'EdgeColor', 'k');
    hold on;
    xline(par.template_sdnum, 'r--', 'LineWidth', 2);
    xlabel('Normalized Winning Distance');
    ylabel('Count');
    title('All Quarantined Spikes');
    text(0.95, 0.95, sprintf('n = %d', length(dist_all)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10);
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
    text(0.95, 0.95, sprintf('n = %d', length(dist_rescued)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10);
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
    text(0.95, 0.95, sprintf('n = %d', length(dist_non_rescued)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10);
    grid on;

    sgtitle(sprintf('Histogram of Normalized Winning Distances - Channel %s (weight=%.1f)', ch_lbl, weight), 'Interpreter', 'none');
    spike_dir = pwd;
    save_folder = fullfile(spike_dir, 'histogram_distances');
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    save_name = fullfile(save_folder, sprintf('%s_wt_%s.png', ch_lbl, string(weight)));
    exportgraphics(fig, save_name, 'Resolution', 300);
    fprintf('Saved normalized histogram to: %s\n', save_name);

    fig_raw = figure;

    % All spikes
    subplot(1,3,1);
    histogram(raw_dist_all, num_bins, 'FaceColor', 'b', 'EdgeColor', 'k');
    xlabel('Raw Winning Distance');
    ylabel('Count');
    title('All Quarantined Spikes');
    text(0.95, 0.95, sprintf('n = %d', length(raw_dist_all)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10);
    grid on;

    % Rescued spikes
    subplot(1,3,2);
    if ~isempty(raw_dist_rescued)
        histogram(raw_dist_rescued, num_bins, 'FaceColor', 'g', 'EdgeColor', 'k');
    end
    xlabel('Raw Winning Distance');
    ylabel('Count');
    title('Rescued Spikes');
    text(0.95, 0.95, sprintf('n = %d', length(raw_dist_rescued)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10);
    grid on;

    % Non-rescued spikes
    subplot(1,3,3);
    if ~isempty(raw_dist_non_rescued)
        histogram(raw_dist_non_rescued, num_bins, 'FaceColor', 'r', 'EdgeColor', 'k');
    end
    xlabel('Raw Winning Distance');
    ylabel('Count');
    title('Non-Rescued Spikes');
    text(0.95, 0.95, sprintf('n = %d', length(raw_dist_non_rescued)), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10);
    grid on;

    sgtitle(sprintf('Histogram of Raw Winning Distances - Channel %s (weight=%.1f)', ch_lbl, weight), 'Interpreter', 'none');
    save_name_raw = fullfile(save_folder, sprintf('%s_raw_wt_%s.png', ch_lbl, string(weight)));
    exportgraphics(fig_raw, save_name_raw, 'Resolution', 300);
    fprintf('Saved raw histogram to: %s\n', save_name_raw);
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
        
        % if width_max < length(spike_x)/2.5
        %     w_vect(left_width:right_width) = pk_weight;
        % end
    end
    
    weights = w_vect';
    normConst = sqrt(length(weights)) / sqrt(sum(weights));
end