function plot_rescue_distances(ch, varargin)
    % plot_rescue_distances - 2D plot of normalized distances (unweighted vs weighted)
    % Shows how distances move when weighting is applied. Points are colored by
    % mask type and symbols indicate rescue status change.
    %
    % Inputs:
    %   ch - channel ID or label
    %   num_spikes - number of random quarantined spikes to plot (default 100)
    %   weight - peak weight for weighted distance (default 3)
    %   include_task - include task-excluded spikes (default false)
    
    p = inputParser;
    addParameter(p, 'num_spikes', 100, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'include_task', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'weight', 3, @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});
    num_spikes = p.Results.num_spikes;
    include_task = p.Results.include_task;
    weight = p.Results.weight;

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

    S = load(fname_times);

    % Load masks from SPK file (same pattern as plot_rescue_histograms and rescue_spikes)
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
    if ~include_task
        mask_quar = ~mask_non_quarantine | ~mask_non_collision;      % Quarantine + collision
    else
        mask_quar = ~mask_non_quarantine | ~mask_non_collision | ~mask_task;  % All three
    end

    if ~any(mask_quar)
        error('No quarantined spikes found for channel %s', ch_lbl);
    end

    % Get quarantined spike indices in index_all
    quar_indices_in_all = find(mask_quar);
    spikes_quar = spikes_all(mask_quar, :);

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
    
    % Select random subset of quarantined spikes
    n_quar = size(spikes_quar, 1);
    if num_spikes > n_quar
        num_spikes = n_quar;
    end
    rand_idx = randperm(n_quar, num_spikes);
    
    selected_spikes = spikes_quar(rand_idx, :);
    selected_indices_in_all = quar_indices_in_all(rand_idx);

    % Set up parameters for distance computation
    par.pk_weight = weight;
    if ~isfield(par, 'amp_dir')
        par.amp_dir = 'neg';
    end

    % Compute normalized distances for all selected spikes (both weighted and unweighted)
    norm_dist_unweighted = zeros(1, num_spikes);
    norm_dist_weighted = zeros(1, num_spikes);
    raw_dist_unweighted = zeros(1, num_spikes);
    raw_dist_weighted = zeros(1, num_spikes);

    for i = 1:num_spikes
        spike = selected_spikes(i, :);
        
        % Compute distances to all centers
        dist_uw = zeros(1, size(centers, 1));
        
        for cls = 1:size(centers, 1)
            center = centers(cls, :);
            
            % Unweighted distance
            dist_uw(cls) = sqrt(sum((spike - center).^2));
        end
        
        % Weighted distance
        [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, par.pk_weight, par.amp_dir);
        dist_w = (normConst_vec .* sqrt(sum(w_matrix .* (repmat(spike, size(centers,1),1) - centers).^2, 2)))';
        
        % Find winning class (min distance) for each
        [~, win_cls_uw] = min(dist_uw);
        [~, win_cls_w] = min(dist_w);
        
        % Raw distances (winning class)
        raw_dist_unweighted(i) = dist_uw(win_cls_uw);
        raw_dist_weighted(i) = dist_w(win_cls_w);
        
        % Normalized by winning class's maxdist
        norm_dist_unweighted(i) = dist_uw(win_cls_uw) / maxdist(win_cls_uw);
        norm_dist_weighted(i) = dist_w(win_cls_w) / maxdist(win_cls_w);
    end

    % Determine mask category and rescue status for each spike
    mask_id = zeros(1, num_spikes);
    status_id = zeros(1, num_spikes);
    sdnum = par.template_sdnum;  % Cutoff threshold

    for i = 1:num_spikes
        idx = selected_indices_in_all(i);
        is_quar = ~mask_non_quarantine(idx);
        is_coll = ~mask_non_collision(idx);
        is_task_excl = ~mask_task(idx);
        
        % Determine mask category (which mask(s) this spike belongs to)
        if is_quar && ~is_coll && ~is_task_excl
            mask_id(i) = 1; % only quarantine
        elseif ~is_quar && is_coll && ~is_task_excl
            mask_id(i) = 2; % only collision
        elseif ~is_quar && ~is_coll && is_task_excl
            mask_id(i) = 3; % only task-excluded
        elseif is_quar && is_coll && ~is_task_excl
            mask_id(i) = 4; % quarantine & collision
        elseif is_quar && ~is_coll && is_task_excl
            mask_id(i) = 5; % quarantine & task-excluded
        elseif ~is_quar && is_coll && is_task_excl
            mask_id(i) = 6; % collision & task-excluded
        elseif is_quar && is_coll && is_task_excl
            mask_id(i) = 7; % all three
        else
            mask_id(i) = 1; % fallback
        end
        
        % Determine rescue status change
        rescued_uw = norm_dist_unweighted(i) < sdnum;
        rescued_w = norm_dist_weighted(i) < sdnum;
        
        if rescued_uw && rescued_w
            status_id(i) = 1; % rescued in both (no change)
        elseif ~rescued_uw && rescued_w
            status_id(i) = 2; % weight ENABLES rescue (unrescued -> rescued)
        elseif rescued_uw && ~rescued_w
            status_id(i) = 3; % weight PREVENTS rescue (rescued -> unrescued)
        else
            status_id(i) = 4; % not rescued in either (no change)
        end
    end

    % Define colors for mask categories
    colors = [
        0.2 0.4 0.8;    % 1: quarantine only - blue
        0.9 0.3 0.2;    % 2: collision only - red
        0.2 0.7 0.3;    % 3: task-excluded only - green
        0.7 0.3 0.7;    % 4: quarantine & collision - purple
        0.2 0.7 0.7;    % 5: quarantine & task - cyan
        0.9 0.6 0.2;    % 6: collision & task - orange
        0.5 0.5 0.5     % 7: all three - gray
    ];
    
    mask_labels = {'Quarantine', 'Collision', 'Task-Excluded', ...
                   'Quar+Coll', 'Quar+Task', 'Coll+Task', 'All Three'};

    % Define markers for rescue status
    markers = {'o', '^', 'v', 'x'};
    marker_labels = {'Both Rescued', 'Weight Enables', 'Weight Prevents', 'Neither Rescued'};

    % Create 2D scatter plot
    fig = figure('Position', [100, 100, 900, 800]);
    hold on;

    % Plot each combination of mask and status
    legend_handles = [];
    legend_texts = {};
    
    for m = 1:7
        for s = 1:4
            idx = (mask_id == m) & (status_id == s);
            if any(idx)
                h = scatter(norm_dist_unweighted(idx), norm_dist_weighted(idx), ...
                    60, colors(m,:), markers{s}, 'LineWidth', 1.5);
                if s == 1 || s == 4  % filled markers for stable states
                    set(h, 'MarkerFaceColor', colors(m,:), 'MarkerFaceAlpha', 0.5);
                end
            end
        end
    end

    % Draw threshold lines at sdnum (typically 3)
    xline(sdnum, 'k--', 'LineWidth', 2);
    yline(sdnum, 'k--', 'LineWidth', 2);

    % Draw diagonal reference line (y = x)
    max_val = max([max(norm_dist_unweighted), max(norm_dist_weighted), sdnum+1]);
    plot([0, max_val], [0, max_val], 'k:', 'LineWidth', 1);

    % Shade the quadrants
    fill([0, sdnum, sdnum, 0], [0, 0, sdnum, sdnum], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Both rescued
    fill([sdnum, max_val, max_val, sdnum], [0, 0, sdnum, sdnum], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Weight enables
    fill([0, sdnum, sdnum, 0], [sdnum, sdnum, max_val, max_val], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Weight prevents
    fill([sdnum, max_val, max_val, sdnum], [sdnum, sdnum, max_val, max_val], 'k', 'FaceAlpha', 0.05, 'EdgeColor', 'none'); % Neither

    % Labels
    xlabel('Normalized Distance (Unweighted)', 'FontSize', 12);
    ylabel('Normalized Distance (Weighted)', 'FontSize', 12);
    title(sprintf('Rescue Distance Movement - Channel %s (weight=%.1f)', ch_lbl, weight), ...
        'FontSize', 14, 'Interpreter', 'none');

    % Add quadrant labels
    text(sdnum/2, sdnum/2, 'Both Rescued', 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Color', [0 0.5 0]);
    text((sdnum + max_val)/2, sdnum/2, 'Weight Enables', 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Color', [0 0 0.8]);
    text(sdnum/2, (sdnum + max_val)/2, 'Weight Prevents', 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Color', [0.8 0 0]);
    text((sdnum + max_val)/2, (sdnum + max_val)/2, 'Neither', 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Color', [0.3 0.3 0.3]);

    % Create custom legend
    % First for colors (mask types)
    unique_masks = unique(mask_id);
    for m = unique_masks
        h = scatter(NaN, NaN, 60, colors(m,:), 'o', 'filled');
        legend_handles = [legend_handles, h];
        legend_texts = [legend_texts, mask_labels{m}];
    end
    
    % Then for markers (status)
    unique_status = unique(status_id);
    for s = unique_status
        h = scatter(NaN, NaN, 60, 'k', markers{s}, 'LineWidth', 1.5);
        legend_handles = [legend_handles, h];
        legend_texts = [legend_texts, marker_labels{s}];
    end

    legend(legend_handles, legend_texts, 'Location', 'eastoutside', 'FontSize', 9);

    % Add statistics text
    n_both_rescued = sum(status_id == 1);
    n_weight_enables = sum(status_id == 2);
    n_weight_prevents = sum(status_id == 3);
    n_neither = sum(status_id == 4);
    
    stats_text = sprintf(['n = %d spikes\n' ...
        'Both rescued: %d (%.1f%%)\n' ...
        'Weight enables: %d (%.1f%%)\n' ...
        'Weight prevents: %d (%.1f%%)\n' ...
        'Neither: %d (%.1f%%)'], ...
        num_spikes, ...
        n_both_rescued, 100*n_both_rescued/num_spikes, ...
        n_weight_enables, 100*n_weight_enables/num_spikes, ...
        n_weight_prevents, 100*n_weight_prevents/num_spikes, ...
        n_neither, 100*n_neither/num_spikes);
    
    annotation('textbox', [0.15, 0.75, 0.25, 0.15], 'String', stats_text, ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'FontSize', 9);

    grid on;
    axis equal;
    xlim([0, max_val]);
    ylim([0, max_val]);
    hold off;

    % Save normalized figure
    spike_dir = pwd;
    save_folder = fullfile(spike_dir, 'rescue_distance_plots');
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end
    save_name = fullfile(save_folder, sprintf('%s_distances_wt_%s.png', ch_lbl, string(weight)));
    exportgraphics(fig, save_name, 'Resolution', 300);
    fprintf('Saved normalized plot to: %s\n', save_name);

    %% ========== RAW (UNNORMALIZED) DISTANCE PLOT ==========
    fig_raw = figure('Position', [100, 100, 900, 800]);
    hold on;

    % Plot each combination of mask and status (using raw distances)
    for m = 1:7
        for s = 1:4
            idx = (mask_id == m) & (status_id == s);
            if any(idx)
                h = scatter(raw_dist_unweighted(idx), raw_dist_weighted(idx), ...
                    60, colors(m,:), markers{s}, 'LineWidth', 1.5);
                if s == 1 || s == 4  % filled markers for stable states
                    set(h, 'MarkerFaceColor', colors(m,:), 'MarkerFaceAlpha', 0.5);
                end
            end
        end
    end

    % Draw diagonal reference line (y = x)
    max_val_raw = max([max(raw_dist_unweighted), max(raw_dist_weighted)]);
    plot([0, max_val_raw], [0, max_val_raw], 'k:', 'LineWidth', 1);

    % Labels
    xlabel('Raw Distance (Unweighted)', 'FontSize', 12);
    ylabel('Raw Distance (Weighted)', 'FontSize', 12);
    title(sprintf('Raw Distance Movement - Channel %s (weight=%.1f)', ch_lbl, weight), ...
        'FontSize', 14, 'Interpreter', 'none');

    % Create custom legend for raw plot
    legend_handles_raw = [];
    legend_texts_raw = {};
    
    for m = unique_masks
        h = scatter(NaN, NaN, 60, colors(m,:), 'o', 'filled');
        legend_handles_raw = [legend_handles_raw, h];
        legend_texts_raw = [legend_texts_raw, mask_labels{m}];
    end
    
    for s = unique_status
        h = scatter(NaN, NaN, 60, 'k', markers{s}, 'LineWidth', 1.5);
        legend_handles_raw = [legend_handles_raw, h];
        legend_texts_raw = [legend_texts_raw, marker_labels{s}];
    end

    legend(legend_handles_raw, legend_texts_raw, 'Location', 'eastoutside', 'FontSize', 9);

    % Add statistics text (same as normalized)
    annotation('textbox', [0.15, 0.75, 0.25, 0.15], 'String', stats_text, ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'FontSize', 9);

    grid on;
    axis equal;
    xlim([0, max_val_raw * 1.05]);
    ylim([0, max_val_raw * 1.05]);
    hold off;

    % Save raw figure
    save_name_raw = fullfile(save_folder, sprintf('%s_distances_raw_wt_%s.png', ch_lbl, string(weight)));
    exportgraphics(fig_raw, save_name_raw, 'Resolution', 300);
    fprintf('Saved raw plot to: %s\n', save_name_raw);
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

function [normConst, weights] = get_weight_matrix(spike_x, tmplt_vect, pk_weight, amp_dir)
    % Compute peak width for spike_x
    spike_width = get_peak_width(spike_x, amp_dir);
    
    n_templates = size(tmplt_vect, 1);
    n_samples = size(tmplt_vect, 2);
    weights = ones(n_templates, n_samples);
    
    % Create spike mask once (static across templates)
    spike_mask = false(1, n_samples);
    if ~isnan(spike_width.left) && ~isnan(spike_width.right)
        spike_mask(spike_width.left:spike_width.right) = true;
    end
    
    for i = 1:n_templates
        template = tmplt_vect(i, :);
        template_width = get_peak_width(template, amp_dir);
        
        % Create template mask
        template_mask = false(1, n_samples);
        if ~isnan(template_width.left) && ~isnan(template_width.right)
            template_mask(template_width.left:template_width.right) = true;
        end
        
        % Overlap: both have width
        overlap = spike_mask & template_mask;
        % Symmetric difference: only one has width
        symmetric_diff = xor(spike_mask, template_mask);
        
        % Set weights: pk_weight in overlap, max(1, pk_weight/2) in symmetric difference
        weights(i, symmetric_diff) = max(1, pk_weight / 2);
        weights(i, overlap) = pk_weight;
    end
    
    normConst = sqrt(n_samples) ./ sqrt(sum(weights, 2));
end

function width_struct = get_peak_width(spike_x, amp_dir)
    if strcmp(amp_dir, 'neg')
        wav = -spike_x;
    else
        wav = spike_x;
    end
    [pks, locs, w, p] = findpeaks(wav);
    
    if ~isempty(pks)
        [pk, peak_loc] = max(pks);
        pk_loc = locs(peak_loc);
        prom_max = p(peak_loc);

        level = pk - prom_max/2;
        above_level = find(wav >= level);
        if ~isempty(above_level)
            % Find connected components
            diff_above = diff(above_level);
            breaks = find(diff_above > 1);
            segments = {};
            start_idx = 1;
            for b = 1:length(breaks)
                segments{end+1} = above_level(start_idx:breaks(b));
                start_idx = breaks(b) + 1;
            end
            segments{end+1} = above_level(start_idx:end);
            % Find segment containing peak_loc
            peak_segment = [];
            for s = 1:length(segments)
                if any(segments{s} == pk_loc)
                    peak_segment = segments{s};
                    break;
                end
            end
            if ~isempty(peak_segment)
                left_width = min(peak_segment);
                right_width = max(peak_segment);
            else
                left_width = NaN;
                right_width = NaN;
            end
        else
            left_width = NaN;
            right_width = NaN;
        end
        if left_width <= 0
            left_width = 1;
        end
        if right_width > length(spike_x)
            right_width = length(spike_x);
        end
    else
        left_width = NaN;
        right_width = NaN;
    end
    
    width_struct.left = left_width;
    width_struct.right = right_width;
end