function plot_rescue_waveforms(ch, varargin)
    % plot_rescue_waveforms - Plot mean waveforms for different spike categories
    % Shows waveforms grouped by rescue status (both, weight enables, weight prevents, neither)
    % and colored by mask type (quarantine, collision, task-excluded, combinations)
    %
    % Inputs:
    %   ch - channel ID or label
    %   max_waveforms - maximum number of waveforms to sample per category (default 2000)
    %   weight - peak weight for weighted distance (default 3)
    %   include_task - include task-excluded spikes (default false)

    p = inputParser;
    addParameter(p, 'max_waveforms', 2000, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'include_task', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'weight', 3, @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});
    max_waveforms = p.Results.max_waveforms;
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
    n_quar = size(spikes_quar, 1);

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
    if ~isfield(par, 'amp_dir')
        par.amp_dir = 'neg';
    end
    
    % Compute normalized distances for all quarantined spikes (both weighted and unweighted)
    norm_dist_unweighted = zeros(1, n_quar);
    norm_dist_weighted = zeros(1, n_quar);

    for i = 1:n_quar
        spike = spikes_quar(i, :);
        
        % Compute distances to all centers
        dist_uw = zeros(1, size(centers, 1));
        dist_w = zeros(1, size(centers, 1));
        
        for cls = 1:size(centers, 1)
            center = centers(cls, :);
            
            % Unweighted distance
            dist_uw(cls) = sqrt(sum((spike - center).^2));
            
            % Weighted distance
            [normConst_w, w] = get_weight_vector(spike, par.pk_weight, par.amp_dir);
            dist_w(cls) = normConst_w * sqrt(sum(w .* (spike - center).^2));
        end
        
        % Find winning class (min distance) for each
        [~, win_cls_uw] = min(dist_uw);
        [~, win_cls_w] = min(dist_w);
        
        % Normalized by winning class's maxdist
        norm_dist_unweighted(i) = dist_uw(win_cls_uw) / maxdist(win_cls_uw);
        norm_dist_weighted(i) = dist_w(win_cls_w) / maxdist(win_cls_w);
    end

    % Determine mask category and rescue status for each spike
    mask_id = zeros(1, n_quar);
    status_id = zeros(1, n_quar);
    sdnum = par.template_sdnum;  % Cutoff threshold

    for i = 1:n_quar
        idx = quar_indices_in_all(i);
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

    status_labels = {'Both Rescued', 'Weight Enables', 'Weight Prevents', 'Neither Rescued'};

    % Save folder setup
    spike_dir = pwd;
    save_folder = fullfile(spike_dir, 'rescue_waveform_plots');
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    fig_summary = figure('Position', [50, 50, 1200, 400]);
    
    % Original clustered spikes
    subplot(1, 5, 1);
    sampled = sample_waveforms(spikes_good_classified, max_waveforms);
    plot_waveform_summary(sampled, [0 0 0], 'Original');
    title(sprintf('Original\n(n=%d)', size(spikes_good_classified, 1)));
    ylabel('Amplitude');
    
    % Each rescue status (all masks combined)
    status_colors = [0.2 0.7 0.3; 0.2 0.4 0.8; 0.9 0.3 0.2; 0.5 0.5 0.5];  % green, blue, red, gray
    for s = 1:4
        subplot(1, 5, s + 1);
        idx = (status_id == s);
        spikes_subset = spikes_quar(idx, :);
        n_subset = sum(idx);
        
        if n_subset > 0
            sampled = sample_waveforms(spikes_subset, max_waveforms);
            plot_waveform_summary(sampled, status_colors(s,:), status_labels{s});
        end
        title(sprintf('%s\n(n=%d)', status_labels{s}, n_subset));
    end
    
    sgtitle(sprintf('Waveforms by Rescue Status - Channel %s (weight=%.1f)', ch_lbl, weight), ...
        'Interpreter', 'none');
    
    save_name_summary = fullfile(save_folder, sprintf('%s_waves_wt%.1f.png', ch_lbl, weight));
    exportgraphics(fig_summary, save_name_summary, 'Resolution', 300);
    fprintf('Saved summary plot to: %s\n', save_name_summary);

    unique_masks = unique(mask_id);
    n_masks = length(unique_masks);
    n_rows = n_masks + 1;  % +1 for original
    
    fig_stable = figure('Position', [50, 50, 700, 200 + 150*n_masks]);
    
    % Original clustered spikes in first row
    subplot(n_rows, 2, 1);
    sampled = sample_waveforms(spikes_good_classified, max_waveforms);
    plot_waveform_summary(sampled, [0 0 0], 'Original Clustered');
    title(sprintf('Original (n=%d)', size(spikes_good_classified, 1)));
    
    subplot(n_rows, 2, 2);
    axis off;
    
    % Plot by mask type - only status 1 (Both Rescued) and status 4 (Neither)
    stable_status = [1, 4];  % Both Rescued, Neither
    stable_labels = {'Both Rescued', 'Neither Rescued'};
    
    for mi = 1:n_masks
        m = unique_masks(mi);
        row = mi + 1;
        
        for si = 1:2
            s = stable_status(si);
            subplot(n_rows, 2, (row-1)*2 + si);
            
            idx = (mask_id == m) & (status_id == s);
            spikes_subset = spikes_quar(idx, :);
            n_subset = sum(idx);
            
            if n_subset > 0
                sampled = sample_waveforms(spikes_subset, max_waveforms);
                plot_waveform_summary(sampled, colors(m,:), mask_labels{m});
            else
                axis off;
            end
            
            if mi == 1
                title(sprintf('%s\n(n=%d)', stable_labels{si}, n_subset));
            else
                title(sprintf('n=%d', n_subset));
            end
            
            if si == 1
                ylabel(mask_labels{m}, 'FontWeight', 'bold', 'Color', colors(m,:));
            end
        end
    end
    
    sgtitle(sprintf('Stable States by Mask - Channel %s (weight=%.1f)', ch_lbl, weight), ...
        'Interpreter', 'none');
    
    save_name_stable = fullfile(save_folder, sprintf('%s_waves_stable_wt%.1f.png', ch_lbl, weight));
    exportgraphics(fig_stable, save_name_stable, 'Resolution', 300);
    fprintf('Saved stable states plot to: %s\n', save_name_stable);

    fig_changed = figure('Position', [50, 50, 700, 200 + 150*n_masks]);
    
    % Original clustered spikes in first row
    subplot(n_rows, 2, 1);
    sampled = sample_waveforms(spikes_good_classified, max_waveforms);
    plot_waveform_summary(sampled, [0 0 0], 'Original Clustered');
    title(sprintf('Original (n=%d)', size(spikes_good_classified, 1)));
    
    subplot(n_rows, 2, 2);
    axis off;
    
    % Plot by mask type - only status 2 (Weight Enables) and status 3 (Weight Prevents)
    changed_status = [2, 3];  % Weight Enables, Weight Prevents
    changed_labels = {'Weight Enables', 'Weight Prevents'};
    
    for mi = 1:n_masks
        m = unique_masks(mi);
        row = mi + 1;
        
        for si = 1:2
            s = changed_status(si);
            subplot(n_rows, 2, (row-1)*2 + si);
            
            idx = (mask_id == m) & (status_id == s);
            spikes_subset = spikes_quar(idx, :);
            n_subset = sum(idx);
            
            if n_subset > 0
                sampled = sample_waveforms(spikes_subset, max_waveforms);
                plot_waveform_summary(sampled, colors(m,:), mask_labels{m});
            else
                axis off;
            end
            
            if mi == 1
                title(sprintf('%s\n(n=%d)', changed_labels{si}, n_subset));
            else
                title(sprintf('n=%d', n_subset));
            end
            
            if si == 1
                ylabel(mask_labels{m}, 'FontWeight', 'bold', 'Color', colors(m,:));
            end
        end
    end
    
    sgtitle(sprintf('Changed States by Mask - Channel %s (weight=%.1f)', ch_lbl, weight), ...
        'Interpreter', 'none');
    
    save_name_changed = fullfile(save_folder, sprintf('%s_waves_changed_wt%.1f.png', ch_lbl, weight));
    exportgraphics(fig_changed, save_name_changed, 'Resolution', 300);
    fprintf('Saved changed states plot to: %s\n', save_name_changed);
end

function plot_waveform_summary(waveforms, color, ~)
    if isempty(waveforms)
        return;
    end
    
    tvec = 1:size(waveforms, 2);
    
    % Plot individual waveforms with transparency (like plot_channel_spikes_with_mask)
    % Use more waveforms but with low alpha for better visualization
    n_show = min(500, size(waveforms, 1));
    if size(waveforms, 1) > n_show
        show_idx = randperm(size(waveforms, 1), n_show);
    else
        show_idx = 1:size(waveforms, 1);
    end
    
    hold on;
    % Plot waveforms with RGBA transparency
    plot(tvec, waveforms(show_idx, :)', 'Color', [color, 0.15], 'LineWidth', 0.8);
    
    % Plot mean waveform in black
    plot(tvec, mean(waveforms, 1), 'k', 'LineWidth', 2.4);
    
    hold off;
    grid on;
    box off;
    set(gca, 'GridAlpha', 0.25, 'LineWidth', 0.8);
    xlabel('Sample');
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
        
        % if width_max < length(spike_x)/2.5
        %     w_vect(left_width:right_width) = pk_weight;
        % end
    end
    
    weights = w_vect';
    normConst = sqrt(length(weights)) / sqrt(sum(weights));
end