function out = plot_spike_distances(orig_spikes, resc_spikes, ch, cluster_num, use_template_weight)
% plot_spike_distances - Compute and plot distance metrics for original vs rescued spikes
% Inputs:
%   orig_spikes        - Matrix of original spike waveforms (n_spikes x n_samples)
%   resc_spikes        - Matrix of rescued spike waveforms (n_spikes x n_samples)
%   ch                 - Channel number or label
%   cluster_num        - Cluster number
%   use_template_weight - (optional, default false) If true, sweeps spike weight then
%                         template weight separately over fixed pairs. If false, uses
%                         the original combined weight sweep up to 1000.
% Outputs:
%   out - Structure with fields:
%         .weights / .spike_weights / .tmpl_weights - weight info
%         .orig - Structure with table for each weight pair
%         .resc - Structure with table for each weight pair

    if nargin < 5
        use_template_weight = false;
    end

    % Create folder (append ' template weight' suffix when using that mode)
    folder_base = sprintf('wave%d_clust%d', ch, cluster_num);
    if use_template_weight
        folder_name = [folder_base ' xOR weight + AND weight'];
    else
        folder_name = folder_base;
    end
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    
    % Load data for distance computation
    ch_lbl = get_channel_label(ch);
    fname_spk = sprintf('%s_spikes.mat', ch_lbl);
    fname_times = sprintf('times_%s.mat', ch_lbl);
    
    if ~exist(fname_spk, 'file') || ~exist(fname_times, 'file')
        error('Data files not found for distance computation: %s and %s', fname_spk, fname_times);
    end
    
    SPK = load(fname_spk);
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
        
        fprintf('Building templates from %d original spikes (excluding %d rescued)\n', ...
            sum(original_good_mask), sum(~original_good_mask));
    else
        original_good_mask = true(size(class_good));
        fprintf('Building templates from all %d spikes (no rescue_mask found)\n', length(class_good));
    end
    
    % Debug: print template info
    [centers, maxdist, ~] = build_templates(class_good(original_good_mask), spikes_good_classified(original_good_mask, :));

    
    par = SPK.par;
    par.amp_dir = 'neg';

    if use_template_weight
        % Template weight sweep mode
        % Phase 1: spike weight increases [1,25,50,100], template held at 1
        % Phase 2: spike held at 100, template increases [5,10,25,50,100]
        spike_weights = [1, 5, 10, 25, 50, 100, 100, 100, 100, 100, 100];
        tmpl_weights  = [1, 1, 1, 1, 1,  1, 5, 10, 25, 50, 100];
        n_pairs = length(spike_weights);
        % Build x-axis labels: (spike_weight, template_weight)
        weight_pair_labels = arrayfun(@(s,t) sprintf('(%d,%d)', s, t), spike_weights, tmpl_weights, 'UniformOutput', false);
        % First index where template weight starts increasing
        tmpl_transition_idx = find(tmpl_weights > tmpl_weights(1), 1);
    else
        weights = [1, 5, 10, 25, 50, 100];
        n_pairs = length(weights);
    end

    % Compute distances for orig_spikes
    n_orig = size(orig_spikes, 1);
    norm_dist_orig = zeros(n_orig, n_pairs);
    raw_dist_orig = zeros(n_orig, n_pairs);
    win_cls_orig = zeros(n_orig, n_pairs);

    % Create tables for all weight pairs
    dist_tables_orig = cell(1, n_pairs);
    table_weight = 300;  % Weight to display in console
    
    for w_idx = 1:n_pairs
        if use_template_weight
            cur_spike_wt = spike_weights(w_idx);
            cur_tmpl_wt  = tmpl_weights(w_idx);
        else
            par.pk_weight = weights(w_idx);
        end
        dist_table_w = zeros(n_orig, size(centers, 1) + 1);
        
        for i = 1:n_orig
            spike = orig_spikes(i, :);
            if use_template_weight
                [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, cur_spike_wt, par.amp_dir, cur_tmpl_wt);
            else
                [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, par.pk_weight, par.amp_dir);
            end
            dist_w = (normConst_vec .* sqrt(sum(w_matrix .* (repmat(spike, size(centers,1),1) - centers).^2, 2)))';
            dist_table_w(i, 1:size(centers, 1)) = dist_w;
            
            % Find conforming clusters (within maxdist) like nearest_neighbor does
            conforming = find(dist_w < maxdist * par.template_sdnum);
            if isempty(conforming)
                win_cls_orig(i, w_idx) = 0;  % No cluster within threshold
            else
                [~, idx] = min(dist_w(conforming));
                win_cls_orig(i, w_idx) = conforming(idx);  % Actual cluster number
            end
            
            % Store winning cluster
            dist_table_w(i, end) = win_cls_orig(i, w_idx);
            
            raw_dist_orig(i, w_idx) = dist_w(cluster_num);
            norm_dist_orig(i, w_idx) = dist_w(cluster_num) / maxdist(cluster_num);
        end
        
        % Create table for this weight
        col_names = cell(1, size(centers, 1) + 1);
        for cls = 1:size(centers, 1)
            col_names{cls} = sprintf('Dist_C%d', cls);
        end
        col_names{end} = 'Winner';
        dist_tables_orig{w_idx} = array2table(dist_table_w, 'VariableNames', col_names);
    end
    
    % Display table for one weight
    % [~, display_idx] = min(abs(weights - table_weight));
    % fprintf('\n=== GOOD SPIKES (Weight=%d) ===\n', weights(display_idx));
    % disp(dist_tables_orig{display_idx});
    
    % Compute distances for resc_spikes
    n_resc = size(resc_spikes, 1);
    norm_dist_resc = zeros(n_resc, n_pairs);
    raw_dist_resc = zeros(n_resc, n_pairs);
    win_cls_resc = zeros(n_resc, n_pairs);
    
    % Create tables for all weight pairs
    dist_tables_resc = cell(1, n_pairs);
    
    for w_idx = 1:n_pairs
        if use_template_weight
            cur_spike_wt = spike_weights(w_idx);
            cur_tmpl_wt  = tmpl_weights(w_idx);
        else
            par.pk_weight = weights(w_idx);
        end
        dist_table_w = zeros(n_resc, size(centers, 1) + 1);
        
        for i = 1:n_resc
            spike = resc_spikes(i, :);
            if use_template_weight
                [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, cur_spike_wt, par.amp_dir, cur_tmpl_wt);
            else
                [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, par.pk_weight, par.amp_dir);
            end
            dist_w = (normConst_vec .* sqrt(sum(w_matrix .* (repmat(spike, size(centers,1),1) - centers).^2, 2)))';
            dist_table_w(i, 1:size(centers, 1)) = dist_w;
            
            % Find conforming clusters (within maxdist) like nearest_neighbor does
            conforming = find(dist_w < maxdist * par.template_sdnum);
            if isempty(conforming)
                win_cls_resc(i, w_idx) = 0;  % No cluster within threshold
            else
                [~, idx] = min(dist_w(conforming));
                win_cls_resc(i, w_idx) = conforming(idx);  % Actual cluster number
            end
            
            % Store winning cluster
            dist_table_w(i, end) = win_cls_resc(i, w_idx);
            
            raw_dist_resc(i, w_idx) = dist_w(cluster_num);
            norm_dist_resc(i, w_idx) = dist_w(cluster_num) / maxdist(cluster_num);
        end
        
        % Create table for this weight
        col_names = cell(1, size(centers, 1) + 1);
        for cls = 1:size(centers, 1)
            col_names{cls} = sprintf('Dist_C%d', cls);
        end
        col_names{end} = 'Winner';
        dist_tables_resc{w_idx} = array2table(dist_table_w, 'VariableNames', col_names);
    end
    
    % Display table for one weight
    % fprintf('\n=== POOR_RESCUE SPIKES (Weight=%d) ===\n', weights(display_idx));
    % disp(dist_tables_resc{display_idx});
    
    % Create output structure with all tables
    if use_template_weight
        out.spike_weights = spike_weights;
        out.tmpl_weights  = tmpl_weights;
        out.weight_pair_labels = weight_pair_labels;
    else
        out.weights = weights;
    end
    out.orig = struct();
    out.resc = struct();
    for w_idx = 1:n_pairs
        if use_template_weight
            field_name = sprintf('S%d_T%d', spike_weights(w_idx), tmpl_weights(w_idx));
        else
            field_name = sprintf('W%d', weights(w_idx));
        end
        out.orig.(field_name) = dist_tables_orig{w_idx};
        out.resc.(field_name) = dist_tables_resc{w_idx};
    end
    
    out_table = fullfile(folder_name, 'distance_tables.mat');
    save(out_table, 'out');
    % fprintf('Distance tables saved to: %s\n', out_table);
    
    % Compute trends for each spike and track indices
    pos_trend_good = 0;
    neutral_trend_good = 0;
    neg_trend_good = 0;
    pos_idx_good = [];
    neutral_neg_idx_good = [];
    slopes = zeros(n_orig, 1);
    for i = 1:n_orig
        % Linear fit: use pair index for template-weight mode, log10(weights) for classic mode
        if use_template_weight
            p = polyfit(1:n_pairs, norm_dist_orig(i, :), 1);
        else
            p = polyfit(log10(weights), norm_dist_orig(i, :), 1);
        end
        slopes(i) = p(1);
        if slopes(i) > 0.1  % Threshold for positive trend
            pos_trend_good = pos_trend_good + 1;
            pos_idx_good = [pos_idx_good; i];
        elseif slopes(i) < -0.1  % Threshold for negative trend
            neg_trend_good = neg_trend_good + 1;
            neutral_neg_idx_good = [neutral_neg_idx_good; i];
        else
            neutral_trend_good = neutral_trend_good + 1;
            neutral_neg_idx_good = [neutral_neg_idx_good; i];
        end
    end
    
    pos_trend_poor = 0;
    neutral_trend_poor = 0;
    neg_trend_poor = 0;
    pos_idx_poor = [];
    neutral_neg_idx_poor = [];
    slopes_resc = zeros(n_resc, 1);
    
    for i = 1:n_resc
        % Linear fit: use pair index for template-weight mode, log10(weights) for classic mode
        if use_template_weight
            p = polyfit(1:n_pairs, norm_dist_resc(i, :), 1);
        else
            p = polyfit(log10(weights), norm_dist_resc(i, :), 1);
        end
        slopes_resc(i) = p(1);
        if slopes_resc(i) > 0.05
            pos_trend_poor = pos_trend_poor + 1;
            pos_idx_poor = [pos_idx_poor; i];
        elseif slopes_resc(i) < -0.05
            neg_trend_poor = neg_trend_poor + 1;
            neutral_neg_idx_poor = [neutral_neg_idx_poor; i];
        else
            neutral_trend_poor = neutral_trend_poor + 1;
            neutral_neg_idx_poor = [neutral_neg_idx_poor; i];
        end
    end
    
    % fprintf('\n=== TREND ANALYSIS ===\n');
    % fprintf('               Positive  Neutral  Negative\n');
    % fprintf('Good:          %8d %8d %9d\n', pos_trend_good, neutral_trend_good, neg_trend_good);
    % fprintf('Poor_Rescue:   %8d %8d %9d\n', pos_trend_poor, neutral_trend_poor, neg_trend_poor);
    
    % Identify special spikes for highlighting and separate plots
    % 1. Top 2 and bottom 2 finishing distances (at last weight)
    final_dists_good = norm_dist_orig(:, end);
    final_dists_poor = norm_dist_resc(:, end);
    
    [~, idx_sorted_good] = sort(final_dists_good);
    [~, idx_sorted_poor] = sort(final_dists_poor);
    
    top2_finish_good = idx_sorted_good(end-1:end);  % Highest 2 distances
    bottom2_finish_good = idx_sorted_good(1:min(2, n_orig));  % Lowest 2 distances
    top2_finish_poor = idx_sorted_poor(end-1:end);
    bottom2_finish_poor = idx_sorted_poor(1:min(2, n_resc));
    
    % 2. Highest and lowest slope spikes
    [~, idx_sorted_slopes_good] = sort(slopes);
    [~, idx_sorted_slopes_poor] = sort(slopes_resc);
    
    highest_slope_good = idx_sorted_slopes_good(end-1:end);  % 2 highest slopes
    lowest_slope_good = idx_sorted_slopes_good(1:min(2, n_orig));  % 2 lowest slopes
    highest_slope_poor = idx_sorted_slopes_poor(end-1:end);
    lowest_slope_poor = idx_sorted_slopes_poor(1:min(2, n_resc));
    
    % Combine for highlighting
    highlight_good = unique([top2_finish_good; bottom2_finish_good; highest_slope_good; lowest_slope_good]);
    highlight_poor = unique([top2_finish_poor; bottom2_finish_poor; highest_slope_poor; lowest_slope_poor]);
    
    % Collect data for table
    % For neutral/negative
    nn_slopes = [];
    nn_widths = [];
    nn_types = {};
    for i = 1:length(neutral_neg_idx_good)
        spike_idx = neutral_neg_idx_good(i);
        spike = orig_spikes(spike_idx, :);
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            width = right_w - left_w;
            nn_slopes = [nn_slopes; slopes(spike_idx)];
            nn_widths = [nn_widths; width];
            nn_types = [nn_types; 'good'];
        end
    end
    for i = 1:length(neutral_neg_idx_poor)
        spike_idx = neutral_neg_idx_poor(i);
        spike = resc_spikes(spike_idx, :);
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            width = right_w - left_w;
            nn_slopes = [nn_slopes; slopes_resc(spike_idx)];
            nn_widths = [nn_widths; width];
            nn_types = [nn_types; 'poor_rescue'];
        end
    end
    % Sort neutral/negative by slope
    [nn_sorted_slopes, nn_idx] = sort(nn_slopes);
    nn_sorted_widths = nn_widths(nn_idx);
    nn_sorted_types = nn_types(nn_idx);
    
    % For positive
    pos_slopes = [];
    pos_widths = [];
    pos_types = {};
    for i = 1:length(pos_idx_good)
        spike_idx = pos_idx_good(i);
        spike = orig_spikes(spike_idx, :);
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            width = right_w - left_w;
            pos_slopes = [pos_slopes; slopes(spike_idx)];
            pos_widths = [pos_widths; width];
            pos_types = [pos_types; 'good'];
        end
    end
    for i = 1:length(pos_idx_poor)
        spike_idx = pos_idx_poor(i);
        spike = resc_spikes(spike_idx, :);
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            width = right_w - left_w;
            pos_slopes = [pos_slopes; slopes_resc(spike_idx)];
            pos_widths = [pos_widths; width];
            pos_types = [pos_types; 'poor_rescue'];
        end
    end
    % Sort positive by slope
    [pos_sorted_slopes, pos_idx_sort] = sort(pos_slopes);
    pos_sorted_widths = pos_widths(pos_idx_sort);
    pos_sorted_types = pos_types(pos_idx_sort);
    
    % Set up x-axis values for the distance plots
    % Always use evenly spaced positions so the first point isn't pinned against 0
    x_vals = 1:n_pairs;
    if use_template_weight
        x_label_str = '(Spike Weight, Template Weight)';
    else
        x_label_str = 'Weight';
    end

    % Figure for normalized distances
    figure;
    hold on;
    h1 = scatter(nan, nan, 'bo', 'filled'); % Dummy for legend - Good
    h2 = scatter(nan, nan, '^b', 'filled'); % Dummy for legend - Good unclassified
    h3 = scatter(nan, nan, 'ro', 'filled'); % Dummy for legend - Poor_Rescue
    h4 = scatter(nan, nan, '^r', 'filled'); % Dummy for legend - Poor_Rescue unclassified

    % Plot good spikes with lines connecting each spike's points
    for i = 1:n_orig
        % Check if this spike should be highlighted
        linewidth = 0.5;
        if ismember(i, highlight_good)
            linewidth = 2.5;  % Bolder line for highlighted spikes
        end

        % Draw line connecting all points for this spike
        plot(x_vals, norm_dist_orig(i, :), 'b-', 'LineWidth', linewidth, 'HandleVisibility', 'off');

        % Plot markers on top
        for w_idx = 1:n_pairs
            if win_cls_orig(i, w_idx) == 0
                scatter(x_vals(w_idx), norm_dist_orig(i, w_idx), '^b', 'filled', 'HandleVisibility', 'off');
            else
                scatter(x_vals(w_idx), norm_dist_orig(i, w_idx), 'bo', 'filled', 'HandleVisibility', 'off');
            end
        end
    end

    % Plot poor_rescue spikes with lines connecting each spike's points
    for i = 1:n_resc
        % Check if this spike should be highlighted
        linewidth = 0.5;
        if ismember(i, highlight_poor)
            linewidth = 2.5;  % Bolder line for highlighted spikes
        end

        % Draw line connecting all points for this spike
        plot(x_vals, norm_dist_resc(i, :), 'r-', 'LineWidth', linewidth, 'HandleVisibility', 'off');

        % Plot markers on top
        for w_idx = 1:n_pairs
            if win_cls_resc(i, w_idx) == 0
                scatter(x_vals(w_idx), norm_dist_resc(i, w_idx), '^r', 'filled', 'HandleVisibility', 'off');
            else
                scatter(x_vals(w_idx), norm_dist_resc(i, w_idx), 'ro', 'filled', 'HandleVisibility', 'off');
            end
        end
    end

    % Add shaded region + dividing line where template weight begins increasing
    if use_template_weight && ~isempty(tmpl_transition_idx)
        yl = ylim;
        patch([x_vals(tmpl_transition_idx)-0.5, x_vals(end)+0.5, x_vals(end)+0.5, x_vals(tmpl_transition_idx)-0.5], ...
              [yl(1), yl(1), yl(2), yl(2)], [0.85 0.85 1.0], ...
              'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        xline(x_vals(tmpl_transition_idx)-0.5, '--', 'Template weight increasing', ...
              'Color', [0.2 0.2 0.8], 'LineWidth', 1.5, ...
              'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
        ylim(yl);  % restore limits after patch
    end

    % X-axis formatting
    if use_template_weight
        xticks(x_vals);
        xticklabels(weight_pair_labels);
        xtickangle(45);
        xlim([x_vals(1), x_vals(end) + 0.5]);
    else
        xticks(1:n_pairs);
        xticklabels(string(weights));
        xlim([1, n_pairs + 0.5]);
    end
    title(sprintf('Normalized Distances vs Weight - Cluster %d', cluster_num), 'Interpreter', 'none');
    xlabel(x_label_str);
    ylabel('Normalized Distance');
    
    % Add trend analysis table as text annotation
    trend_text = sprintf(['Trend Analysis:\n' ...
        '                Pos   Neut   Neg\n' ...
        'Good:          %3d   %3d   %3d\n' ...
        'Poor_Rescue:   %3d   %3d   %3d'], ...
        pos_trend_good, neutral_trend_good, neg_trend_good, ...
        pos_trend_poor, neutral_trend_poor, neg_trend_poor);
    annotation('textbox', [0.72, 0.70, 0.2, 0.15], 'String', trend_text, ...
        'FontName', 'Courier', 'FontSize', 8, 'BackgroundColor', 'white', ...
        'EdgeColor', 'black', 'FitBoxToText', 'on', 'Interpreter', 'none');
    
    legend([h1 h2 h3 h4], 'Good', 'Good (unclassified)', 'Poor_Rescue', 'Poor_Rescue (unclassified)', 'Location', 'eastoutside', 'Interpreter', 'none');
    grid on;
    hold off;
    
    % Save normalized distances
    save_path_norm = sprintf('%s/norm_distances_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path_norm);
    close(gcf);
    fprintf('Normalized distance plot saved to: %s\n', save_path_norm);
    
    % Figure for raw distances
    figure;
    hold on;
    h1 = scatter(nan, nan, 'bo', 'filled'); % Dummy for legend - Good
    h2 = scatter(nan, nan, '^b', 'filled'); % Dummy for legend - Good unclassified
    h3 = scatter(nan, nan, 'ro', 'filled'); % Dummy for legend - Poor_Rescue
    h4 = scatter(nan, nan, '^r', 'filled'); % Dummy for legend - Poor_Rescue unclassified

    % Plot good spikes with lines connecting each spike's points
    for i = 1:n_orig
        % Check if this spike should be highlighted
        linewidth = 0.5;
        if ismember(i, highlight_good)
            linewidth = 2.5;  % Bolder line for highlighted spikes
        end

        % Draw line connecting all points for this spike
        plot(x_vals, raw_dist_orig(i, :), 'b-', 'LineWidth', linewidth, 'HandleVisibility', 'off');

        % Plot markers on top
        for w_idx = 1:n_pairs
            if win_cls_orig(i, w_idx) == 0
                scatter(x_vals(w_idx), raw_dist_orig(i, w_idx), '^b', 'filled', 'HandleVisibility', 'off');
            else
                scatter(x_vals(w_idx), raw_dist_orig(i, w_idx), 'bo', 'filled', 'HandleVisibility', 'off');
            end
        end
    end

    % Plot poor_rescue spikes with lines connecting each spike's points
    for i = 1:n_resc
        % Check if this spike should be highlighted
        linewidth = 0.5;
        if ismember(i, highlight_poor)
            linewidth = 2.5;  % Bolder line for highlighted spikes
        end

        % Draw line connecting all points for this spike
        plot(x_vals, raw_dist_resc(i, :), 'r-', 'LineWidth', linewidth, 'HandleVisibility', 'off');

        % Plot markers on top
        for w_idx = 1:n_pairs
            if win_cls_resc(i, w_idx) == 0
                scatter(x_vals(w_idx), raw_dist_resc(i, w_idx), '^r', 'filled', 'HandleVisibility', 'off');
            else
                scatter(x_vals(w_idx), raw_dist_resc(i, w_idx), 'ro', 'filled', 'HandleVisibility', 'off');
            end
        end
    end

    % Add shaded region + dividing line where template weight begins increasing
    if use_template_weight && ~isempty(tmpl_transition_idx)
        yl = ylim;
        patch([x_vals(tmpl_transition_idx)-0.5, x_vals(end)+0.5, x_vals(end)+0.5, x_vals(tmpl_transition_idx)-0.5], ...
              [yl(1), yl(1), yl(2), yl(2)], [0.85 0.85 1.0], ...
              'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        xline(x_vals(tmpl_transition_idx)-0.5, '--', 'Template weight increasing', ...
              'Color', [0.2 0.2 0.8], 'LineWidth', 1.5, ...
              'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
        ylim(yl);
    end

    % X-axis formatting
    if use_template_weight
        xticks(x_vals);
        xticklabels(weight_pair_labels);
        xtickangle(45);
        xlim([x_vals(1), x_vals(end) + 0.5]);
    else
        xticks(1:n_pairs);
        xticklabels(string(weights));
        xlim([1, n_pairs + 0.5]);
    end
    title(sprintf('Raw Distances vs Weight - Cluster %d', cluster_num), 'Interpreter', 'none');
    xlabel(x_label_str);
    ylabel('Raw Distance');
    
    % Add trend analysis table as text annotation
    trend_text = sprintf(['Trend Analysis:\n' ...
        '                Pos   Neut   Neg\n' ...
        'Good:          %3d   %3d   %3d\n' ...
        'Poor_Rescue:   %3d   %3d   %3d'], ...
        pos_trend_good, neutral_trend_good, neg_trend_good, ...
        pos_trend_poor, neutral_trend_poor, neg_trend_poor);
    annotation('textbox', [0.72, 0.70, 0.2, 0.15], 'String', trend_text, ...
        'FontName', 'Courier', 'FontSize', 8, 'BackgroundColor', 'white', ...
        'EdgeColor', 'black', 'FitBoxToText', 'on', 'Interpreter', 'none');
    
    legend([h1 h2 h3 h4], 'Good', 'Good (unclassified)', 'Poor_Rescue', 'Poor_Rescue (unclassified)', 'Location', 'eastoutside', 'Interpreter', 'none');
    grid on;
    hold off;
    
    % Save raw distances
    save_path_raw = sprintf('%s/raw_distances_wave%d_clust%d.png', folder_name, ch, cluster_num);
    % saveas(gcf, save_path_raw);
    close(gcf);
    fprintf('Raw distance plot saved to: %s\n', save_path_raw);
    
    % Calculate common y-axis limits for waveform plots
    all_waveforms = [orig_spikes(:); resc_spikes(:)];
    y_min = min(all_waveforms);
    y_max = max(all_waveforms);
    y_limits = [y_min, y_max];
    offset = (y_max - y_min) * 0.05;
    
    % Extract template for this cluster
    template_center = centers(cluster_num, :);
    template_std = maxdist(cluster_num);
    template_width_struct = get_peak_width(template_center, par.amp_dir);
    
    % FIGURE 1: Neutral and Negative trend waveforms
    if ~isempty(neutral_neg_idx_good) || ~isempty(neutral_neg_idx_poor)
        figure;
        
        % Top: Good spikes (neutral + negative)
        subplot(2, 1, 1);
        hold on;
        if ~isempty(neutral_neg_idx_good)
            n_spikes = length(neutral_neg_idx_good);
            colors = lines(n_spikes);
            for i = 1:n_spikes
                spike_idx = neutral_neg_idx_good(i);
                spike = orig_spikes(spike_idx, :);
                plot(spike, 'Color', colors(i, :), 'LineWidth', 1);
                
                % Get and plot peak width region
                width_struct = get_peak_width(spike, par.amp_dir);
                left_w = width_struct.left;
                right_w = width_struct.right;
                if ~isnan(left_w)
                    % Plot dotted line at peak width boundaries
                    plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                    plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                end
            end
        end
        % Plot template
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
        if ~isnan(template_width_struct.left)
            plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
            plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
        end
        hold off;
        title(sprintf('Good Spikes - Neutral & Negative Trends - Cluster %d', cluster_num), 'Interpreter', 'none');
        xlabel('Sample');
        ylabel('Amplitude');
        ylim(y_limits);
        
        % Bottom: Poor_rescue spikes (neutral + negative)
        subplot(2, 1, 2);
        hold on;
        if ~isempty(neutral_neg_idx_poor)
            n_spikes = length(neutral_neg_idx_poor);
            colors = lines(n_spikes);
            for i = 1:n_spikes
                spike_idx = neutral_neg_idx_poor(i);
                spike = resc_spikes(spike_idx, :);
                plot(spike, 'Color', colors(i, :), 'LineWidth', 1);
                
                % Get and plot peak width region
                width_struct = get_peak_width(spike, par.amp_dir);
                left_w = width_struct.left;
                right_w = width_struct.right;
                if ~isnan(left_w)
                    % Plot dotted line at peak width boundaries
                    plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                    plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                end
            end
        end
        % Plot template
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
        if ~isnan(template_width_struct.left)
            plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
            plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
        end
        hold off;
        title(sprintf('Poor_Rescue Spikes - Neutral & Negative Trends - Cluster %d', cluster_num), 'Interpreter', 'none');
        xlabel('Sample');
        ylabel('Amplitude');
        ylim(y_limits);
        
        save_path = sprintf('%s/waveforms_neutral_neg_wave%d_clust%d.png', folder_name, ch, cluster_num);
        saveas(gcf, save_path);
        close(gcf);
        fprintf('Neutral & Negative trend waveforms saved to: %s\n', save_path);
    end
    
    % FIGURE 2: Positive trend waveforms with width markers
    if ~isempty(pos_idx_good) || ~isempty(pos_idx_poor)
        figure;
        
        % Top: Good spikes (positive)
        subplot(2, 1, 1);
        hold on;
        if ~isempty(pos_idx_good)
            n_spikes = length(pos_idx_good);
            colors = lines(n_spikes);
            for i = 1:n_spikes
                spike_idx = pos_idx_good(i);
                spike = orig_spikes(spike_idx, :);
                plot(spike, 'Color', colors(i, :), 'LineWidth', 1);
                
                % Get and plot peak width region
                width_struct = get_peak_width(spike, par.amp_dir);
                left_w = width_struct.left;
                right_w = width_struct.right;
                if ~isnan(left_w)
                    % Plot dotted line at peak width boundaries
                    plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                    plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                end
            end
        end
        % Plot template
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
        if ~isnan(template_width_struct.left)
            plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
            plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
        end
        hold off;
        title(sprintf('Good Spikes - Positive Trends - Cluster %d', cluster_num), 'Interpreter', 'none');
        xlabel('Sample');
        ylabel('Amplitude');
        ylim(y_limits);
        
        % Bottom: Poor_rescue spikes (positive)
        subplot(2, 1, 2);
        hold on;
        if ~isempty(pos_idx_poor)
            n_spikes = length(pos_idx_poor);
            colors = lines(n_spikes);
            for i = 1:n_spikes
                spike_idx = pos_idx_poor(i);
                spike = resc_spikes(spike_idx, :);
                plot(spike, 'Color', colors(i, :), 'LineWidth', 1);
                
                % Get and plot peak width region
                width_struct = get_peak_width(spike, par.amp_dir);
                left_w = width_struct.left;
                right_w = width_struct.right;
                if ~isnan(left_w)
                    % Plot dotted line at peak width boundaries
                    plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                    plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                end
            end
        end
        % Plot template
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
        if ~isnan(template_width_struct.left)
            plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
            plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
        end
        hold off;
        title(sprintf('Poor_Rescue Spikes - Positive Trends - Cluster %d', cluster_num), 'Interpreter', 'none');
        xlabel('Sample');
        ylabel('Amplitude');
        ylim(y_limits);
        
        save_path = sprintf('%s/waveforms_positive_wave%d_clust%d.png', folder_name, ch, cluster_num);
        saveas(gcf, save_path);
        close(gcf);
        fprintf('Positive trend waveforms saved to: %s\n', save_path);
    end
    
    % FIGURE 3: Combined waveforms by trend
    figure;
    
    % Top: ALL neutral + negative trends
    subplot(2, 1, 1);
    hold on;
    % Plot good spikes
    if ~isempty(neutral_neg_idx_good)
        n_spikes = length(neutral_neg_idx_good);
        colors = lines(n_spikes + length(neutral_neg_idx_poor));
        for i = 1:n_spikes
            spike_idx = neutral_neg_idx_good(i);
            spike = orig_spikes(spike_idx, :);
            plot(spike, 'Color', colors(i, :), 'LineWidth', 1);
            
            % Get and plot peak width region
            width_struct = get_peak_width(spike, par.amp_dir);
            left_w = width_struct.left;
            right_w = width_struct.right;
            if ~isnan(left_w)
                plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
            end
        end
    end
    % Plot poor_rescue spikes
    if ~isempty(neutral_neg_idx_poor)
        n_offset = length(neutral_neg_idx_good);
        n_spikes = length(neutral_neg_idx_poor);
        colors = lines(n_offset + n_spikes);
        for i = 1:n_spikes
            spike_idx = neutral_neg_idx_poor(i);
            spike = resc_spikes(spike_idx, :);
            plot(spike, 'Color', colors(n_offset + i, :), 'LineWidth', 1);
            
            % Get and plot peak width region
            width_struct = get_peak_width(spike, par.amp_dir);
            left_w = width_struct.left;
            right_w = width_struct.right;
            if ~isnan(left_w)
                plot([left_w left_w], y_limits, ':', 'Color', colors(n_offset + i, :), 'LineWidth', 1);
                plot([right_w right_w], y_limits, ':', 'Color', colors(n_offset + i, :), 'LineWidth', 1);
            end
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('All Spikes - Neutral & Negative Trends - Cluster %d', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    
    % Bottom: ALL positive trends with width markers
    subplot(2, 1, 2);
    hold on;
    % Plot good spikes
    if ~isempty(pos_idx_good)
        n_spikes = length(pos_idx_good);
        colors = lines(n_spikes + length(pos_idx_poor));
        for i = 1:n_spikes
            spike_idx = pos_idx_good(i);
            spike = orig_spikes(spike_idx, :);
            plot(spike, 'Color', colors(i, :), 'LineWidth', 1);
            
            % Get and plot peak width region
            width_struct = get_peak_width(spike, par.amp_dir);
            left_w = width_struct.left;
            right_w = width_struct.right;
            if ~isnan(left_w)
                plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
            end
        end
    end
    % Plot poor_rescue spikes
    if ~isempty(pos_idx_poor)
        n_offset = length(pos_idx_good);
        n_spikes = length(pos_idx_poor);
        colors = lines(n_offset + n_spikes);
        for i = 1:n_spikes
            spike_idx = pos_idx_poor(i);
            spike = resc_spikes(spike_idx, :);
            plot(spike, 'Color', colors(n_offset + i, :), 'LineWidth', 1);
            
            % Get and plot peak width region
            width_struct = get_peak_width(spike, par.amp_dir);
            left_w = width_struct.left;
            right_w = width_struct.right;
            if ~isnan(left_w)
                plot([left_w left_w], y_limits, ':', 'Color', colors(n_offset + i, :), 'LineWidth', 1);
                plot([right_w right_w], y_limits, ':', 'Color', colors(n_offset + i, :), 'LineWidth', 1);
            end
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('All Spikes - Positive Trends - Cluster %d', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    
    save_path = sprintf('%s/waveforms_combined_trends_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path);
    close(gcf);
    fprintf('Combined trend waveforms saved to: %s\n', save_path);
    
    % Figure for tables
    figure;
    % Left: Neutral/Negative
    subplot(1, 2, 1);
    nn_table_str = sprintf('%-6s  %-5s  %-12s\n', 'Slope', 'Width', 'Type');
    for i = 1:length(nn_sorted_slopes)
        nn_table_str = sprintf('%s%-6.3f  %-5.1f  %-12s\n', nn_table_str, nn_sorted_slopes(i), nn_sorted_widths(i), nn_sorted_types{i});
    end
    fprintf('Neutral & Negative Trends Table:\n%s\n', nn_table_str);
    text(0.5, 0.5, nn_table_str, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontName', 'Courier', 'FontSize', 8, 'Interpreter', 'none');
    title('Neutral & Negative Trends', 'Interpreter', 'none');
    axis off;
    
    % Right: Positive
    subplot(1, 2, 2);
    pos_table_str = sprintf('%-6s  %-5s  %-12s\n', 'Slope', 'Width', 'Type');
    for i = 1:length(pos_sorted_slopes)
        pos_table_str = sprintf('%s%-6.3f  %-5.1f  %-12s\n', pos_table_str, pos_sorted_slopes(i), pos_sorted_widths(i), pos_sorted_types{i});
    end
    fprintf('Positive Trends Table:\n%s\n', pos_table_str);
    text(0.5, 0.5, pos_table_str, 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontName', 'Courier', 'FontSize', 8, 'Interpreter', 'none');
    title('Positive Trends', 'Interpreter', 'none');
    axis off;
    
    table_save_path = sprintf('%s/trend_tables_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, table_save_path);
    close(gcf);
    fprintf('Trend tables saved to: %s\n', table_save_path);
    
    % NEW FIGURE: Top 2 and Bottom 2 finishing distances
    figure;
    subplot(2, 2, 1);
    hold on;
    colors = lines(2);
    for idx = 1:length(top2_finish_good)
        i = top2_finish_good(idx);
        spike = orig_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good - Top 2 Finishing Distances (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: %.2f', i, final_dists_good(i)), top2_finish_good, 'UniformOutput', false), 'Interpreter', 'none');
    
    subplot(2, 2, 2);
    hold on;
    colors = lines(2);
    for idx = 1:length(top2_finish_poor)
        i = top2_finish_poor(idx);
        spike = resc_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Poor_Rescue - Top 2 Finishing Distances (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: %.2f', i, final_dists_poor(i)), top2_finish_poor, 'UniformOutput', false), 'Interpreter', 'none');
    
    subplot(2, 2, 3);
    hold on;
    colors = lines(2);
    for idx = 1:length(bottom2_finish_good)
        i = bottom2_finish_good(idx);
        spike = orig_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good - Bottom 2 Finishing Distances (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: %.2f', i, final_dists_good(i)), bottom2_finish_good, 'UniformOutput', false), 'Interpreter', 'none');
    
    subplot(2, 2, 4);
    hold on;
    colors = lines(2);
    for idx = 1:length(bottom2_finish_poor)
        i = bottom2_finish_poor(idx);
        spike = resc_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Poor_Rescue - Bottom 2 Finishing Distances (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: %.2f', i, final_dists_poor(i)), bottom2_finish_poor, 'UniformOutput', false), 'Interpreter', 'none');
    
    save_path = sprintf('%s/waveforms_finishing_distances_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path);
    close(gcf);
    fprintf('Finishing distances waveforms saved to: %s\n', save_path);
    
    % NEW FIGURE: Highest and Lowest slope spikes
    figure;
    subplot(2, 2, 1);
    hold on;
    colors = lines(2);
    for idx = 1:length(highest_slope_good)
        i = highest_slope_good(idx);
        spike = orig_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good - Highest Slopes (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: slope=%.3f', i, slopes(i)), highest_slope_good, 'UniformOutput', false), 'Interpreter', 'none');
    
    subplot(2, 2, 2);
    hold on;
    colors = lines(2);
    for idx = 1:length(highest_slope_poor)
        i = highest_slope_poor(idx);
        spike = resc_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Poor_Rescue - Highest Slopes (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: slope=%.3f', i, slopes_resc(i)), highest_slope_poor, 'UniformOutput', false), 'Interpreter', 'none');
    
    subplot(2, 2, 3);
    hold on;
    colors = lines(2);
    for idx = 1:length(lowest_slope_good)
        i = lowest_slope_good(idx);
        spike = orig_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good - Lowest Slopes (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: slope=%.3f', i, slopes(i)), lowest_slope_good, 'UniformOutput', false), 'Interpreter', 'none');
    
    subplot(2, 2, 4);
    hold on;
    colors = lines(2);
    for idx = 1:length(lowest_slope_poor)
        i = lowest_slope_poor(idx);
        spike = resc_spikes(i, :);
        plot(spike, 'Color', colors(idx, :), 'LineWidth', 1.5);
        
        % Get and plot peak width region
        width_struct = get_peak_width(spike, par.amp_dir);
        left_w = width_struct.left;
        right_w = width_struct.right;
        if ~isnan(left_w)
            % Plot dotted line at peak width boundaries
            plot([left_w left_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
            plot([right_w right_w], y_limits, ':', 'Color', colors(idx, :), 'LineWidth', 1);
        end
    end
    % Plot template
    plot(template_center, 'k-', 'LineWidth', 2.5);
    plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
    plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    if ~isnan(template_width_struct.left)
        plot([template_width_struct.left template_width_struct.left], y_limits, 'k:', 'LineWidth', 1.5);
        plot([template_width_struct.right template_width_struct.right], y_limits, 'k:', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Poor_Rescue - Lowest Slopes (Cluster %d)', cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend(arrayfun(@(i) sprintf('Spike %d: slope=%.3f', i, slopes_resc(i)), lowest_slope_poor, 'UniformOutput', false), 'Interpreter', 'none');
    
    save_path = sprintf('%s/waveforms_extreme_slopes_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path);
    close(gcf);
    fprintf('Extreme slopes waveforms saved to: %s\n', save_path);
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

function ch_lbl = get_channel_label(ch)
    if isnumeric(ch)
        files = dir(sprintf('*_%d_spikes.mat', ch));
        if ~isempty(files)
            [~, name, ~] = fileparts(files(1).name);
            if endsWith(name, '_spikes')
                ch_lbl = name(1:end-length('_spikes'));
            else
                ch_lbl = name;
            end
        else
            ch_lbl = num2str(ch);
        end
    elseif ischar(ch) || isstring(ch)
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

function [normConst, weights] = get_weight_matrix(spike_x, tmplt_vect, pk_weight, amp_dir, template_weight)
    % Compute peak width for spike_x
    % pk_weight      - weight applied to the spike XOR region (and overlap)
    % template_weight - (optional) separate weight for the template-only XOR region.
    %                   When omitted, falls back to old behavior: symmetric_diff = max(1,pk_weight/2)
    if nargin < 5
        template_weight = [];   % empty = use old combined-weight behaviour
    end

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

        % Overlap: both spike and template have width
        overlap = spike_mask & template_mask;

        if isempty(template_weight)
            % --- Original behaviour: symmetric XOR gets half weight ---
            symmetric_diff = xor(spike_mask, template_mask);
            % weights(i, symmetric_diff) = max(1, pk_weight);
            % weights(i, overlap) = pk_weight;
            weights(i,spike_mask) = pk_weight;
        else
            % --- New behaviour: separate spike-only and template-only XOR weights ---
            spike_only    = spike_mask    & ~template_mask;
            template_only = template_mask & ~spike_mask;
            weights(i, spike_only)    = pk_weight;        % spike XOR region
            weights(i, template_only) = template_weight;  % template XOR region
            weights(i, overlap)       = pk_weight;        % overlap unchanged (uses spike weight)
        end
    end

    normConst = sqrt(n_samples) ./ sqrt(sum(weights, 2));
end
