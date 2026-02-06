function out = plot_spike_distances(orig_spikes, resc_spikes, ch, cluster_num)
% plot_spike_distances - Compute and plot distance metrics for original vs rescued spikes
% Inputs:
%   orig_spikes - Matrix of original spike waveforms (n_spikes x n_samples)
%   resc_spikes - Matrix of rescued spike waveforms (n_spikes x n_samples)
%   ch - Channel number or label
%   cluster_num - Cluster number
% Outputs:
%   out - Structure with fields:
%         .weights - Array of weight values
%         .orig - Structure with table for each weight (W1, W5, W10, etc.)
%         .resc - Structure with table for each weight

    % Create folder
    folder_name = sprintf('wave%d_clust%d', ch, cluster_num);
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
    weights = [1, 5, 10, 25, 50, 100, 200, 300, 500, 1000];
    
    % Compute distances for orig_spikes
    n_orig = size(orig_spikes, 1);
    norm_dist_orig = zeros(n_orig, length(weights));
    raw_dist_orig = zeros(n_orig, length(weights));
    win_cls_orig = zeros(n_orig, length(weights));
    
    % Create tables for all weights
    dist_tables_orig = cell(1, length(weights));
    table_weight = 300;  % Weight to display in console
    
    for w_idx = 1:length(weights)
        par.pk_weight = weights(w_idx);
        dist_table_w = zeros(n_orig, size(centers, 1) + 1);
        
        for i = 1:n_orig
            spike = orig_spikes(i, :);
            dist_w = zeros(1, size(centers, 1));
            for cls = 1:size(centers, 1)
                center = centers(cls, :);
                [normConst_w, w] = get_weight_vector(spike, par.pk_weight, par.amp_dir);
                dist_w(cls) = normConst_w * sqrt(sum(w .* (spike - center).^2));
            end
            
            % Store distances for this weight
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
    norm_dist_resc = zeros(n_resc, length(weights));
    raw_dist_resc = zeros(n_resc, length(weights));
    win_cls_resc = zeros(n_resc, length(weights));
    
    % Create tables for all weights
    dist_tables_resc = cell(1, length(weights));
    
    for w_idx = 1:length(weights)
        par.pk_weight = weights(w_idx);
        dist_table_w = zeros(n_resc, size(centers, 1) + 1);
        
        for i = 1:n_resc
            spike = resc_spikes(i, :);
            dist_w = zeros(1, size(centers, 1));
            for cls = 1:size(centers, 1)
                center = centers(cls, :);
                [normConst_w, w] = get_weight_vector(spike, par.pk_weight, par.amp_dir);
                dist_w(cls) = normConst_w * sqrt(sum(w .* (spike - center).^2));
            end
            
            % Store distances for this weight
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
    out.weights = weights;
    out.orig = struct();
    out.resc = struct();
    for w_idx = 1:length(weights)
        field_name = sprintf('W%d', weights(w_idx));
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
        % Linear fit to log(weights) vs normalized distance
        p = polyfit(log10(weights), norm_dist_orig(i, :), 1);
        slopes(i) = p(1);
        if slopes(i) > 0.05  % Threshold for positive trend
            pos_trend_good = pos_trend_good + 1;
            pos_idx_good = [pos_idx_good; i];
        elseif slopes(i) < -0.05  % Threshold for negative trend
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
        % Linear fit to log(weights) vs normalized distance
        p = polyfit(log10(weights), norm_dist_resc(i, :), 1);
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
    
    % Collect data for table
    % For neutral/negative
    nn_slopes = [];
    nn_widths = [];
    nn_types = {};
    for i = 1:length(neutral_neg_idx_good)
        spike_idx = neutral_neg_idx_good(i);
        spike = orig_spikes(spike_idx, :);
        [left_w, right_w] = get_peak_width(spike, par.amp_dir);
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
        [left_w, right_w] = get_peak_width(spike, par.amp_dir);
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
        [left_w, right_w] = get_peak_width(spike, par.amp_dir);
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
        [left_w, right_w] = get_peak_width(spike, par.amp_dir);
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
    
    % Figure for normalized distances
    figure;
    hold on;
    h1 = scatter(nan, nan, 'bo', 'filled'); % Dummy for legend - Good
    h2 = scatter(nan, nan, '^b', 'filled'); % Dummy for legend - Good unclassified
    h3 = scatter(nan, nan, 'ro', 'filled'); % Dummy for legend - Poor_Rescue
    h4 = scatter(nan, nan, '^r', 'filled'); % Dummy for legend - Poor_Rescue unclassified
    
    % Plot good spikes with lines connecting each spike's points
    for i = 1:n_orig
        % Draw line connecting all points for this spike
        plot(weights, norm_dist_orig(i, :), 'b-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        
        % Plot markers on top
        for w_idx = 1:length(weights)
            if win_cls_orig(i, w_idx) == 0
                scatter(weights(w_idx), norm_dist_orig(i, w_idx), '^b', 'filled', 'HandleVisibility', 'off');
            else
                scatter(weights(w_idx), norm_dist_orig(i, w_idx), 'bo', 'filled', 'HandleVisibility', 'off');
            end
        end
    end
    
    % Plot poor_rescue spikes with lines connecting each spike's points
    for i = 1:n_resc
        % Draw line connecting all points for this spike
        plot(weights, norm_dist_resc(i, :), 'r-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        
        % Plot markers on top
        for w_idx = 1:length(weights)
            if win_cls_resc(i, w_idx) == 0
                scatter(weights(w_idx), norm_dist_resc(i, w_idx), '^r', 'filled', 'HandleVisibility', 'off');
            else
                scatter(weights(w_idx), norm_dist_resc(i, w_idx), 'ro', 'filled', 'HandleVisibility', 'off');
            end
        end
    end
    set(gca, 'XScale', 'log');
    xticks(weights);
    xticklabels(string(weights));
    title(sprintf('Normalized Distances vs Weight - Cluster %d', cluster_num), 'Interpreter', 'none');
    xlabel('Weight');
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
        % Draw line connecting all points for this spike
        plot(weights, raw_dist_orig(i, :), 'b-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        
        % Plot markers on top
        for w_idx = 1:length(weights)
            if win_cls_orig(i, w_idx) == 0
                scatter(weights(w_idx), raw_dist_orig(i, w_idx), '^b', 'filled', 'HandleVisibility', 'off');
            else
                scatter(weights(w_idx), raw_dist_orig(i, w_idx), 'bo', 'filled', 'HandleVisibility', 'off');
            end
        end
    end
    
    % Plot poor_rescue spikes with lines connecting each spike's points
    for i = 1:n_resc
        % Draw line connecting all points for this spike
        plot(weights, raw_dist_resc(i, :), 'r-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        
        % Plot markers on top
        for w_idx = 1:length(weights)
            if win_cls_resc(i, w_idx) == 0
                scatter(weights(w_idx), raw_dist_resc(i, w_idx), '^r', 'filled', 'HandleVisibility', 'off');
            else
                scatter(weights(w_idx), raw_dist_resc(i, w_idx), 'ro', 'filled', 'HandleVisibility', 'off');
            end
        end
    end
    set(gca, 'XScale', 'log');
    xticks(weights);
    xticklabels(string(weights));
    title(sprintf('Raw Distances vs Weight - Cluster %d', cluster_num), 'Interpreter', 'none');
    xlabel('Weight');
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
    saveas(gcf, save_path_raw);
    close(gcf);
    fprintf('Raw distance plot saved to: %s\n', save_path_raw);
    
    % Calculate common y-axis limits for waveform plots
    all_waveforms = [orig_spikes(:); resc_spikes(:)];
    y_min = min(all_waveforms);
    y_max = max(all_waveforms);
    y_limits = [y_min, y_max];
    offset = (y_max - y_min) * 0.05;
    
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
                plot(orig_spikes(neutral_neg_idx_good(i), :), 'Color', colors(i, :), 'LineWidth', 1);
            end
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
                plot(resc_spikes(neutral_neg_idx_poor(i), :), 'Color', colors(i, :), 'LineWidth', 1);
            end
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
                [left_w, right_w] = get_peak_width(spike, par.amp_dir);
                if ~isnan(left_w)
                    % Plot dotted line at peak width boundaries
                    plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                    plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                end
            end
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
                [left_w, right_w] = get_peak_width(spike, par.amp_dir);
                if ~isnan(left_w)
                    % Plot dotted line at peak width boundaries
                    plot([left_w left_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                    plot([right_w right_w], y_limits, ':', 'Color', colors(i, :), 'LineWidth', 1);
                end
            end
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
            plot(orig_spikes(neutral_neg_idx_good(i), :), 'Color', colors(i, :), 'LineWidth', 1);
        end
    end
    % Plot poor_rescue spikes
    if ~isempty(neutral_neg_idx_poor)
        n_offset = length(neutral_neg_idx_good);
        n_spikes = length(neutral_neg_idx_poor);
        colors = lines(n_offset + n_spikes);
        for i = 1:n_spikes
            plot(resc_spikes(neutral_neg_idx_poor(i), :), 'Color', colors(n_offset + i, :), 'LineWidth', 1);
        end
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
            [left_w, right_w] = get_peak_width(spike, par.amp_dir);
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
            [left_w, right_w] = get_peak_width(spike, par.amp_dir);
            if ~isnan(left_w)
                plot([left_w left_w], y_limits, ':', 'Color', colors(n_offset + i, :), 'LineWidth', 1);
                plot([right_w right_w], y_limits, ':', 'Color', colors(n_offset + i, :), 'LineWidth', 1);
            end
        end
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
end

function [left_width, right_width] = get_peak_width(spike_x, amp_dir)
    % Extract peak width boundaries used in get_weight_vector
    if strcmp(amp_dir, 'neg')
        wav = -spike_x;
    else
        wav = spike_x;
    end
    [pks, locs, w, p] = findpeaks(wav);
    
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
    else
        left_width = NaN;
        right_width = NaN;
    end
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
