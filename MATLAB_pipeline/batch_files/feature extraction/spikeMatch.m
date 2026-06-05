function select_spike_match = spikeMatch(select_all, threshQ2, threshQ4, g, coeffs, basename, min_coeff_count, corr_thresh)
    
    lSel = size(select_all,1);
    spike_idx = cell(lSel,1);
    
    for i = 1:lSel
        coeff_sel = select_all(i,3);
        gauss_select = select_all(i,4);
        mu = g{coeff_sel}.mu;
        sigma = sqrt(squeeze(g{coeff_sel}.Sigma(:)));
        mu_sel = mu(gauss_select);
        s_select = sigma(gauss_select);
        range_upp = mu_sel + 2*s_select;
        range_low = mu_sel - 2*s_select;
        % Find spike indices that fall within the 2-sigma range
        spike_idx{i} = find(coeffs(:,coeff_sel) >= range_low & ...
                            coeffs(:,coeff_sel) <= range_upp);
    end
    
    % build pairwise overlap matrix
    n = numel(spike_idx);
    overlapMat = zeros(n);
    maxSizes = zeros(n);
    minSizes = zeros(n);
     for i = 1:n
        Ai = spike_idx{i};
        for j = i+1:n
            Aj = spike_idx{j};
            if isempty(Ai) || isempty(Aj)
                overlapMat(i,j) = 0;
            else
                inter = numel(intersect(Ai, Aj));
                szAi = numel(Ai);
                szAj = numel(Aj);
                
                minSz = min(szAj,szAi);
                minSizes(i,j) = minSz;
                maxSizes(i,j) = max(szAj,szAi);
                % Overlap normalized by the size of the smaller set
                overlapMat(i,j) = inter / minSz;
            end
        end
    end
    
    % Calculate Normalized Minimum Distance to Boundary
    X = select_all(:,1);
    Y = select_all(:,2);
    
    x_min = min(X); x_max = max(X); x_range = x_max - x_min;
    y_min = min(Y); y_max = max(Y); y_range = y_max - y_min;
    
    if x_range == 0, x_range = 1; end
    if y_range == 0, y_range = 1; end
    
    X_norm = (X - x_min) / x_range;
    Y_norm = (Y - y_min) / y_range;
    
    % Normalize threshold lines
    m2_norm = threshQ2(1) * (x_range / y_range);
    b2_norm = (threshQ2(2) - y_min) / y_range;
    
    m4_norm = threshQ4(1) * (x_range / y_range);
    b4_norm = (threshQ4(2) - y_min) / y_range;
    
    % Calculate perpendicular distance to each boundary line
    dist_Q2 = abs(m2_norm * X_norm - Y_norm + b2_norm) ./ sqrt(m2_norm^2 + (-1)^2);
    dist_Q4 = abs(m4_norm * X_norm - Y_norm + b4_norm) ./ sqrt(m4_norm^2 + (-1)^2);
    
    x_inter = select_all(end,1);
    
    % Get IDs for labeling and correlation
    CoeffIDs = select_all(:, 3);
    GaussIDs = select_all(:, 4);
    % Ranking metric: minimum distance to either boundary line
    minDist_vec_original = min(dist_Q4,dist_Q2);
    
    % The following plotting block remains as in the original code (sorted ascending)
    minDist_vec = minDist_vec_original; 
    [minDist_sort, minIdx] = sort(minDist_vec);
    % dist_tot = figure;
    % hold on;
    % for k = 1:n
    %     idx = minIdx(k);
    %     label = sprintf('C%dG%d', CoeffIDs(idx), GaussIDs(idx));
    %     scatter(k, minDist_sort(k), 30, 'o', ...
    %             'filled', 'MarkerFaceAlpha', 0.5); 
    %     % text(k, minDist_sort(k) + 0.05, label, ...
    %     %      'FontSize', 7, 'HorizontalAlignment', 'center');
    % 
    % end
    % saveas(dist_tot, fullfile(folderSpikeMatch, ['totDistNoLbl_' basename '.png']), 'png');
    maxDistances = zeros(n);
    minDistances = zeros(n);
   % close(dist_tot);
    % Use the correct vector for pairwise distance comparisons
    for i = 1:n
        for j = i+1:n
            Di = minDist_vec_original(i); % Fix 1: Use the original distances
            Dj = minDist_vec_original(j); % Fix 1: Use the original distances
            
            minDistances(i,j) = min(Di, Dj);
            maxDistances(i,j) = max(Di, Dj);
        end
    end
    
    % Make overlapMat fully symmetric
    overlapMatFull = overlapMat + overlapMat' + eye(n);
    
    % Calculate the correlation between rows of the overlap matrix
    overlapCorrMat = calculateOverlapRowCorrelation(overlapMatFull, GaussIDs);
    
    % Define the threshold for filtering (used for plotting size/distance plots)
    threshold = 0.01;
    
    % PLOTTING DATA PREPARATION
    
    % Get the indices of the calculated upper triangle pairs
    upper_tri_indices = triu(ones(n), 1) == 1;
    
    % Extract the overlap values only for the calculated pairs
    overlap_vec = overlapMat(upper_tri_indices);
    
    % Identify the indices of pairs that meet the overlap threshold
    meets_threshold_indices = overlap_vec >= threshold;
    
    % Filter all data vectors based on the threshold
    X_minSizes = minSizes(upper_tri_indices);
    Y_maxSizes = maxSizes(upper_tri_indices);
    X_minDist = minDistances(upper_tri_indices);
    Y_maxDist = maxDistances(upper_tri_indices);
    
    X_minSizes_F = X_minSizes(meets_threshold_indices);
    Y_maxSizes_F = Y_maxSizes(meets_threshold_indices);
    X_minDist_F = X_minDist(meets_threshold_indices);
    Y_maxDist_F = Y_maxDist(meets_threshold_indices);
    % Filter the pair indices for labeling
    all_pair_indices = zeros(sum(upper_tri_indices(:)), 2); 
    k = 1;
    for i = 1:n
        for j = i+1:n
            all_pair_indices(k,:) = [i, j];
            k = k + 1;
        end
    end
    pair_indices_F = all_pair_indices(meets_threshold_indices, :);
    % Get the new total number of pairs to plot
    n_pairs_F = size(pair_indices_F, 1);
    
    % Generate colors based on the new, smaller number of pairs
    
    all_hues = hsv(n_pairs_F);
    shuffled_order = randperm(n_pairs_F);
    pair_colors_F = all_hues(shuffled_order, :);
    
    % Create combined labels C[coeff]G[gauss] for the heatmap axes
    
    heatmap_labels = cell(n, 1);
    for idx = 1:n
        % Format: C[Coefficient ID]G[Gaussian ID]
        heatmap_labels{idx} = sprintf('C%dG%d', CoeffIDs(idx), GaussIDs(idx));
    end
    % visualize the overlap
    % hmap = figure;
    % heatmap(overlapMat, 'ColorLimits', [0, 1], 'Title', sprintf('Pairwise Spike Overlap: %s', basename), ...
    %          'XLabel', 'Coefficient (C) and Gaussian (G) ID', ... 
    %          'YLabel', 'Coefficient (C) and Gaussian (G) ID', ... 
    %          'XDisplayLabels', heatmap_labels, ... 
    %          'YDisplayLabels', heatmap_labels); 
    % saveas(hmap, fullfile(folderSpikeMatch, ['overlap_' basename '.png']), 'png');
    % close(hmap);
    % visualize the overlap row correlation
    % hCorr = figure;
    % %overlapMatFull = overlapMatFull + overlapMatFull' + eye(n);
    % 
    % heatmap(overlapCorrMat, 'ColorLimits', [-1, 1], 'Title', sprintf('Overlap Row Correlation (Skipping Zeros/Same Gaussians): %s', basename), ...
    %          'XLabel', 'Coefficient (C) and Gaussian (G) ID', ...
    %          'YLabel', 'Coefficient (C) and Gaussian (G) ID', ...
    %          'XDisplayLabels', heatmap_labels, ...
    %          'YDisplayLabels', heatmap_labels);
    % saveas(hCorr, fullfile(folderSpikeMatch, ['correlation_overlap' basename '.png']), 'png');
    % close(hCorr);
    
    % max and min spike count plot
   %  sizePlt = figure('Name', 'Spike Count Plot');
   %  ax_size = axes(sizePlt);
   %  hold(ax_size, 'on');
   % 
   %  % Set limits and plot diagonal starting from 0,0
   %  max_size = max([X_minSizes_F; Y_maxSizes_F]);
   %  lim_size = max_size * 1.1; % 10% buffer
   %  xlim(ax_size, [0, lim_size]);
   %  ylim(ax_size, [0, lim_size]);
   %  plot(ax_size, [0, lim_size], [0, lim_size], 'k--', 'LineWidth', 1.5);
   % 
   %  for p = 1:n_pairs_F
   %      i_idx = pair_indices_F(p, 1);
   %      j_idx = pair_indices_F(p, 2);
   %      label = sprintf('C%dG%d, C%dG%d', CoeffIDs(i_idx), GaussIDs(i_idx), CoeffIDs(j_idx), GaussIDs(j_idx));
   %     % label_ind = sprintf('i: %d, j: %d',i_idx,j_idx);
   %      % Scatter plot: Correct syntax uses 30 SizeData and color as positional args
   %      scatter(ax_size, X_minSizes_F(p), Y_maxSizes_F(p), 30, pair_colors_F(p,:), 'o', ...
   %              'filled', 'MarkerFaceAlpha', 0.5); 
   %      text(ax_size, X_minSizes_F(p), Y_maxSizes_F(p) + 0.03*lim_size, label, ...
   %           'FontSize', 7, 'Color', pair_colors_F(p,:), 'HorizontalAlignment', 'center');
   %  end
   % 
   %  hold(ax_size, 'off');
   % 
   %  xlabel(ax_size, 'Minimum Spike Count minSz');
   %  ylabel(ax_size, 'Maximum Spike Count maxSz');
   %  title(ax_size, sprintf('Comparison of Spike Set Sizes for Overlapping Pairs: %s', basename));
   %  saveas(sizePlt, fullfile(folderSpikeMatch, ['size_ind' basename '.png']), 'png');
   % % close(sizePlt);
   % 
   %  % max and min boundary distance plot
   % distPlt = figure('Name', 'Boundary Distance Plot');
   %  ax_dist = axes(distPlt);
   %  hold(ax_dist, 'on');
   % % Set limits and plot diagonal starting from 0,0
   %  max_dist = max([X_minDist_F; Y_maxDist_F]);
   %  lim_dist = max_dist * 1.1; % 10% buffer
   %  xlim(ax_dist, [0, lim_dist]);
   %  ylim(ax_dist, [0, lim_dist]);
   %  plot(ax_dist, [0, lim_dist], [0, lim_dist], 'k--', 'LineWidth', 1.5);
   % 
   %  for p = 1:n_pairs_F
   %      i_idx = pair_indices_F(p, 1);
   %      j_idx = pair_indices_F(p, 2);
   %      label = sprintf('C%dG%d, C%dG%d', CoeffIDs(i_idx), GaussIDs(i_idx), CoeffIDs(j_idx), GaussIDs(j_idx));
   % 
   %      % Scatter plot: Correct syntax uses 30 SizeData and color as positional args
   %      scatter(ax_dist, X_minDist_F(p), Y_maxDist_F(p), 30, pair_colors_F(p,:), 'o', ...
   %              'filled', 'MarkerFaceAlpha', 0.5); 
   % 
   %      text(ax_dist, X_minDist_F(p), Y_maxDist_F(p) + 0.03*lim_dist, label, ...
   %           'FontSize', 7, 'Color', pair_colors_F(p,:), 'HorizontalAlignment', 'center');
   %  end
   % 
   %  hold(ax_dist, 'off');
   % 
   %  xlabel(ax_dist, 'Minimum Normalized Distance to Boundary');
   %  ylabel(ax_dist, 'Maximum Normalized Distance to Boundary');
   %  title(ax_dist, sprintf('Pairwise Distance to Selection Boundary: %s', basename));
   %  saveas(distPlt, fullfile(folderSpikeMatch, ['dist_' basename '.png']), 'png'); 
   %  close(distPlt);
 
    select_spike_match = filterByDistanceAndCorrelation(select_all, overlapCorrMat, minDist_vec_original, min_coeff_count, corr_thresh, threshQ2, threshQ4);
    
    % Old logic preserved below, commented out
    % find groups of highly similar coefficients
    % isSimilar = overlapMat >= threshold;
    % G = digraph(isSimilar);
    % groups = conncomp(G);
    % 
    % % keep 1 representative per group
    % numGroups = max(groups);
    % reps = zeros(numGroups,1);
    % for gnum = 1:numGroups
    %     idx = find(groups == gnum);
    %     lens = cellfun(@numel, spike_idx(idx));
    %     [~, bestIdx] = max(lens);
    %     reps(gnum) = idx(bestIdx);
    % end
    % select_spike_match = select_all(reps,:);
    
    numOriginal = numel(unique(select_all(:,3)));
    numReduced  = numel(unique(select_spike_match(:,3)));
    fprintf('Reduced from %d to %d coefficients using distance rank and correlation criteria\n', numOriginal, numReduced);
end

% Helper function to perform selection based on distance ranking, correlation threshold, and unique coefficient count
function selected_set = filterByDistanceAndCorrelation(select_all, overlapCorrMat, rank_metric, min_coeff_count, corr_thresh, threshQ2, threshQ4)
    
    % Sort points by distance ranking metric (descending)
    [~, sortIdx] = sort(rank_metric, 'descend');
    
    % Reorder inputs based on ranking
    ranked_select_all = select_all(sortIdx, :);
    ranked_corr_mat = overlapCorrMat(sortIdx, sortIdx);
    
    n = size(ranked_select_all, 1);
    selected_indices = false(n, 1);
    
    % Select the highest ranked point
    if n > 0
        selected_indices(1) = true;
    end
    
    % Iterative correlation filtering
    for i = 2:n
        current_selection_rank_indices = find(selected_indices);
        correlations_to_selected = ranked_corr_mat(i, current_selection_rank_indices);
        
        % Check if max absolute correlation is below threshold
        if max(abs(correlations_to_selected)) < corr_thresh
            selected_indices(i) = true;
        end
    end
    
    % Separate filtered sets
    correlation_filtered_set = ranked_select_all(selected_indices, :);
    unselected_set = ranked_select_all(~selected_indices, :);
    selected_set = correlation_filtered_set;
    
    % Initialize backfilled tracking array with correct column dimension
    backfilled_rows_to_track = zeros(0, size(ranked_select_all, 2)); % Fix 2: Initialize with correct columns
    
    unique_coeffs = unique(selected_set(:, 3));
    
    % Backfilling: Add non-selected points (from highest rank) until minimum unique coefficient count is achieved
    k = 1; 
    while numel(unique_coeffs) < min_coeff_count && k <= size(unselected_set, 1)
        
        candidate_row = unselected_set(k, :);
        candidate_coeff_id = candidate_row(3);
        
        % Only add if it introduces a new coefficient ID
        if ~ismember(candidate_coeff_id, unique_coeffs)
            selected_set = [selected_set; candidate_row]; 
            unique_coeffs = unique(selected_set(:, 3));
            backfilled_rows_to_track = [backfilled_rows_to_track; candidate_row];
        end
        k = k + 1;
    end
    
    % Prepare classification map for plotting:
    % 1: Core Selected (Green Circle)
    % 2: Backfilled (Green X)
    % 3: Rejected (Black X)
    
    [~, core_sel_idx] = ismember(correlation_filtered_set, select_all, 'rows');
    core_sel_idx = core_sel_idx(core_sel_idx~=0); 

    [~, backfilled_idx] = ismember(backfilled_rows_to_track, select_all, 'rows');
    backfilled_idx = backfilled_idx(backfilled_idx~=0); 

    classification = 3 * ones(size(select_all, 1), 1);
    classification(core_sel_idx) = 1; 
    classification(backfilled_idx) = 2;

    %plot_selection_results(select_all, classification, threshQ2, threshQ4, folderSpikeMatch, basename);
    
end


% Local helper function to plot the selection process and boundaries
function plot_selection_results(select_all, classification, threshQ2, threshQ4, folderSpikeMatch, basename)

    % Get the raw coordinates
    X = select_all(:, 1);
    Y = select_all(:, 2);
    
    h_select = figure('Name', 'Final Selection Plot');
    ax = axes(h_select);
    hold(ax, 'on');
    
    % Define X-range for plotting the lines
    x_min = min(X); x_max = max(X);
    x_plot = linspace(x_min, x_max, 100);
    
    % Plot Boundary Q2
    y_plot_Q2 = threshQ2(1) * x_plot + threshQ2(2);
    plot(ax, x_plot, y_plot_Q2, 'r--', 'LineWidth', 1, 'DisplayName', 'Threshold Q2');
    
    % Plot Boundary Q4
    y_plot_Q4 = threshQ4(1) * x_plot + threshQ4(2);
    plot(ax, x_plot, y_plot_Q4, 'b--', 'LineWidth', 1, 'DisplayName', 'Threshold Q4');

    m2 = threshQ2(1);
    b2 = threshQ2(2);
    m4 = threshQ4(1);
    b4 = threshQ4(2);
    
    % Check if slopes are significantly different to avoid division by zero
    if abs(m2 - m4) > 1e-6
        % Calculate x-coordinate of intersection
        x_i = (b4 - b2) / (m2 - m4);
        
        % Calculate y-coordinate of intersection using the Q2 equation
        y_i = m2 * x_i + b2;
        
        % Plot the intersection point (Yellow filled circle with black edge)
        scatter(ax, x_i, y_i, 100, 'y', 'o', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Intersection');
    else
        % Slopes are equal (lines are parallel or coincident).
        warning('Threshold lines are parallel or coincident (slopes are equal). Intersection point not calculated or plotted.');
    end
    idx_rejected = (classification == 3);
    scatter(ax, X(idx_rejected), Y(idx_rejected), 40, 'k', 'x', ...
            'LineWidth', 1.5, 'DisplayName', 'Rejected (High Correlation)');
        
    idx_backfilled = (classification == 2);
    scatter(ax, X(idx_backfilled), Y(idx_backfilled), 60, 'g', 'x', ...
            'LineWidth', 2, 'DisplayName', 'Backfilled (Unique Coeff)');

    idx_selected = (classification == 1);
    scatter(ax, X(idx_selected), Y(idx_selected), 70, 'g', 'o', ...
            'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'g', 'LineWidth', 2, ...
            'DisplayName', 'Core Selected (Low Correlation)');
        
    % Finalize Plot
    xlabel(ax, 'med dist');
    ylabel(ax, 'k distance');
    title(ax, sprintf('Final Selection Results: %s', basename));
    legend(ax, 'Location', 'best');
    grid(ax, 'on');
    hold(ax, 'off');
    
    % Save the figure
    saveas(h_select, fullfile(folderSpikeMatch, ['final_selection_' basename '.png']), 'png');
    close(h_select)
end

