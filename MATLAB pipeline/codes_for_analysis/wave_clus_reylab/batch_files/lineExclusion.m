function [select_gauss_orig, select_gauss_1pct, select_gauss_2_5pct, select_gauss_5pct, select_gauss_10pct, select_all, x_vals, y_vals, gauss_ids, x_inter, y_inter] = lineExclusion(medDist_vec, medDist_select, k_select, kDist_vec, folderName, channelNum)
% LINEEXCLUSION (Center Trim: Trims based on Normalized Euclidean Distance to the Intersection Point)
% Builds Q2/Q4 threshold lines, iteratively trims based on distance to center point (intersection),
% and returns selected Gaussians while optionally saving two figures.
%
% MODIFIED: Now also returns plot-critical variables (x_vals, y_vals, etc.)
%           The call to plotSelectedVsAll has been REMOVED and must be called
%           from the main script AFTER spikeMatch.

if nargin < 5, folderName = ''; end 
if nargin < 6, channelNum = ''; end 
%plot_and_save = ~isempty(folderName) && ~isempty(channelNum); 
plot_and_save = false;
%% initialization
maxKv = max(kDist_vec(:,1));
minKv = min(kDist_vec(:,1));
maxMedDist = max(medDist_vec(:,1));
minMedDist = min(medDist_vec(:,1));
x_inter = medDist_select(end,1);
y_inter = k_select(end,1);
[commonPairs, idxMed, idxK] = intersect(medDist_vec(:,2:3), kDist_vec(:,2:3), 'rows');
x_vals = medDist_vec(idxMed, 1);
y_vals = kDist_vec(idxK, 1);
gauss_ids = commonPairs; 
coeffGauss = medDist_vec(idxMed,2:3); 
%% normalization 
x_min = min(x_vals);
x_max = max(x_vals);
y_min = min(y_vals);
y_max = max(y_vals);
x_range = x_max - x_min;
y_range = y_max - y_min;
x_vals_norm = (x_vals - x_min) / x_range;
y_vals_norm = (y_vals - y_min) / y_range;
% Normalize the center point coordinates
x_inter_norm = (x_inter - x_min) / x_range;
y_inter_norm = (y_inter - y_min) / y_range;
% --- NEW TRIMMING LOGIC: CALCULATE DISTANCE TO CENTER POINT ---
% Euclidean distance in the normalized [0, 1] space.
dist_to_center_norm = sqrt((x_vals_norm - x_inter_norm).^2 + (y_vals_norm - y_inter_norm).^2);
%% original boundary
% Original Boundaries (0% trim) - UN-NORMALIZED (Needed for the line itself)
slope_Q2 = (maxKv - y_inter) / (minMedDist - x_inter);
slope_Q4 = (minKv - y_inter) / (maxMedDist - x_inter);
int_Q2 = y_inter - slope_Q2 * x_inter;
int_Q4 = y_inter - slope_Q4 * x_inter;
% Perform Original Selection (0% trim, no masking applied yet)
y_line_Q2 = slope_Q2*x_vals + int_Q2;
y_line_Q4 = slope_Q4*x_vals + int_Q4;
in_Q2 = (x_vals <= x_inter) & (x_vals >= minMedDist);
in_Q4 = (x_vals >= x_inter) & (x_vals <= maxMedDist);
select_mask_orig = ((y_vals >= y_line_Q2) & in_Q2) | ((y_vals >= y_line_Q4) & in_Q4);
select_gauss_orig = [x_vals(select_mask_orig), y_vals(select_mask_orig), gauss_ids(select_mask_orig,:)];
select_all.original = struct('threshQ2',[slope_Q2,int_Q2],...
                             'threshQ4',[slope_Q4,int_Q4],...
                             'select_gauss',select_gauss_orig,...
                             'removed_points',0,...
                             'keep_mask', true(size(x_vals)));
                             
%% trimming by percentage for dist removal
trim_list = [1, 2.5, 5, 10];
colors = {[0 0 1],[0.5 0 0.5],[0 1 1],[0 0.5 0]}; % Kept your colors
field_names = {'trim1_0pct','trim2_5pct','trim5_0pct','trim10_0pct'};
select_sets = cell(1,numel(trim_list));
allSelectedGauss = [select_gauss_orig(:,3:4)]; 
for ii = 1:numel(trim_list)
    trimPct = trim_list(ii);
    
    % Remove points with the largest distance to the center.
    cutoff_norm = prctile(dist_to_center_norm, 100 - trimPct);
    keep_mask = dist_to_center_norm <= cutoff_norm;
    
    num_total_points = length(x_vals);
    num_kept = sum(keep_mask);
    num_removed = num_total_points - num_kept;
    % ---------------------------------
    x_keep = x_vals(keep_mask); 
    y_keep = y_vals(keep_mask);
    
    % Handle edge case where trimming fails
    if isempty(x_keep) || isempty(y_keep) || (min(x_keep) == x_inter) || (max(x_keep) == x_inter)
        if ii == 1
            prev_struct = select_all.original;
        else
            prev_struct = select_all.(field_names{ii-1});
        end
        slope_Q2r = prev_struct.threshQ2(1); int_Q2r = prev_struct.threshQ2(2);
        slope_Q4r = prev_struct.threshQ4(1); int_Q4r = prev_struct.threshQ4(2);
    else
        % Recalculate boundary based on the tightest remaining points
        slope_Q2r = (max(y_keep) - y_inter) / (min(x_keep) - x_inter);
        slope_Q4r = (min(y_keep) - y_inter) / (max(x_keep) - x_inter);
        int_Q2r = y_inter - slope_Q2r*x_inter;
        int_Q4r = y_inter - slope_Q4r*x_inter;
    end
    
    y_line_Q2r = slope_Q2r*x_vals + int_Q2r;
    y_line_Q4r = slope_Q4r*x_vals + int_Q4r;
    in_Q2r = (x_vals <= x_inter) & (x_vals >= minMedDist); 
    in_Q4r = (x_vals >= x_inter) & (x_vals <= maxMedDist);
    select_mask_r = ((y_vals >= y_line_Q2r) & in_Q2r) | ((y_vals >= y_line_Q4r) & in_Q4r);
    select_gauss_refined = [x_vals(select_mask_r), y_vals(select_mask_r), gauss_ids(select_mask_r,:)];
    
    select_all.(field_names{ii}) = struct('threshQ2',[slope_Q2r,int_Q2r],...
                                          'threshQ4',[slope_Q4r,int_Q4r],...
                                          'select_gauss',select_gauss_refined,...
                                          'removed_points',num_removed,...
                                          'keep_mask', keep_mask); % Store the mask that *generated* this line
    select_sets{ii} = select_gauss_refined;
    allSelectedGauss = [allSelectedGauss; select_gauss_refined(:,3:4)];
end
% Assign individual outputs
select_gauss_1pct   = select_sets{1};
select_gauss_2_5pct = select_sets{2};
select_gauss_5pct   = select_sets{3};
select_gauss_10pct  = select_sets{4};
%% optional plots
if plot_and_save
    
    if isnumeric(channelNum), channelNumStr = num2str(channelNum); else channelNumStr = channelNum; end
    if ~exist(folderName,'dir'), mkdir(folderName); end
    
    % FIG1: Boundary & Selection Plot (Detailed Labeling)
    figure('Visible','on'); hold on; grid on; box on;
    h_legend = []; % Initialize legend handle array
    
    % Plot ALL points (gray background) - this should always be first
    h_all = scatter(x_vals, y_vals, 25, [0.7 0.7 0.7], 'filled', 'DisplayName', 'All Points'); 
    h_legend(end+1) = h_all; % Add to legend
    
    % Add coeff# labels for all points (on top of all points)
    for i = 1:length(x_vals)
        pointID = sprintf('C%d,G%d',coeffGauss(i,1),coeffGauss(i,2));
        text(x_vals(i), y_vals(i)+0.7, pointID, 'HorizontalAlignment', ...
            'center', 'FontSize',7, 'FontWeight','bold', 'Color','k');
    end
    
    h_inter = plot(x_inter, y_inter, 'ko', 'MarkerFaceColor','y', 'MarkerSize',8, 'DisplayName','Intersection');
    h_legend(end+1) = h_inter; 
    
    % This ensures the most restrictive selections are drawn last and are visible.
    trim_list_rev = fliplr(trim_list); % [10, 5, 2.5, 1]
    field_names_rev = fliplr(field_names); % corresponding field names
    colors_rev = fliplr(colors); % corresponding colors
    boundaryDisplayNames = {'1%','2.5%','5%','10%'}; % These labels should match the original order
    boundaryDisplayNames_rev = fliplr(boundaryDisplayNames);
    
    for ii = 1:numel(trim_list_rev)
        % Get the current trimming level's data
        current_field_name = field_names_rev{ii};
        trim_struct = select_all.(current_field_name);
        color = colors_rev{ii}; % Use the reversed color array
    
        % Get the points that were actually used to calculate this boundary
        keep_mask_ii = trim_struct.keep_mask;
        x_keep = x_vals(keep_mask_ii);
        y_keep = y_vals(keep_mask_ii);
        
        if ~isempty(x_keep)
            % Get the generator points for this specific boundary
            x_lim_Q2 = min(x_keep); 
            y_lim_Q2 = max(y_keep); 
            
            x_lim_Q4 = max(x_keep);
            y_lim_Q4 = min(y_keep); 
            
            % Plot Q2 boundary segment and capture its handle
            h_trim_line = plot([x_lim_Q2, x_inter], [y_lim_Q2, y_inter], '-', 'Color', color, 'LineWidth',1.5, 'DisplayName', sprintf('%s Trimmed Boundary', boundaryDisplayNames_rev{ii}));
            
            % Add the handle to the legend array
            h_legend(end+1) = h_trim_line;
            
            % Plot Q4 boundary segment (no legend entry needed)
            plot([x_inter, x_lim_Q4], [y_inter, y_lim_Q4], '-', 'Color', color, 'LineWidth',1.5, 'HandleVisibility','off');
        end
        
        % Plot the selected points for THIS trimmed boundary (on top)
        scatter(trim_struct.select_gauss(:,1), trim_struct.select_gauss(:,2), 50, color, 'filled', 'HandleVisibility','off');
    end
    
    % Plot Original Boundary (0%) and selected points (Black) - should be drawn last
    % Get the generator points for the *original* boundary
    x_lim_Q2_orig = minMedDist;
    y_lim_Q2_orig = maxKv;
    x_lim_Q4_orig = maxMedDist;
    y_lim_Q4_orig = minKv;
    
    % Plot Q2 segment
    h_orig_q2 = plot([x_lim_Q2_orig, x_inter], [y_lim_Q2_orig, y_inter], '-', 'Color', [0 0 0], 'LineWidth',1.5, 'DisplayName', 'Original Boundary');
    % Plot Q4 segment
    plot([x_inter, x_lim_Q4_orig], [y_inter, y_lim_Q4_orig], '-', 'Color', [0 0 0], 'LineWidth',1.5, 'HandleVisibility','off');
    
    h_legend(end+1) = h_orig_q2; % Add to legend
    
    % Plot original selected points (these should be on top of all other selected points)
    selected_orig_pts = select_all.original.select_gauss;
    scatter(selected_orig_pts(:,1), selected_orig_pts(:,2), 50, [0 0 0], 'filled', 'HandleVisibility','off');
    
    % Plot remaining points (Same Coeff/Unrelated) - these should be on top of everything
    allSelectedGauss = unique(allSelectedGauss, 'rows'); % Ensure unique list for comparison
    
    h_same_tmp = []; h_unrelated_tmp = [];
    for k = 1:length(x_vals)
        % Check if this gaussian ID is in any of the *selected* sets (including original and trimmed)
        is_selected = ismember(gauss_ids(k,:), allSelectedGauss, 'rows');
        
        % If it's not selected, then it's either "Same Coeff" or "Unrelated"
        if ~is_selected
            coeff_num = coeffGauss(k,1);
            % Check if its coefficient is present in any selected gaussian's coefficient
            if any(ismember(allSelectedGauss(:,1), coeff_num))
                h_same_tmp = [h_same_tmp, scatter(x_vals(k), y_vals(k), 35, [0.5 0.5 1], 'filled', 'HandleVisibility','off')];
            else
                h_unrelated_tmp = [h_unrelated_tmp, scatter(x_vals(k), y_vals(k), 35, [1 0 0], 'x', 'LineWidth',1.5, 'HandleVisibility','off')];
            end
        end
    end
    
    % Only create legend entries if there are actual points to display
    if ~isempty(h_same_tmp)
        h_same = scatter(NaN, NaN, 35, [0.5 0.5 1], 'filled', 'DisplayName', 'Same coeff'); % Dummy plot for legend
        h_legend(end+1) = h_same;
    end
    if ~isempty(h_unrelated_tmp)
        h_unrelated = scatter(NaN, NaN, 35, [1 0 0], 'x', 'LineWidth',1.5, 'DisplayName', 'Unrelated'); % Dummy for legend
        h_legend(end+1) = h_unrelated;
    end
    
    legend(h_legend,'Location','bestoutside');
    
    % --- Add text under legend showing summary counts ---
    all_selected = select_all.original.select_gauss; % Summary based on original (as per previous logic)
    num_gauss = size(all_selected, 1);
    num_unique_coeff = numel(unique(all_selected(:,3)));
    
    % Create annotation textbox
    txt_summary = sprintf('Selected Gaussians: %d\nUnique Coefficients: %d', num_gauss, num_unique_coeff);
    annotation('textbox',[0.83, 0.15, 0.15, 0.07], 'String',txt_summary, ...
    'FitBoxToText','on', 'EdgeColor','none', 'FontSize',9, 'VerticalAlignment','top');
    
    xlabel('Median Distance');
    ylabel('K-value');
    title('Median Distance vs K-value with Sequential Boundaries');
    
    % Save figure 1
    filename1 = fullfile(folderName, sprintf('ch%s_medD_v_kv_boundaries.png', channelNumStr));
    exportgraphics(gcf, filename1, 'Resolution',300);
    close(gcf); % Close the current figure
    % FIG 2: Removal Plot (Shows points removed for trimming)
    figure('Visible','on'); hold on; grid on; box on;
    % Use the 10% mask to define removed points
    trim_struct_10 = select_all.trim10_0pct;
    removed_10pct_mask = ~trim_struct_10.keep_mask;
    
    % Plot all points that were kept (gray)
    h_kept = scatter(x_vals(~removed_10pct_mask), y_vals(~removed_10pct_mask), 18, [0.7 0.7 0.7], 'filled', 'DisplayName','Kept Points (10% Trim)');
    
    % Plot removed points (red 'x')
    h_removed = scatter(x_vals(removed_10pct_mask), y_vals(removed_10pct_mask), 30, [1 0 0], 'x', 'LineWidth', 1.5, 'DisplayName','Removed Points (10% Trim)');
    
    % Plot Boundaries (Original and Final 10% trim)
    
    % Get generator points for Original Boundary (already have from Fig 1)
    plot([x_lim_Q2_orig, x_inter], [y_lim_Q2_orig, y_inter], 'k--', 'LineWidth', 1.4, 'DisplayName', 'Original Boundary');
    plot([x_inter, x_lim_Q4_orig], [y_inter, y_lim_Q4_orig], 'k--', 'LineWidth', 1.4, 'HandleVisibility', 'off');
    
    % Get generator points for 10% Trim Boundary
    keep_mask_10 = trim_struct_10.keep_mask;
    x_keep_10 = x_vals(keep_mask_10);
    y_keep_10 = y_vals(keep_mask_10);
    
    if ~isempty(x_keep_10)
        x_lim_Q2_10 = min(x_keep_10); y_lim_Q2_10 = max(y_keep_10);
        x_lim_Q4_10 = max(x_keep_10); y_lim_Q4_10 = min(y_keep_10);
    
        plot([x_lim_Q2_10, x_inter], [y_lim_Q2_10, y_inter], 'g-', 'LineWidth', 1.5, 'DisplayName', '10% Trimmed Boundary');
        plot([x_inter, x_lim_Q4_10], [y_inter, y_lim_Q4_10], 'g-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    plot(x_inter, y_inter, 'ko', 'MarkerFaceColor','y', 'MarkerSize',8, 'DisplayName','Intersection','HandleVisibility','off');
    
    
    % Get the 10% cutoff distance (in normalized units)
    cutoff_10pct_norm = prctile(dist_to_center_norm, 90); 
    
    % To plot the circle in UN-NORMALIZED coordinates:
    theta = linspace(0, 2*pi, 100);
    x_circle_norm = x_inter_norm + cutoff_10pct_norm * cos(theta);
    y_circle_norm = y_inter_norm + cutoff_10pct_norm * sin(theta);
    
    x_circle = x_circle_norm * x_range + x_min;
    y_circle = y_circle_norm * y_range + y_min;
    
    h_circle = plot(x_circle, y_circle, 'b:', 'LineWidth', 1.5, 'DisplayName', '10% Trim Distance Circle');
    % ---------------------------------------------------------------------------------
    
    xlabel('Median Distance');
    ylabel('K-value');
    title(sprintf('Points Removed by 10%% Center Distance Cutoff (Channel %s)', channelNumStr));
    legend([h_kept, h_removed, h_circle],'Location','bestoutside'); % Added h_circle to legend
    
    % Save figure 2
    filename2 = fullfile(folderName, sprintf('ch%s_lineExclusion_removed_points.png', channelNumStr));
    exportgraphics(gcf, filename2, 'Resolution',300);
    close(gcf);

end
%% summary
    fprintf('\n=== LineExclusion Summary ===\n');
    fprintf('Original: %d selected\n', size(select_gauss_orig,1));
    fprintf('Trim 1%%: %d selected, %d removed\n', size(select_gauss_1pct,1), select_all.trim1_0pct.removed_points);
    fprintf('Trim 2.5%%: %d selected, %d removed\n', size(select_gauss_2_5pct,1), select_all.trim2_5pct.removed_points);
    fprintf('Trim 5%%: %d selected, %d removed\n', size(select_gauss_5pct,1), select_all.trim5_0pct.removed_points);
    fprintf('Trim 10%%: %d selected, %d removed\n', size(select_gauss_10pct,1), select_all.trim10_0pct.removed_points);
    fprintf('=============================\n\n');
end