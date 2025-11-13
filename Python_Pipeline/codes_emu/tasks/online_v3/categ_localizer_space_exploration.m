function categ_localizer_space_exploration(pics_path)
    xls_file = dir(fullfile(pics_path,'*.xlsx'));
    xls_path = fullfile(pics_path, xls_file.name);
    max_stim_cat_cols = 5;
    CUSTOM_PICS_PER_CAT = 18;
    not_enough_count = 0;
    tot_spaces_count = 0;
    % Set a correlation threshold - combinations with correlations above this will be excluded
    MI_THRESHOLD = 0.5; 
    CRAMERS_V_THRESHOLD = 0.4;
    custom_pics_path = [pics_path filesep 'custom_pics'];
    img_info_table = readtable(xls_path, 'Sheet', 'AllCat');
    uniq_stim_cats = unique(img_info_table.StimCat);
    sort_by_coverage_only = true;
    task_other_only = false;
    task_recdec_only = false;
    best_only = true;
    plot_stat = true;
    b_plot_confusion_matrices = true;
    % th_stat = 'plateau';
    th_stat = 'quant_95';
    extra_txt = '';
    if task_recdec_only
        extra_txt = '_recdec';
    elseif task_other_only
        extra_txt = '_other';
    end


    % Create a folder to store the figures
    folder_name = 'spaces_dist_figures';
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    recdec_cats = {'Adult', 'Car', 'House', 'Faces'};
    for i_stim_cat = 1:numel(uniq_stim_cats)        
        stim_category = uniq_stim_cats(i_stim_cat);
        stim_category = stim_category{1};
        if strcmp(stim_category, 'Adult') || ...
            strcmp(stim_category, 'Child') || ...
            ~strcmp(stim_category, 'Animal')
            continue
        end
        if ~ismember(stim_category, recdec_cats) && (task_other_only)
            continue;
        end
        if ~ismember(stim_category, recdec_cats) && (task_recdec_only)
            continue;
        end
        stim_cat_cols = img_info_table.Properties.VariableNames( ...
                startsWith(img_info_table.Properties.VariableNames, stim_category));
        all_column_names = strrep(stim_cat_cols, '_', '-');
        all_column_names = strrep(all_column_names, [stim_category '-'], '');
        
        % Get the data for correlation analysis
        stim_cat_data = img_info_table(ismember(img_info_table.StimCat, stim_category),:);
        if task_other_only
            stim_cat_data = stim_cat_data(ismember(stim_cat_data.TaskGroup, 'Other'), :);
        elseif task_recdec_only
            stim_cat_data = stim_cat_data(ismember(stim_cat_data.TaskGroup, 'RecDec'), :);
        end
        stim_cat_data = stim_cat_data(:, stim_cat_cols);
        
        % if strcmp(stim_category, "Adult")
        %     non_uniform_feats = [];%[5,6];
        % else
        %     non_uniform_feats = [];
        % end
        non_uniform_feats = [];
        non_uniform_feats_cv = [];
        
        % histogram to see the distribution of the data
        feat_hist_fig = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
        t = tiledlayout(2, ceil(numel(stim_cat_cols)/2));%, 'TileSpacing', 'compact', 'Padding', 'compact');
        for i = 1:numel(stim_cat_cols)
            nexttile;
            % subplot(2, ceil(numel(stim_cat_cols)/2), i);
            % histogram(categorical(stim_cat_data.(stim_cat_cols{i})),"Normalization","probability");
            % title(all_column_names{i});
            [b_not_uniform, cv] = analyze_feature_distribution(categorical(stim_cat_data.(stim_cat_cols{i})), all_column_names{i});
            if b_not_uniform
                non_uniform_feats = [non_uniform_feats, i];
                non_uniform_feats_cv = [non_uniform_feats_cv, cv];
            end
        end
        saveas(feat_hist_fig, fullfile(folder_name, [stim_category sprintf('%s_histograms.png',extra_txt)]));
        
        if best_only
            if strcmp(stim_category, 'Faces')
                non_uniform_feats = [];
            elseif strcmp(stim_category, 'Car')
                non_uniform_feats = [2, 10, 11];
            elseif strcmp(stim_category, 'House')
                non_uniform_feats = [10,13,15,16,17];
            elseif strcmp(stim_category, 'Instrument')
                non_uniform_feats = [];
            elseif strcmp(stim_category, 'Limb')
                non_uniform_feats = [];
            elseif strcmp(stim_category, 'Animal')
                non_uniform_feats = [1,7,10,14,15,16,17,18,20,21]; % remove wings, groundHome 
            end
        end
        % Remove non-uniform features from the analysis
        if ~isempty(non_uniform_feats)
            available_cols_count = numel(all_column_names) - numel(non_uniform_feats);
            if available_cols_count < 5
                [~, non_uniform_feats_sorted_indices] = sort(non_uniform_feats_cv, 'descend');
                % non_uniform_feats = non_uniform_feats(non_uniform_feats_sorted_indices);
                % non_uniform_feats = non_uniform_feats(1:(5-available_cols_count));
            end
            fprintf('Removing non-uniform features from %s: ', stim_category);
            for i = 1:numel(non_uniform_feats)
                fprintf('%s ', all_column_names{non_uniform_feats(i)});
            end
            fprintf('\n');
            stim_cat_cols(non_uniform_feats) = [];
            all_column_names(non_uniform_feats) = [];
        end
        

        % Calculate correlation matrix between features more efficiently
        corr_matrix = zeros(numel(stim_cat_cols));
        cramers_v = zeros(numel(stim_cat_cols));
        mutual_info = zeros(numel(stim_cat_cols));
        
        % Create a figure for confusion matrices at the start
        n_pairs = (numel(stim_cat_cols) * (numel(stim_cat_cols)-1)) / 2;
        % conf_fig = figure('Visible', 'off', 'Name', ['Confusion Matrices - ' stim_category]);
        n_rows = ceil(sqrt(n_pairs));
        n_cols = ceil(n_pairs/n_rows);
        % Base size for a single subplot in pixels
        subplotWidth = 300; 
        subplotHeight = 300;
        
        % Add margins and adjust for labels
        marginWidth = 100;
        marginHeight = 100;
        
        % Calculate figure size
        figWidth = n_cols * subplotWidth + marginWidth;
        figHeight = n_rows * subplotHeight + marginHeight;
        
        if b_plot_confusion_matrices
            % Create figure
            conf_fig = figure('Position', [100, 100, figWidth, figHeight],'Visible', 'off', 'Units', 'pixels');
            t = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
            subplot_idx = 1;
        end
        glob_min = 0;
        glob_max = 0;
        for i = 1:numel(stim_cat_cols)
            for j = i:numel(stim_cat_cols) % Only calculate upper triangle
                % Convert categorical variables to numeric if needed
                col_i = stim_cat_data.(stim_cat_cols{i});
                col_j = stim_cat_data.(stim_cat_cols{j});
        
                % Calculate Cramer's V for categorical variables
                cramers_v(i, j) = calc_cramers_v(col_i, col_j);
                cramers_v(j, i) = cramers_v(i, j); % Mirror value
        
                % Calculate mutual information
                mutual_info(i, j) = calc_mutual_information(col_i, col_j);
                mutual_info(j, i) = mutual_info(i, j); % Mirror value
                
                if iscategorical(col_i) || iscell(col_i)
                    col_i_cat = categorical(col_i);
                    col_i_cat = double(col_i_cat);
                end
                
                if iscategorical(col_j) || iscell(col_j)
                    col_j_cat = categorical(col_j);
                    col_j_cat = double(col_j_cat);
                end
                
                % Calculate correlation
                corr_matrix(i, j) = abs(corr(col_i_cat, col_j_cat, 'Type','Kendall', 'rows', 'complete'));
                corr_matrix(j, i) = corr_matrix(i, j); % Mirror value

                % if i ~= j && b_plot_confusion_matrices
                %     plot_confusion_matrices(subplot_idx, i,j, col_i, col_j, ...
                %                             all_column_names, ...
                %                             corr_matrix, cramers_v, mutual_info)
                %     subplot_idx = subplot_idx + 1;
                % end
                confusion_mat = crosstab(categorical(col_i), categorical(col_j));
                if glob_min == 0 && glob_max == 0
                    glob_min = min(confusion_mat(:));
                    glob_max = max(confusion_mat(:));
                else 
                    if glob_min > min(confusion_mat(:))
                        glob_min = min(confusion_mat(:));
                    end
                    if glob_max < max(confusion_mat(:))
                        glob_max = max(confusion_mat(:));
                    end
                end
            end
        end

        for i = 1:numel(stim_cat_cols)
            for j = i:numel(stim_cat_cols) % Only calculate upper triangle
                % Convert categorical variables to numeric if needed
                col_i = stim_cat_data.(stim_cat_cols{i});
                col_j = stim_cat_data.(stim_cat_cols{j});

                if i ~= j && b_plot_confusion_matrices
                    plot_confusion_matrices(subplot_idx, i,j, col_i, col_j, ...
                                            all_column_names, ...
                                            corr_matrix, cramers_v, mutual_info, glob_min, glob_max)
                    subplot_idx = subplot_idx + 1;
                end
            end
        end
        
        if b_plot_confusion_matrices
            % Adjust figure layout and save
            sgtitle(['Confusion Matrices for ' stim_category], 'FontSize', 14);
            % set(conf_fig, 'Position', [100 100 1200 800]);
            saveas(conf_fig, fullfile(folder_name, [stim_category sprintf('%s_confusion_matrices.png',extra_txt)]));
            
            close(conf_fig);
        end
        
        % Display correlation matrix
        % fprintf('Correlation matrix for %s features:\n', stim_category);
        % disp(array2table(corr_matrix, 'VariableNames', all_column_names, 'RowNames', all_column_names));
        % fprintf('Mutual info for %s features:\n', stim_category);
        % disp(array2table(mutual_info, 'VariableNames', all_column_names, 'RowNames', all_column_names));
        
        MI_THRESHOLD = estimate_threshold(stim_category, mutual_info, th_stat, extra_txt);
        % MI_THRESHOLD = estimate_threshold(stim_category, mutual_info, 'quant_90');

        % print highly correlated features and their correlation values, consider that correlation is symmetric
        fprintf('%s, Highly correlated features (MI >= %.2f):\n', stim_category, MI_THRESHOLD);
        for i = 1:numel(stim_cat_cols)
            for j = i+1:numel(stim_cat_cols)
                if mutual_info(i, j) >= MI_THRESHOLD
                    fprintf('%s and %s. MI: %.2f\n', all_column_names{i}, all_column_names{j}, mutual_info(i, j));
                end
                % if cramers_v(i, j) > CRAMERS_V_THRESHOLD
                %     fprintf('%s and %s. cramers_v: %.2f\n', stim_cat_cols{i}, stim_cat_cols{j}, cramers_v(i, j));
                % end
            end
        end
        
        if numel(stim_cat_cols) >= max_stim_cat_cols
            all_combs = nchoosek(1:numel(stim_cat_cols), max_stim_cat_cols);
            % Remove combinations with highly correlated features
            valid_combinations = true(size(all_combs, 1), 1);
            for i = 1:size(all_combs, 1)
                combo = all_combs(i, :);
                % Check all pairs within this combination
                for j = 1:length(combo)
                    for k = j+1:length(combo)
                        if mutual_info(combo(j), combo(k)) >= MI_THRESHOLD% || ...
                           %ismember(combo(j), non_uniform_feats) || ismember(combo(k), non_uniform_feats)
                            valid_combinations(i) = false;
                            break;
                        end
                    end
                    if ~valid_combinations(i)
                        break;
                    end
                end
            end
            
            
            
            % If we've eliminated all combinations, relax the threshold
            if sum(valid_combinations) == 0
                
                fprintf('%s: All combinations have high correlation items. \n', stim_category);
                valid_combinations = true(size(all_combs, 1), 1);
                % continue;
                % fprintf('All combinations have high correlations. Relaxing threshold...\n');
                % Progressively increase threshold until we have at least some combinations
                % while sum(valid_combinations) == 0 && MI_THRESHOLD < 1.0
                %     MI_THRESHOLD = MI_THRESHOLD + 0.05;
                %     valid_combinations = true(size(all_combs, 1), 1);
                %     for i = 1:size(all_combs, 1)
                %         combo = all_combs(i, :);
                %         for j = 1:length(combo)
                %             for k = j+1:length(combo)
                %                 if mutual_info(combo(j), combo(k)) >= MI_THRESHOLD
                %                     valid_combinations(i) = false;
                %                     break;
                %                 end
                %             end
                %             if ~valid_combinations(i)
                %                 break;
                %             end
                %         end
                %     end
                % end
                % fprintf('New threshold: %.2f, Valid combinations: %d\n', ...
                %     MI_THRESHOLD, sum(valid_combinations));
            end
            if best_only
                if strcmp(stim_category, 'Adult')
                    valid_combs_idxs = [20,53,54,24,9];
                elseif strcmp(stim_category, 'Faces')
                    valid_combs_idxs = [7,18];
                elseif strcmp(stim_category, 'Car')
                    valid_combs_idxs = [4];
                elseif strcmp(stim_category, 'House')
                    valid_combs_idxs = [49,64];
                elseif strcmp(stim_category, 'Animal')
                    valid_combs_idxs = [38,39,63,456];
                elseif strcmp(stim_category, 'Body')
                    valid_combs_idxs = [47];
                elseif strcmp(stim_category, 'Corridor')
                    valid_combs_idxs = [3];
                elseif strcmp(stim_category, 'Instrument')
                    valid_combs_idxs = [6,19];
                elseif strcmp(stim_category, 'Limb')
                    valid_combs_idxs = [7];
                end
                valid_combinations = false(size(all_combs, 1), 1);
                for i = 1:numel(valid_combs_idxs)
                    valid_combinations(valid_combs_idxs(i)) = true;
                end
            else
                % Get indices of the valid combinations
                valid_combs_idxs = find(valid_combinations);
            end
            % Keep only valid combinations
            selected_combs = all_combs(valid_combinations, :);
            % fprintf('%s valid combinations:\n', stim_category);
            % for i = 1:size(all_combs, 1)
            %     fprintf('%d. %s\n', i, strjoin(all_column_names(all_combs(i, :)), ', '));
            % end
            fprintf('Removed %d/%d combinations with correlation > %.2f\n', ...
                sum(~valid_combinations), numel(valid_combinations), MI_THRESHOLD);
        else
            selected_combs = 1:numel(stim_cat_cols);
        end
        
        % fprintf("Valid combinations:\n")
        % % print all valid combinations
        % ctr = 1;
        % for i = 1:numel(valid_combinations)
        %     if valid_combinations(i)
        %         fprintf('%d.(%d) %s\n', ctr, i, strjoin(all_column_names(all_combs(i, :)), ', '));
        %         ctr = ctr + 1;
        %     end
        % end
        % fprintf("Invalid combinations:\n")
        % % print all invalid combinations
        % ctr = 1;
        % for i = 1:numel(valid_combinations)
        %     if ~valid_combinations(i)
        %         fprintf('%d.(%d) %s\n', ctr, i, strjoin(all_column_names(all_combs(i, :)), ', '));
        %         ctr = ctr + 1;
        %     end
        % end  

        % keep_comb_with_col = [3];
        % logi_selected_combs = ismember(selected_combs, keep_comb_with_col);
        % logi_selected_combs_idx = [];
        % for i=1:length(logi_selected_combs)
        %     if sum(logi_selected_combs(i))
        %         logi_selected_combs_idx = [logi_selected_combs_idx, i];
        %     end
        % end
        % % remove combinations that do not contain the column with index 4
        % selected_combs = selected_combs(logi_selected_combs_idx, :);
        
        best_comb = [];
        best_not_enough_perc = 100;
        num_combs = size(selected_combs, 1);
        distributions = cell(num_combs, 1);
        comb_imgs = cell(num_combs, 1);
        ks_statistics = zeros(size(distributions));
        uniformity_scores = zeros(size(distributions));
        coverage_perc = zeros(num_combs,1);
        column_uniques = cell(num_combs, 1);
        comb_names = cell(num_combs, 1);
        cat_tot_combs = zeros(num_combs,1);
        cat_uniq_combs = zeros(num_combs,1);
        ylim_cat = 0;
        
        plot_corr_mat(mutual_info, all_column_names, MI_THRESHOLD, th_stat, ...
                      stim_category, folder_name, extra_txt, false);
        plot_corr_mat(mutual_info, all_column_names, MI_THRESHOLD, th_stat, ...
                      stim_category, folder_name, extra_txt, true);
        
        % best_tot_spaces_count_cat = 0;
        % best_not_enough_count_cat = 0;
        % best_colnames = [];
        for i_perm = 1:size(selected_combs, 1)
            % Get the current combination of columns to fix
            current_cols_idxs = selected_combs(i_perm, :);
            col_names = stim_cat_cols(current_cols_idxs);
            stim_cat_cols_curr = stim_cat_cols(current_cols_idxs);
            stim_cat_cols_curr{end+1} = 'TaskGroup';
            stim_cat_cols_curr{end+1} = 'StimNum';
            task_img_info_table = img_info_table(ismember(img_info_table.StimCat, stim_category), stim_cat_cols_curr);
            % Other taskgroup
            if task_other_only
                task_img_info_table = task_img_info_table(ismember(task_img_info_table.TaskGroup, 'Other'), :);
            elseif task_recdec_only
                task_img_info_table = task_img_info_table(ismember(task_img_info_table.TaskGroup, 'RecDec'), :);
            end
            % Get the number of unique values in each column
            num_unique_values = zeros(1, size(task_img_info_table, 2) - 2);
            for i = 1:size(task_img_info_table, 2) - 2
                num_unique_values(i) = numel(unique(task_img_info_table{:, i}));
            end
            column_uniques{i_perm} = num_unique_values;
            % Calculate the total number of combinations
            total_combinations = prod(num_unique_values);
            cat_tot_combs(i_perm) = total_combinations;
            
            [u,~,IC] = unique(task_img_info_table(:,1:end-2),'rows','stable');
            
            % if height(u) < CUSTOM_PICS_PER_CAT
            %     uniformity_scores(i_perm) = -1;
            %     continue;
            % end
            
            % categ_info_cell = cell(height(u), 1);
            recdeccount = [];
            othercount = [];
            cat_cust_pics_count = 0;
            not_enough_count_cat = 0;
            tot_spaces_count_cat = height(u);
            tot_spaces_count = tot_spaces_count + tot_spaces_count_cat;
            item_counts = zeros(size(u, 1), 1);
            comb_imgs_paths = cell(size(u, 1), 1);
            % item_counts = zeros(total_combinations, 1);
            % Get rows that match the unique rows
            for iu = 1: height(u)
                gp_rows = task_img_info_table(find(IC==iu),:);
                recdeccount = [recdeccount sum(ismember(gp_rows.TaskGroup, 'RecDec'))];
                othercount  = [othercount sum(ismember(gp_rows.TaskGroup, 'Other'))];
                % categ_info_cell{iu} = gp_rows;
                % gp_rows = gp_rows(ismember(gp_rows.TaskGroup, 'Other'),:);
                item_counts(iu) = height(gp_rows);
                if height(gp_rows) < 2
                    not_enough_count_cat = not_enough_count_cat + 1;
                end
                % Randomly pick one row from gp_rows
                random_index = randi(height(gp_rows));
                pic_row = gp_rows(random_index, :);
                pic_path = fullfile(pics_path, lower(stim_category), sprintf("%s-%d.jpg",lower(stim_category), pic_row.StimNum));
                pic_path = fullfile(pic_path);
                comb_imgs_paths{iu} = pic_path;
                if cat_cust_pics_count < CUSTOM_PICS_PER_CAT && height(gp_rows) >=2
                    
                    % copyfile(pic_path, custom_pics_path);
                    cat_cust_pics_count = cat_cust_pics_count + 1;
                end
            end
            not_enough_count = not_enough_count + not_enough_count_cat;
            not_enough_perc = not_enough_count_cat/tot_spaces_count_cat*100;
            cat_uniq_combs(i_perm) = height(u)- not_enough_count_cat;
            coverage_perc(i_perm) = (cat_uniq_combs(i_perm)/ total_combinations) * 100;            
            
            if max(item_counts) > ylim_cat
                ylim_cat = max(item_counts);
            end
            % Store the distribution for each combination            
            [distributions{i_perm}, sorted_col_indices] = sort(item_counts,'descend');
            comb_names{i_perm} = u(sorted_col_indices,:);
            comb_imgs{i_perm} = comb_imgs_paths(sorted_col_indices);

            % Calculate the uniformity score for each combination
            % uniformity_scores(i_perm) = 1 - (std(item_counts) / mean(item_counts));

            % Perform K-S test
            [h, p_value, ks_statistics(i_perm), cv] = kstest(item_counts, 'CDF', ...
                makedist('Uniform', 'Lower', min(item_counts), 'Upper', max(item_counts)));
            uniformity_scores(i_perm) = p_value;
            
            % Calculate average pairwise correlation for this combination
            combo_cols = current_cols_idxs;
            pairwise_corrs = [];
            for j = 1:length(combo_cols)
                for k = j+1:length(combo_cols)
                    pairwise_corrs = [pairwise_corrs; mutual_info(combo_cols(j), combo_cols(k))];
                end
            end
            avg_correlation = mean(pairwise_corrs);
            
            % Adjust uniformity score based on correlation
            % Lower correlation is better, so we penalize high correlation
            % corr_penalty = avg_correlation^2; % Square to emphasize high correlations
            % uniformity_scores(i_perm) = uniformity_scores(i_perm) * (1 - corr_penalty);
            
            if cat_cust_pics_count == CUSTOM_PICS_PER_CAT
                if not_enough_perc < best_not_enough_perc
                    best_not_enough_perc = not_enough_perc;
                    best_colnames = col_names;
                    best_not_enough_count_cat = not_enough_count_cat;
                    best_tot_spaces_count_cat = tot_spaces_count_cat;
                end
            end
        end
        
        if sort_by_coverage_only && ~ best_only
            % Sort by a combination of coverage and correlation
            [~, sorted_indices] = sort(coverage_perc, 'descend');
            if numel(sorted_indices) > 100
                cut_off = min(100, sum(coverage_perc>=50));
                sorted_indices = sorted_indices(1:cut_off);
            end
            fprintf('%s: valid combs indices: %s\n', stim_category, mat2str(valid_combs_idxs(sorted_indices)));
        else
            % Sort the uniformity scores and group them by p-value ranges
            p_values = uniformity_scores;
            group1_indices = find(p_values > 0.05);
            group2_indices = find(p_values >= 0.01 & p_values <= 0.05);
            group3_indices = find(p_values > 0.0 & p_values < 0.01);
            group4_indices = find(p_values == 0.0);
    
            group1_coverage_perc = coverage_perc(group1_indices);
            group2_coverage_perc = coverage_perc(group2_indices);
            group3_coverage_perc = coverage_perc(group3_indices);
            group4_coverage_perc = coverage_perc(group4_indices);
    
            [~, sorted_group1_indices] = sort(group1_coverage_perc, 'descend');
            [~, sorted_group2_indices] = sort(group2_coverage_perc, 'descend');
            [~, sorted_group3_indices] = sort(group3_coverage_perc, 'descend');
            [~, sorted_group4_indices] = sort(group4_coverage_perc, 'descend');

            % Get the first 10 items from each group
            sorted_group1_indices = sorted_group1_indices(1:min(10, numel(sorted_group1_indices)));
            sorted_group2_indices = sorted_group2_indices(1:min(10, numel(sorted_group2_indices)));
            sorted_group3_indices = sorted_group3_indices(1:min(10, numel(sorted_group3_indices)));
            sorted_group4_indices = sorted_group4_indices(1:min(10, numel(sorted_group4_indices)));
    
            % Get the original indices for each sorted group
            sorted_group1_original_indices = group1_indices(sorted_group1_indices);
            sorted_group2_original_indices = group2_indices(sorted_group2_indices);
            sorted_group3_original_indices = group3_indices(sorted_group3_indices);
            sorted_group4_original_indices = group4_indices(sorted_group4_indices);
    
            % Combine the sorted indices
            sorted_indices = [sorted_group1_original_indices; 
                              sorted_group2_original_indices; 
                              sorted_group3_original_indices;
                              sorted_group4_original_indices];
            fprintf('%s: valid combs indices: %s\n', stim_category, mat2str(valid_combs_idxs(sorted_indices)));
        end
        
        if best_only
            if strcmp(stim_category, 'Faces')
                sorted_indices = [1,2];
            elseif strcmp(stim_category, 'Car')
                sorted_indices = [1];
            elseif strcmp(stim_category, 'House')
                sorted_indices = [1,2];
            elseif strcmp(stim_category, 'Animal')
                sorted_indices = [1,2,3,4];
            elseif strcmp(stim_category, 'Body')
                sorted_indices = [1];
            elseif strcmp(stim_category, 'Corridor')
                sorted_indices = [1];
            elseif strcmp(stim_category, 'Instrument')
                sorted_indices = [1,2];
            elseif strcmp(stim_category, 'Limb')
                sorted_indices = [1];
            end
        end
        
        if best_only
            cat_folder = [folder_name filesep stim_category sprintf('_best%s_%s',extra_txt, th_stat)];     
        elseif sort_by_coverage_only
            cat_folder = [folder_name filesep stim_category sprintf('%s_cov_%s', extra_txt, th_stat)];
        else
            cat_folder = [folder_name filesep stim_category sprintf('%s_%s', extra_txt, th_stat)];
        end
        if ~exist(cat_folder, 'dir')
            mkdir(cat_folder);
        end

        % Calculate the number of subplots needed
        num_subplots = numel(sorted_indices);
        subplots_per_img = 5;
        if best_only
            num_images = num_subplots;
            subplots_per_img = 1;
        else
            num_images = ceil(num_subplots / subplots_per_img);
        end
        fig = figure('Visible', 'off');
        
        cat_title = sprintf('%s (tot imgs:%d)cols(%d)(5 combs:%d/%d):\n%s', stim_category, height(task_img_info_table), ...
                            numel(all_column_names), num_combs, numel(valid_combinations),strjoin(all_column_names,', '));
        
        % Loop through each set of 5 items
        item_idx = 1;
        for i = 1:num_images
            % Calculate the start and end indices for the current set
            start_index = (i - 1) * subplots_per_img + 1;
            end_index = min(i * subplots_per_img, num_subplots);
            
            % Create a figure with 5 subplots
            clf(fig);
            
            sgtitle(sprintf('%d. %s', i, cat_title));  % Set the main title to the category name

            if best_only
                lyt_rows = 5;
                lyt_cols = 10;
                % Create a subplot for the current item
                t = tiledlayout(lyt_rows, lyt_cols);%, 'TileSpacing', 'compact', 'Padding', 'compact');
            else
                % Create a subplot for the current item
                t = tiledlayout(subplots_per_img, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
            end
            
            % Loop through each item in the current set
            for j = start_index:end_index 
                if best_only
                    nexttile(1, [1 lyt_cols])
                else
                    nexttile(j - start_index + 1);   
                end

                % Data for the bar plot
                data = distributions{sorted_indices(j)};                
                
                % Calculate the average
                average_value = mean(data);
                                
                b = bar(data);
                b.FaceAlpha = 0.5;
                b.EdgeColor = 'none';

                % Add the horizontal dotted line at the average value
                indexOfAvg = find(data < average_value, 1, 'first');
                xline(indexOfAvg - 0.5, '--', sprintf('avg:%.2f',average_value), 'Color', 'r', 'LineWidth', 1, 'LabelOrientation', 'aligned');

                indexOfLastTwo = find(data < 2, 1, 'first');

                if ~isempty(indexOfLastTwo)
                    xline(indexOfLastTwo - 0.5, 'b', 'LineWidth', 2, 'LineStyle', '--');
                end
                
                % text(1:length(distributions{sorted_indices(j)}), distributions{sorted_indices(j)}, ...
                %     num2str(distributions{sorted_indices(j)}), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                % Create tick labels with counts in brackets
                num_bars = length(data);
                tick_labels = cell(1, num_bars);
                for idx = 1:num_bars
                    tick_labels{idx} = sprintf('%d(%d)', idx, data(idx));
                end

                ylim([0, ylim_cat]);
                
                set(gca, 'XGrid', 'off', ...
                    'YGrid', 'off', ...
                    'TickLength', [0 0], ...  % Remove tick marks
                    'XTickLabel', tick_labels, ...
                    'XTick', 1:num_bars);
                ylabel('Item Count');
                % Calculate evenly spaced positions using y limits
                y_positions = linspace(ylim_cat, 0, 7);
                y_positions = y_positions(2:6);
                curr_comb = comb_names{sorted_indices(j)};
                % Add combination values within each bar
                for row_idx=1:height(curr_comb)
                    % Get row
                    row_values = table2cell(curr_comb(row_idx,:));                    
                    
                    % Add each value as a separate line
                    for val_idx = 1:length(row_values)
                        text(row_idx, y_positions(val_idx), row_values{val_idx}, ...
                            'HorizontalAlignment', 'center', ...
                            'Color', 'k', ...
                            'FontSize', 9);
                    end
                end

                % plot images for best only as a subplot
                if best_only
                    %for img_idx = 1:length(comb_imgs{sorted_indices(j)})
                    for img_idx = 1:indexOfLastTwo-1
                        img_path = comb_imgs{sorted_indices(j)}{img_idx};
                        if exist(img_path, 'file')
                            % get image name from path
                            [~, img_name, ext] = fileparts(img_path);
                            img = imread(img_path);
                            ax = nexttile(img_idx + lyt_cols);                            
                            imshow(img, 'Parent', ax);
                            title(img_name, 'Interpreter', 'none', 'FontSize', 8);
                            axis(ax, 'off'); % Hide axes
                        end
                    end
                end

                % Calculate the average correlation for this combination
                combo_cols = selected_combs(sorted_indices(j), :);
                pairwise_corrs = [];
                for c1 = 1:length(combo_cols)
                    for c2 = c1+1:length(combo_cols)
                        pairwise_corrs = [pairwise_corrs; mutual_info(combo_cols(c1), combo_cols(c2))];
                    end
                end
                avg_correlation = mean(pairwise_corrs);

                column_names = strrep(all_column_names(selected_combs(sorted_indices(j),:)), '_', '-');
                combined_info = join(cellfun(@(col, unique) sprintf('%s(%d)', col, unique), ...
                    column_names, num2cell(column_uniques{sorted_indices(j)}), 'UniformOutput', false), ', ');
                less_than_2 = cat_tot_combs(sorted_indices(j)) - cat_uniq_combs(sorted_indices(j));
                zeros_count = cat_tot_combs(sorted_indices(j)) - numel(data);
                if cat_uniq_combs(sorted_indices(j)) >= 18 && zeros_count/cat_tot_combs(sorted_indices(j)) <= .25
                    title_color = [0, 0.5, 0];  % green
                elseif cat_uniq_combs(sorted_indices(j)) >= 18 && zeros_count/cat_tot_combs(sorted_indices(j)) > .25
                    title_color = [0.5, 0.5, 0];%[0.7725, 0.8902, 0.5176]; % yellow-green
                elseif cat_uniq_combs(sorted_indices(j)) >= 16 && zeros_count/cat_tot_combs(sorted_indices(j)) > .25
                    title_color = [1, 0.5, 0];  % orange
                elseif cat_uniq_combs(sorted_indices(j)) >= 16
                    title_color = [1, 0.5, 0];  % orange
                else
                    title_color = 'r';
                end
                title(t, sprintf('%d(%d). kstest p-value: %.3f Coverage:%.2f%%(%d/%d) (<2:%d)(1:%d)(0:%d) AvgMI:%.2f\nColumns: %s ', ...
                            item_idx, valid_combs_idxs(sorted_indices(j)), uniformity_scores(sorted_indices(j)), ...
                            coverage_perc(sorted_indices(j)), ...
                            cat_uniq_combs(sorted_indices(j)), cat_tot_combs(sorted_indices(j)), ...
                            less_than_2, less_than_2-zeros_count, zeros_count, ...
                            avg_correlation, ...
                            combined_info{1}), ...
                            'Color', title_color);
                
                item_idx = item_idx + 1;
            end
            
            % Adjust the subplot layout
            set(fig, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
            % set(gca, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
            
            % Save the figure
            % saveas(gcf, fullfile(cat_folder, sprintf('%s_%d.png', stim_category, i)))
            
            exportgraphics(fig,fullfile(cat_folder, sprintf('%s_%d.png', stim_category, i)),'Resolution',75)

        end

        % Find the combination with the highest uniformity score
        [~, best_comb_idx] = max(uniformity_scores);
        best_comb = selected_combs(best_comb_idx, :);
        
        % Calculate correlation for best combination
        best_combo_cols = best_comb;
        best_pairwise_corrs = [];
        for c1 = 1:length(best_combo_cols)
            for c2 = c1+1:length(best_combo_cols)
                best_pairwise_corrs = [best_pairwise_corrs; mutual_info(best_combo_cols(c1), best_combo_cols(c2))];
            end
        end
        best_avg_correlation = mean(best_pairwise_corrs);
        
        % Print the results
        fprintf('Best combination for category %s: %s (Avg. Correlation: %.2f)\n', ...
            stim_category, strjoin(stim_cat_cols(best_comb), ', '), best_avg_correlation);
        % fprintf("Added %d/%d custom pics. <2 items in %d/%d(%.2f%%) spaces in category:%s ,cols: %s\n", ...
        %             18, best_tot_spaces_count_cat, best_not_enough_count_cat, ...
        %             best_tot_spaces_count_cat, best_not_enough_perc, ...
        %             stim_category, strjoin(best_colnames, ', '));
    end
end

function cramers_v = calc_cramers_v(x, y)
    % Ensure inputs are properly treated as categorical
    if ~iscategorical(x)
        x = categorical(x);
    end
    if ~iscategorical(y)
        y = categorical(y);
    end
    
    % Create the confusion matrix (contingency table)
    confusion_matrix = crosstab(x, y);
    
    % Calculate expected frequencies
    row_totals = sum(confusion_matrix, 2);
    col_totals = sum(confusion_matrix, 1);
    total = sum(confusion_matrix(:));
    expected = (row_totals * col_totals) / total;
    
    % Calculate Chi-squared statistic
    % Avoid division by zero
    valid_indices = expected(:) > 0;
    chi2 = sum(((confusion_matrix(valid_indices) - expected(valid_indices)).^2) ./ expected(valid_indices));
    
    % Total number of samples
    n = total;
    
    % Calculate Cramér's V
    % Ensure we don't divide by zero
    min_dim = min(size(confusion_matrix)) - 1;
    if min_dim == 0 || n == 0
        cramers_v = 0;
    else
        cramers_v = sqrt(chi2 / (n * min_dim));
    end
end

function u = theil_u(x, y)
    % Calculate Theil's U (y given x) - how well x predicts y
    x_cat = categorical(x);
    y_cat = categorical(y);
    
    contingency = crosstab(x_cat, y_cat);
    n = sum(contingency(:));
    
    % Calculate entropy of y
    p_y = sum(contingency, 1) / n;
    H_y = -sum(p_y .* log2(p_y + eps));
    
    % Calculate conditional entropy of y given x
    H_y_given_x = 0;
    p_x = sum(contingency, 2) / n;
    
    for i = 1:size(contingency, 1)
        if p_x(i) > 0
            p_y_given_x = contingency(i,:) / sum(contingency(i,:));
            h = -sum(p_y_given_x .* log2(p_y_given_x + eps));
            H_y_given_x = H_y_given_x + p_x(i) * h;
        end
    end
    
    % Calculate Theil's U
    if H_y > 0
        u = (H_y - H_y_given_x) / H_y;
    else
        u = 1; % If y has no entropy, then x perfectly predicts y
    end
end

function nmi = calc_mutual_information(x, y)
    % Calculate mutual information and normalized mutual information
    % between two categorical variables
    % 
    % Parameters:
    %   x, y: Categorical variables to compare
    % 
    % Returns:
    %   mi: Raw mutual information value
    %   nmi: Normalized mutual information (ranges from 0 to 1)
    
    % Ensure inputs are properly treated as categorical
    if ~iscategorical(x)
        x = categorical(x);
    end
    if ~iscategorical(y)
        y = categorical(y);
    end
    
    % Remove any undefined or missing values
    valid = ~isundefined(x) & ~isundefined(y) & ~ismissing(x) & ~ismissing(y);
    x = x(valid);
    y = y(valid);
    
    % Create the contingency table
    confusion_matrix = crosstab(x, y);
    
    % Calculate joint and marginal probabilities
    n = sum(confusion_matrix(:));
    p_xy = confusion_matrix / n;
    p_x = sum(p_xy, 2);
    p_y = sum(p_xy, 1);
    
    % Calculate entropy for each variable
    H_x = -sum(p_x .* log2(p_x + eps));  % eps prevents log(0)
    H_y = -sum(p_y .* log2(p_y + eps));
    
    % Calculate mutual information
    mi = 0;
    [rows, cols] = size(p_xy);
    for i = 1:rows
        for j = 1:cols
            if p_xy(i,j) > 0
                mi = mi + p_xy(i,j) * log2(p_xy(i,j) / (p_x(i) * p_y(j)));
            end
        end
    end
    
    % Calculate normalized mutual information
    % Use min entropy for normalization (other options include max or average)
    if min(H_x, H_y) > 0
        nmi = mi / min(H_x, H_y);
    else
        % If one variable has zero entropy (constant), set NMI to either 0 or 1
        % If both have zero entropy and identical, NMI = 1
        % If both have zero entropy and different, NMI is undefined (set to 0)
        if H_x == 0 && H_y == 0
            if numel(unique(x)) == 1 && numel(unique(y)) == 1
                all_x_same = all(x == x(1));
                all_y_same = all(y == y(1));
                if all_x_same && all_y_same
                    nmi = 1;  % Both variables are constant and identical
                else
                    nmi = 0;  % Both variables are constant but different
                end
            else
                nmi = 0;
            end
        else
            nmi = 0;  % One variable has zero entropy
        end
    end
    
    % Ensure NMI is between 0 and 1 (numerical precision issues)
    nmi = max(0, min(1, nmi));
end

function plot_corr_mat(corr_matrix, all_column_names, MI_THRESHOLD, th_stat, ...
                       stim_category, folder_name, extra_txt, b_binary)
    % Plot the correlation matrix
    figure('Visible', 'off');
    % imagesc(corr_matrix);
    lower_triangle = tril(corr_matrix); % Keep only the upper triangle
    imagesc(lower_triangle);
    if b_binary
        custom_colormap = [0 0 1; 1 1 0.3; 0 1 0]; % Blue for below threshold, Yellow for above

        % Create a binary matrix for thresholding
        thresholded_matrix = lower_triangle;
        thresholded_matrix(lower_triangle < MI_THRESHOLD) = 0; % Below threshold
        thresholded_matrix(lower_triangle >= MI_THRESHOLD & lower_triangle < 1) = 0.5;  % Above threshold
        thresholded_matrix(lower_triangle == 1) = 1; % Exactly 1

        % Plot the thresholded matrix
        imagesc(thresholded_matrix);
        colormap(custom_colormap);
    else
        imagesc(lower_triangle);
        colormap('parula');
        colorbar;
    end
    
    % Set axis labels and title
    xticks(1:size(corr_matrix, 2));
    yticks(1:size(corr_matrix, 1));
    xticklabels(all_column_names);
    yticklabels(all_column_names);
    xtickangle(45);
    
    % Add title
    if b_binary
        title(sprintf('%s (MI >= %.2f(%s))', stim_category, MI_THRESHOLD, th_stat), ...
              'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
    else
        title(sprintf('%s MI', stim_category), ...
              'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
    end

    % % Add correlation values as text in each cell (only for upper triangle)
    % for i = 1:size(corr_matrix, 1)
    %     for j = i:size(corr_matrix, 2) % Start from diagonal to avoid lower triangle
    %         background_val = lower_triangle(i, j) / max(lower_triangle(:));
    %         text_color = 'white';
    %         if background_val > 0.5
    %             text_color = 'black';
    %         end
    %         text(j, i, sprintf('%.2f', corr_matrix(i, j)), ...
    %             'HorizontalAlignment', 'center', ...
    %             'Color', text_color, ...
    %             'FontSize', 7);
    %     end
    % end

    % Add correlation values as text in each cell (only for lower triangle)
    for i = 1:size(corr_matrix, 1)
        for j = 1:i % Only for lower triangle
            background_val = lower_triangle(i, j) / max(lower_triangle(:));
            text_color = 'white';
            if background_val > 0.5
                text_color = 'black';
            end
            if b_binary
                if lower_triangle(i, j) >= MI_THRESHOLD
                    text_color = 'black'; % Change text color for above threshold
                else
                    text_color = 'white'; % Change text color for below threshold
                end
            end
            text(j, i, sprintf('%.2f', corr_matrix(i, j)), ...
                'HorizontalAlignment', 'center', ...
                'Color', text_color, ...
                'FontSize', 7);
        end
    end
    
    if b_binary
        saveas(gcf, fullfile(folder_name, [stim_category sprintf('%s_MI_%s_binary.png',extra_txt, th_stat)]));
    else
        saveas(gcf, fullfile(folder_name, [stim_category sprintf('%s_MI.png', extra_txt)]));
    end
    
end

function plot_confusion_matrices(subplot_idx, i,j, col_i, col_j, ...
                                 all_column_names, ...
                                 corr_mat, cramers_v, mutual_info, glob_min, glob_max)
    nexttile(subplot_idx);
    confusion_mat = crosstab(categorical(col_i), categorical(col_j));

    % Create heatmap
    h = imagesc(confusion_mat);
    % axis square;
    % axis image;
    colormap(gca, 'parula');
    c = colorbar;
    c.Label.String = 'Count';

    % Set consistent color axis range
    clim([glob_min, glob_max]);

    row_cats = categories(categorical(col_i));
    col_cats = categories(categorical(col_j));
    xticks(1:length(col_cats));
    yticks(1:length(row_cats));
    xticklabels(col_cats);
    yticklabels(row_cats);
    xtickangle(45);

    set(gca, 'FontSize', 8, 'FontWeight', 'bold');

    % Add title
    title(sprintf('%s vs %s\nr=%.2f, V=%.2f, MI=%.2f', ...
        all_column_names{i}, all_column_names{j}, ...
        corr_mat(i,j), cramers_v(i,j), mutual_info(i,j)), ...
        'FontSize', 8, 'FontWeight', 'bold');

    % Display counts in each cell
    for row = 1:size(confusion_mat,1)
        for col = 1:size(confusion_mat,2)
            val = num2str(confusion_mat(row,col));
            background_val = confusion_mat(row,col) / max(confusion_mat(:));
            text_color = 'white';
            if background_val > 0.5
                text_color = 'black';
            end
            text(col, row, val, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Color', text_color, ...
                'FontWeight', 'bold', ...
                'FontSize', 10);
        end
    end
    
end

% Estimate threshold for correlation, mutual information, any measure
% that can be used to filter out highly correlated features
function threshold = estimate_threshold(stim_category, corr_matrix, ...
                                        method, extra_txt, task_recdec_only)
    % Estimate the threshold for filtering out highly correlated features
    % 
    % Parameters:
    %   corr_matrix: Input correlation/similarity matrix
    %   method: Method to use for threshold estimation ('mean', 'median', 'std', 'quantile')
    % 
    % Returns:
    %   threshold: Estimated threshold value
    
    % Get upper triangle of matrix excluding diagonal
    n = size(corr_matrix, 1);
    upper_tri = zeros(1, (n^2-n)/2);  % Pre-allocate array
    idx = 1;
    for i = 1:n
        for j = (i+1):n
            upper_tri(idx) = corr_matrix(i,j);
            idx = idx + 1;
        end
    end
    
    % Calculate the threshold based on the method using only upper triangle values
    
    
    % Create figure with 2 subplots
    figure('Visible', 'off', 'Position', [100 100 1200 400]);
    
    % First subplot - Probability histogram
    subplot(1, 2, 1);
    [f, x] = ecdf(upper_tri);
    plot(x, f, 'LineWidth', 1);
    hold on;
    
    % Calculate slopes between consecutive points
    slopes = diff(f) ./ diff(x);
    % diff_x = diff(x);
    
    slope_threshold = prctile(slopes, 10); 
    plateau_indices = find(slopes < slope_threshold);
    
    % First plateau
    plateau_x = x(min(plateau_indices));
    
    % Add vertical lines for different metrics
    xline(mean(upper_tri), '--r', sprintf('Mean: %.2f', mean(upper_tri)), 'LineWidth', 1.5);
    xline(median(upper_tri), '--g', sprintf('Median: %.2f', median(upper_tri)), 'LineWidth', 1.5);
    xline(plateau_x, '--c', sprintf('Plateau: %.2f', plateau_x), 'LineWidth', 1.5);
    xline(quantile(upper_tri, 0.90), '--b', sprintf('90th: %.2f', quantile(upper_tri, 0.90)));
    xline(quantile(upper_tri, 0.95), '--b', sprintf('95th: %.2f', quantile(upper_tri, 0.95)));
    
    title('CDF of MI (Probability)');
    xlabel('MI');
    ylabel('Probability');
    grid on;
    
    % Second subplot - Count histogram
    subplot(1, 2, 2);
    h = histogram(upper_tri,100);
    hold on;
    
    % Add count labels on top of each bar
    for i = 1:length(h.Values)
        if h.Values(i) > 0
            x = (h.BinEdges(i) + h.BinEdges(i+1))/2;
            text(x, h.Values(i), num2str(h.Values(i)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom');
        end
    end
    
    % Add vertical lines for different metrics
    xline(mean(upper_tri), '--r', sprintf('Mean: %.2f', mean(upper_tri)));
    xline(median(upper_tri), '--g', sprintf('Median: %.2f', median(upper_tri)));
    xline(quantile(upper_tri, 0.75), '--b', sprintf('75th: %.2f', quantile(upper_tri, 0.75)));
    xline(quantile(upper_tri, 0.90), '--b', sprintf('90th: %.2f', quantile(upper_tri, 0.90)));
    xline(quantile(upper_tri, 0.95), '--b', sprintf('95th: %.2f', quantile(upper_tri, 0.95)));
    xline(plateau_x, '--c', sprintf('Plateau: %.2f', plateau_x));
    
    title('Distribution of MI (Counts)');
    xlabel('MI');
    ylabel('Count');
    grid on;
    
    % Save the figure in the same folder as other plots
    folder_name = 'spaces_dist_figures';
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    saveas(gcf, fullfile(folder_name, [stim_category sprintf('%s_CDF_distribution.png', extra_txt)]));
    
    
    close(gcf);
    
    % Print all metrics for debugging
    fprintf('Threshold (excluding diagonal),plateau: %.2f, mean: %.2f, median: %.2f, std: %.2f, quant_75: %.2f, quant_90: %.2f\n', ...
            plateau_x, mean(upper_tri), median(upper_tri), std(upper_tri), ...
            quantile(upper_tri, 0.75), quantile(upper_tri, 0.90));
    
    switch method
        case 'mean'
            threshold = mean(upper_tri);
        case 'median'
            threshold = median(upper_tri);
        case 'std'
            threshold = std(upper_tri);
        case 'quant_90'
            threshold = quantile(upper_tri, 0.90);
        case 'quant_95'
            threshold = quantile(upper_tri, 0.95);
        case 'plateau'
            threshold = plateau_x;
        otherwise
            error('Invalid method: %s', method);
    end
    threshold = floor(threshold*100)/100;
end

function [is_not_uniform, cv] = analyze_feature_distribution(feat_data, feat_name)
    % Get the histogram data
    [counts, edges] = histcounts(feat_data);
    
    % Plot histogram
    histogram(feat_data, 'Normalization', 'probability');
    
    % Chi-squared test using raw counts
    n_bins = length(counts);
    total_counts = sum(counts);
    expected_counts = ones(1, n_bins) * total_counts / n_bins;

    % Avoid division by zero
    valid = expected_counts > 0;
    chi2_stat = sum((counts(valid) - expected_counts(valid)).^2 ./ expected_counts(valid));
    p_value = 1 - chi2cdf(chi2_stat, n_bins - 1);

    % Coefficient of variation on probabilities (for visualization)
    prob_counts = counts / total_counts;
    if mean(prob_counts) == 0
        cv = Inf;
    else
        cv = std(prob_counts) / mean(prob_counts);
    end
    
    
    % Add reference line for uniform distribution
    yline(mean(counts), '--r', 'Mean');

    ylim([0, 1]);

    % Define thresholds for uniformity
    CV_THRESHOLD = 0.8;
    UNIFORMITY_THRESHOLD = 0.05;

    is_not_uniform = p_value < UNIFORMITY_THRESHOLD && ...
                cv > CV_THRESHOLD;

    if is_not_uniform
        title(sprintf('%s\nχ²(p=%.3f)\nCV=%.2f', ...
        feat_name, p_value, cv),'FontSize', 6, "Color",'r');
    else
        title(sprintf('%s\nχ²(p=%.3f)\nCV=%.2f', ...
        feat_name, p_value, cv), 'FontSize', 6);
    end
    % is_not_uniform = cv > 1.0;
end