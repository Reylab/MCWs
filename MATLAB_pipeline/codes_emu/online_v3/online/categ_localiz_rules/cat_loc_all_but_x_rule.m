function [all_categ_all_but_x_cell, categ_localiz_history, experiment] = cat_loc_all_but_x_rule(selected2explore, experiment, ...
                                                   categ_localiz_history, ...
                                                   img_info_table, nonfixed_feats_count, ...
                                                   max_pics_per_rule, select_status)

all_categ_all_but_x_cell = cell(numel(categ_localiz_history), 1);
for i_stim_cat = 1:numel(categ_localiz_history)
    all_categ_all_but_x_cell{i_stim_cat}.category = categ_localiz_history{i_stim_cat}.category; 
    all_categ_all_but_x_cell{i_stim_cat}.pic_ids = [];
end

for c = selected2explore(:)'
    % Get row for img_info_table for the pic
    pic_row_idx = find(cellfun(@(x)strcmp(x,experiment.ImageNames.name{c}),img_info_table.name));
    pic_info = img_info_table(pic_row_idx, :);
    if height(pic_info) == 1
        stim_cat = pic_info.StimCat;
    else
        continue;
    end
    cat_id = find(cellfun(@(x)strcmp(x.category, stim_cat),all_categ_all_but_x_cell));
    categ_all_but_x_feats = all_categ_all_but_x_cell{cat_id};

    cat_idx = find(cellfun(@(x)strcmp(x.category, stim_cat),categ_localiz_history));
    cat_history = categ_localiz_history{cat_idx}.history;
    % Get columns that start with stim_cat
    stim_cat_cols = img_info_table.Properties.VariableNames( ...
        startsWith(img_info_table.Properties.VariableNames, stim_cat));
    if numel(stim_cat_cols) >= 5
        stim_cat_cols= stim_cat_cols(1:5);
    end

    % Get the number of columns in the input table
    num_cols = width(stim_cat_cols);
    if num_cols <= nonfixed_feats_count
        fprintf('Not enough feats/cols for executing all_but_%d for %s\n', ...
                nonfixed_feats_count, experiment.ImageNames.name{c})
        continue;
    end 
    
    cols_to_fix = nchoosek(1:num_cols, num_cols - nonfixed_feats_count);
    
    same_feats_idx = true(size(cat_history,1),1);    
    all_but_x_idxs  = false(size(cat_history,1),1);

    % Remove row with same feats as pic_info from cat_history
    for stim_cat_col = stim_cat_cols
        same_feats_idx = same_feats_idx & cellfun(@(x) strcmp(x,pic_info.(stim_cat_col{1}){1}), ...
                                                      cat_history.(stim_cat_col{1}));            
    end
    without_same_feats_idx = ~same_feats_idx;
    
    % Loop through each combination of columns to fix
    for i = 1:size(cols_to_fix, 1)
        fixed_feats_idx = true(size(cat_history,1),1);
        nonfixed_feats_idx = true(size(cat_history,1),1);

        % Get the current combination of columns to fix
        current_cols_to_fix = cols_to_fix(i, :);
        
        % Get the columns that are not fixed
        non_fixed_cols = setdiff(1:num_cols, current_cols_to_fix);     
        
        % all the rows that have the same values for current_cols_to_fix as
        % pic_info
        for stim_cat_col = stim_cat_cols(current_cols_to_fix)
            fixed_feats_idx = fixed_feats_idx & cellfun(@(x) strcmp(x,pic_info.(stim_cat_col{1}){1}), ...
                                                          cat_history.(stim_cat_col{1}));            
        end
%         fprintf("Features of %s \n", pic_info.name{1});
%         disp(pic_info(:,stim_cat_cols));
%         fprintf("Same feat rows in cat history with %d cols fixed\n", feats_fixed_count);
%         disp(stim_cat_cols(current_cols_to_fix))
        
%         disp(cat_history(fixed_feats_idx,:));
        % all the rows that have diff values for non_fixed_cols as
        % pic_info
        for stim_cat_col = stim_cat_cols(non_fixed_cols)
            nonfixed_feats_idx = nonfixed_feats_idx & cellfun(@(x) ~strcmp(x,pic_info.(stim_cat_col{1}){1}), ...
                                                          cat_history.(stim_cat_col{1}));            
        end
        
%         fprintf("Diff feat rows in cat history with %d cols nonfixed\n", num_cols - feats_fixed_count);
%         disp(stim_cat_cols(non_fixed_cols));
%         disp(cat_history(nonfixed_feats_idx,:));

        curr_all_but_x_idx = fixed_feats_idx & nonfixed_feats_idx;
%         fprintf("Matches in cat history with %d cols nonfixed\n", num_cols - feats_fixed_count);
%         disp(stim_cat_cols(non_fixed_cols));
%         disp(cat_history(curr_all_but_x_idx,:));

        all_but_x_idxs = all_but_x_idxs | curr_all_but_x_idx;
    end    
    
    % Get rid of same feats
    all_but_x_idxs = without_same_feats_idx & all_but_x_idxs;

    if sum(all_but_x_idxs) == 0
        fprintf('No pictures with all_but_%d for %s added\n', ...
                nonfixed_feats_count, experiment.ImageNames.name{c})
        continue
    end
    all_but_x_table = cat_history(all_but_x_idxs,:);
    [all_but_x_table, pic_id_list, experiment] = add_max_pics_per_rule(experiment, all_but_x_table, ...
                                            stim_cat, max_pics_per_rule, select_status);
    
    % Update categ_localiz_history
    categ_localiz_history{cat_idx}.history(all_but_x_idxs,:) = all_but_x_table;
    categ_all_but_x_feats.pic_ids = [categ_all_but_x_feats.pic_ids; pic_id_list(:)];
    all_categ_all_but_x_cell{cat_id} = categ_all_but_x_feats;
    str_pics_added = "";
    for du_idx = 1:numel(pic_id_list)
        pic_id = pic_id_list(du_idx);
        str_pics_added = sprintf("%s %s", str_pics_added, experiment.ImageNames(pic_id,:).name{1});
    end
    fprintf('%d pictures with all_but_%d as %s added (%s)\n', ...
            numel(pic_id_list), nonfixed_feats_count, experiment.ImageNames.name{c}, str_pics_added)

%     disp(pic_info(:,stim_cat_cols));
%     disp(all_but_x_table);    
end
tot_pics_added = sum(cellfun(@(x)numel(x.pic_ids), all_categ_all_but_x_cell));
fprintf("all_but_%d rule summary(tot pics added = %d):\n", nonfixed_feats_count, tot_pics_added);
for i_stim_cat = 1:numel(all_categ_all_but_x_cell)
    categ_all_but_x_feats = all_categ_all_but_x_cell{i_stim_cat};
    fprintf('%d pictures added with all_but_%d rule for category %s\n', ...
                numel(categ_all_but_x_feats.pic_ids), nonfixed_feats_count, categ_all_but_x_feats.category)
end
% fprintf('%d pictures same_unit in total\n', numel(same_feats_list))

end