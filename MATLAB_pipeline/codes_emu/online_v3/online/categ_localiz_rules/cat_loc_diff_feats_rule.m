function [all_categ_diff_feats_cell, categ_localiz_history, experiment] = cat_loc_diff_feats_rule(selected2explore, experiment, ...
                                                   categ_localiz_history, ...
                                                   img_info_table, max_pics_per_rule, ...
                                                   select_status)

all_categ_diff_feats_cell = cell(numel(categ_localiz_history), 1);
for i_stim_cat = 1:numel(categ_localiz_history)
    all_categ_diff_feats_cell{i_stim_cat}.category = categ_localiz_history{i_stim_cat}.category; 
    all_categ_diff_feats_cell{i_stim_cat}.pic_ids = [];
end

for c = selected2explore(:)'
    % Get row for img_info_table for the pic
    pic_row_idx = cellfun(@(x)strcmp(x,experiment.ImageNames.name{c}),img_info_table.name);
    pic_info = img_info_table(pic_row_idx, :);
    if height(pic_info) == 1
        stim_cat = pic_info.StimCat;
    else
        continue;
    end
    cat_id = find(cellfun(@(x)strcmp(x.category, stim_cat),all_categ_diff_feats_cell));
    categ_diff_feats = all_categ_diff_feats_cell{cat_id};

    cat_idx = cellfun(@(x)strcmp(x.category, stim_cat),categ_localiz_history);
    cat_history = categ_localiz_history{cat_idx}.history;
    % Get columns that start with stim_cat
    stim_cat_cols = img_info_table.Properties.VariableNames( ...
        startsWith(img_info_table.Properties.VariableNames, stim_cat));
    if numel(stim_cat_cols) >= 5
        stim_cat_cols= stim_cat_cols(1:5);
    end

    % Get all pics that have the same features as pic_info
    diff_feats_idx = true(size(cat_history,1),1);
    for stim_cat_col = stim_cat_cols
        diff_feats_idx = diff_feats_idx & cellfun(@(x) ~strcmp(x,pic_info.(stim_cat_col{1}){1}), ...
                                                    cat_history.(stim_cat_col{1}));
    end
    if sum(diff_feats_idx) == 0
        fprintf('No pictures with diff features as %s added\n', experiment.ImageNames.name{c})
        continue
    end
    diff_feats_table = cat_history(diff_feats_idx,:);
    [diff_feats_table, pic_id_list, experiment] = add_max_pics_per_rule(experiment, diff_feats_table, ...
                                             stim_cat, max_pics_per_rule, select_status);
    
    % Update categ_localiz_history
    categ_localiz_history{cat_idx}.history(diff_feats_idx,:) = diff_feats_table;
    categ_diff_feats.pic_ids = [categ_diff_feats.pic_ids; pic_id_list(:)];
    all_categ_diff_feats_cell{cat_id} = categ_diff_feats;
    str_pics_added = "";
    for du_idx = 1:numel(pic_id_list)
        pic_id = pic_id_list(du_idx);
        str_pics_added = sprintf("%s %s", str_pics_added, experiment.ImageNames(pic_id,:).name{1});
    end
    fprintf('%d pictures with diff features as %s added (%s)\n', numel(pic_id_list), experiment.ImageNames.name{c}, str_pics_added)
end
tot_pics_added = sum(cellfun(@(x)numel(x.pic_ids), all_categ_diff_feats_cell));
fprintf("diff features rule summary(tot pics added = %d):\n", tot_pics_added);
for i_stim_cat = 1:numel(all_categ_diff_feats_cell)
    categ_diff_feats = all_categ_diff_feats_cell{i_stim_cat};
    fprintf('%d pictures added with diff feats rule for category %s\n', numel(categ_diff_feats.pic_ids), categ_diff_feats.category)
end

end