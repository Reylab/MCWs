function [all_categ_same_feats_cell, categ_localiz_history, experiment] = cat_loc_same_feats_rule(selected2explore, experiment, ...
                                                   categ_localiz_history, ...
                                                   img_info_table, max_pics_per_rule, select_status)
all_categ_same_feats_cell = cell(numel(categ_localiz_history), 1);
for i_stim_cat = 1:numel(categ_localiz_history)
    all_categ_same_feats_cell{i_stim_cat}.category = categ_localiz_history{i_stim_cat}.category; 
    all_categ_same_feats_cell{i_stim_cat}.pic_ids = [];
end

for c = selected2explore(:)'
    % Get row for img_info_table for the pic
    pic_row_idx = find(cellfun(@(x)strcmp(x,experiment.ImageNames.name{c}), ...
                                   img_info_table.name));
    pic_info = img_info_table(pic_row_idx, :);
    if height(pic_info) == 1
        stim_cat = pic_info.StimCat;
    else
        continue;
    end
    cat_id = find(cellfun(@(x)strcmp(x.category, stim_cat),all_categ_same_feats_cell));
    categ_same_feats = all_categ_same_feats_cell{cat_id};
    cat_idx = find(cellfun(@(x)strcmp(x.category, stim_cat),categ_localiz_history));
    cat_history = categ_localiz_history{cat_idx}.history;
    % Get columns that start with stim_cat
    stim_cat_cols = img_info_table.Properties.VariableNames( ...
        startsWith(img_info_table.Properties.VariableNames, stim_cat));
    if numel(stim_cat_cols) >= 5
        stim_cat_cols= stim_cat_cols(1:5);
    end

    % Get all pics that have the same features as pic_info
    same_feats_idx = true(size(cat_history,1),1);
    for stim_cat_col = stim_cat_cols
        same_feats_idx = same_feats_idx & cellfun(@(x)strcmp(x,pic_info.(stim_cat_col{1}){1}), ...
                                                    cat_history.(stim_cat_col{1}));
    end
    same_feats = cat_history(same_feats_idx,:);
    if same_feats.Visited{1}
        continue;
    end
    same_feats.Visited = {true};

    % Add max_pics_per_rule pics to the list
    if numel(same_feats.Unused{1}) >= max_pics_per_rule
        same_unit_to_add = same_feats.Unused{1}(1:max_pics_per_rule);        
    elseif numel(same_feats.Unused{1}) > 0
        same_unit_to_add = same_feats.Unused{1}; 
    else
        same_unit_to_add = [];
        continue;
    end
    same_feats.Used = {[same_feats.Used{1}; same_unit_to_add]};
    same_feats.Unused = {setdiff(same_feats.Unused{1}, same_unit_to_add)};

    % Update categ_localiz_history
    categ_localiz_history{cat_idx}.history(same_feats_idx,:) = same_feats;

    % This piece of code just converts the same_unit
    % indices to match indices in experiment.ImageNames
    % table
    su_idx_list = [];
    str_pics_added = "";
    if 0 < numel(same_unit_to_add)
        for su_idx = 1:numel(same_unit_to_add)
            pic_id = find(cellfun(@(x)strcmp(x, sprintf("%s-%d.jpg",lower(stim_cat{1}), ...
                                             same_unit_to_add(su_idx))), ...
                                             experiment.ImageNames.name));
            experiment.ImageNames(pic_id,:).selectable = select_status;
            str_pics_added = sprintf("%s %s", str_pics_added, experiment.ImageNames(pic_id,:).name{1});
            su_idx_list = [su_idx_list; pic_id];
        end
    end
    categ_same_feats.pic_ids = [categ_same_feats.pic_ids; su_idx_list(:)];
    
    all_categ_same_feats_cell{cat_id} = categ_same_feats;
    fprintf('%d pictures with same features as %s added (%s)\n', ...
            numel(same_unit_to_add), experiment.ImageNames.name{c}, str_pics_added)
    % disp(same_feats)
end
tot_pics_added = sum(cellfun(@(x)numel(x.pic_ids), all_categ_same_feats_cell));
fprintf("same features rule summary(tot pics added = %d):\n", tot_pics_added);
for i_stim_cat = 1:numel(all_categ_same_feats_cell)
    categ_same_feats = all_categ_same_feats_cell{i_stim_cat};
    fprintf('%d pictures added with same feats rule for category %s\n', ...
            numel(categ_same_feats.pic_ids), categ_same_feats.category)
end

end