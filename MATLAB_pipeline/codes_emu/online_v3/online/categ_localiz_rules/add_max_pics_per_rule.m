function [sel_cat_hist_table, pic_id_list, experiment] = add_max_pics_per_rule(experiment, sel_cat_hist_table, ...
                                                 stim_cat, max_pics_per_rule, select_status)
% used with diff, all_but_x rule.
added_pics_count = 0;
pic_id_list = [];
pic_ctr = 1;
b_use_visited = false;
rand_idxs = randperm(height(sel_cat_hist_table));
while added_pics_count < max_pics_per_rule
    if pic_ctr > height(sel_cat_hist_table)
        pic_ctr = 1;
        b_use_visited = true;
        if sum(cellfun(@numel,sel_cat_hist_table.Unused)) == 0
            break;
        end
    end
    pic_idx = rand_idxs(pic_ctr);
    pic_ctr = pic_ctr + 1;
    pic_feats = sel_cat_hist_table(pic_idx,:);
    if pic_feats.Visited{1} && ~b_use_visited
        continue;
    else
        pic_feats.Visited = {true};
    end        

    % Add max_pics_per_rule pics to the list
    if numel(pic_feats.Unused{1}) > 0           
    
        % This piece of code just converts the
        % indices to match indices in experiment.ImageNames
        % table 
        pic_id = find(cellfun(@(x)strcmp(x, sprintf("%s-%d.jpg",lower(stim_cat{1}), ...
                                         pic_feats.Unused{1}(1))), ...
                                         experiment.ImageNames.name));
        experiment.ImageNames(pic_id,:).selectable = select_status;
        pic_id_list = [pic_id_list; pic_id];

        added_pics_count = added_pics_count + 1;
        pic_feats.Used = {[pic_feats.Used{1}; pic_feats.Unused{1}(1)]};
        pic_feats.Unused = {setdiff(pic_feats.Unused{1}, ...
                                     pic_feats.Unused{1}(1))};
            
        sel_cat_hist_table(pic_idx,:) = pic_feats;
    else
        continue;
    end            
end

end