function [scr_fetched_cell, experiment, categ_localiz_history]  = get_categ_localiz_pics( ...
                                                      n_scr, experiment, img_info_table, ...
                                                      categ_localiz_history, selected2explore)
scr_fetched_cell = {};
tot_pics_added = 0;
max_pics_per_rule = 2;
cat_loc_rules = struct('SAME', 1, 'DIFF', 2, 'ALL_BUT_X', 3);
cat_loc_rules_start = tic();

disp("same features rule:")
same_feats_rule_start = tic();
[same_feats_list, categ_localiz_history, experiment] = cat_loc_same_feats_rule( ...
                                                       selected2explore, experiment, ...
                                                       categ_localiz_history, img_info_table, ...
                                                       max_pics_per_rule, cat_loc_rules.SAME);
tot_pics_added = tot_pics_added + sum(cellfun(@(x)numel(x.pic_ids), same_feats_list));

time_taken = toc(same_feats_rule_start);
time_taken_mins = floor(time_taken / 60);
time_taken_secs = mod(time_taken, 60);
fprintf("same features rule executed for screening %d in %d mins %0.0f seconds\n", ...
                                    n_scr, time_taken_mins, time_taken_secs);
disp("------------------------------------------------------------")
scr_fetched_cell{end+1}.rule_name = 'same_feats';
scr_fetched_cell{end}.categ_info = same_feats_list;

disp("different features rule:")
diff_feats_rule_start = tic();
[diff_feats_list, categ_localiz_history, experiment] = cat_loc_diff_feats_rule( ...
                                                       selected2explore, experiment, ...
                                                       categ_localiz_history, img_info_table, ...
                                                       max_pics_per_rule, cat_loc_rules.DIFF);
tot_pics_added = tot_pics_added + sum(cellfun(@(x)numel(x.pic_ids), diff_feats_list));

time_taken = toc(diff_feats_rule_start);
time_taken_mins = floor(time_taken / 60);
time_taken_secs = mod(time_taken, 60);
fprintf("different features rule executed for screening %d in %d mins %0.0f seconds\n", ...
                                    n_scr, time_taken_mins, time_taken_secs);
disp("------------------------------------------------------------")
scr_fetched_cell{end+1}.rule_name = 'diff_feats';
scr_fetched_cell{end}.categ_info = diff_feats_list;

max_feat_length = 0;
for cat_idx = 1:numel(categ_localiz_history)
    categ_hist = categ_localiz_history{cat_idx};
    cat_sz = size(categ_hist.history);
    if cat_sz > max_feat_length
        max_feat_length = cat_sz(2) - 3; % last 3 columns are not features
    end
end
max_feat_length = 5;
nonfixed_feats_count = 1;
while nonfixed_feats_count < max_feat_length
    fprintf("all_but_%d rule:\n", nonfixed_feats_count);
    all_but_x_rule_start = tic();
    [all_but_x_list, categ_localiz_history, experiment] = cat_loc_all_but_x_rule( ...
                                                          selected2explore, experiment, ...
                                                          categ_localiz_history, img_info_table, ...
                                                          nonfixed_feats_count, max_pics_per_rule, ...
                                                          cat_loc_rules.ALL_BUT_X);
    tot_pics_added = tot_pics_added + sum(cellfun(@(x)numel(x.pic_ids), all_but_x_list));
    
    time_taken = toc(all_but_x_rule_start);
    time_taken_mins = floor(time_taken / 60);
    time_taken_secs = mod(time_taken, 60);
    fprintf("all_but_%d rule executed for screening %d in %d mins %0.0f seconds\n", ...
                                    nonfixed_feats_count, n_scr, time_taken_mins, time_taken_secs);
    disp("------------------------------------------------------------")
    scr_fetched_cell{end+1}.rule_name = sprintf("all_but_%d", nonfixed_feats_count);
    scr_fetched_cell{end}.categ_info = all_but_x_list;
    nonfixed_feats_count = nonfixed_feats_count + 1;
end
time_taken = toc(cat_loc_rules_start);
time_taken_mins = floor(time_taken / 60);
time_taken_secs = mod(time_taken, 60);
fprintf("all rules executed for screening %d (tot_pics_added=%d) in %d mins %0.0f seconds\n", ...
                                    n_scr, tot_pics_added, time_taken_mins, time_taken_secs);
end