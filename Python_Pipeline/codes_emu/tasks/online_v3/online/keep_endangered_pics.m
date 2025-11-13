function [next_scr_pics, pics_removed, categ_localiz_history] = keep_endangered_pics(experiment, scr_id, next_available_pics, ...
                                                   unused_available_pics, extra_used_stims, num_datat_best, selected2notremove, ...
                                                   selected2explore_cell,...
                                                   fetched_pics_cell, scr_config, ...
                                                   exp_time_taken, b_use_all_pics, categ_localiz_history)
if ~exist("categ_localiz_history", "var")
    categ_localiz_history = {};
end
pics_removed = [];
if b_use_all_pics
    pics_not_to_be_shown = next_available_pics;
    others_not_to_be_shown = extra_used_stims;
    num_not_to_be_shown = numel(pics_not_to_be_shown);
else
    pics_not_to_be_shown = next_available_pics(experiment.NPICS(scr_id+1) + 1:end);
    selected2notremove_not_to_be_shown = pics_not_to_be_shown(ismember(pics_not_to_be_shown, selected2notremove));
    others_not_to_be_shown             = pics_not_to_be_shown(ismember(pics_not_to_be_shown, extra_used_stims));
    num_not_to_be_shown = numel(selected2notremove_not_to_be_shown) + ...
                      numel(others_not_to_be_shown);
end

next_scr_pics = [];
if ~b_use_all_pics
    next_scr_pics = next_available_pics(1:experiment.NPICS(scr_id+1));
end

if num_not_to_be_shown > 0
    
    b_last_block = numel(experiment.NPICS) == scr_id+1;
    
    [pics_to_keep, pics_removed, categ_localiz_history] = choose_pics_to_keep(experiment, next_scr_pics,...
                                                       scr_config.ISI, scr_id+1, ...
                                                       pics_not_to_be_shown,...
                                                       unused_available_pics,...
                                                       selected2explore_cell, ...
                                                       fetched_pics_cell, ...
                                                       others_not_to_be_shown, ...
                                                       b_last_block, exp_time_taken, ...
                                                       categ_localiz_history, num_datat_best);
    if numel(pics_to_keep) > 0
        next_scr_pics = [next_scr_pics(:); pics_to_keep(:)];
    end    
    
end

end