function categ_localiz_history = update_categ_localiz_history(experiment, ...
                                                              categ_localiz_history, ...
                                                              pics_removed)
% pics_removed = pics_removed(ismember(pics_removed, curr_fetched));

for pic_rm_id=pics_removed'
    pic_name = split(experiment.ImageNames(pic_rm_id,:).name,'-');
    pic_cat_name = pic_name{1};
    pic_id = str2num(pic_name{2}(1:end-4));
    for cat_id=1:numel(categ_localiz_history)
        cat_info = categ_localiz_history{cat_id};
        cat_name = lower(cat_info.category);        
        if strcmp(cat_name, ...
                  pic_cat_name)
            cat_history = cat_info.history;
            for feat_id=1:height(cat_history)
                if any(cat_history(feat_id,:).Used{1} == pic_id)
                    categ_localiz_history{cat_id}.history(feat_id,:).Unused = {[cat_history(feat_id,:).Unused{1};pic_id]};
                    categ_localiz_history{cat_id}.history(feat_id,:).Used = {setdiff(cat_history(feat_id,:).Used{1},pic_id)};
                end
                
            end
        end
    end
end

end