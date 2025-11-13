function [scr_fetched_cell, experiment] = get_same_unit_categ_pics(n_scr, experiment, tbl_unused_pics, ...
                                                     selected2explore, selected2notremove, new_pics2load)
scr_fetched_cell = {};
same_units = cell(1,1);
same_units{end}.category = '';
same_units{end}.pic_ids = [];

same_categories = cell(1,1);
same_categories{end}.category = '';
same_categories{end}.pic_ids = [];

for c = selected2explore(:)'
    same_category = [];
    same_unit = find(cellfun(@(x)strcmp(x,experiment.ImageNames.concept_name{c}),tbl_unused_pics.concept_name));

    for i = 1:height(tbl_unused_pics)
        if tbl_unused_pics.concept_number(i) ~= 1 || any(i==same_unit)
            continue
        end
        this_categories = tbl_unused_pics.concept_categories{i};
        for xi = 1:numel(this_categories)
            share_category = any(strcmp(experiment.ImageNames.concept_categories{c},this_categories{xi}));
            if share_category
                same_category(end+1) = i;
                break
            end
        end
    end

    % This piece of code just converts the same_unit
    % indices to match indices in experiment.ImageNames
    % table
    if 0 < numel(same_unit)
        su_idx_list = [];
        for su_idx = 1:length(same_unit)
            su_idx_list = [su_idx_list; find(cellfun(@(x)strcmp(x, tbl_unused_pics.name(same_unit(su_idx))), experiment.ImageNames.name))];
        end
        same_unit = su_idx_list;
    end
    % This piece of code just converts the same_category
    % indices to match indices in experiment.ImageNames
    % table
    if 0 < numel(same_category)
        sc_idx_list = [];
        for sc_idx = 1:length(same_category)
            sc_idx_list = [sc_idx_list; find(cellfun(@(x)strcmp(x, tbl_unused_pics.name(same_category(sc_idx))), experiment.ImageNames.name))];
        end
        same_category = sc_idx_list;
    end

    same_units{end}.pic_ids = [same_units{end}.pic_ids ; same_unit(:)];
    same_categories{end}.pic_ids = [same_categories{end}.pic_ids ; same_category(:)];
    fprintf('\n%d pictures same_unit to %s\n', numel(same_unit), experiment.ImageNames.concept_name{c})
    for pic_idx=same_unit
        fprintf('%s ', experiment.ImageNames.name{pic_idx})
    end
    fprintf('\n%d pictures same_category to %s\n', numel(same_category), experiment.ImageNames.concept_name{c})
    for pic_idx=same_category
        fprintf('%s ', experiment.ImageNames.name{pic_idx})
    end
end

fprintf('%d pictures same_unit in total\n', numel(same_units{end}.pic_ids))
fprintf('%d pictures same_category  in total\n', numel(same_categories{end}.pic_ids))

experiment.ImageNames.selectable(same_units{end}.pic_ids) = 1; % to color magenta in best resp popup
experiment.ImageNames.selectable(same_categories{end}.pic_ids) = 2; % to color blue in best resp popup

% num_pics_to_add = min(experiment.NPICS(n_scr+1)-numel(selected2notremove),new_pics2load(n_scr));
% % Show a dialogue if there's anything in same_categories
% if numel(same_units{end}.pic_ids) + numel(same_categories{end}.pic_ids) >= num_pics_to_add && ...
%                        numel(same_categories{end}.pic_ids) > 0
%     choice = keep_same_cat_dlg(n_scr+1, experiment.NPICS(n_scr+1), ...
%             numel(selected2explore), numel(same_units{end}.pic_ids), ...
%             numel(same_categories{end}.pic_ids), ...
%             num_pics_to_add);
%     if choice == 0
%         same_categories{end}.pic_ids = [];
%     end
% end

scr_fetched_cell{1}.rule_name = 'same_units';
scr_fetched_cell{1}.categ_info = same_units;
scr_fetched_cell{2}.rule_name = 'same_categories';
scr_fetched_cell{2}.categ_info = same_categories;
end