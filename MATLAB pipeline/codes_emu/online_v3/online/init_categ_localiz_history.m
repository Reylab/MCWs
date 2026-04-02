function [categ_localiz_history, img_info_table] = init_categ_localiz_history(experiment, stims_used, task_group)
    img_info_table = read_img_info_table(experiment, task_group);
    % Get all categories (Adult, Car, House)
    uniq_stim_cats = unique(img_info_table.StimCat);
    categ_localiz_history = cell(numel(uniq_stim_cats), 1);
    for i_stim_cat = 1:numel(uniq_stim_cats)
        stim_cat = uniq_stim_cats(i_stim_cat);
        stim_cat = stim_cat{1};
        % Get columns that start with stim_cat
        stim_cat_cols = img_info_table.Properties.VariableNames( ...
            startsWith(img_info_table.Properties.VariableNames, stim_cat));
        if numel(stim_cat_cols) >= 5
            stim_cat_cols= stim_cat_cols(1:5);
        end
        stim_cat_cols{end+1} = 'name';
        stim_cat_cols{end+1} = 'StimNum';
        cat_info_table = img_info_table(ismember(img_info_table.StimCat, stim_cat), stim_cat_cols);
        [u,~,IC] = unique(cat_info_table(:,stim_cat_cols(1:end-2)),'rows','stable');
        categ_unused_cell = cell(height(u), 1);
        categ_used_cell = cell(height(u), 1);
        categ_visited_cell = cell(height(u), 1);
        % Get rows that match the unique rows
        for iu = 1: height(u)
            % disp(u(iu,:));
            gp_rows = cat_info_table(find(IC==iu),:);
            b_used_ids = ismember(gp_rows.name, experiment.ImageNames(stims_used,:).name);
            categ_unused_cell{iu} = gp_rows.StimNum(~b_used_ids);
            categ_used_cell{iu} = gp_rows.StimNum(b_used_ids);
            categ_visited_cell{iu} = false;
%             if sum(b_used_ids)
%                 categ_visited_cell{iu} = true;
%             else
%                 categ_visited_cell{iu} = false;
%             end
        end
        u = addvars(u, categ_unused_cell, categ_used_cell, categ_visited_cell, 'NewVariableNames', {'Unused', 'Used', 'Visited'});
        categ_localiz_history{i_stim_cat}.category = stim_cat;
        categ_localiz_history{i_stim_cat}.history = u;
    end
end

function img_info_table = read_img_info_table(experiment, task_group)
xls_file = dir(fullfile(experiment.task_pics_folder,'*.xlsx'));

% create_categ_localizer_custom_pics(xls_file, Path_pics);

img_info_table = readtable(fullfile(experiment.task_pics_folder, xls_file.name), 'Sheet', 'AllCat', 'Format', 'auto');
if strcmp(task_group, 'RecDec')
    img_info_table = img_info_table(ismember(img_info_table.TaskGroup, 'RecDec'), :);
elseif strcmp(task_group, 'Other')
    img_info_table = img_info_table(ismember(img_info_table.TaskGroup, 'Other'), :);
end

img_names_list = cellfun(@(x, y)[lower(x) '-' num2str(y) '.jpg'], img_info_table.StimCat, ...
                                 num2cell(img_info_table.StimNum), 'UniformOutput', false);
img_info_table = [table(img_names_list, 'VariableNames', {'name'}) img_info_table];

end