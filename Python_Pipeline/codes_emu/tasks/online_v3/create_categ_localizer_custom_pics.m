function create_categ_localizer_custom_pics(pics_path)
xls_file = dir(fullfile(pics_path,'*.xlsx'));
xls_path = fullfile(pics_path, xls_file.name);
max_stim_cat_cols = 5;
custom_pics_per_cat = 18;
custom_pics_path = [pics_path filesep 'custom_pics'];
img_info_table = readtable(xls_path, 'Sheet', 'AllCat');
uniq_stim_cats = unique(img_info_table.StimCat);
all_categ_info_cell = cell(numel(uniq_stim_cats), 1);
all_categ_info_cell2 = cell(numel(uniq_stim_cats), 1);
for i_stim_cat = 1:numel(uniq_stim_cats)
    stim_category = uniq_stim_cats(i_stim_cat);
    stim_category = stim_category{1};
    stim_cat_cols = img_info_table.Properties.VariableNames( ...
            startsWith(img_info_table.Properties.VariableNames, stim_category));
    if numel(stim_cat_cols) >= max_stim_cat_cols
        stim_cat_cols= stim_cat_cols(1:max_stim_cat_cols);
    end
    stim_cat_cols{end+1} = 'TaskGroup';
    stim_cat_cols{end+1} = 'StimNum';
    cat_info_table = img_info_table(ismember(img_info_table.StimCat, stim_category), stim_cat_cols);
    % task_img_info_table = cat_info_table;
    % Other taskgroup
    task_img_info_table = cat_info_table(ismember(cat_info_table.TaskGroup, 'Other'), :);
    [u,~,IC] = unique(task_img_info_table(:,1:end-2),'rows','stable');
    categ_info_cell = cell(height(u), 1);
    fprintf("Only %d unique items in %s\n", height(u), stim_category);
    if numel(u) < custom_pics_per_cat
        fprintf("Only %d unique items in %s\n", numel(u), stim_category);
    end
    recdeccount = [];
    othercount = [];
    cat_cust_pics_count = 0;
    % Get rows that match the unique rows
    for iu = 1: height(u)
%         disp(u(iu,:));
        gp_rows = task_img_info_table(find(IC==iu),:);
        recdeccount = [recdeccount sum(ismember(gp_rows.TaskGroup, 'RecDec'))];
        othercount  = [othercount sum(ismember(gp_rows.TaskGroup, 'Other'))];
        categ_info_cell{iu} = gp_rows;
%         disp(height(gp_rows))
%         disp(gp_rows);
        if cat_cust_pics_count <= custom_pics_per_cat && height(gp_rows) >=2
            pic_row = gp_rows(1,:);
            pic_path = fullfile(pics_path, lower(stim_category), sprintf("%s-%d.jpg",lower(stim_category), pic_row.StimNum));
            pic_path = fullfile(pic_path);
            copyfile(pic_path, custom_pics_path);
            cat_cust_pics_count = cat_cust_pics_count + 1;
        end
    end
    u = addvars(u, recdeccount', othercount', 'NewVariableNames', {'RecDecCount', 'OtherCount'});
    all_categ_info_cell{i_stim_cat} = categ_info_cell;
    all_categ_info_cell2{i_stim_cat} = u;
end

save('all_categ_info.mat', "all_categ_info_cell", '-mat')
save('all_categ_inf.mat', "all_categ_info_cell2", '-mat')