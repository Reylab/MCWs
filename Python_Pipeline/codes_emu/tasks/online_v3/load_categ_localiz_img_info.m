function img_info_table = load_categ_localiz_img_info(Path_pics)
xls_file = dir(fullfile(Path_pics,'*.xlsx'));

% create_categ_localizer_custom_pics(xls_file, Path_pics);

img_info_table = readtable(fullfile(Path_pics, xls_file.name), 'Sheet', 'AllCat');
img_info_table = img_info_table(ismember(img_info_table.TaskGroup, 'Other'), :);

img_names_list = cellfun(@(x, y)[lower(x) '-' num2str(y) '.jpg'], img_info_table.StimCat, ...
                                 num2cell(img_info_table.StimNum), 'UniformOutput', false);
img_info_table = [table(img_names_list, 'VariableNames', {'name'}) img_info_table];

% use name column to merge with ImageNames
% idx = ismember(img_info_table.name, ImageNames.name);
% img_info = img_info_table(idx,3:end);

end