function new_data = merge_clusters(data, merge_list, varargin)
% MERGE_AND_REPORT - Merge specified clusters and save the result
%
% Usage:
%   new_data = merge_and_report(data, merge_list, ...)
%
% Inputs:
%   data       - struct with cluster_class, spikes, inspk (or filename string to load)
%   merge_list - vector of cluster IDs to merge, e.g., [1, 2, 3]
%                Must have length >= 2 and < total number of clusters
%
% Optional Parameters:
%   'overwrite', true/false  - If true, backup originals and save to original filename
%                              If false, save with suffix _merge[IDs] (default: false)
%
% Outputs:
%   new_data - merged data structure (with merge provenance in new_data.par)
%
% Examples:
%   % Inspect result without saving:
%   new_data = merge_and_report(data, [2, 4], 'test', true);
%
%   % Save with suffix (e.g. times_ch01_merge[24].mat):
%   merge_and_report(data, [2, 4], 'test', false);
%
%   % Overwrite original file (backs up first):
%   merge_and_report(data, [2, 4], 'test', false, 'overwrite', true);

    p = inputParser;
    addRequired(p, 'data');
    addRequired(p, 'merge_list', @(x) isnumeric(x) && isvector(x) && length(x) >= 2);
    addParameter(p, 'overwrite', false, @islogical);
    parse(p, data, merge_list, varargin{:});

    %  Locate active times directory ---
    dates_times = dir(fullfile(pwd, 'times*'));
    dates_times = dates_times([dates_times.isdir]);
    if isempty(dates_times), error('No times folders found.'); end
    [~, idx_t] = max([dates_times.datenum]);
    active_times_dir = fullfile(pwd, dates_times(idx_t).name);

    %  Load data if a filename was passed ---
    if ischar(data) || isstring(data)
        filename = char(data);
        if ~exist(filename, 'file')
            filename = fullfile(active_times_dir, filename);
        end
        if ~exist(filename, 'file')
            error('File not found: %s', filename);
        end
        data = load(filename);
        [pathstr, name, ext] = fileparts(filename);
        if isempty(pathstr), pathstr = active_times_dir; end
    else
        if isfield(data, 'filename') && ~isempty(data.filename)
            [pathstr, name, ext] = fileparts(data.filename);
            if isempty(pathstr), pathstr = active_times_dir; end
            if isempty(ext), ext = '.mat'; end
        elseif isfield(data, 'fullpath') && ~isempty(data.fullpath)
            [pathstr, name, ext] = fileparts(data.fullpath);
        else
            pathstr = active_times_dir;
            name    = 'merged_data';
            ext     = '.mat';
        end
    end

    %  Validate data structure 
    if ~isstruct(data) || ~isfield(data, 'cluster_class') || ~isfield(data, 'spikes') || ~isfield(data, 'inspk')
        error('data must be a struct with fields: cluster_class, spikes, inspk');
    end

    %  Validate merge_list 
    merge_list      = unique(merge_list(:))';
    cluster_ids     = data.cluster_class(:, 1);
    unique_clusters = unique(cluster_ids);

    if length(merge_list) < 2
        error('merge_list must contain at least 2 clusters to merge');
    end
    if length(merge_list) >= length(unique_clusters)
        error('merge_list must be less than total number of clusters (%d)', length(unique_clusters));
    end
    if ~all(ismember(merge_list, unique_clusters))
        missing = merge_list(~ismember(merge_list, unique_clusters));
        error('Clusters not found in data: %s', mat2str(missing));
    end

    %  Perform merge 
    fprintf('Merging clusters: %s\n', mat2str(merge_list));

    % Copy all relevant fields
    new_data = struct();
    fields_to_copy = {'cluster_class','spikes','inspk','par','spikes_all','index_all', ...
                      'forced','rescue_mask','cluster_class_pre_rescue','mask_nonart', ...
                      'mask_non_quarantine','mask_taskspks','filename','fullpath'};
    for fi = 1:numel(fields_to_copy)
        f = fields_to_copy{fi};
        if isfield(data, f)
            new_data.(f) = data.(f);
        end
    end

    % Assign all source clusters to the target (first ID in merge_list)
    target_cluster  = merge_list(1);
    source_clusters = merge_list(2:end);
    new_cluster_ids = new_data.cluster_class(:, 1);
    for s = source_clusters
        new_cluster_ids(new_cluster_ids == s) = target_cluster;
    end

    % Renumber to contiguous 0-indexed IDs
    unique_new  = unique(new_cluster_ids);
    cluster_map = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    for i = 1:length(unique_new)
        cluster_map(unique_new(i)) = i - 1;
    end
    renumbered = arrayfun(@(x) cluster_map(x), new_cluster_ids);
    new_data.cluster_class(:, 1) = renumbered;

    final_clusters = unique(renumbered);
    fprintf('  Before: %d clusters  →  After: %d clusters\n', ...
            length(unique_clusters), length(final_clusters));

    %  Record merge provenance in par ---
    if ~isfield(new_data, 'par') || isempty(new_data.par)
        new_data.par = struct();
    end
    new_merged_id = cluster_map(target_cluster);
    merge_event = struct( ...
        'timestamp',         datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
        'original_clusters', merge_list, ...
        'merged_into_id',    new_merged_id, ...
        'cluster_map_before', unique_clusters(:)', ...
        'cluster_map_after',  final_clusters(:)');
    if isfield(new_data.par, 'merge_history') && ~isempty(new_data.par.merge_history)
        new_data.par.merge_history(end+1) = merge_event;
    else
        new_data.par.merge_history = merge_event;
    end
    new_data.par.is_merged       = true;
    new_data.par.merge_list_last = merge_list;

    %  Save 
    merge_suffix = sprintf('_merge%s', strrep(mat2str(merge_list), ' ', ''));

    fprintf('Saving merged data...\n');

    if p.Results.overwrite
        % Back up the original before overwriting
        backup_dir = fullfile(pathstr, 'backup_originals');
        if ~exist(backup_dir, 'dir'), mkdir(backup_dir); end
        original_mat = fullfile(pathstr, [name, ext]);
        if exist(original_mat, 'file')
            try
                copyfile(original_mat, fullfile(backup_dir, [name, ext]));
                fprintf('  Backed up: %s\n', original_mat);
            catch ME_bak
                warning('Failed to backup %s: %s', original_mat, ME_bak.message);
            end
        end
        out_base = name;
        out_mat  = original_mat;
    else
        merged_dir = fullfile(pathstr, 'merged');
        if ~exist(merged_dir, 'dir'), mkdir(merged_dir); end
        out_base = [name, merge_suffix];
        out_mat  = fullfile(merged_dir, [out_base, ext]);
    end

    new_data.filename = out_base;
    try
        save(out_mat, '-struct', 'new_data');
        fprintf('  Saved: %s\n', out_mat);
    catch ME_save
        warning('Failed to save merged data: %s', ME_save.message);
    end
    

    fprintf('Merge complete\n');
end