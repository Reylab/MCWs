function [new_data, figs, df_metrics, SS] = merge_and_report(data, merge_groups, varargin)
% MERGE_CLUSTERS - Merge specified clusters with optional reporting
%
% Input:
%   data - struct from .mat file with cluster_class, spikes, inspk
%          OR filename string to load .mat file
%   merge_groups - clusters to merge: [1,2,3] or {[1,2], [3,4]}
%
% Optional:
%   'make_plots', true/false (default: false)
%   'save_plots', false (default: false)
%   'calc_metrics', true/false (default: false)
%   'output_file', '' (default: '' - no save)
%
% Output:
%   new_data - merged data structure
%   figs - figure handles (empty if no plots)
%   df_metrics - metrics table (empty if no metrics)
%   SS - silhouette matrix (empty if no metrics)

% Parse inputs
p = inputParser;
addParameter(p, 'make_plots', false, @islogical);
addParameter(p, 'save_plots', false, @islogical);
addParameter(p, 'calc_metrics', false, @islogical);
addParameter(p, 'output_file', '', @ischar);
addParameter(p, 'test', true, @islogical);            % save a test merge variant (default true)
addParameter(p, 'apply_merge', false, @islogical);    % accept/apply previously saved test merge
addParameter(p, 'backup_original', false, @islogical);% backup originals before overwriting when applying
addParameter(p, 'test_suffix', '_testMerge', @ischar);% suffix used for test files
addParameter(p, 'outdir', '', @ischar);               % optional explicit output directory
addParameter(p, 'parallel', false, @islogical);       % run merges across channels in parallel
addParameter(p, 'n_workers', 0, @isscalar);           % number of workers for parallel (0 = auto)
addParameter(p, 'apply_report', false, @islogical);   % accept/apply previously saved test report files
parse(p, varargin{:});

% Handle file input
if isnumeric(data)
    % treat numeric input as channel list (one or more channels)
    channels = data(:)';
    % Build file list for each channel
    file_list = {};
    for c = channels
        % try several patterns used in this repo: chNN, _NN, plain NN
        pats = {sprintf('times_*ch%d*.mat', c), sprintf('times_*_%d*.mat', c), sprintf('times_*%d*.mat', c)};
        found = {};
        for pi = 1:length(pats)
            d = dir(pats{pi});
            if ~isempty(d)
                found = [found, {d.name}]; %#ok<AGROW>
            end
        end
        if isempty(found)
            warning('No file found for channel %d (looked for patterns: %s)', c, strjoin(pats, ', '));
        else
            % use unique and pick first match if multiple
            found = unique(found);
            file_list{end+1} = found{1}; %#ok<AGROW>
        end
    end

    if isempty(file_list)
        error('No files found for provided channel(s)');
    end

    % Run merge_and_report on each found file (optionally in parallel)
    if p.Results.parallel
        if p.Results.n_workers > 0
            n_workers = p.Results.n_workers;
        else
            n_workers = feature('numcores');
        end
        if isempty(gcp('nocreate'))
            parpool(n_workers);
        end
        parfor i = 1:length(file_list)
            try
                merge_and_report(file_list{i}, merge_groups, varargin{:});
            catch MEp
                fprintf('Error processing %s in parallel: %s\n', file_list{i}, MEp.message);
            end
        end
    else
        for i = 1:length(file_list)
            merge_and_report(file_list{i}, merge_groups, varargin{:});
        end
    end

    % we handled everything for numeric input
    new_data = struct(); figs = []; df_metrics = []; SS = [];
    return;
elseif ischar(data) || isstring(data)
    filename = data;
    data = load(filename);
    fprintf('Loaded: %s\n', filename);
else
    filename = '';
end

% Determine original file base name/path to drive saved filenames
if ~isempty(filename)
    [pathstr, name, ext] = fileparts(filename);
    if isempty(pathstr), pathstr = pwd; end
else
    if isfield(data, 'filename') && ~isempty(data.filename)
        [pathstr, name, ext] = fileparts(data.filename);
        if isempty(pathstr), pathstr = pwd; end
    else
        pathstr = pwd; name = 'merged'; ext = '.mat';
    end
end

% Initialize outputs
figs = [];
df_metrics = [];
SS = [];

% Handle different input formats for merge_groups
if isnumeric(merge_groups)
    % Single merge group provided as vector
    merge_groups = {merge_groups};
elseif ~iscell(merge_groups)
    error('merge_groups must be a cell array or numeric vector');
end

% Create deep copy
new_data = struct();
new_data.cluster_class = data.cluster_class;
if isfield(data, 'spikes')
    new_data.spikes = data.spikes;
end
if isfield(data, 'inspk')
    new_data.inspk = data.inspk;
end
if isfield(data, 'par')
    new_data.par = data.par;
end

cluster_class = new_data.cluster_class;
cluster_ids = cluster_class(:, 1);

original_clusters = unique(cluster_ids);
fprintf('Merging: %d clusters → ', length(original_clusters));

% Apply merges
if ~isempty(merge_groups)
    for m = 1:length(merge_groups)
        merge_group = merge_groups{m};
        
        if length(merge_group) < 2
            continue;
        end
        
        target_cluster = merge_group(1);
        source_clusters = merge_group(2:end);
        
        for s = 1:length(source_clusters)
            source_cluster = source_clusters(s);
            mask = (cluster_ids == source_cluster);
            cluster_ids(mask) = target_cluster;
        end
    end
end

% Renumber clusters to be contiguous (0 to N)
unique_clusters = unique(cluster_ids);
cluster_mapping = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
for i = 1:length(unique_clusters)
    cluster_mapping(unique_clusters(i)) = i - 1; % Start from 0
end

% Apply renumbering
new_cluster_ids = arrayfun(@(x) cluster_mapping(x), cluster_ids);
cluster_class(:, 1) = new_cluster_ids;
new_data.cluster_class = cluster_class;

final_clusters = unique(new_cluster_ids);
fprintf('%d clusters\n', length(final_clusters));

% Show what happened
if length(original_clusters) > length(final_clusters)
    fprintf('  Merged: ');
    for i = 1:length(merge_groups)
        fprintf('[%s] ', num2str(merge_groups{i}));
    end
    fprintf('\n');
end

% Compute metrics and plots if requested
if p.Results.calc_metrics || p.Results.make_plots
    fprintf('Computing metrics...\n');

    % Choose output directory for saved figures/metrics
    outdir = pathstr;
    if ~isempty(p.Results.outdir)
        outdir = p.Results.outdir;
    end

    % If running in "test" mode, compute metrics/plots using a temporary
    % filename that includes the test suffix so saved figures/plots include it.
    if p.Results.test
        temp_data = new_data;
        temp_data.filename = sprintf('%s%s', name, p.Results.test_suffix);
        plot_params = struct('outdir', outdir, 'test', p.Results.test, 'test_suffix', p.Results.test_suffix, ...
            'apply_report', p.Results.apply_report, 'backup_original', p.Results.backup_original, 'save_figs', p.Results.save_plots);
        [df_metrics, SS, figs] = compute_cluster_metrics(temp_data, ...
            'make_plots', p.Results.make_plots, ...
            'save_plots', p.Results.save_plots, ...
            'calc_metrics', p.Results.calc_metrics, ...
            'plot_params', plot_params);

        % Save the merged data as a test file next to the original
        test_file = fullfile(outdir, sprintf('%s%s%s', name, p.Results.test_suffix, ext));
        try
            save(test_file, '-struct', 'new_data');
            fprintf('Saved test merge file: %s\n', test_file);
        catch ME
            fprintf('Failed saving test merge file %s: %s\n', test_file, ME.message);
        end

        % Save metrics/SS for test
        if p.Results.calc_metrics && ~isempty(df_metrics)
            test_metrics_file = fullfile(outdir, sprintf('%s%s_metrics%s', name, p.Results.test_suffix, ext));
            try
                save(test_metrics_file, 'df_metrics', 'SS');
                fprintf('Saved test metrics: %s\n', test_metrics_file);
            catch ME2
                fprintf('Failed saving test metrics %s: %s\n', test_metrics_file, ME2.message);
            end
        end

    else
        % Not test mode: compute metrics/plots using the final name
        new_data.filename = name;
        plot_params = struct('outdir', outdir, 'test', p.Results.test, 'test_suffix', p.Results.test_suffix, ...
            'apply_report', p.Results.apply_report, 'backup_original', p.Results.backup_original, 'save_figs', p.Results.save_plots);
        [df_metrics, SS, figs] = compute_cluster_metrics(new_data, ...
            'make_plots', p.Results.make_plots, ...
            'save_plots', p.Results.save_plots, ...
            'calc_metrics', p.Results.calc_metrics, ...
            'plot_params', plot_params);
    end
end

% Save output if requested

% If the user requested to "apply" the merge (accept the test merge), move
% the test files into place (optionally backing up originals). This will
% replace original .mat, metrics and any saved figure files that use the
% filename prefix.
if p.Results.apply_merge
    % Determine target directory/base
    target_dir = pathstr;
    if ~isempty(p.Results.output_file)
        [target_dir_tmp, ~, ~] = fileparts(p.Results.output_file);
        if ~isempty(target_dir_tmp), target_dir = target_dir_tmp; end
    end

    % Find test files produced earlier
    test_pattern = fullfile(outdir, [name p.Results.test_suffix '*']);
    test_files = dir(test_pattern);

    if isempty(test_files)
        fprintf('No test files found to apply for "%s" (expected pattern %s)\n', name, test_pattern);
    else
        % Optionally create backup directory
        if p.Results.backup_original
            backup_dir = fullfile(target_dir, 'backup_originals');
            if ~exist(backup_dir, 'dir')
                mkdir(backup_dir);
            end
        end

        for k = 1:length(test_files)
            tf = test_files(k).name;
            src = fullfile(outdir, tf);
            dest_name = strrep(tf, p.Results.test_suffix, '');
            dest = fullfile(target_dir, dest_name);

            % If dest exists, handle backup or deletion
            if exist(dest, 'file')
                if p.Results.backup_original
                    try
                        movefile(dest, fullfile(backup_dir, dest_name));
                        fprintf('Backed up original: %s -> %s\n', dest, backup_dir);
                    catch
                        fprintf('Warning: failed to back up %s\n', dest);
                    end
                else
                    try
                        delete(dest);
                    catch
                        % ignore
                    end
                end
            end

            % Move test file into place (rename by removing suffix)
            try
                movefile(src, dest);
                fprintf('Applied: %s -> %s\n', src, dest);
            catch ME3
                fprintf('Failed to move %s -> %s: %s\n', src, dest, ME3.message);
            end
        end
    end
end

fprintf('✅ Merge complete!\n');
end