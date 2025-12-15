function [new_data, figs, df_metrics, SS] = merge_and_report(data, merge_list, varargin)
% MERGE_AND_REPORT - Merge specified clusters and optionally compute/save metrics
%
% Usage:
%   [new_data, figs, df_metrics, SS] = merge_and_report(data, merge_list, ...)
%
% Inputs:
%   data         - struct with cluster_class, spikes, inspk (or filename to load)
%   merge_list   - vector of cluster IDs to merge, e.g., [1, 2, 3]
%                  Must have length >= 2 and < total number of clusters
%
% Optional Parameters:
%   'test', true/false        - If true, don't save anything (default: true)
%   'show_figs', true/false   - If true, show figures (default: false)
%   'overwrite', true/false   - If true, backup originals and replace them (default: false)
%                               If false, save with suffix _mergeXYZ
%   'rescue', true/false      - If true, use 'metrics rescue/' folder for metrics (default: false)
%
% Outputs:
%   new_data   - merged data structure
%   figs       - figure handles (if generated)
%   df_metrics - metrics table (if computed)
%   SS         - silhouette matrix (if computed)
%
% Examples:
%   % Test merge (no save, no show):
%   merge_and_report(data, [1, 2, 3], 'test', true);
%
%   % Save with suffix (figures hidden):
%   merge_and_report(data, [1, 2, 3], 'test', false, 'overwrite', false);
%
%   % Overwrite originals with backup:
%   merge_and_report(data, [1, 2, 3], 'test', false, 'overwrite', true);
%
%   % Use rescue folder for metrics:
%   merge_and_report(data, [1, 2, 3], 'test', false, 'rescue', true);

    % Parse inputs
    p = inputParser;
    addRequired(p, 'data');
    addRequired(p, 'merge_list', @(x) isnumeric(x) && isvector(x) && length(x) >= 2);
    addParameter(p, 'test', true, @islogical);
    addParameter(p, 'show_figs', false, @islogical);
    addParameter(p, 'overwrite', false, @islogical);
    addParameter(p, 'exclude_cluster_0', true, @islogical);
    addParameter(p, 'rescue', false, @islogical);
    parse(p, data, merge_list, varargin{:});
    
    % --- Load data if filename provided ---
    if ischar(data) || isstring(data)
        filename = char(data);
        if ~exist(filename, 'file')
            error('File not found: %s', filename);
        end
        data = load(filename);
        [pathstr, name, ext] = fileparts(filename);
        if isempty(pathstr), pathstr = pwd; end
    else
        % Data is struct - try to get filename from it
        filename = '';
        if isfield(data, 'filename') && ~isempty(data.filename)
            [pathstr, name, ext] = fileparts(data.filename);
            if isempty(pathstr), pathstr = pwd; end
        elseif isfield(data, 'fullpath') && ~isempty(data.fullpath)
            [pathstr, name, ext] = fileparts(data.fullpath);
        else
            pathstr = pwd;
            name = 'merged_data';
            ext = '.mat';
        end
    end
    
    % Validate data structure
    if ~isstruct(data) || ~isfield(data, 'cluster_class') || ~isfield(data, 'spikes') || ~isfield(data, 'inspk')
        error('data must be struct with fields: cluster_class, spikes, inspk');
    end
    
    % --- Validate merge_list ---
    merge_list = unique(merge_list(:))'; % ensure unique and row vector
    cluster_ids = data.cluster_class(:, 1);
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
    
    % --- Perform merge ---
    fprintf('Merging clusters: %s\n', mat2str(merge_list));
    
    % Deep copy data
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
    
    % Merge: assign all spikes in merge_list to the first cluster ID
    target_cluster = merge_list(1);
    source_clusters = merge_list(2:end);
    
    new_cluster_ids = new_data.cluster_class(:, 1);
    for s = source_clusters
        mask = (new_cluster_ids == s);
        new_cluster_ids(mask) = target_cluster;
    end
    
    % Renumber to be contiguous (0, 1, 2, ...)
    unique_new = unique(new_cluster_ids);
    cluster_map = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    for i = 1:length(unique_new)
        cluster_map(unique_new(i)) = i - 1; % 0-indexed
    end
    
    renumbered = arrayfun(@(x) cluster_map(x), new_cluster_ids);
    new_data.cluster_class(:, 1) = renumbered;
    
    final_clusters = unique(renumbered);
    fprintf('  Before: %d clusters → After: %d clusters\n', length(unique_clusters), length(final_clusters));
    
    % --- Determine metrics directory based on rescue parameter ---
    if p.Results.rescue
        metrics_dir = fullfile(pathstr, 'metrics rescue');
    else
        metrics_dir = fullfile(pathstr, 'metrics');
    end
    
    if ~exist(metrics_dir, 'dir')
        mkdir(metrics_dir);
    end
    
    % --- Compute metrics and plots ---
    figs = [];
    df_metrics = [];
    SS = [];
    
    % Determine visibility
    if p.Results.show_figs
        vis_str = 'on';
    else
        vis_str = 'off';
    end
    
    % Always compute metrics/plots (but visibility controlled by show_figs)
    fprintf('Computing metrics for merged data...\n');
    
    % Temporarily set filename for plotting
    merge_suffix = sprintf('_merge%s', strrep(mat2str(merge_list), ' ', ''));
    new_data.filename = [name, merge_suffix];
    
    try
        [df_metrics, SS, figs] = compute_cluster_metrics(new_data, ...
            'exclude_cluster_0', p.Results.exclude_cluster_0, ...
            'show_plots', true, ...
            'save_plots', false, ...
            'plot_params', struct('base_name', [name, merge_suffix], 'outdir', metrics_dir));
        
        % Set figure visibility
        for i = 1:length(figs)
            if isgraphics(figs{i})
                set(figs{i}, 'Visible', vis_str);
            end
        end
    catch ME
        warning('Failed to compute metrics: %s', ME.message);
    end
    
    % --- Save logic ---
    if ~p.Results.test
        fprintf('Saving merged data and metrics...\n');
        
        % Determine output filenames
        if p.Results.overwrite
            % Overwrite mode: backup originals, then save to original names
            backup_dir = fullfile(pathstr, 'backup_originals');
            if ~exist(backup_dir, 'dir')
                mkdir(backup_dir);
            end
            
            % Backup original .mat file
            original_mat = fullfile(pathstr, [name, ext]);
            if exist(original_mat, 'file')
                backup_mat = fullfile(backup_dir, [name, ext]);
                try
                    copyfile(original_mat, backup_mat);
                    fprintf('  Backed up: %s → %s\n', original_mat, backup_mat);
                catch ME_bak
                    warning('Failed to backup %s: %s', original_mat, ME_bak.message);
                end
            end
            
            % Backup original metrics from appropriate metrics directory
            original_metrics = fullfile(metrics_dir, [name, '_metrics.mat']);
            if exist(original_metrics, 'file')
                backup_metrics = fullfile(backup_dir, [name, '_metrics.mat']);
                try
                    copyfile(original_metrics, backup_metrics);
                    fprintf('  Backed up: %s → %s\n', original_metrics, backup_metrics);
                catch
                end
            end
            
            % Backup original plots (PNG files) from metrics directory
            plot_pattern = fullfile(metrics_dir, [name, '_plots*.png']);
            plot_files = dir(plot_pattern);
            for k = 1:length(plot_files)
                original_plot = fullfile(metrics_dir, plot_files(k).name);
                backup_plot = fullfile(backup_dir, plot_files(k).name);
                try
                    copyfile(original_plot, backup_plot);
                    fprintf('  Backed up: %s\n', plot_files(k).name);
                catch
                end
            end
            
            % Save to original filenames
            out_mat = original_mat;
            out_metrics = fullfile(metrics_dir, [name, '_metrics.mat']);
            out_base = name;
            
        else
            % Non-overwrite mode: save with _mergeXYZ suffix
            out_base = [name, merge_suffix];
            out_mat = fullfile(pathstr, [out_base, ext]);
            out_metrics = fullfile(metrics_dir, [out_base, '_metrics.mat']);
        end
        
        % Save merged data
        try
            new_data.filename = out_base; % update filename in struct
            save(out_mat, '-struct', 'new_data');
            fprintf('  Saved merged data: %s\n', out_mat);
        catch ME_save
            warning('Failed to save merged data: %s', ME_save.message);
        end
        
        % Save metrics
        if ~isempty(df_metrics)
            try
                metrics_table = df_metrics; % rename for consistency with compute_cluster_metrics
                save(out_metrics, 'metrics_table', 'SS');
                fprintf('  Saved metrics: %s\n', out_metrics);
            catch ME_met
                warning('Failed to save metrics: %s', ME_met.message);
            end
        end
        
        % Save plots to metrics directory
        if ~isempty(figs)
            try
                plot_params = struct('base_name', out_base, 'outdir', metrics_dir);
                save_plot_files(figs, new_data, plot_params);
                fprintf('  Saved %d plot pages to %s\n', length(figs), metrics_dir);
            catch ME_plot
                warning('Failed to save plots: %s', ME_plot.message);
            end
        end
        
    else
        fprintf('Test mode: no files saved\n');
    end
    
    % Close figures if not showing
    if ~p.Results.show_figs && ~isempty(figs)
        for i = 1:length(figs)
            if isgraphics(figs{i})
                try
                    close(figs{i});
                catch
                end
            end
        end
    end
    
    fprintf('✅ Merge complete!\n');
end

function save_plot_files(figs, data, plot_params)
% SAVE_PLOT_FILES - Save figure handles to PNG files
    if isempty(figs), return; end
    
    if ~iscell(figs)
        figs = {figs};
    end
    
    % Get output directory and base name
    if isfield(plot_params, 'outdir') && ~isempty(plot_params.outdir)
        outdir = plot_params.outdir;
    else
        outdir = pwd;
    end
    
    if isfield(plot_params, 'base_name') && ~isempty(plot_params.base_name)
        base_name = plot_params.base_name;
    elseif isfield(data, 'filename') && ~isempty(data.filename)
        base_name = data.filename;
    else
        base_name = 'merged_plots';
    end
    
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    
    % Save each figure
    for i = 1:length(figs)
        fig = figs{i};
        if ~isgraphics(fig, 'figure')
            continue;
        end
        
        try
            fname = sprintf('%s_plots%03d.png', base_name, i);
            fpath = fullfile(outdir, fname);
            
            drawnow nocallbacks;
            set(fig, 'PaperPositionMode', 'auto');
            print(fig, fpath, '-dpng', '-r150');
            
        catch ME
            warning('Failed to save figure %d: %s', i, ME.message);
        end
    end
end