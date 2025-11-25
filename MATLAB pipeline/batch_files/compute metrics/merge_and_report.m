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
parse(p, varargin{:});

% Handle file input
if ischar(data) || isstring(data)
    filename = data;
    data = load(filename);
    fprintf('Loaded: %s\n', filename);
else
    filename = '';
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
    [df_metrics, SS, figs] = compute_cluster_metrics(new_data, ...
        'make_plots', p.Results.make_plots, ...
        'save_plots', p.Results.save_plots, ...
        'calc_metrics', p.Results.calc_metrics);
end

% Save output if requested
if ~isempty(p.Results.output_file)
    save(p.Results.output_file, '-struct', 'new_data');
    fprintf('Saved: %s\n', p.Results.output_file);
    
    % Save metrics separately if computed
    if p.Results.calc_metrics && ~isempty(df_metrics)
        [path, name, ext] = fileparts(p.Results.output_file);
        metrics_file = fullfile(path, [name '_metrics' ext]);
        save(metrics_file, 'df_metrics', 'SS');
        fprintf('Saved metrics: %s\n', metrics_file);
    end
end

fprintf('✅ Merge complete!\n');
end