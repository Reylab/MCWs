function [all_metrics, all_SS] = compute_metrics_batch(file_pattern, varargin)
% COMPUTE_METRICS_BATCH - Compute quality metrics for multiple times files in parallel
%
% Input:
%   file_pattern - string pattern for times files (e.g., 'times_*.mat') 
%                  OR cell array of specific filenames
%                  OR 'all' to process all times_*.mat files in current directory
%
%   Optional parameters:
%   'exclude_cluster_0', true/false (default: true)
%   'n_neighbors', 5 (default: 5)
%   'bin_duration', 60000.0 (default: 60000.0 ms)
%   'parallel', true/false (default: true)
%   'n_workers', N (default: auto-detect)
%   'save', true/false (default: false)        
%   'quar', true/false (default: false)     run clustering with quarantined spikes added
%
% Output:
%   all_metrics - struct array with metrics for each file
%   all_SS - cell array of silhouette matrices for each file

% Parse optional inputs
p = inputParser;
addParameter(p, 'exclude_cluster_0', true, @islogical);
addParameter(p, 'n_neighbors', 5, @isscalar);
addParameter(p, 'bin_duration', 60000.0, @isscalar);
addParameter(p, 'parallel', true, @islogical);
addParameter(p, 'n_workers', 0, @isscalar); % 0 = auto-detect
addParameter(p, 'save', false, @islogical);  % save metrics and SS to .mat when true
addParameter(p, 'quar', false, @islogical);  % run metrics with quarantined spikes added & clustered
parse(p, varargin{:});

% Get file list
if ischar(file_pattern)
    if strcmp(file_pattern, 'all')
        file_list = dir('times_*.mat');
        file_list = {file_list.name};
    else
        file_list = dir(file_pattern);
        file_list = {file_list.name};
    end
elseif iscell(file_pattern)
    file_list = file_pattern;
else
    error('file_pattern must be string pattern, cell array of filenames, or ''all''');
end

if isempty(file_list)
    error('No files found matching pattern');
end

fprintf('Found %d files to process\n', length(file_list));

% Set up parallel processing
if p.Results.parallel
    if p.Results.n_workers > 0
        n_workers = p.Results.n_workers;
    else
        n_workers = feature('numcores'); % Use all available cores
    end
    
    if isempty(gcp('nocreate'))
        parpool(n_workers);
    end
    fprintf('Using parallel processing with %d workers\n', n_workers);
end

% Preallocate results
all_metrics = cell(length(file_list), 1);
all_SS = cell(length(file_list), 1);

% Process files
if p.Results.parallel
    parfor i = 1:length(file_list)
        [all_metrics{i}, all_SS{i}] = process_single_file(file_list{i}, p.Results);
    end
else
    for i = 1:length(file_list)
        [all_metrics{i}, all_SS{i}] = process_single_file(file_list{i}, p.Results);
    end
end

fprintf('Batch processing complete! Processed %d files\n', length(file_list));
end

function [metrics_table, SS] = process_single_file(filename, params)
% PROCESS_SINGLE_FILE - Process one times file
    try
        %fprintf('Processing: %s\n', filename);
        
        % Load data
        data = load(filename);
        
        % add filename/path fields to the loaded struct so downstream code that
        % expects these fields doesn't error (prevents "Unrecognized field name 'filename'")
        [fullpath, name, ext] = fileparts(filename);
        if isempty(fullpath)
            fullpath = pwd;
        end
        data.filename = name;
        data.fullpath = fullfile(fullpath, [name ext]);
        
        % Extract channel info from filename (supports names like
        % times_mProbe_raw_ch12 or times_mProbe_raw_12 and names with spaces)
        name_clean = regexprep(name, '^times[_\-]*', '', 'ignorecase');
        channel_id = NaN;
        tok = regexp(name_clean, '[cC][hH](\d+)', 'tokens', 'once');
        if isempty(tok)
            tok = regexp(name_clean, '[_\s](\d+)$', 'tokens', 'once'); % trailing _NN or space NN
        end
        if isempty(tok)
            tok = regexp(name_clean, '(\d+)', 'tokens', 'once'); % fallback: any digits
        end
        if ~isempty(tok)
            channel_id = str2double(tok{1});
        end
        
        % Compute metrics (single run). Catch errors locally so we don't call
        % reporting functions without metrics.
        try
            [metrics_table, SS] = compute_cluster_metrics(data, ...
                'exclude_cluster_0', params.exclude_cluster_0, ...
                'n_neighbors', params.n_neighbors, ...
                'bin_duration', params.bin_duration, ...
                'show_plots', false, ...
                'save_plots',true);
        catch ME_cm
            fprintf('  ✗ compute_cluster_metrics failed for %s: %s\n', filename, ME_cm.message);
            % Print full stack to help locate "Too many output arguments" source
            try
                fprintf('%s\n', getReport(ME_cm, 'extended'));
            catch
                % fallback
                disp(ME_cm);
            end
             metrics_table = table();
             SS = [];
        end
        
        % Add filename and channel info to metrics table if it's non-empty
        if istable(metrics_table) && height(metrics_table) > 0
            metrics_table.filename = repmat({filename}, height(metrics_table), 1);
            metrics_table.channel_id = repmat(channel_id, height(metrics_table), 1);
        end
        
        % Save results if requested. If params.quar is true, save with a "_quar" suffix.
        if params.save
            try
                out_dir = pwd; % save to current working directory; change if desired
                suffix = '';
                if params.quar
                    suffix = '_quar';
                end
                metrics_fname = fullfile(out_dir, sprintf('%s_metrics%s.mat', name, suffix));
                save(metrics_fname, 'metrics_table', 'SS');
                
                % Also save any open figures (produced by compute_cluster_metrics) with the same suffix
                figHandles = findall(0, 'Type', 'figure');
                savedCount = 0;
                for f = 1:numel(figHandles)
                    fig = figHandles(f);
                    % skip invisible figures
                    if isprop(fig, 'Visible') && ~strcmp(get(fig, 'Visible'), 'on')
                        continue;
                    end
                    % consider figure non-empty only if at least one axes has children
                    axesList = findall(fig, 'Type', 'axes');
                    nonEmpty = false;
                    for a = 1:numel(axesList)
                        if ~isempty(get(axesList(a), 'Children'))
                            nonEmpty = true;
                            break;
                        end
                    end
                    % also treat figures with text objects as non-empty
                    if ~nonEmpty
                        if ~isempty(findall(fig, 'Type', 'text'))
                            nonEmpty = true;
                        end
                    end
                    if ~nonEmpty
                        continue;
                    end
                    try
                        fig_name = sprintf('%s_fig%d%s.png', name, fig.Number, suffix);
                        fig_path = fullfile(out_dir, fig_name);
                        if exist('exportgraphics', 'file') == 2
                            exportgraphics(fig, fig_path, 'BackgroundColor', 'white');
                        else
                            saveas(fig, fig_path);
                        end
                        savedCount = savedCount + 1;
                    catch saveFigErr
                        fprintf(' Failed to save figure %d for %s: %s\n', fig.Number, filename, saveFigErr.message);
                    end
                end
               % fprintf('  Saved %d figure(s)\n', savedCount);
                
               % fprintf('  Saved metrics%s for %s\n', suffix, filename);
            catch ME3
                fprintf('  Failed saving results for %s: %s\n', filename, ME3.message);
            end
        end
        
       % fprintf('Completed: %s (%d clusters)\n', filename, height(metrics_table));
        
    catch ME
        fprintf('Error processing %s: %s\n', filename, ME.message);
        metrics_table = table();
        SS = [];
    end
end

function compare_tbl = build_metrics_comparison(metrics_orig, metrics_quar)
% BUILD_METRICS_COMPARISON - Simple comparator between two metrics tables
% - tries to match clusters by a common key ('cluster_id' or 'cluster'), otherwise
%   compares statistics aggregated across tables.
    try
        % Identify possible key columns
        key_candidates = {'cluster_id', 'cluster', 'clusterLabel', 'id'};
        key = '';
        for k = 1:numel(key_candidates)
            if any(strcmp(metrics_orig.Properties.VariableNames, key_candidates{k})) && ...
               any(strcmp(metrics_quar.Properties.VariableNames, key_candidates{k}))
                key = key_candidates{k};
                break;
            end
        end
        
        if ~isempty(key)
            % join on key
            A = metrics_orig;
            B = metrics_quar;
            % ensure key variable has same name in both for join
            A.Properties.VariableNames{strcmp(A.Properties.VariableNames, key)} = 'KEY_';
            B.Properties.VariableNames{strcmp(B.Properties.VariableNames, key)} = 'KEY_';
            JT = outerjoin(A, B, 'Keys', 'KEY_', 'MergeKeys', true);
            
            % compute differences for numeric columns present on both sides
            varnames = JT.Properties.VariableNames;
            compare_rows = table();
            compare_rows.KEY = JT.KEY_;
            for vi = 1:numel(varnames)
                v = varnames{vi};
                if endsWith(v, '_A') || endsWith(v, '_B')
                    base = v(1:end-2);
                    colA = [base '_A'];
                    colB = [base '_B'];
                    if any(strcmp(varnames, colA)) && any(strcmp(varnames, colB))
                        a = JT.(colA);
                        b = JT.(colB);
                        if isnumeric(a) && isnumeric(b)
                            diffcol = b - a;
                            % sanitize name
                            newname = sprintf('%s_diff', base);
                            compare_rows.(newname) = diffcol;
                        end
                    end
                end
            end
            compare_tbl = compare_rows;
        else
            % No key: compute simple aggregate differences for numeric vars
            numA = varfun(@mean, metrics_orig, 'InputVariables', @isnumeric);
            numB = varfun(@mean, metrics_quar, 'InputVariables', @isnumeric);
            % align variable names
            namesA = numA.Properties.VariableNames;
            namesB = numB.Properties.VariableNames;
            common = intersect(namesA, namesB);
            compare_tbl = table();
            for ii = 1:numel(common)
                v = common{ii};
                if ~strcmp(v, 'GroupCount')
                    a = numA.(v);
                    b = numB.(v);
                    compare_tbl.([v '_diff']) = b - a;
                end
            end
        end
    catch
        compare_tbl = table();
    end
end