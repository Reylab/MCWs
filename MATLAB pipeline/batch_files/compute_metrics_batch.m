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
addParameter(p, 'save', true, @islogical);  % save metrics and SS to .mat when true
addParameter(p, 'show_plots', false, @islogical);  % show plots during processing
addParameter(p, 'quar', false, @islogical);  % run metrics with quarantined spikes added & clustered
parse(p, varargin{:});

% Get file list
if ischar(file_pattern) || isStringScalar(file_pattern)
    % string pattern or 'all'
    fp = char(file_pattern);
    if strcmp(fp, 'all')
        file_list_struct = dir('times_*.mat');
        file_list = {file_list_struct.name};
    else
        file_list_struct = dir(fp);
        file_list = {file_list_struct.name};
    end
elseif isnumeric(file_pattern) || (iscell(file_pattern) && all(cellfun(@isnumeric, file_pattern)))
    % numeric vector or cell of numbers -> interpret as channel ids
    if iscell(file_pattern)
        chans = unique(cell2mat(file_pattern));
    else
        chans = unique(file_pattern(:)');
    end
    % find all times_*.mat files and match by extracted channel id
    all_times = dir('times_*.mat');
    all_names = {all_times.name};
    file_list = {};
    for ci = chans(:)'
        found = false;
        for k = 1:numel(all_names)
            nm = all_names{k};
            name_clean = regexprep(nm, '^times[_\-]*', '', 'ignorecase');
            tok = regexp(name_clean, '[cC][hH](\d+)', 'tokens', 'once');
            if isempty(tok)
                tok = regexp(name_clean, '[_\s](\d+)\.mat$', 'tokens', 'once'); % trailing _NN.mat or space NN.mat
            end
            if isempty(tok)
                tok = regexp(name_clean, '(\d+)', 'tokens', 'once'); % fallback: any digits
            end
            if ~isempty(tok)
                file_chan = str2double(tok{1});
                if ~isnan(file_chan) && file_chan == double(ci)
                    file_list{end+1} = nm; %#ok<AGROW>
                    found = true;
                end
            end
        end
        if ~found
            warning('No times_*.mat file found for requested channel %d', ci);
        end
    end
    % de-duplicate and preserve order
    if ~isempty(file_list)
        file_list = unique(file_list, 'stable');
    end
elseif iscell(file_pattern)
    % cell array of filename strings
    file_list = file_pattern;
else
    error('file_pattern must be string pattern, ''all'', numeric channel list, or cell array of filenames');
end

if isempty(file_list)
    error('No files found matching pattern');
end

fprintf('Found %d files to process\n', length(file_list));

% Prevent figure pop-ups during batch processing by turning off the default
% figure visibility. Restore the previous value on function exit using
% onCleanup. This ensures even helper functions that call `figure()` without
% visibility args won't show UI windows.
try
    prevFigVis = get(groot, 'DefaultFigureVisible');
catch
    prevFigVis = 'on';
end
set(groot, 'DefaultFigureVisible', 'off');
cleanupFigVis = onCleanup(@() set(groot, 'DefaultFigureVisible', prevFigVis));

% Also set the legacy root (0) DefaultFigureVisible for code that uses
% the older root handle API. Restore on exit as well.
try
    prevFigVis0 = get(0, 'DefaultFigureVisible');
catch
    prevFigVis0 = 'on';
end
try
    set(0, 'DefaultFigureVisible', 'off');
    cleanupFigVis0 = onCleanup(@() set(0, 'DefaultFigureVisible', prevFigVis0));
catch
    % ignore if not supported
end

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
    % Ensure worker sessions also create invisible figures by default to avoid
    % pop-ups when plotting on workers. Use pctRunOnAll to set the root
    % DefaultFigureVisible on each worker.
    try
        pctRunOnAll('try; set(groot, "DefaultFigureVisible", ''off''); end');
        % Also set the legacy root (0) DefaultFigureVisible on workers
        pctRunOnAll('try; set(0, "DefaultFigureVisible", ''off''); end');
    catch
        % pctRunOnAll may not be available in some environments; ignore if so.
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
        prev_vis = get(0, 'DefaultFigureVisible');
        set(0, 'DefaultFigureVisible', 'off');
        % Restore visibility when this function finishes (or errors)
        clean_worker_vis = onCleanup(@() set(0, 'DefaultFigureVisible', prev_vis));
    catch
        % Fallback for older MATLAB versions
    end

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
            % Call compute_cluster_metrics. Request invisible figures and ask
            % the function to save plots when 'params.save' is true. This keeps
            % saving centralized inside compute_cluster_metrics and avoids
            % pop-ups or duplicate saves.
            [metrics_table, SS, figs] = compute_cluster_metrics(data, ...
                'exclude_cluster_0', params.exclude_cluster_0, ...
                'n_neighbors', params.n_neighbors, ...
                'bin_duration', params.bin_duration, ...
                'show_plots', params.show_plots, ...
                'save_plots', params.save, ...
                'plot_params', struct('base_name', name_clean)); % pass basename for consistent naming


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
            figs = {};
        end
        
        % Add filename and channel info to metrics table if it's non-empty
        if istable(metrics_table) && height(metrics_table) > 0
            metrics_table.filename = repmat({filename}, height(metrics_table), 1);
            metrics_table.channel_id = repmat(channel_id, height(metrics_table), 1);
        end
        
            % Save metrics centrally (single place). If quarantine flag is set,
            % append '_quar' to the filename.
            % Determine output directory and base name
            out_dir = pwd;
            if isfield(data, 'fullpath') && ~isempty(data.fullpath)
                out_dir = fileparts(data.fullpath);
            end
            base_name = name; % from fileparts above

            % Save metrics centrally (single place). If quarantine flag is set,
            % append '_quar' to the filename.
            if params.save
                try
                    outname = sprintf('%s_metrics.mat', base_name);
                    if params.quar
                        outname = sprintf('%s_quar_metrics.mat', base_name);
                    end
                    save(fullfile(out_dir, outname), 'metrics_table', 'SS');
                    %fprintf('  Saved metrics to %s\n', fullfile(out_dir, outname));
                catch ME
                    warning('Failed to save metrics for %s: %s', filename, ME.message);
                end
            end

            % Plots (if any) are saved inside compute_cluster_metrics when
            % 'save_plots' is true. Do not re-save them here to avoid
            % duplicate files.
        
       % fprintf('Completed: %s (%d clusters)\n', filename, height(metrics_table));
        
    catch ME
        fprintf('Error processing %s: %s\n', filename, ME.message);
        metrics_table = table();
        SS = [];
    end
end