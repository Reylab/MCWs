function Do_features(input, varargin)
    % PROGRAM Do_features.
    % Extracts features (wavelets) from *_spikes.mat files and appends
    % 'inspk' and 'coeff' directly to the same *_spikes.mat file.
    %
    % This is Step 1 of the clustering process.
    % Step 2 is Do_clustering.m (which loads inspk/coeff from the spikes file).

    min_spikes4SPC = 64; % if less than this number of spikes, features won't be calculated.
    
    % Parse input arguments
    p = inputParser;
    addParameter(p, 'par', struct, @isstruct);
    addParameter(p, 'parallel', false, @islogical);
    parse(p, varargin{:});
    
    par_input = p.Results.par;
    parallel = p.Results.parallel;
    run_par_for = parallel;
    filenames = {};

    dates = dir(fullfile(pwd, 'spikes*'));
    dates = dates([dates.isdir]);
    if isempty(dates)
        error('No folders starting with ''spikes'' found in the current working directory.');
    end
    [~, idx] = max([dates.datenum]);
    target_spikes_folder = fullfile(pwd, dates(idx).name);
    fprintf('Locking feature extraction to spikes folder: %s\n', dates(idx).name);
    
    % Gather and build absolute file paths using the locked target directory
    if isnumeric(input) || any(strcmp(input,'all'))
        filenames_all = {};
        dirnames = dir(fullfile(target_spikes_folder, '*_spikes.mat'));
        dirnames = {dirnames.name};
    
        for i = 1:length(dirnames)
            fname = dirnames{i};
            filenames_all{end+1} = fullfile(target_spikes_folder, fname);
        end
        
        if isnumeric(input)
            for i=1:length(input)
                chan_cells = regexp(filenames_all, ['CSC' num2str(input(i)) '_spikes\.mat|NSX' num2str(input(i)) '_spikes\.mat'], 'match');
                chan_cells = [chan_cells{:}];
                filenames = [filenames chan_cells];
            end
        else
            filenames = filenames_all;
        end
        
    elseif iscell(input)
        for i = 1:length(input)
            filenames{i} = fullfile(target_spikes_folder, input{i});
        end
    elseif ischar(input) && length(input) > 4 && strcmp(input(end-3:end), '.txt')
        f_list = fopen(input);
        while ~feof(f_list)
            fil = fgetl(f_list);
            if ischar(fil) && ~isempty(fil)
                filenames{end+1} = fullfile(target_spikes_folder, fil);
            end
        end
        fclose(f_list);
    elseif ischar(input) && length(input) > 11 && strcmp(input(end-10:end), '_spikes.mat')
        filenames{1} = fullfile(target_spikes_folder, input);
    end

    % Get parameters file name
    if exist('set_parameters.m','file')
        par_file = 'set_parameters';
    else
        par_file = [];
    end

    feature_start_time = tic;
    num_files = length(filenames);
    fprintf('Found %d files to process.\n', num_files);

    % Main execution loop (serial or parallel)
    if run_par_for == true
        if exist('matlabpool','file')
            try
                matlabpool('open');
            catch
                run_par_for = false;
            end
        else
            poolobj = gcp('nocreate');
            if isempty(poolobj)
                try
                    parpool;
                catch
                    run_par_for = false;
                end
            end
        end
    end

    if run_par_for == true
        parfor fnum = 1:num_files
            filename = filenames{fnum};
            fprintf('Processing file %d of %d: %s (Parallel)\n', fnum, num_files, filename);
            do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum);
        end
    else
        for fnum = 1:num_files
            filename = filenames{fnum};
            fprintf('Processing file %d of %d: %s (Serial)\n', fnum, num_files, filename);
            do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum);
        end
    end

    % Cleanup Parallel Pool
    if parallel == true
        if exist('matlabpool','file')
            matlabpool('close')
        else
            poolobj = gcp('nocreate');
            if ~isempty(poolobj)
                delete(poolobj);
            end
        end
    end

    time_taken = toc(feature_start_time);
    fprintf('Feature extraction done in %0.2f seconds.\n', time_taken);

end

function do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum)

    par = struct;
    par = update_parameters(par,par_file,'clus');
    par = update_parameters(par,par_input,'clus');
    par.filename = filename;

    % Because readInData will invoke find_latest_spikes, passing an absolute path
    % ensures readInData loads from the explicitly targeted file immediately.
    data_handler = readInData(par);
    par = data_handler.par;
    
    if data_handler.with_spikes
        [spikes, index, spikes_all, index_all] = data_handler.load_spikes_withCollisions();
    else
        warning('File: %s doesn''t include spikes', filename);
        return
    end

    % Check spike count
    nspk = size(spikes,1);
    if nspk < min_spikes4SPC
        warning('Not enough spikes in %s (found %d, need %d)', filename, nspk, min_spikes4SPC);
        return
    end

    [inspk, coeff] = wave_features(spikes, par);

    % Append features directly back into the targeted spike file
    try
        save(filename, 'inspk', 'coeff', '-append');
    catch
        % Fallback if file becomes large or version needs enforcement
        save(filename, 'inspk', 'coeff', '-append', '-v7.3');
    end

end