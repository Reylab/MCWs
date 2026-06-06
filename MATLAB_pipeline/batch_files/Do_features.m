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

    dates_spikes = dir(fullfile(pwd, 'spikes*'));
    dates_spikes = dates_spikes([dates_spikes.isdir]);
    if isempty(dates_spikes)
        error('No folders starting with ''spikes'' found in the current working directory.');
    end
    [~, idx_spk] = max([dates_spikes.datenum]);
    target_spikes_folder = fullfile(pwd, dates_spikes(idx_spk).name);
    fprintf('Locking feature extraction to spikes folder: %s\n', dates_spikes(idx_spk).name);
    
    % NEW: Resolve the latest times folder to check against during saving
    dates_times = dir(fullfile(pwd, 'times*'));
    dates_times = dates_times([dates_times.isdir]);
    if ~isempty(dates_times)
        [~, idx_times] = max([dates_times.datenum]);
        latest_times_folder = fullfile(pwd, dates_times(idx_times).name);
        fprintf('Found existing times folder: %s. Will route features here if times files exist.\n', dates_times(idx_times).name);
    else
        latest_times_folder = '';
        fprintf('No times folder found. Will save features to spikes files by default.\n');
    end
    
    % Gather and build absolute file paths using the locked target directory
    if isnumeric(input) || any(strcmp(input,'all'))
        dirnames = dir(fullfile(target_spikes_folder, '*_spikes.mat'));
        filenames_all = cell(1, length(dirnames));
    
        for i = 1:length(dirnames)
            filenames_all{i} = fullfile(target_spikes_folder, dirnames(i).name);
        end
        
        if isnumeric(input)
            % Loop through each requested channel number safely
            for i = 1:length(input)
                % Create a flexible pattern that looks for the number followed by '_spikes.mat'
                % Matches: '1_spikes.mat', 'ch1_spikes.mat', 'CSC1_spikes.mat', 'NSX1_spikes.mat'
                pattern = [num2str(input(i)) '_spikes.mat'];
                
                % Check which absolute filenames contain this specific channel pattern
                matches = contains(filenames_all, pattern);
                
                if any(matches)
                    filenames = [filenames, filenames_all(matches)];
                end
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
        par_file = set_parameters();
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
            do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum, latest_times_folder);
        end
    else
        for fnum = 1:num_files
            filename = filenames{fnum};
            fprintf('Processing file %d of %d: %s (Serial)\n', fnum, num_files, filename);
            do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum, latest_times_folder);
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

function do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum, latest_times_folder)
    par = struct;
    par = update_parameters(par,par_file,'clus');
    par = update_parameters(par,par_input,'clus');
    par.filename = filename;

    % Because readInData will invoke find_latest_spikes, passing an absolute path
    % ensures readInData loads from the explicitly targeted file immediately.
    data_handler = readInData(par);
    par = data_handler.par;
    nick_name = data_handler.nick_name;
    
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

    [features] = wave_features(spikes, par);
    inspk = features.inspk;
    coeff = features.coeff;

    % Determine save target - times file if it exists, otherwise spikes file
    target_save_file = filename;
    if ~isempty(latest_times_folder)
        possible_times_file = fullfile(latest_times_folder, ['times_' nick_name '.mat']);
        if exist(possible_times_file, 'file')
            target_save_file = possible_times_file;
        end
    end

    % Use matfile to overwrite features in place without touching other variables
    m = matfile(target_save_file, 'Writable', true);
    m.inspk = inspk;
    m.coeff = coeff;
    m.features = features;

    % Always also update the spikes file itself for legacy functions
    if ~strcmp(target_save_file, filename)
        m_spk = matfile(filename, 'Writable', true);
        m_spk.inspk = inspk;
        m_spk.coeff = coeff;
        m_spk.features = features;
    end

end