function Do_features(input, varargin)
    % PROGRAM Do_features.
    % Extracts features (wavelets) from *_spikes.mat files and saves them 
    % to 'features_*.mat'. 
    % 
    % This is Step 1 of the clustering process.
    % Step 2 is Do_clustering.m (which loads the file created here).

    min_spikes4SPC = 64; % if less than this number of spikes, features won't be calculated.
    
    par_input = struct;
    parallel = false;
    
    nvar = length(varargin);
    for v = 1:nvar
        if strcmp(varargin{v},'par')
            if (nvar>=v+1) && isstruct(varargin{v+1})
                par_input = varargin{v+1};
            else
                error('Error in ''par'' optional input.')
            end
        elseif strcmp(varargin{v},'parallel')
            if (nvar>=v+1) && islogical(varargin{v+1})
                parallel = varargin{v+1};
            else
                error('Error in ''parallel'' optional input.')
            end
        end
    end

    run_par_for = parallel;
    filenames = {};

    % Get a cell of filenames from the input
    if isnumeric(input) || any(strcmp(input,'all'))  % cases for numeric or 'all' input
        
        filenames_all = {};
        dirnames = dir();
        dirnames = {dirnames.name};
    
        for i = 1:length(dirnames)
            fname = dirnames{i};
    
            if length(fname) < 12
                continue
            end
            if ~ strcmp(fname(end-10:end),'_spikes.mat')
                continue
            end
            if strcmp(input,'all')
                filenames = [filenames {fname}];
            else
                aux = regexp(fname(1:end-11), '\d+$', 'match');
                if ~isempty(aux) && ismember(str2num(aux{1}),input)
                    filenames = [filenames {fname}];
                end
            end
            filenames_all = [filenames_all {fname}];
        end
    
    elseif ischar(input) && length(input) > 4
        if  strcmp (input(end-3:end),'.txt')   % case for .txt input
            filenames =  textread(input,'%s');
        else
            filenames = {input};               % case for cell input
        end
    
    elseif iscellstr(input)
        filenames = input;
    else
        ME = MException('MyComponent:noValidInput', 'Invalid input arguments');
        throw(ME)
    end
    
    feature_start_time = tic;
    par_file = set_parameters();
    
    if parallel == true
        if exist('matlabpool','file')
            try
                matlabpool('open');
            catch
                parallel = false;
            end
        else
            poolobj = gcp('nocreate'); % If no pool, do not create new one.
            if isempty(poolobj)
                parpool
            else
                parallel = false;
            end
        end
    end

    initial_date = now;
    Nfiles = length(filenames);
    
    if run_par_for == true
        parfor fnum = 1:Nfiles
            filename = filenames{fnum};
            do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum);
        end
    else
        for fnum = 1:Nfiles
            filename = filenames{fnum};
            do_features_single(filename, min_spikes4SPC, par_file, par_input, fnum);
        end
    end

    % Cleanup Parallel Pool
    if parallel == true
        if exist('matlabpool','file')
            matlabpool('close')
        else
            poolobj = gcp('nocreate');
            delete(poolobj);
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
    
    outfile_name = ['features_' data_handler.nick_name '.mat'];
    
    par_saved = par; 
    save_helper(outfile_name, spikes, index, spikes_all, index_all, inspk, coeff, par_saved);

    fprintf('Features saved: %s\n', outfile_name);
end

function save_helper(fname, spikes, index, spikes_all, index_all, inspk, coeff, par)
    try
        save(fname, 'spikes', 'index', 'spikes_all', 'index_all', 'inspk', 'coeff', 'par');
    catch
        save(fname, 'spikes', 'index', 'spikes_all', 'index_all', 'inspk', 'coeff', 'par', '-v7.3');
    end
end