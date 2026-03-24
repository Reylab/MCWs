function batch_clust_resp(input,varargin)
% this function will test a given channel over various
% methods of doing the template matching
% comparisons will be made via response profiling
% make sure changes to be made will be made on a copied spike file in the same foldre


    p = inputParser;
    addParameter(p, 'par', struct, @isstruct);
    addParameter(p, 'parallel', false, @islogical);
    parse(p, varargin{:});
    
    par_input = p.Results.par;
    parallel = p.Results.parallel;
    
    if isnumeric(input) || any(strcmp(input,'all'))  %cases for numeric or 'all' input
        
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
        if  strcmp (input(end-3:end),'.txt')   %case for .txt input
            filenames =  textread(input,'%s');
        else
            filenames = {input};               %case for cell input
        end
    
    elseif iscellstr(input)
        filenames = input;
    else
        ME = MException('MyComponent:noValidInput', 'Invalid input arguments');
        throw(ME)
    end

    par_file = set_parameters();
    %% first step is feature extraction
    % same for all methods - can be saved in same folder bc/ new
    
    do_features_single(input, 64, par_file, par_input, 1);

    %% create files for all methods
    
    folder_sd_1 = sprintf('sdnum_1');
    if ~exist(folder_sd_1, 'dir')
        mkdir(folder_sd_1);
    end


    folder_sd_3 = sprintf('sdnum_%d', sdnum);
    if ~exist(folder_sd_3, 'dir')
        mkdir(folder_sd_3);
    end


    %% Copy spikes file to all folders
    spikes_file = filenames{1};
    orig_cluster_temp = {folder_sd_1, folder_sd_3};
    
    for i = 1:length(orig_cluster_temp)
        dest_file = fullfile(orig_cluster_temp{i}, spikes_file);
        copyfile(spikes_file, dest_file);
    end

    %% clustering is where things change - the original clustering is done already
 
    cd(orig_cluster_temp{1});
    do_clustering(current_file, 'par', par_input, 'parallel', parallel, 'sdnum', 1);
    compute_metrics_batch(input,'parallel',false, 'save',true);

    cd('../' + orig_cluster_temp{2});
    do_clustering(current_file, 'par', par_input, 'parallel', parallel, 'sdnum', 3);
    compute_metrics_batch(input,'parallel',false, 'save',true);
    cd('../');

    %% create folders for varying logic of template matching and copy clustered files there
    folder_sd_1_t_3 = sprintf('sdnum_1_t_3');
    if ~exist(folder_sd_1_t_3, 'dir')
        mkdir(folder_sd_1_t_3);
    end

    folder_algo1_strt_sd1 = sprintf('algo1_strt_sd1');
    if ~exist(folder_algo1_strt_sd1, 'dir')
        mkdir(folder_algo1_strt_sd1);
    end 
    
    folder_algo1_strt_sd3 = sprintf('algo1_strt_sd3');
    if ~exist(folder_algo1_strt_sd3, 'dir')
        mkdir(folder_algo1_strt_sd3);
    end 
    
    folder_algo2_strt_sd1 = sprintf('algo2_strt_sd1');
    if ~exist(folder_algo2_strt_sd1, 'dir')
        mkdir(folder_algo2_strt_sd1);
    end

    folder_algo2_strt_sd3 = sprintf('algo2_strt_sd3');
    if ~exist(folder_algo2_strt_sd3, 'dir')
        mkdir(folder_algo2_strt_sd3);
    end

    folder_algo3_strt_sd1 = sprintf('algo3_strt_sd1');
    if ~exist(folder_algo3_strt_sd1, 'dir')
        mkdir(folder_algo3_strt_sd1);
    end

    folder_algo3_strt_sd3 = sprintf('algo3_strt_sd3');
    if ~exist(folder_algo3_strt_sd3, 'dir')
        mkdir(folder_algo3_strt_sd3);
    end

    folder_algo4_strt_sd1 = sprintf('algo4_strt_sd1');
    if ~exist(folder_algo4_strt_sd1, 'dir')
        mkdir(folder_algo4_strt_sd1);
    end

    folder_algo4_strt_sd3 = sprintf('algo4_strt_sd3');
    if ~exist(folder_algo4_strt_sd3, 'dir')
        mkdir(folder_algo4_strt_sd3);
    end

    folder_algo5_strt_sd1 = sprintf('algo5_strt_sd1');
    if ~exist(folder_algo5_strt_sd1, 'dir')
        mkdir(folder_algo5_strt_sd1);
    end

    folder_algo5_strt_sd3 = sprintf('algo5_strt_sd3');
    if ~exist(folder_algo5_strt_sd3, 'dir')
        mkdir(folder_algo5_strt_sd3);
    end

    %% Copy clustering results to appropriate folders
    % Copy sd_1 results to folders that start with sd_1
    sd1_folders = {folder_sd_1_t_3, folder_algo1_strt_sd1, folder_algo2_strt_sd1, folder_algo3_strt_sd1, folder_algo4_strt_sd1, folder_algo5_strt_sd1};
    sd3_folders = {folder_algo1_strt_sd3, folder_algo2_strt_sd3, folder_algo3_strt_sd3, folder_algo4_strt_sd3, folder_algo5_strt_sd3};
    all_algo_folders = [sd1_folders, sd3_folders];
    
    % Copy all files from sd_1 to sd_1-based folders
    sd1_files = dir(fullfile(folder_sd_1, '*'));
    for i = 1:length(sd1_files)
        if ~sd1_files(i).isdir  % Copy only files, not directories
            for j = 1:length(sd1_folders)
                source = fullfile(folder_sd_1, sd1_files(i).name);
                dest = fullfile(sd1_folders{j}, sd1_files(i).name);
                copyfile(source, dest);
            end
        end
    end
    
    % Copy all files from sd_3 to sd_3-based folders
    sd3_files = dir(fullfile(folder_sd_3, '*'));
    for i = 1:length(sd3_files)
        if ~sd3_files(i).isdir  % Copy only files, not directories
            for j = 1:length(sd3_folders)
                source = fullfile(folder_sd_3, sd3_files(i).name);
                dest = fullfile(sd3_folders{j}, sd3_files(i).name);
                copyfile(source, dest);
            end
        end
    end
    
    
    %% Now you can run your different template matching algorithms in the respective folders

    for i = 1:length(all_algo_folders)
        cd(all_algo_folders{i});
        
        % Load the clustering results
        load('*times*.mat');
        f_in  = spikes(classes~=0,:);
        f_out = spikes(classes==0,:);
        class_in = classes(classes~=0);
        par.template_sdnum = 3;
        
        if contains(all_algo_folders{i}, 'algo1')
            algo = 'algo1';
        elseif contains(all_algo_folders{i}, 'algo2')
            algo = 'algo2';
        elseif contains(all_algo_folders{i}, 'algo3')
            algo = 'algo3';
        elseif contains(all_algo_folders{i}, 'algo4')
            algo = 'algo4';
        elseif contains(all_algo_folders{i}, 'algo5')
            algo = 'algo5';
        else
            algo = 'algo0';
        end
        
        % Apply force membership with tracking
        class_out = force_membership_wc(f_in, class_in, f_out, par, algo);
        forced = classes==0;  % Mark which were originally unclassified
        classes(classes==0) = class_out;
        forced(classes==0) = 0;  % Unmark the newly classified ones
        
        % Update cluster_class with new classifications
        cluster_class(:,1) = classes';
        
        % Save updated results to times file
        times_file = dir('*times*.mat');
        if ~isempty(times_file)
            fname_times = times_file(1).name;
            save(fname_times, 'classes', 'cluster_class', 'forced', '-append');
        end
        compute_metrics_batch(input,'parallel',false, 'save',true);
        cd('../');
    end

    %% need to do response profile still then comparisons can be made visually across all methods by
    % comparing images
    all_folders = [orig_cluster_temp, all_algo_folders];
    for i = 1:length(all_folders)
        cd(all_folders{i});
        do_structure_sorted_BCM_online3(input, par.use_blanks, par.circshiftblanks, par.is_online)                

        plot_grapes_as_online('grapes_offline',true,'channels2plot', 'all', 'stim_list', 'all', 'order_by_rank', true, ...
                                'is_online', par.is_online, 'plot_best_stims_only', par.plot_best_stims_only, ...
                                'copy2miniscrfolder', par.copy2miniscrfolder, 'show_sel_count', par.show_sel_count, ...
                                'show_best_stims_wins', par.show_best_stims_wins, 'best_stims_nwins', 8, ...
                                'ch_grapes_nwins', 3, 'extra_lbl', '', 'use_blanks', par.use_blanks, ...
                                'circshiftblanks', par.circshiftblanks);
        cd('../');
    end


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

    % Append features directly to the spikes file instead of a separate file
    try
        save(filename, 'inspk', 'coeff', '-append');
    catch
        save(filename, 'inspk', 'coeff', '-append', '-v7.3');
    end

    fprintf('Features appended to: %s\n', filename);
end