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
    filenames = {};
    
    if isnumeric(input) || any(strcmp(input,'all'))  %cases for numeric or 'all' input
        
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

    if ~isempty(filenames)
        fprintf('Detected %d input spike file(s).\n', numel(filenames));
    end

    par_file = set_parameters();
    par = struct;
    par_groups = {'clus', 'relevant', 'batch_plot'};
    for k = 1:numel(par_groups)
        par = update_parameters(par, par_file, par_groups{k});
    end
    for k = 1:numel(par_groups)
        par = update_parameters(par, par_input, par_groups{k});
    end



    %% create files for all methods
    
    folder_sd_1 = sprintf('sdnum_1');
    if ~exist(folder_sd_1, 'dir')
        mkdir(folder_sd_1);
    end


    folder_sd_3 = sprintf('sdnum_3');
    if ~exist(folder_sd_3, 'dir')
        mkdir(folder_sd_3);
    end

    orig_cluster_temp = {folder_sd_1, folder_sd_3};

    % Stage shared files once per sd folder. Spike files are included so
    % downstream response-profile code can load *_spikes.mat locally.
    shared_patterns = {'*times*.mat', '*NSx*.mat', '*stimulus*.mat', ...
                       '*finalevents*.mat', '*experiment_properties_online3*.mat'};
    shared_files = collect_files_from_patterns(shared_patterns);
    if isempty(shared_files)
        error('No shared pipeline files found in top-level directory');
    end
    if isempty(filenames)
        spike_files = {dir('*_spikes.mat').name};
    else
        spike_files = filenames;
    end
    if ~isempty(spike_files)
        shared_files = unique([shared_files, spike_files], 'stable');
    end

    for i = 1:length(orig_cluster_temp)
        if ~exist(orig_cluster_temp{i}, 'dir')
            mkdir(orig_cluster_temp{i});
        end
        for j = 1:length(shared_files)
            copy_or_link_file(shared_files{j}, fullfile(orig_cluster_temp{i}, shared_files{j}));
        end
    end

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
    
    required_names = unique(shared_files, 'stable');

    % Populate sd_1-based folders with required files only.
    for i = 1:length(required_names)
        source = fullfile(folder_sd_1, required_names{i});
        if ~exist(source, 'file')
            continue
        end
        for j = 1:length(sd1_folders)
            dest = fullfile(sd1_folders{j}, required_names{i});
            copy_or_link_file(source, dest);
        end
    end
    
    % Populate sd_3-based folders with required files only.
    for i = 1:length(required_names)
        source = fullfile(folder_sd_3, required_names{i});
        if ~exist(source, 'file')
            continue
        end
        for j = 1:length(sd3_folders)
            dest = fullfile(sd3_folders{j}, required_names{i});
            copy_or_link_file(source, dest);
        end
    end
    
    
    %% Now you can run your different template matching algorithms in the respective folders

    for i = 1:length(all_algo_folders)
        cd(all_algo_folders{i});
        
        % Load the clustering results
        times_file = dir('*times*.mat');
        if isempty(times_file)
            warning('No times file found in %s. Skipping.', all_algo_folders{i});
            cd('../');
            continue
        end
        fname_times = times_file(1).name;
        data = load(fname_times);
        if ~isfield(data, 'spikes') || ~isfield(data, 'classes') || ~isfield(data, 'cluster_class')
            warning('Missing required variables in %s. Skipping.', fname_times);
            cd('../');
            continue
        end
        spikes = data.spikes;
        classes = data.classes;
        cluster_class = data.cluster_class;
        if isfield(data, 'forced')
            forced = data.forced;
        else
            forced = [];
        end

        % Start each trial from original clustering by undoing previously forced assignments.
        classes = classes(:)';
        if exist('forced', 'var') && numel(forced) == numel(classes)
            forced_mask = logical(forced(:))';
            classes(forced_mask) = 0;
        end

        f_in  = spikes(classes~=0,:);
        f_out = spikes(classes==0,:);
        class_in = classes(classes~=0);
        if contains(all_algo_folders{i}, 'sd1')
            par.template_sdnum = 1;
        else
            par.template_sdnum = 3;
        end
        
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
        cluster_class(:,1) = classes(:);
        
        % Save updated results to times file
        save(fname_times, 'classes', 'cluster_class', 'forced', '-append');
        compute_metrics_batch(input,'parallel',parallel, 'save',true);
        cd('../');
    end

    %% need to do response profile still then comparisons can be made visually across all methods by
    % comparing images
    all_folders = [orig_cluster_temp, all_algo_folders];
    for i = 1:length(all_folders)
        cd(all_folders{i});
        do_structure_mu_BCM_online3(input,'RSVP_online', true, false,false)

        do_structure_sorted_BCM_online3(input, true,false, false)                

        plot_grapes_as_online('grapes_offline',true,'channels2plot', 'all', 'stim_list', 'all', 'order_by_rank', true, ...
                                'is_online', false, 'plot_best_stims_only', false, ...
                                'copy2miniscrfolder',false, 'show_sel_count', true, ...
                                'show_best_stims_wins', true, 'best_stims_nwins', 8, ...
                                'ch_grapes_nwins', 3, 'extra_lbl', '', 'use_blanks', true, ...
                                'circshiftblanks', false);
        cd('../');
    end


end

function files = collect_files_from_patterns(patterns)
files = {};
for k = 1:numel(patterns)
    d = dir(patterns{k});
    d = d(~[d.isdir]);
    if isempty(d)
        continue
    end
    files = [files, {d.name}]; %#ok<AGROW>
end
if isempty(files)
    files = {};
else
    files = unique(files, 'stable');
end
end

function copy_or_link_file(source, dest)
if exist(dest, 'file')
    return
end

dest_dir = fileparts(dest);
if ~isempty(dest_dir) && ~exist(dest_dir, 'dir')
    mkdir(dest_dir);
end

if ~exist(source, 'file')
    warning('Source file not found: %s', source);
    return
end

% Prefer hard links for large shared files to avoid redundant storage.
if ispc
    cmd = sprintf('cmd /c mklink /H "%s" "%s"', dest, source);
    [status, ~] = system(cmd);
    did_link = (status == 0);
else
    cmd = sprintf('ln "%s" "%s"', source, dest);
    [status, ~] = system(cmd);
    did_link = (status == 0);
end

if ~did_link
    copyfile(source, dest);
end
end