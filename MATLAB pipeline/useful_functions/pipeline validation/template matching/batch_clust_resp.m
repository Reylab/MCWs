function batch_clust_resp(input,varargin)
% this function will test a given channel over various
% methods of doing the template matching
% comparisons will be made via response profiling
% make sure changes to be made will be made on a copied spike file in the same foldre


    p = inputParser;
    addParameter(p, 'par', struct, @isstruct);
    addParameter(p, 'parallel', false, @islogical);
    addParameter(p, 'class', [], @(x) isnumeric(x) || ischar(x));
    parse(p, varargin{:});
    
    par_input = p.Results.par;
    parallel = p.Results.parallel;
    target_class = p.Results.class;
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
                       '*finalevents*.mat', '*experiment_properties_online3*.mat', '*.dg_01*'};
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
    sd1_folders = {folder_sd_1, folder_sd_1_t_3, folder_algo1_strt_sd1, folder_algo2_strt_sd1, folder_algo3_strt_sd1, folder_algo4_strt_sd1, folder_algo5_strt_sd1};
    sd3_folders = {folder_sd_3, folder_algo1_strt_sd3, folder_algo2_strt_sd3, folder_algo3_strt_sd3, folder_algo4_strt_sd3, folder_algo5_strt_sd3};
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
    base_dir = pwd;

    % Optional: open parallel pool if user passed parallel=true
    if parallel && isempty(gcp('nocreate'))
        parpool;
    end

    % Use parfor to process the different algorithm folders in parallel
    parfor i = 1:length(all_algo_folders)
        fprintf('\n=== Processing folder %d/%d: %s ===\n', i, length(all_algo_folders), all_algo_folders{i});
        cd(fullfile(base_dir, all_algo_folders{i}));
        fprintf('  Current directory: %s\n', pwd);
        
        % Build the correct times filename based on input channel
        % input is the channel number, construct the filename
        if isnumeric(input)
            % Convert to string representation to search for times file
            input_str = num2str(input(1)); % if input is array, use first element
        else
            input_str = input;
        end
        
        % Search for the specific times file matching this channel
        times_pattern = sprintf('times*%s*.mat', input_str);
        times_file = dir(times_pattern);
        
        if isempty(times_file)
            warning('No times file found matching pattern %s in %s. Skipping.', times_pattern, all_algo_folders{i});
            cd(base_dir);
            continue
        end
        fname_times = times_file(1).name;
        fprintf('  Found times file: %s\n', fname_times);
        data = load(fname_times);
        if ~isfield(data, 'spikes') || ~isfield(data, 'cluster_class')
            warning('Missing required variables in %s. Skipping.', fname_times);
            cd(base_dir);
            continue
        end
        spikes = data.spikes;
        cluster_class = data.cluster_class;
        if isfield(data, 'forced')
            forced = data.forced;
        else
            forced = [];
        end
        
        % Start each trial from original clustering by undoing previously forced assignments.
        % Extract classes from cluster_class (first column has cluster IDs)
        classes = cluster_class(:,1)';
        if exist('forced', 'var') && numel(forced) == numel(classes)
            forced_mask = logical(forced(:))';
            classes(forced_mask) = 0;
        end
        
        f_in  = spikes(classes~=0,:);
        f_out = spikes(classes==0,:);
        class_in = classes(classes~=0);
        
        local_par = par; % Avoid modifying broadcast variable 'par' in parfor
        if contains(all_algo_folders{i}, 'sd1') || contains(all_algo_folders{i}, 'sdnum_1')
            local_par.template_sdnum = 1;
        else
            local_par.template_sdnum = 3;
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
        try
            class_out = force_membership_wc(f_in, class_in, f_out, local_par, algo);
        catch ME
            fprintf('ERROR in force_membership_wc for %s: %s\n', all_algo_folders{i}, ME.message);
            cd(base_dir);
            continue
        end
        forced = classes==0;  % Mark which were originally unclassified
        classes(classes==0) = class_out;
        forced(classes==0) = 0;  % Unmark the newly classified ones
        
        % Update cluster_class with new classifications
        cluster_class(:,1) = classes(:);
        
        % Save updated results to times file
        save(fname_times, 'classes', 'cluster_class', 'forced', '-append');
        fprintf('  Calling Do_clustering...\n');
        param.min_clus = 15;
        param.max_spk = 30000;
        param.mintemp = 0.00;                  % minimum temperature for SPC
        param.maxtemp = 0.251;                 % maximum temperature for SPC
        param.tempstep = 0.01;
        param.max_std_templates = 3;
        param.max_spikes_plot = par.max_spikes_plot; % Default: 5000
        
        Do_clustering(input,'parallel',false,'make_times',false,'make_templates',false,'make_plots',true,'par',param)    

        fprintf('  Calling compute_metrics_batch...\n');
        % Pass 'parallel', false to inner functions since outer loop is parallelized
        compute_metrics_batch(input,'parallel',false, 'save',true);
        fprintf('  Done with algorithms in this folder.\n');
        cd(base_dir);
    end

    %% need to do response profile still then comparisons can be made visually across all methods by
    % comparing images
    all_folders = [orig_cluster_temp, all_algo_folders];
    parfor i = 1:length(all_folders)
        cd(fullfile(base_dir, all_folders{i}));
        
        % DELTE ANY LEFTOVER GRAPES FILES TO PREVENT REPEATED RASTERS
        if exist('grapes_blanks.mat', 'file')
            delete 'grapes_blanks.mat'
        end
        if exist('grapes.mat', 'file')
            delete 'grapes.mat'
        end

        do_structure_mu_BCM_online3(input,'RSVP_online', true, false,false)

        do_structure_sorted_BCM_online3(input, true,false, false)

        stimlist = [7	8	12	25	28	39	43	49	81	82	86	95	96	103	125	150	176	177	204	228	236	266	292	324	414	420	472	482	503	531	587	611	613	615	696	738	744	770	785];
        plot_grapes_as_online('grapes_offline',true,'channels2plot',input, 'stim_list', stimlist, 'order_by_rank', false, ...
                                'is_online', false, 'plot_best_stims_only', false, ...
                                'copy2miniscrfolder',false, 'show_sel_count', true, ...
                                'show_best_stims_wins', true, 'best_stims_nwins', 8, ...
                                'ch_grapes_nwins', 3, 'extra_lbl', '', 'use_blanks', true, ...
                                'circshiftblanks', false);
        plot_grapes_as_online('grapes_offline',true,'channels2plot',input, 'stim_list', 'all', 'order_by_rank', true, ...
                                'is_online', false, 'plot_best_stims_only', false, ...
                                'copy2miniscrfolder',false, 'show_sel_count', true, ...
                                'show_best_stims_wins', true, 'best_stims_nwins', 8, ...
                                'ch_grapes_nwins', 3, 'extra_lbl', '', 'use_blanks', true, ...
                                'circshiftblanks', false);
        cd(base_dir);
    end

    %% Collect comparison images for requested class
    if ~isempty(target_class)
        if isnumeric(target_class)
            class_str = num2str(target_class(1));
        else
            class_str = target_class;
        end
        
        comp_folder = fullfile(base_dir, sprintf('Comparison_Class_%s', class_str));
        if ~exist(comp_folder, 'dir')
            mkdir(comp_folder);
        end
        
        fprintf('\n=== Collecting comparison images for Class %s ===\n', class_str);
        for i = 1:length(all_folders)
            algo_name = all_folders{i};
            % Use recursive search (**) to find files in any subfolder with "classX"
            search_pattern = fullfile(base_dir, algo_name, '**', sprintf('*class*%s*.*', class_str));
            found_images = dir(search_pattern);
            
            for f = 1:length(found_images)
                % Filter out folders and non-images
                [~, ~, ext] = fileparts(found_images(f).name);
                valid_exts = {'.png', '.fig', '.jpg', '.jpeg', '.pdf', '.tif', '.bmp', '.emf'};
                
                % Strictly ensure it is pulling from the plot_grapes results folders
                if found_images(f).isdir || ~any(strcmpi(ext, valid_exts)) || ~contains(lower(found_images(f).folder), 'grapes')
                    continue;
                end
                
                src_file = fullfile(found_images(f).folder, found_images(f).name);
                % Prepend algorithm/SD folder name to prevent overwriting
                dest_name = sprintf('%s_%s', algo_name, found_images(f).name);
                copyfile(src_file, fullfile(comp_folder, dest_name));
            end
        end
        fprintf('Saved copied comparison images to: %s\n', comp_folder);
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
    files = [files, {d.name}]; 
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

% Create true copies for each folder to ensure independent processing
% and proper OneDrive syncing.
copyfile(source, dest);
end