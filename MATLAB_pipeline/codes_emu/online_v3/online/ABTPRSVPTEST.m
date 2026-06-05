function ABTPRSVPTEST(subtask, varargin)
% ABTPRSVP - Unified RSVP screening task with integrated online processing
% This function combines visual stimulus presentation and online neural 
% processing in a single MATLAB instance, eliminating the need for 
% inter-MATLAB communication.
%
% Available subtasks:
% 'DynamicScr',
% 'DynamicScrTest',
% 'DynamicSeman',
% 'DynamicSemanTest',
% 'CategLocaliz',
% 'FirstTime',
% 'OnlineMiniScr',
% 'OnlineMiniScrMusic', 
% 'Test',
% 'FreqTag_Faces',
% 'FreqTag_NonFaces',
% 'FreqTag_Bruno'
%
% Usage:
%   ABTPRSVPTEST('DynamicScr')
%   ABTPRSVPTEST('DynamicScr', 'auto_resp', true)
%   ABTPRSVPTEST('DynamicScrTest', 'auto_resp', true)
%   ABTPRSVPTEST('OnlineMiniScr', 'location', 'MCW-BEH-RIP')
%   ABTPRSVPTEST('FirstTime', 'location', 'MCW-BEH-RIP')
%
% Parameters:
%   subtask          - (required) Task type string
%   'sub_ID'         - Subject ID string (auto-detected from .map file if empty)
%   'run_num'        - Run number integer (auto-detected if empty)
%   'EMU_num'        - EMU number integer (auto-detected if empty)
%   'location'       - Location string (default: 'MCW-FH-RIP')
%   'is_online'      - Enable online processing (default: true)
%   'auto_resp'      - Auto-press subject inputs (default: false)
%   'ptb_debug'      - PTB debug mode (default: false)
%   'Nrep'           - Override default repetitions
%   'which_nsp_micro'- NSP for micros (BRK only)
%   'lang'           - Language: 'english', 'spanish', 'french'
%   'device_resp'    - Response device: 'keyboard', 'gamepad'
%   'system'         - Acquisition system: 'RIP', 'BRK'
%   'acq_network'    - Network recording control (default: location default)
%   'with_acq_folder'- Access to raw data (default: location default)
%   'templates_required' - Require templates (default: task dependent)
%   'use_daq'        - Use DAQ device (default: location default)
%   'mapfile'        - Map file name
%   'online_notches' - Load calculated notches (default: location default)
%   'use_photodiodo' - Use photodiode (default: location default)
%   'disable_interactions' - Disable matconnect interactions (default: false)
%   'debug'          - Debug mode (default: 0)

%% =======================================================================
%  SECTION 1: INPUT PARSING AND PARAMETER SETUP
%  =======================================================================

ipr = inputParser;
addParameter(ipr, 'sub_ID', []);
addParameter(ipr, 'run_num', []);
addParameter(ipr, 'EMU_num', []);
addParameter(ipr, 'location', 'MCW-ABTL-RIP');
addParameter(ipr, 'is_online', true);
addParameter(ipr, 'auto_resp', false);
addParameter(ipr, 'ptb_debug', false);
addParameter(ipr, 'Nrep', []);
addParameter(ipr, 'which_nsp_micro', []);
addParameter(ipr, 'lang', []);
addParameter(ipr, 'device_resp', []);
addParameter(ipr, 'system', []);
addParameter(ipr, 'acq_network', []);
addParameter(ipr, 'with_acq_folder', []);
addParameter(ipr, 'templates_required', []);
addParameter(ipr, 'use_daq', []);
addParameter(ipr, 'mapfile', []);
addParameter(ipr, 'online_notches', []);
addParameter(ipr, 'use_photodiodo', []);
addParameter(ipr, 'disable_interactions', false);
addParameter(ipr, 'offline_processing', true);  % Enable Tower-based offline processing
addParameter(ipr, 'use_async_processing', true);  % Use async processing (separate pools for screening/processing)
addParameter(ipr, 'async_num_workers', []);  % Number of workers for async processing (auto if empty)
addParameter(ipr, 'use_background_collection', true);  % Use dedicated worker for data collection (separates device_com from main thread)
addParameter(ipr, 'collection_poll_interval', 0.010);  % Poll interval for background collection (10ms default)
addParameter(ipr, 'test_device', false);  % Run device test before starting (verifies connectivity)
addParameter(ipr, 'test_duration', 3);  % Duration of device test in seconds
addParameter(ipr, 'debug', 0);
parse(ipr, varargin{:});

% Add paths - navigate up from online/ -> online_v3/ -> tasks/ -> codes_emu/
codes_emu_path = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(codes_emu_path);
custompath = reylab_custompath({'useful_functions', 'useful_functions/tasks_tools', ...
    'useful_functions/task_tools', 'tasks/.', 'JoyMEX', 'wave_clus_reylab', ...
    'wave_split', 'codes_for_analysis', 'mex', 'tasks/online_v3', 'tasks/online_v3/online'});

% Get location-specific parameters
params = location_setup(ipr.Results.location);

% Override params with user-provided values
inputs = fields(ipr.Results);
for i = 1:numel(inputs)
    pstr = inputs{i};
    if any(strcmp(ipr.UsingDefaults, pstr)) && isempty(ipr.Results.(pstr)) && isfield(params, pstr)
        continue
    end
    params.(pstr) = ipr.Results.(pstr);
end

custompath.add(params.additional_paths, true);
params.do_sorting = true;

%% =======================================================================
%  SECTION 2: SUBTASK CONFIGURATION
%  =======================================================================

scale_factor = 1;
size_point = 10;
MAX_TRIALS = 15;  % Default, can be overwritten by subtask

if strcmp(subtask, 'DynamicScr')
    rel_path_pics = fullfile('pics_space','custom_pics');
    P2REMOVE = [80 90 60 60 0]; 
    MANUAL_SELECT = [true, true, true, false, false];
    NREP = [6 3 3 4 5]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [180 180 180 120 60];
    MIN_SAFE_TRIALS = 15;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    
elseif strcmp(subtask, 'DynamicScrTest')
    rel_path_pics = fullfile('pics_space', 'custom_pics');
    P2REMOVE = [70 50 0];
    MANUAL_SELECT = [true, true, false];
    NREP = [1 1 2]; if ~isempty(params.Nrep), NREP = params.Nrep; end
    NPICS = [120 60 30];
    MIN_SAFE_TRIALS = 3;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required = false; end
    params.remove_pictures = false;
    
elseif contains(subtask, 'FreqTag')
    rel_path_pics = fullfile('pics_space', 'freq_tag');
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    NREP = 12; if ~isempty(params.Nrep), NREP = params.Nrep; end
    scale_factor = 2;
    size_point = 12;
    if contains(subtask, 'Bruno')
        scale_factor = 2.5;
        size_point = 24;
        NPICS = 180;
        N_NF = 144;
        N_F = NPICS - N_NF;
    elseif contains(subtask, 'NonFaces')
        NPICS = 75;
        N_RNF = 15;
        N_NRF = 15;
        N_NRNF = NPICS - N_RNF - N_NRF;
    elseif contains(subtask, 'Faces')
        NPICS = 75;
        N_RF = 15;
        N_NRNF = NPICS - N_RF;
    end
    MIN_SAFE_TRIALS = 3;
    params.use_only_main_pics = true;
    if isempty(params.templates_required), params.templates_required = true; end
    params.remove_pictures = false;
    

elseif strcmp(subtask, 'DynamicSeman')
    rel_path_pics = fullfile('pics_space', 'seman_pics');
    P2REMOVE = [0, 0, 0];
    MANUAL_SELECT = [false, false, false];
    NREP = [4, 3, 3]; if ~isempty(params.Nrep), NREP = params.Nrep; end
    NPICS = [300, 300, 300];
    MIN_SAFE_TRIALS = 10;
    params.use_only_main_pics = true;
    if isempty(params.templates_required), params.templates_required = false; end
    params.remove_pictures = false;
    
elseif strcmp(subtask, 'DynamicSemanTest')
    rel_path_pics = fullfile('pics_space', 'seman_pics');
    P2REMOVE = [0, 0, 0];
    MANUAL_SELECT = [false, false, false];
    NREP = [2, 1, 1]; if ~isempty(params.Nrep), NREP = params.Nrep; end
    NPICS = [300, 300, 300];
    MIN_SAFE_TRIALS = 15;
    params.use_only_main_pics = true;
    if isempty(params.templates_required), params.templates_required = false; end
    params.remove_pictures = false;
    
elseif strcmp(subtask,'CategLocalizTest')
    rel_path_pics = fullfile('pics_space','categ_localiz');
    P2REMOVE = [100,0,0]; %old P2REMOVE = [50 50 50 60 60 0];
    MANUAL_SELECT = [true,true,false];
    NREP = [1,1,3]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     NREP = [2,1,1]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [180,180,1];
    %     NPICS = [120,120,120];
    MIN_SAFE_TRIALS = 5;
    MAX_TRIALS = 5;
    %params.acq_network=1;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    %if there are not enought pictures show the amount added from pic us
elseif strcmp(subtask,'CategLocaliz')
    rel_path_pics = fullfile('pics_space','categ_localiz');
    P2REMOVE = [100,0,0]; %old P2REMOVE = [50 50 50 60 60 0];
    MANUAL_SELECT = [true,true,false];
    NREP = [6,3,3]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     NREP = [2,1,1]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [180,180,1];
    %     NPICS = [120,120,120];
    MIN_SAFE_TRIALS = 10;
    MAX_TRIALS = 12;
    %params.acq_network=1;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;

    
elseif strcmp(subtask, 'FirstTime')
    rel_path_pics = 'picsfirst';
    params.is_online = false;
    params.EMU_num = NaN;
    params.do_sorting = false;
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    NREP = 9; if ~isempty(params.Nrep), NREP = params.Nrep; end
    NPICS = 60;  % Will be updated based on actual image count
    MIN_SAFE_TRIALS = max(NREP);
    params.acq_network = 0;
    params.with_acq_folder = 0;
    params.use_only_main_pics = true;
    if isempty(params.templates_required), params.templates_required = false; end
    params.remove_pictures = false;
    
elseif strcmp(subtask, 'OnlineMiniScr')
    rel_path_pics = 'miniscr_pics';
    NPICS = 60;  % Will be updated based on actual image count
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    params.use_only_main_pics = true;
    if isempty(params.templates_required), params.templates_required = true; end
    params.remove_pictures = false;
    
elseif strcmp(subtask, 'OnlineMiniScrMusic')
    rel_path_pics = 'miniscr_pics';
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    params.use_only_main_pics = true;
    params.templates_required = false;
    params.online_notches = false;
    params.remove_pictures = false;
    
elseif strcmp(subtask, 'Test')
    params.acq_network = 0;
    params.with_acq_folder = 1;
    rel_path_pics = fullfile('pics_space', 'custom_pics');
    P2REMOVE = [2 20 0];
    NREP = [2 2 6]; if ~isempty(params.Nrep), NREP = params.Nrep; end
    NPICS = [30 30 10];
    MIN_SAFE_TRIALS = 4;
    MANUAL_SELECT = [false, false, false];
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required = false; end
    params.remove_pictures = false;
    
else
    error('Subtask not found. Available: DynamicScr, DynamicScrTest, DynamicSeman, DynamicSemanTest, CategLocaliz, FirstTime, OnlineMiniScr, OnlineMiniScrMusic, Test, FreqTag_*')
end

min_seq_length = 60;
n_blks = length(NPICS);

%% =======================================================================
%  SECTION 3: SUBJECT AND RUN IDENTIFICATION
%  =======================================================================

if isempty(params.sub_ID)
    map_files = dir(fullfile(params.beh_rec_metadata, '*.map'));
    if isempty(map_files)
        error('No .map file found in rec_metadata folder: %s', params.beh_rec_metadata);
    end
    map_file_path = fullfile(params.beh_rec_metadata, map_files(1).name);
    params.sub_ID = get_patient_id_from_map(map_file_path);
    if isempty(params.sub_ID)
        error('Could not find patient_id in map file: %s', map_file_path);
    end
end

if isunix, system('nmcli radio wifi off'); end

if params.ptb_debug
    PsychDebugWindowConfiguration;
end

% Check remote folder access
if ~params.acq_is_processing && ~test_remote_folder(params.acq_remote_folder_in_processing)
    if isempty(params.EMU_num)
        custompath.rm()
        error('EMU_num not given and remote acq folder not detected')
    end
    if params.with_acq_folder
        custompath.rm()
        error('remote acq folder not detected')
    end
end

% Define EMU folder path (needed for scanning and experiment folder creation)
emu_folder = fullfile(params.backup_path, params.sub_ID, 'EMU');

% Auto-detect EMU and run numbers using Task History CSV
% This replaces folder-based detection with a centralized CSV tracking system
% NOTE: We do NOT register yet - only after folder is created with files
if isempty(params.EMU_num) || isempty(params.run_num)
    % Include emu_folder explicitly for scanning existing runs
    scan_paths = {params.acq_remote_folder_in_beh, params.backup_path, emu_folder};
    [params.EMU_num, params.run_num, params.task_history_csv] = get_task_history(...
        params.sub_ID, subtask, params.beh_rec_metadata, ...
        'scan_paths', scan_paths, ...
        'force_emu', params.EMU_num, ...
        'force_run', params.run_num, ...
        'acq_folder', params.acq_remote_folder_in_beh, ...
        'backup_path', params.backup_path, ...
        'register', false);  % Don't register yet
else
    % If both provided, just get the CSV path
    [~, ~, params.task_history_csv] = get_task_history(...
        params.sub_ID, subtask, params.beh_rec_metadata, ...
        'force_emu', params.EMU_num, ...
        'force_run', params.run_num, ...
        'acq_folder', params.acq_remote_folder_in_beh, ...
        'backup_path', params.backup_path, ...
        'register', false);  % Don't register yet
end

% Create experiment name
if params.acq_network
    experiment.fname = sprintf('EMU-%.3d_subj-%s_task-RSVP%s_run-%.2d', ...
        params.EMU_num, params.sub_ID, subtask, params.run_num);
else
    experiment.fname = 'dyn_scr_test';
end

% Create experiment folder directly in backup location (EMU folder)
if ~isfolder(emu_folder)
    mkdir(emu_folder);
end
experiment.folder_name = fullfile(emu_folder, experiment.fname);
if ~isfolder(experiment.folder_name)
    mkdir(experiment.folder_name);
end

% NOW register the run in task history (after folder is created)
% Use the SAME emu/run numbers that were determined earlier
get_task_history(params.sub_ID, subtask, params.beh_rec_metadata, ...
    'force_emu', params.EMU_num, ...
    'force_run', params.run_num, ...
    'register', true, ...
    'experiment_folder', experiment.folder_name);

fprintf('\n=== ABTPRSVP Unified Task ===\n');
fprintf('Subject: %s\n', params.sub_ID);
fprintf('Subtask: %s\n', subtask);
fprintf('EMU: %d, Run: %d\n', params.EMU_num, params.run_num);
fprintf('Experiment: %s\n', experiment.fname);
fprintf('==============================\n\n');

%% =======================================================================
%  SECTION 4: IMAGE LOADING AND PREPARATION
%  =======================================================================

afolders = [];

% Handle FreqTag-specific image selection
if strcmp(subtask, 'FreqTag_Faces')
    Path_pics_RF = [rel_path_pics filesep 'RespFaces'];
    Path_pics_NRNF = [rel_path_pics filesep 'NRNonFaces'];
    
    aRF = dir([params.pics_root_beh filesep Path_pics_RF]);
    aRF = aRF(arrayfun(@(x) contains(lower(x.name), '.jp'), aRF));
    temp_imgs = randperm(length(aRF));
    aRF = aRF(temp_imgs(1:N_RF));
    afolders = [afolders; repmat({Path_pics_RF}, length(aRF), 1)];
    
    aNRNF = dir([params.pics_root_beh filesep Path_pics_NRNF]);
    aNRNF = aNRNF(arrayfun(@(x) contains(lower(x.name), '.jp'), aNRNF));
    odd_imgs = 1:1:length(aNRNF);
    even_imgs = 2:1:length(aNRNF);
    temp_imgs_O = randperm(length(odd_imgs));
    temp_imgs_E = randperm(length(even_imgs));
    aNRNF_final = [aNRNF(temp_imgs_O(1:floor(N_NRNF/2))); aNRNF(temp_imgs_E(1:ceil(N_NRNF/2)))];
    afolders = [afolders; repmat({Path_pics_NRNF}, length(aNRNF_final), 1)];
    
    a = [aRF; aNRNF_final];
    
elseif strcmp(subtask, 'FreqTag_NonFaces')
    Path_pics_RNF = [rel_path_pics filesep 'RespNonFaces'];
    Path_pics_NRF = [rel_path_pics filesep 'NRFaces'];
    Path_pics_NRNF = [rel_path_pics filesep 'NRNonFaces'];
    
    aRNF = dir([params.pics_root_beh filesep Path_pics_RNF]);
    aRNF = aRNF(arrayfun(@(x) contains(lower(x.name), '.jp'), aRNF));
    temp_imgs = randperm(length(aRNF));
    aRNF = aRNF(temp_imgs(1:N_RNF));
    afolders = [afolders; repmat({Path_pics_RNF}, length(aRNF), 1)];
    
    aNRF = dir([params.pics_root_beh filesep Path_pics_NRF]);
    aNRF = aNRF(arrayfun(@(x) contains(lower(x.name), '.jp'), aNRF));
    temp_imgs = randperm(length(aNRF));
    aNRF = aNRF(temp_imgs(1:N_NRF));
    afolders = [afolders; repmat({Path_pics_NRF}, length(aNRF), 1)];
    
    aNRNF = dir([params.pics_root_beh filesep Path_pics_NRNF]);
    aNRNF = aNRNF(arrayfun(@(x) contains(lower(x.name), '.jp'), aNRNF));
    odd_imgs = 1:1:length(aNRNF);
    even_imgs = 2:1:length(aNRNF);
    temp_imgs_O = randperm(length(odd_imgs));
    temp_imgs_E = randperm(length(even_imgs));
    aNRNF_final = [aNRNF(temp_imgs_O(1:floor(N_NRNF/2))); aNRNF(temp_imgs_E(1:ceil(N_NRNF/2)))];
    afolders = [afolders; repmat({Path_pics_NRNF}, length(aNRNF_final), 1)];
    
    a = [aRNF; aNRF; aNRNF_final];
    
elseif strcmp(subtask, 'FreqTag_Bruno')
    Path_pics_NRF = [rel_path_pics filesep 'NRFaces_Bruno'];
    Path_pics_NRNF = [rel_path_pics filesep 'NRNonFaces_Bruno'];
    
    aNRF = dir([params.pics_root_beh filesep Path_pics_NRF]);
    aNRF = aNRF(arrayfun(@(x) contains(lower(x.name), '.jp'), aNRF));
    temp_imgs = randperm(length(aNRF));
    aNRF = aNRF(temp_imgs(1:N_F));
    afolders = [afolders; repmat({Path_pics_NRF}, length(aNRF), 1)];
    
    aNRNF = dir([params.pics_root_beh filesep Path_pics_NRNF]);
    aNRNF = aNRNF(arrayfun(@(x) contains(lower(x.name), '.jp'), aNRNF));
    odd_imgs = 1:1:length(aNRNF);
    even_imgs = 2:1:length(aNRNF);
    temp_imgs_O = randperm(length(odd_imgs));
    temp_imgs_E = randperm(length(even_imgs));
    aNRNF_final = [aNRNF(temp_imgs_O(1:floor(N_NF/2))); aNRNF(temp_imgs_E(1:ceil(N_NF/2)))];
    afolders = [afolders; repmat({Path_pics_NRNF}, length(aNRNF_final), 1)];
    
    a = [aNRF; aNRNF_final];
elseif contains(subtask, 'CategLocaliz')
    % CategLocaliz: recursive image loading with custom_pics support (has subfolders)
    all_pics_path = [params.pics_root_beh filesep rel_path_pics];
    experiment.task_pics_folder = all_pics_path;
    custom_pics_path = [all_pics_path filesep 'custom_pics'];
    Path_pics = all_pics_path;
    
    % Get custom pics first
    custom_pics_list = table();
    if isfolder(custom_pics_path)
        all_files = dir(fullfile(custom_pics_path, '**', '*.*'));
        all_files = all_files(~[all_files.isdir]);
        custom_pics_list = all_files(endsWith({all_files.name}, {'.jpg', '.jpeg'}, 'IgnoreCase', true));
        if ~isempty(custom_pics_list)
            custom_pics_list = struct2table(custom_pics_list);
            custom_pics_list = custom_pics_list(:, 1:2); % Keep only name, folder
            custom_pics_list.selectable = zeros(height(custom_pics_list), 1);
        else
            custom_pics_list = table();
        end
    end
    
    % Get all pics recursively
    all_pics_list = table();
    if isfolder(all_pics_path)
        all_files = dir(fullfile(all_pics_path, '**', '*.*'));
        all_files = all_files(~[all_files.isdir]);
        all_pics_list = all_files(endsWith({all_files.name}, {'.jpg', '.jpeg'}, 'IgnoreCase', true));
        all_pics_list = all_pics_list(randperm(length(all_pics_list))); % random order
        all_pics_list = struct2table(all_pics_list);
        all_pics_list = all_pics_list(:, 1:2); % Keep only name, folder
        all_pics_list.selectable = zeros(height(all_pics_list), 1);
        if height(custom_pics_list) > 0
            % Move custom_pics to top of the table
            all_pics_list = vertcat(custom_pics_list, all_pics_list);
            [~, uniq_idxs] = unique(all_pics_list.name, 'stable');
            all_pics_list = all_pics_list(uniq_idxs, :);
        end
    end
    
    if isempty(all_pics_list) || height(all_pics_list) == 0
        error('No pictures for this session in %s', all_pics_path);
    end
    
    % For compatibility, set a to the struct array equivalent
    a = table2struct(all_pics_list);
    afolders = all_pics_list.folder;
else
    % Standard image loading (non-recursive) - for DynamicScr and other tasks
    Path_pics = [params.pics_root_beh filesep rel_path_pics];
    a = dir(Path_pics);
    filt_a = arrayfun(@(x) contains(lower(x.name), '.jp'), a);
    a = a(filt_a);
    afolders = repmat({rel_path_pics}, length(a), 1);
end

if isempty(a)
    error('No pictures for this session in %s', Path_pics);
end

% Handle OnlineMiniScr timing estimation
if contains(subtask, 'OnlineMiniScr')
    N = length(a);
    [NREP, NSEQ, seq_length_est, estimated_duration] = calculate_miniscr_time(N);
    fprintf('Estimated duration: %d pics (%d trials each) in %d sequences of %.1f secs: %.1f min\n', ...
        N, NREP, NSEQ, seq_length_est * 0.5, estimated_duration);
    if ~isempty(params.Nrep), NREP = params.Nrep; end
    MIN_SAFE_TRIALS = max(NREP);
end

% Build ImageNames table
if contains(subtask, 'CategLocaliz')
    % For CategLocaliz, all_pics_list is already built with recursive loading
    totalp2load = NPICS(1);
    if height(all_pics_list) < totalp2load
        error('Not enough pictures in folders, add at least %d.', totalp2load - height(all_pics_list))
    end
    all_pics_list.stim_trial_count = zeros(height(all_pics_list), 1);
    ImageNames = all_pics_list;
    stim_trial_counter = ImageNames.stim_trial_count;
elseif ~params.use_only_main_pics
    new_pics2load = NPICS(2:end) - (NPICS(1:end-1) - P2REMOVE(1:end-1));
    totalp2load = NPICS(1) + sum(new_pics2load);
    
    a_common = dir(sprintf('%s', [params.pics_root_beh filesep params.additional_pics]));
    filt_a = arrayfun(@(x) contains(lower(x.name), '.jp'), a_common);
    a_common = a_common(filt_a);
    
    % Remove duplicates
    a_common_keep_idx = [];
    for k = 1:length(a_common)
        a_comm_str = a_common(k).name;
        flag_match = false;
        for l = 1:length(a)
            a_str = a(l).name;
            if strcmp(a_comm_str, a_str)
                flag_match = true;
            end
        end
        if ~flag_match
            a_common_keep_idx = [a_common_keep_idx, k];
        end
    end
    a_common = a_common(a_common_keep_idx, :);
    a_common = a_common(randperm(length(a_common)));
    
    afolders = [repmat({rel_path_pics}, length(a), 1); repmat({params.additional_pics}, length(a_common), 1)];
    a = [{a.name}, {a_common.name}];
    
    [~, ia, ~] = unique(a, 'stable');
    if length(ia) < totalp2load
        error('Not enough pictures in folders, add at least %d.', totalp2load - length(ia))
    end
    stim_trial_counter = zeros(numel(ia), 1);
    ImageNames = array2table([a(ia)', afolders(ia)], 'VariableNames', {'name', 'folder'});
else
    a = {a.name};
    stim_trial_counter = zeros(numel(a), 1);
    ImageNames = array2table([a', afolders], 'VariableNames', {'name', 'folder'});
    totalp2load = numel(a);
    if ~exist('NPICS', 'var') || length(NPICS) == 1
        NPICS = numel(a);
    end
end

% Extract concept information (common for all tasks)
ImageNames.concept_name = cellfun(@(x) regexpi(x, '^.*(?=(_\d*(?s)\D*$))', 'match', 'once'), ImageNames.name, 'UniformOutput', false);
without_numbers = cellfun('isempty', ImageNames.concept_name);
ImageNames.concept_name(without_numbers) = cellfun(@(x) regexpi(x, '^.*(?=((?s)\D*$))', 'match', 'once'), ImageNames.name(without_numbers), 'UniformOutput', false);

% Additional processing only for DynamicScr (not for CategLocaliz)
if contains(subtask, 'DynamicScr')
    ImageNames.concept_number = cellfun(@(x) str2double(cell2mat(regexpi(x, '_(\d*).\D*$', 'tokens', 'once'))), ImageNames.name);
    ImageNames.concept_number(isnan(ImageNames.concept_number)) = 1;

    [u, ~, IC] = unique(ImageNames.concept_name, 'stable');
    for iu = 1:numel(u)
        ImageNames.concept_number(IC == iu) = 1:sum(IC == iu);
    end

    ImageNames = sortrows(ImageNames, 'concept_number');
    ImageNames.concept_categories = cellfun(@(x) strsplit(x, '~'), ImageNames.concept_name, 'UniformOutput', false);

    % Remove unwanted categories
    categories2remove = {'animal', 'hobby'};
    for i = 1:height(ImageNames)
        ctgs = ImageNames.concept_categories{i};
        valid_ctgs = cellfun(@(x) ~any(strcmp(x, categories2remove)), ctgs);
        ImageNames.concept_categories{i} = ctgs(valid_ctgs);
    end

    % Add face/non-face classification
    [ImageNames, unclassified_count, unclassified_concepts] = add_face_classification_to_imagenames(ImageNames, []);
    if unclassified_count > 0
        resp = questdlg(sprintf('%d images are not classified as Faces or Non Faces.\nContinue?', unclassified_count), ...
            'Unclassified Images Warning', 'Continue', 'Save Unknown', 'Abort', 'Continue');
        if strcmp(resp, 'Abort')
            error('Task aborted due to unclassified images.');
        elseif strcmp(resp, 'Save Unknown')
            [filename, pathname] = uiputfile('unclassified_concepts.csv', 'Save Unclassified Concepts');
            if filename ~= 0
                save_unclassified_concepts_csv(fullfile(pathname, filename), unclassified_concepts);
            end
            resp2 = questdlg('Continue with the task?', 'Continue?', 'Continue', 'Abort', 'Continue');
            if strcmp(resp2, 'Abort')
                error('Task aborted due to unclassified images.');
            end
        end
    end
elseif ~contains(subtask, 'CategLocaliz')
    % Standard processing for other tasks (not DynamicScr, not CategLocaliz)
    ImageNames.concept_number = cellfun(@(x) str2double(cell2mat(regexpi(x, '_(\d*).\D*$', 'tokens', 'once'))), ImageNames.name);
    ImageNames.concept_number(isnan(ImageNames.concept_number)) = 1;

    [u, ~, IC] = unique(ImageNames.concept_name, 'stable');
    for iu = 1:numel(u)
        ImageNames.concept_number(IC == iu) = 1:sum(IC == iu);
    end

    ImageNames = sortrows(ImageNames, 'concept_number');
    ImageNames.concept_categories = cellfun(@(x) strsplit(x, '~'), ImageNames.concept_name, 'UniformOutput', false);
end

total_figures = height(ImageNames);
fprintf('Loaded %d images for task\n', total_figures);

%% =======================================================================
%  SECTION 5: ONLINE PROCESSING SETUP
%  =======================================================================

if params.is_online
    % Online processing configuration
    b_use_blanks = false;
    b_circshiftblanks = false;
    b_remove_collisions = true;
    b_collision_warning_shown = false;
    b_make_coll_plots = false;
    floc_task_group = 'Other'; % For CategLocaliz: all, RecDec or Other
    
    % nwins_best_stims: 6 for CategLocaliz, 12 for DynamicScr
    if contains(subtask, 'CategLocaliz')
        nwins_best_stims = 6;
    else
        nwins_best_stims = 12;
    end
    
    % Channel configuration
    remove_channels_by_label = {'^[^[mc]](.*)$', '^(micro(.*))$', '(ref-\d*)$'};
    chs_th_abs = [];
    chs_th_pos = [];
    remove_channels = [];
    priority_channels = [];
    priority_chs_ranking = [321:328 361:368];
    not_online_channels = [];
    NOTCHES_USED = 25;
    
    if strcmp(params.location, 'MCW-FH-RIP')
        MAX_SORTING_CORES = 35;
    else
        MAX_SORTING_CORES = 5;
    end
    
    % Sorting parameters
    sorting_done = false;
    DO_SORTING = params.do_sorting;
    SP2SORT = 500000;
    nstd2plots = 3;
    
    if contains(subtask, 'Test')
        ntrial2sort = 2;
        par.stdmax = 250;
    else
        ntrial2sort = 17;
        par.stdmax = 50;
    end
    
    mu_only = false;
    if contains(subtask, 'OnlineMiniScr') && ~params.templates_required
        mu_only = true;
    end
    
    % Detection parameters
    par.only_det_filter = true;
    par.stdmin = 5;
    par.preprocessing = params.online_notches;
    par.detect_order = 4;
    par.sort_order = 2;
    par.w_pre = 20;
    par.w_post = 44;
    par.int_factor = 5;
    par.detect_fmin = 300;
    par.detect_fmax = 3000;
    par.sort_fmin = 300;
    par.sort_fmax = 3000;
    par.sr = 30000;
    par.ref_ms = 1.5;
    
    % Timing parameters
    WAIT_LOOP = 1;
    TIME_PRE = 1e3;
    TIME_POS = 2e3;
    RASTER_SIMILARITY_THR = 0.85;
    MAX_NTRIALS = 15;
    
    % Device configuration
    address = {'192.168.137.3', '192.168.137.178'};
    PHOTODIODE_LEVEL = '1900mV';
    NSP_TYPE = 265;
    
    if strcmp(params.system, 'BRK')
        if NSP_TYPE == 265
            PHOTODIODE = 257;
        else
            PHOTODIODE = 129;
        end
        MAPFILE = [];
    else
        PHOTODIODE = 1;
        mapfiles = dir([params.processing_rec_metadata filesep params.mapfile]);
        if numel(mapfiles) > 1
            error('multiple mapfiles found')
        elseif numel(mapfiles) == 1
            MAPFILE = [params.processing_rec_metadata filesep mapfiles(1).name];
        else
            error('mapfile not found')
        end
    end
    
    % =====================================================================
    % DEVICE TEST - Run early to verify connectivity before full setup
    % =====================================================================
    if params.test_device && params.is_online
        fprintf('\n*** RUNNING DEVICE TEST ***\n');
        fprintf('This will verify device connectivity before starting the experiment.\n\n');
        
        if strcmp(params.system, 'RIP')
            [test_success, test_results] = BackgroundDataCollectorWorker.test('RIP', ...
                'mapfile', MAPFILE, ...
                'duration', params.test_duration, ...
                'poll_interval', params.collection_poll_interval);
        else
            [test_success, test_results] = BackgroundDataCollectorWorker.test('BRK', ...
                'address', address{params.which_nsp_micro}, ...
                'instance', params.which_nsp_micro - 1, ...
                'nsp_type', NSP_TYPE, ...
                'duration', params.test_duration, ...
                'poll_interval', params.collection_poll_interval);
        end
        
        if ~test_success
            error('Device test FAILED. Please check device connectivity and try again.\nTest results saved in test_results variable.');
        end
        
        fprintf('Device test passed. Continuing with experiment setup...\n\n');
        pause(1);  % Brief pause to let user see the result
    end
    
    % Response configuration
    MAX_RASTERS_PER_STIM = 2;
    resp_conf = struct;
    resp_conf.from_onset = 1;
    resp_conf.smooth_bin = 1500;
    resp_conf.min_spk_median = 1;
    resp_conf.tmin_median = 200;
    resp_conf.tmax_median = 700;
    resp_conf.psign_thr = 0.05;
    resp_conf.t_down = 20;
    resp_conf.over_threshold_time = 75;
    resp_conf.below_threshold_time = 100;
    resp_conf.nstd = 3;
    resp_conf.win_cent = 1;
    resp_conf.sigma_gauss = 10;
    resp_conf.alpha_gauss = 3.035;
    resp_conf.ifr_resolution_ms = 1;
    resp_conf.sr = par.sr;
    resp_conf.TIME_PRE = TIME_PRE;
    resp_conf.TIME_POS = TIME_POS;
    resp_conf.tmin_base = -900;
    resp_conf.tmax_base = -100;
    resp_conf.FR_resol = 10;
    resp_conf.min_ifr_thr = 4;
    
    % Read priority channels from map file
    map_files = dir(fullfile(params.processing_rec_metadata, '*.map'));
    if ~isempty(map_files)
        map_file_path = fullfile(params.processing_rec_metadata, map_files(1).name);
        priority_chs_ranking = get_priority_channels_from_map(map_file_path);
        if isempty(priority_chs_ranking)
            warning('priority_channel not found in map file, using default');
            priority_chs_ranking = [321:328 361:368];
        end
    else
        warning('No .map file found, using default priority_chs_ranking');
    end
    
    % Determine if this is a screening task (DynamicScr/DynamicSeman) or a follow-up task
    is_screening_task = contains(subtask, 'DynamicScr') || contains(subtask, 'DynamicSeman');
    
    % For non-screening tasks, find latest screening folder.
    % Check both tower and local paths; use whichever has the latest screening.
    tower_emu_path = fullfile('/media/tower', params.sub_ID, 'EMU');
    local_emu_path = fullfile('/home/user/ReyLab/experimental_files', params.sub_ID, 'EMU');
    latest_screening_folder = '';
    if ~is_screening_task
        candidate_folders = {};
        emu_paths_checked = {};
        % Check tower
        if isfolder(tower_emu_path)
            emu_paths_checked{end+1} = tower_emu_path;
            f = find_latest_dynamicscr_folder(tower_emu_path, params.sub_ID);
            if ~isempty(f), candidate_folders{end+1} = f; end
        end
        % Check local
        if isfolder(local_emu_path)
            emu_paths_checked{end+1} = local_emu_path;
            f = find_latest_dynamicscr_folder(local_emu_path, params.sub_ID);
            if ~isempty(f), candidate_folders{end+1} = f; end
        end
        % Pick the one with the highest EMU number
        if ~isempty(candidate_folders)
            best_emu = -1;
            for ci = 1:length(candidate_folders)
                [~, fname] = fileparts(candidate_folders{ci});
                emu_tok = regexp(fname, 'EMU-(\d+)_', 'tokens');
                if ~isempty(emu_tok)
                    n = str2double(emu_tok{1}{1});
                    if n > best_emu
                        best_emu = n;
                        latest_screening_folder = candidate_folders{ci};
                    end
                end
            end
            fprintf('Selected screening folder: %s\n', latest_screening_folder);
        end
        if isempty(emu_paths_checked)
            emu_paths_checked = {tower_emu_path, local_emu_path};
        end
    end
    
    % Template loading - different logic for screening tasks vs follow-up tasks
    TEMPLATES_MS = [params.processing_rec_metadata filesep 'templates_ms.mat'];
    TEMPLATES_WC = [params.processing_rec_metadata filesep 'templates_wc_offline.mat'];
    templates_source = '';
    templates_loaded_from_fallback = false;
    
    if is_screening_task && ~params.templates_required
        % Screening tasks (DynamicScr/DynamicSeman): don't need templates, clean any old ones
        if isfile(TEMPLATES_WC)
            delete(TEMPLATES_WC);
        end
        if isfile(TEMPLATES_MS)
            delete(TEMPLATES_MS);
        end
    elseif params.templates_required
        % Follow-up tasks that require templates
        if is_screening_task
            % Screening task that requires templates - only look in rec_metadata
            if isfile(TEMPLATES_WC)
                TEMPLATES_FILE = TEMPLATES_WC;
                templates_source = 'rec_metadata (templates_wc_offline.mat)';
                disp('Templates loaded from rec_metadata TEMPLATES_WC')
            elseif isfile(TEMPLATES_MS)
                TEMPLATES_FILE = TEMPLATES_MS;
                templates_source = 'rec_metadata (templates_ms.mat)';
                disp('Templates loaded from rec_metadata TEMPLATES_MS')
            else
                error('Templates required but not found in rec_metadata')
            end
        else
            % Non-screening task - look in candidate screening folders for templates
            % Try all candidate folders (tower + local) in order, not just the selected one
            if ~isempty(candidate_folders)
                templates_found = false;
                for cfi = 1:length(candidate_folders)
                    cf = candidate_folders{cfi};
                    % First try templates_wc_offline in main task folder
                    fallback_templates_wc_main = fullfile(cf, 'templates_wc_offline.mat');
                    % Then try results folder
                    fallback_templates_wc = fullfile(cf, 'results', 'templates_wc_offline.mat');
                    fallback_templates_ms = fullfile(cf, 'results', 'templates_ms.mat');
                    
                    if isfile(fallback_templates_wc_main)
                        TEMPLATES_FILE = fallback_templates_wc_main;
                        templates_source = sprintf('latest screening (main folder): %s', cf);
                        templates_loaded_from_fallback = true;
                        templates_found = true;
                        fprintf('Templates loaded from: %s\n', fallback_templates_wc_main);
                        break;
                    elseif isfile(fallback_templates_wc)
                        TEMPLATES_FILE = fallback_templates_wc;
                        templates_source = sprintf('latest screening (results): %s', cf);
                        templates_loaded_from_fallback = true;
                        templates_found = true;
                        fprintf('Templates loaded from: %s\n', fallback_templates_wc);
                        break;
                    elseif isfile(fallback_templates_ms)
                        TEMPLATES_FILE = fallback_templates_ms;
                        templates_source = sprintf('latest screening (results): %s', cf);
                        templates_loaded_from_fallback = true;
                        templates_found = true;
                        fprintf('Templates loaded from: %s\n', fallback_templates_ms);
                        break;
                    end
                end
                if ~templates_found
                    error('Templates required but not found in any candidate folder:\n%s', ...
                        strjoin(candidate_folders, '\n'))
                end
            else
                error('Templates required but no screening folder found in: %s', strjoin(emu_paths_checked, ', '))
            end
        end
    end
    
    % IFR Calculator
    ifr_calculator = IFRCalculator(resp_conf.alpha_gauss, resp_conf.sigma_gauss, ...
        resp_conf.ifr_resolution_ms, resp_conf.sr, TIME_PRE, TIME_POS);
    
    % Preprocessing/notches loading - different logic for screening tasks vs follow-up tasks
    if par.preprocessing
        preprocessing_source = '';
        preprocessing_loaded_from_fallback = false;
        preprocessing_file = [params.processing_rec_metadata filesep 'pre_processing_info.mat'];
        
        if is_screening_task
            % Screening tasks (DynamicScr/DynamicSeman): read from rec_metadata only
            if exist(preprocessing_file, 'file')
                load(preprocessing_file, 'process_info')
                preprocessing_source = sprintf('rec_metadata: %s', preprocessing_file);
                disp('Preprocessing info loaded from rec_metadata.')
            else
                error('pre_processing_info.mat not found in rec_metadata (%s). Required for screening tasks.', ...
                    preprocessing_file)
            end
        else
            % Non-screening tasks: look in candidate screening folders for preprocessing
            if ~isempty(candidate_folders)
                preprocessing_found = false;
                for cfi = 1:length(candidate_folders)
                    cf = candidate_folders{cfi};
                    fallback_preprocessing = fullfile(cf, 'results', 'pre_processing_info.mat');
                    fprintf('Looking for preprocessing in: %s\n', fallback_preprocessing);
                    if isfile(fallback_preprocessing)
                        load(fallback_preprocessing, 'process_info')
                        preprocessing_source = sprintf('latest screening: %s', cf);
                        preprocessing_loaded_from_fallback = true;
                        preprocessing_found = true;
                        fprintf('Preprocessing info loaded from: %s\n', fallback_preprocessing);
                        break;
                    end
                end
                if ~preprocessing_found
                    error('pre_processing_info.mat not found in any candidate folder results:\n%s', ...
                        strjoin(candidate_folders, '\n'))
                end
            else
                error('pre_processing_info.mat required but no screening folder found in: %s', ...
                    strjoin(emu_paths_checked, ', '))
            end
        end
    end
    
    % Show configuration dialog
    if ~exist('preprocessing_source', 'var'), preprocessing_source = ''; end
    
    [priority_chs_ranking, user_cancelled] = show_config_dialog(templates_source, preprocessing_source, priority_chs_ranking);
    if user_cancelled
        error('User cancelled configuration dialog');
    end
    fprintf('Priority channels: [%s]\n', num2str(priority_chs_ranking));


    % Start parallel pool BEFORE opening hardware (cbmex/Ripple)
    % parpool spawns worker processes that can corrupt MEX shared memory
    % if hardware connections are already open
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        num_cores = feature('numcores');
        num_workers = max(2, num_cores - min(4, ceil(num_cores * 0.2)));
        cluster = parcluster('local');
        max_cluster_workers = cluster.NumWorkers;
        num_workers = min(num_workers, max_cluster_workers);
        fprintf('Starting parallel pool with %d workers (system has %d cores, cluster max %d)...\n', ...
            num_workers, num_cores, max_cluster_workers);
        poolobj = parpool('local', num_workers);
    else
        fprintf('Using existing parallel pool with %d workers\n', poolobj.NumWorkers);
    end
   
    % Device setup
    if strcmp(params.system, 'BRK')
        which_nsp = params.which_nsp_micro;
        inst_num = which_nsp - 1;
        address = address{which_nsp};
        channel_id_offset = (inst_num > 0) * (inst_num + 1) * 1000;
    else
        channel_id_offset = 0;
        inst_num = 0;
        address = [];
    end
    
    if strcmp(params.system, 'BRK')
        ev_channels = PHOTODIODE;
    elseif strcmp(params.system, 'RIP')
        ev_channels = 1;
    else
        error('Unsupported device')
    end
    
    % Store device parameters in params for worker
    params.mapfile = MAPFILE;
    params.nsp_type = NSP_TYPE;
    params.nsp_address = address;
    params.photodiode_channel = PHOTODIODE;  % Pass correct photodiode channel to worker
    
    % Determine whether to use dedicated worker for data collection
    use_background_collection = params.use_background_collection;
    
    if use_background_collection && params.is_online
        % === WORKER-BASED DATA COLLECTION ===
        % Device_com runs in a separate parfeval worker, completely independent
        % from the main thread. This is similar to the two-MATLAB architecture.
        
        fprintf('\n=== STARTING DATA COLLECTION WORKER ===\n');
        fprintf('Device_com will run in separate worker process.\n');
        fprintf('Poll interval: %.3f sec\n', params.collection_poll_interval);
        
        bg_collector = BackgroundDataCollectorWorker(params, ...
            'poll_interval', params.collection_poll_interval, ...
            'remove_channels', remove_channels, ...
            'remove_channels_by_label', remove_channels_by_label);
        
        % Wait for worker to open device and get channel info
        bg_collector.wait_ready();
        worker_info = bg_collector.get_channel_info();
        
        % Extract channel info from worker
        channels = worker_info.channels;
        chan_label = worker_info.chan_label;
        conversion = worker_info.conversion;
        ev_channels = worker_info.ev_channels;
        info = worker_info.full_info;
        
        fprintf('Worker initialized with %d channels.\n', numel(channels));
        fprintf('==========================================\n\n');
        
        params.xippmex_connected = true;  % Flag for recording_handler
        
    else
        % === MAIN THREAD DATA COLLECTION (original behavior) ===
        % Open device communication on main thread
        device_com('open', params.system, 'address', address, 'instance', inst_num, ...
            'mapfile', MAPFILE, 'nsp_type', NSP_TYPE);

        params.xippmex_connected = true;  % Flag for recording_handler to skip xippmex('tcp')
        
        % Get channel info
        info = device_com('get_chs_info');
        channels = [];
        conversion = [];
        chan_label = {};
        
        for ci = 1:numel(info.ch)
            if any(info.ch(ci) == remove_channels)
                continue
            end
            rem_ch = false;
            for i = 1:numel(remove_channels_by_label)
                if ~isempty(regexp(info.label{ci}, remove_channels_by_label{i}, 'match'))
                    rem_ch = true;
                    break;
                end
            end
            if rem_ch
                continue
            end
            if info.ismicro(ci)
                channels(end+1) = info.ch(ci);
                chan_label{end+1} = info.label{ci};
                conversion(end+1) = info.conversion(ci);
            end
        end
        
        device_com('enable_chs', channels, true, ev_channels);
        device_com('clear_buffer');
        pause(0.2)
        
        % Test stream read
        lastwarn('', '');
        streams = device_com('get_stream');
        [warnMsg, warnId] = lastwarn();
        if ~isempty(warnMsg)
            error(warnMsg, warnId);
        end
        
        % For non-worker mode, bg_collector will be created later if needed
        bg_collector = [];
    end
    
    % Calculate filters
    [b_filter, a_filter] = ellip(par.detect_order, 0.1, 40, [par.detect_fmin par.detect_fmax] * 2 / par.sr);
    [z_det, p_det, k_det] = tf2zpk(b_filter, a_filter);
    
    if DO_SORTING && ~par.only_det_filter
        [b_sort_filter, a_sort_filter] = ellip(par.sort_order, 0.1, 40, [par.sort_fmin par.sort_fmax] * 2 / par.sr);
        [z_sort_det, p_sort_det, k_sort_det] = tf2zpk(b_sort_filter, a_sort_filter);
    end
    
    % Setup channel filters
    ch_filters = {};
    for ci = 1:numel(info.ch)
        if par.preprocessing
            index = find([process_info(:).chID] == (info.ch(ci) + channel_id_offset));
            if isempty(index)
                preprocessing_info = [];
            else
                preprocessing_info = process_info(index);
            end
        else
            preprocessing_info = [];
        end
        
        if par.preprocessing && ~isempty(preprocessing_info)
            if par.only_det_filter
                [sos, g] = calc_sosg(preprocessing_info.notches, z_det, p_det, k_det, NOTCHES_USED);
                ch_filters{end+1}.det = {sos, g};
            else
                ch_filters{end+1}.det = {b_filter, a_filter};
                if DO_SORTING
                    ch_filters{end}.sort = {b_sort_filter, a_sort_filter};
                end
                [sos_notch, g_notch] = calc_sosg(preprocessing_info.notches, [], [], 1, NOTCHES_USED);
                ch_filters{end}.notches = {sos_notch, g_notch};
            end
        else
            ch_filters{end+1}.det = {b_filter, a_filter};
            if DO_SORTING && ~par.only_det_filter
                ch_filters{end}.sort = {b_sort_filter, a_sort_filter};
            end
        end
    end
    
    % Photodiode configuration
    if params.use_photodiodo && strcmp(params.system, 'BRK')
        pause(0.05)
        cbmex('config', PHOTODIODE, 'spkfilter', 0, 'spkthrlevel', PHOTODIODE_LEVEL, 'instance', inst_num)
        pause(0.05)
    end
    
    det_conf = -1 * ones(size(channels));
    det_conf(ismember(channels, chs_th_pos)) = 1;
    det_conf(ismember(channels, chs_th_abs)) = 0;
    
    % Set async processing flag based on parameter
    use_async = params.use_async_processing;
    if use_async
        fprintf('\n=== ASYNC PROCESSING MODE ENABLED ===\n');
        fprintf('Neural processing will run in background while PTB presents stimuli.\n');
        fprintf('This simulates having 2 separate MATLAB instances.\n');
        fprintf('==========================================\n\n');
    end
    
    fprintf('Online processing initialized with %d channels\n', numel(channels));
    
    % Print timing source info
    if params.use_photodiodo
        fprintf('Timing source: PHOTODIODE (with DAQ fallback)\n');
    else
        fprintf('Timing source: DAQ (parallel port) only\n');
    end
    
    % Initialize output info for tracking issues
    outputinfo = {};
end



bg_collector = [];

% Create temp folder for file operations
temp_folder = fullfile(params.backup_path, 'temp');
if ~exist(temp_folder, 'dir')
    mkdir(temp_folder);
end

% Clear miniscr folder for DynamicScr
if contains(subtask, 'DynamicScr')
    miniscr_folder = fullfile(params.pics_root_processing, 'miniscr_pics');
    try
        delete([miniscr_folder filesep '*']);
    catch
        warning('Deleting miniscr pics failed.');
    end
end

%% =======================================================================
%  SECTION 6: PSYCHTOOLBOX SETUP AND EXPERIMENT CONFIGURATION
%  =======================================================================

% Recording setup
if params.acq_network
    recording = recording_handler(params, experiment.fname);
    params.recording_name = recording.rec_name;
end

% Keyboard setup
[kbs, products_names] = GetKeyboardIndices;
dev_used = [];
for i = 1:numel(params.keyboards)
    if isnumeric(params.keyboards{i})
        dev_used(end+1) = params.keyboards{i};
    else
        kbix = strcmp(params.keyboards{i}, products_names);
        if ~any(kbix)
            warning('Keyboard %s not found', params.keyboards{i});
        else
            dev_used = [dev_used kbs(kbix)];
        end
    end
end
if IsWin
    dev_used = [0];
end
if isempty(dev_used)
    if ~isempty(kbs)
        dev_used = kbs;
    else
        error('Keyboards not found')
    end
end

KbName('UnifyKeyNames');
Screen('Preference', 'VisualDebugLevel', 3);
AssertOpenGL;

exitKey = KbName('F2');
startKey = KbName('s');
continueKey = KbName('c');

% Messages
message_begin = {'Ready to begin?'; 'Listo para empezar?'; 'Etes-vous pret pour commencer?'};
message_continue = {'Ready to continue?'; 'Listo para continuar?'; 'Etes-vous pret pour continuer?'};
message_wait = {'Take a short break.\nWe will resume shortly'; 'Take a short break.\nWe will resume shortly'; 'Take a short break.\nWe will resume shortly'};
message_final = {'That would be all.\nThank you!'; 'Eso es todo.\nGracias!'; 'C''est tout.\nMerci!'};

% Timing and visual parameters
wait_reset = 0.1;
value_reset = 0;
min_blank = 1.25;
max_rand_blank = 0.5;
min_lines_onoff = 0.5;
max_rand_lines_onoff = 0.2;
size_line = 5;
colorOval = [[255, 0, 0]; [255 255 0]];
gamepadname = 'Logitech';

% Event codes
pic_onoff = [[1 4 16]; [2 8 32]];
bits_for_break = [];
blank_on = 11;
lines_onoff = 13;
continue_msg_on = 19;
lines_flip_blank = 103;
lines_flip_pic = 22;
trial_on = 26;
data_signature_on = 64;
data_signature_off = 128;
stim_off = 35;

msgs = {'blank on'; 'lines on'; 'pic change'; 'lines change blank'; 'lines change pic'; 'lines off'; 'trial ended'};

rng('shuffle', 'twister');

% Build experiment structure
experiment.pwd = pwd;
experiment.params = params;
experiment.subtask = subtask;
experiment.date = datetime(now, 'ConvertFrom', 'datenum');
experiment.pic = pic_onoff;
experiment.blank_on = blank_on;
experiment.lines_onoff = lines_onoff;
experiment.continue_msg_on = continue_msg_on;
experiment.lines_flip_blank = lines_flip_blank;
experiment.lines_flip_pic = lines_flip_pic;
experiment.trial_on = trial_on;
experiment.stim_off = stim_off;
experiment.bits_for_break = bits_for_break;
experiment.data_signature = [data_signature_on data_signature_off];
experiment.value_reset = value_reset;
experiment.wait_reset = wait_reset;
experiment.ImageNames = ImageNames;
experiment.msgs = msgs;
experiment.deviceresp = params.device_resp;
experiment.with_reset = false;
experiment.P2REMOVE = P2REMOVE;
experiment.NREP = NREP;
experiment.NPICS = NPICS;
experiment.MANUAL_SELECT = MANUAL_SELECT;
experiment.MAX_TRIALS = MAX_TRIALS;
experiment.N_BLKS = n_blks;

% Save experiment properties
exp_prop_file = fullfile(experiment.folder_name, 'experiment_properties_online3.mat');
save(exp_prop_file, 'experiment');

% Language selection
Screen('Preference', 'SkipSyncTests', double(IsWin));
if strcmp(params.lang, 'english')
    ind_lang = 1;
elseif strcmp(params.lang, 'spanish')
    ind_lang = 2;
elseif strcmp(params.lang, 'french')
    ind_lang = 3;
else
    ind_lang = 1;
end

% Gamepad setup
gamepad_ix = [];
if strcmp(params.device_resp, 'gamepad')
    if IsWin
        clear JoyMEX;
        JoyMEX('init', 0);
    elseif IsLinux
        numGamepads = Gamepad('GetNumGamepads');
        if numGamepads == 0
            error('Gamepad not connected');
        else
            [~, gamepad_name] = GetGamepadIndices;
            idx = find(contains(gamepad_name, gamepadname, 'IgnoreCase', true), 1);
            gamepad_name = gamepad_name{idx};
            gamepad_ix = Gamepad('GetGamepadIndicesFromNames', gamepad_name);
        end
    else
        error('gamepad not coded for this OS');
    end
end

% Create results folder
result_folder = fullfile(experiment.folder_name, 'results');
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

% Initialize diary
diary_filename = [datestr(now, 'mm-dd-yy') '_' datestr(now, 'HH_MM_SSAM') '_ABTPRSVP_diary.txt'];
diary_file = fullfile(result_folder, [experiment.fname '_' diary_filename]);
diary_file = fullfile(diary_file(~isspace(diary_file)));
diary(diary_file);

disp([experiment.fname ':BEGIN']);
drawnow;

%% =======================================================================
%  SECTION 7: MAIN EXPERIMENT LOOP
%  =======================================================================

% Initialize tracking variables
scr_config_cell = {};
scr_end_cell = {};
available_pics = 1:totalp2load;
available_pics_cell = {};
stim_rm_cell = {};
selected2explore_cell = {};
selected2rm_cell = {};
selected2notremove = [];
selected2notremove_cell = {};
stim_rm_max_trials_cell = {};
same_units_cell = {};
same_categories_cell = {};
unused_pics = 1:height(ImageNames);

% CategLocaliz-specific tracking
categ_localiz_history = [];
img_info_table = [];
fetched_pics_cell = {};

if params.is_online
    % Online-specific initialization
    wc_sp_index_cell = cell(0, 1);
    pics_onset_cell = cell(0, 1);
    stimulus_cell = cell(0, 1);
    
    % Setup grapes structure
    grapes = struct;
    grapes.exp_type = 'ONLINE_RSVPSCR';
    grapes.time_pre = TIME_PRE;
    grapes.time_pos = TIME_POS;
    grapes.ImageNames = ImageNames;
    % For CategLocaliz, folder already contains full path; for others, prepend pics_root_processing
    if ~contains(subtask, 'CategLocaliz')
        grapes.ImageNames.folder = fullfile(params.pics_root_processing, grapes.ImageNames.folder);
    end
    
    % Setup sorter
    if DO_SORTING
        sorter = online_sorter(channels, priority_channels, SP2SORT, true, MAX_SORTING_CORES);
        if exist('TEMPLATES_FILE', 'var')
            sorter.load_sorting_results(TEMPLATES_FILE);
            sorting_done = true;
        end
    end
    
    % Initialize AsyncTrialProcessor if async mode is enabled
    if use_async
        async_num_workers = params.async_num_workers;
        if isempty(async_num_workers)
            % Use 70% of pool workers for async processing
            async_num_workers = max(2, floor(poolobj.NumWorkers * 0.7));
        end
        
        async_processor = AsyncTrialProcessor(channels, ch_filters, par, det_conf, ...
            'DO_SORTING', DO_SORTING, ...
            'sorter', sorter, ...
            'chan_label', chan_label, ...
            'b_remove_collisions', b_remove_collisions, ...
            'b_make_coll_plots', b_make_coll_plots, ...
            'use_dedicated_pool', false);  % Share pool with main
        async_spikes_added = [];  % Track trial IDs that already had spikes added to sorter
        
        fprintf('AsyncTrialProcessor ready with %d workers\n', async_num_workers);
    end
    
    % Note: BackgroundDataCollectorWorker is already initialized earlier
    % in the device setup section if use_background_collection is true.
    % The worker runs device_com in a separate parfeval process.
end



abort = 0;
n_scr = 0;
pre_times = NaN(1, 0);

% Initialize variables for cleanup (will be populated later)
window = [];
dev_used = [];
dig_out = [];
async_processor = [];
bg_collector = [];
cleanup_done = false;

% Setup cleanup handler for graceful shutdown on error/Ctrl+C
% Uses nested function to access workspace variables
cleanup_obj = onCleanup(@do_cleanup);

try
    % Open screen
    screens = Screen('Screens');
    whichScreen = max(screens);
    
    if params.use_daq
        dig_out = TTL_device(params.ttl_device);
    end
    
    if isfield(params, 'windowRect')
        [window, windowRect] = Screen(whichScreen, 'OpenWindow', 0, params.windowRect);
    else
        [window, windowRect] = Screen(whichScreen, 'OpenWindow', 0);
    end
    
    % Load textures
    tex_all = zeros(1, total_figures);
    imageRect = cell(total_figures, 1);
    destRect_all = cell(total_figures, 1);
    
    for i = 1:total_figures
        % For CategLocaliz, folder contains full path; for others, need to prepend pics_root_beh
        if contains(subtask, 'CategLocaliz')
            pic_path = fullfile(ImageNames.folder{i}, ImageNames.name{i});
        else
            pic_path = fullfile(params.pics_root_beh, ImageNames.folder{i}, ImageNames.name{i});
        end
        Im = imread(pic_path);
        nRows = size(Im, 1);
        nCols = size(Im, 2);
        
        % Validate dimensions based on subtask
        if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'DynamicSeman')
            if ~(nRows == 320 && nCols == 320)
                error('Picture %s with wrong dimensions', pic_path)
            end
        elseif contains(subtask, 'Bruno')
            if ~(nRows == 256 && nCols == 256)
                error('Picture %s with wrong dimensions', pic_path)
            end
        else
            if ~(nRows == 160 && nCols == 160)
                % Allow other sizes but warn
            end
        end
        
        imageRect{i} = SetRect(0, 0, nCols * scale_factor, nRows * scale_factor);
        destRect_all{i} = CenterRect(imageRect{i}, windowRect);
        tex_all(i) = Screen('MakeTexture', window, Im);
    end
    
    % Screen properties
    xcenter = windowRect(3) / 2;
    ycenter = windowRect(4) / 2;
    Priority(params.ptb_priority_high);
    ifi = Screen('GetFlipInterval', window, 200);
    slack = ifi / 8;
    flicker_duration = 2 * ifi;
    Priority(params.ptb_priority_normal);
    frame_rate = 1 / ifi;
    
    white = WhiteIndex(window);
    black = BlackIndex(window);
    if white == 1
        grey = 138 / 255;
    else
        grey = 138;
    end
    
    bgnd_color = black;
    if contains(subtask, 'FreqTag')
        bgnd_color = grey;
    end
    
    Screen('TextSize', window, 32);
    flickerSquare = flickerSquareLoc(windowRect, 24, 2, 'BottomLeft');
    
    experiment.xcenter = xcenter;
    experiment.ycenter = ycenter;
    experiment.frame_duration = ifi;
    experiment.frame_rate = frame_rate;
    experiment.flickerSquare = flickerSquare;
    
    % Keyboard queue setup
    keysOfInterest = zeros(1, 256);
    keysOfInterest([exitKey startKey continueKey]) = 1;
    
    save(exp_prop_file, 'experiment');
    
    % Ready prompt
    if ~params.auto_resp
        dresp = questdlg('Ready to begin?', 'Subject Ready?', 'OK', 'OK');
        if isempty(dresp)
            error('Dialog closed');
        end
    end
    
    % Save workspace before starting
    if ~strcmp(subtask, 'FirstTime')
        save(fullfile(experiment.folder_name, 'RSVP_SCR_workspace.mat'), '-regexp', '^(?!(M_PTB|f|backup_worker|bg_collector|async_processor|poolobj|collector_future|data_queue|collection_timer|sorter|dig_out|recording|custompath)$).');
    end
    
    k = 1;
    init_time = tic();
    exp_start_time = tic();  % For tracking total experiment time
    
    % Start recording
    if params.acq_network
        try
            recording.start();
            pre_times(k) = GetSecs;
            k = k + 1;
        catch ME
            errMsg = getReport(ME);
            disp(errMsg);
            error('Failed to start recording');
        end
        WaitSecs(6);
        fprintf('Recording started\n');
        fprintf('===============================\n');
    end
    
    % Data signature
    if params.use_daq
        dig_out.send(data_signature_on);
        WaitSecs(0.05);
        dig_out.send(data_signature_off);
        WaitSecs(0.45);
        dig_out.send(data_signature_on);
        WaitSecs(0.05);
        dig_out.send(data_signature_off);
        WaitSecs(0.45);
        dig_out.send(data_signature_on);
        WaitSecs(0.05);
        dig_out.send(data_signature_off);
        pre_times(k) = GetSecs;
        k = k + 1;
    end
    
    for d = dev_used
        KbQueueCreate(d, keysOfInterest);
        KbQueueStart(d);
    end
    
    lines_offset = 50;
    
    % DynamicSeman list handling
    if contains(subtask, 'DynamicSeman')
        lists_path = fullfile(params.pics_root_beh, 'pics_space', 'seman_lists');
        lists = dir(fullfile(lists_path, '*.*'));
        lists = lists(~[lists.isdir]);
        if length(lists) > 100
            lists = lists(end-99:end);
        end
        
        selectedLists = {};
        first_lists = {};
        last_list = {};
        
        for i = 1:length(lists)
            if params.run_num == 1
                for y = 1:length(lists)
                    if contains(lists(y).name, '-A-') && ~contains(lists(y).name, '-10-')
                        first_lists{y} = lists(y).name;
                    elseif contains(lists(y).name, '-A-10-')
                        last_list{y} = lists(y).name;
                    end
                end
            else
                for y = 1:length(lists)
                    if contains(lists(y).name, '-B-') && ~contains(lists(y).name, '-10-')
                        first_lists{y} = lists(y).name;
                    elseif contains(lists(y).name, '-B-10-')
                        last_list{y} = lists(y).name;
                    end
                end
            end
        end
        sorted_lists = [first_lists, last_list];
        sorted_lists = sorted_lists(~cellfun('isempty', sorted_lists));
    end
    
    %% ===================================================================
    %  SECTION 8: SUBSCREENING LOOP
    %  ===================================================================
    
    ini = [];
    fin = [];
    fail_safe_mode = ~params.is_online;
    safe_not_done = true;
    
    while (n_scr < length(NPICS)) && safe_not_done && ~abort
        n_scr = n_scr + 1;
        
        % Skip subscreenings with 0 pictures
        if NPICS(n_scr) == 0
            fprintf('Skipping screening %d (NPICS=0)\n', n_scr);
            continue;
        end
        
        % Determine pics to use for this subscreening
        pics2use = available_pics(1:NPICS(n_scr));
        if isempty(ini)
            ini = 1;
        else
            ini = ini + NREP(n_scr - 1);
        end
        fin = ini + NREP(n_scr) - 1;
        
        stim_trial_counter(pics2use) = stim_trial_counter(pics2use) + NREP(n_scr);
        
        % Load textures for current pics
        tex = tex_all(pics2use);
        destRect = destRect_all(pics2use);
        
        % Generate screening configuration
        if contains(subtask, 'DynamicSeman')
            scr_config = shuffle_rsvpSCR_online3(NREP(n_scr), NPICS(n_scr), subtask, ImageNames, sorted_lists, n_scr, ini, fin, lists_path);
        elseif contains(subtask, 'FreqTag')
            scr_config = shuffle_rsvpSCR_online3_freqtag(NREP(n_scr), NPICS(n_scr), bgnd_color, subtask, ImageNames);
        elseif contains(subtask, 'CategLocaliz') && n_scr == n_blks
            % Last block of CategLocaliz: use shuffle_rsvp_dynamic_2 to match
            % the preview from choose_pics_to_keep. This correctly uses
            % MAX_TRIALS and stim_trial_count to determine how many more
            % repetitions each picture needs, instead of using NREP blindly.
            [experiment, scr_config, extra_rep_pics] = shuffle_rsvp_dynamic_2(experiment, pics2use, n_scr);
            if numel(extra_rep_pics) > 0
                fprintf('%d pics will be shown an extra time: %s\n', ...
                    numel(extra_rep_pics), strjoin(experiment.ImageNames.name(extra_rep_pics), ' '));
            end
        else
            scr_config = shuffle_rsvpSCR_online3(NREP(n_scr), NPICS(n_scr), subtask);
        end
        
        scr_config.pics2use = pics2use;
        scr_config.fail_safe_mode = fail_safe_mode;
        scr_config.manual_select = MANUAL_SELECT(min(n_scr, length(MANUAL_SELECT)));
        scr_config_cell{end+1} = scr_config;
        
        order_pic = scr_config.order_pic;
        order_ISI = scr_config.order_ISI;
        ISI = scr_config.ISI;
        seq_length = scr_config.seq_length;
        Nseq = scr_config.Nseq;
        lines_change = scr_config.lines_change;
        NISI = numel(ISI);
        
        % Initialize timing arrays
        times = NaN * ones(1, Nseq * (NISI + 1 + 2 + NISI * seq_length + 6 + 1));
        times(1:length(pre_times)) = pre_times;
        k = numel(pre_times) + 1;
        t_stimon = NaN * ones(1, Nseq * seq_length);
        t_fliptime = NaN * ones(1, Nseq * seq_length);
        t_DAQpic = NaN * ones(1, Nseq * seq_length);
        time_wait = NaN * ones(1, Nseq);
        inds_pics = zeros(1, seq_length * NISI * Nseq);
        inds_start_seq = zeros(1, Nseq);
        times_break = [];
        
        randTime_blank = min_blank + max_rand_blank * rand(NISI + 1, Nseq);
        randTime_lines_on = min_lines_onoff + max_rand_lines_onoff * rand(1, Nseq);
        randTime_lines_off = randTime_blank(NISI + 1, :) - (min_lines_onoff + max_rand_lines_onoff * rand(1, Nseq));
        
        % Online processing setup for this subscreening
        if params.is_online
            TRIAL_LEN = (scr_config.ISI(1) * scr_config.seq_length + 3 + 5) * 1.2;
            if params.debug
                TRIAL_LEN = 5 * TRIAL_LEN;
            end
            
            data_n = length(channels);
            data = cell(data_n, 1);
            for i = 1:length(data)
                data{i} = zeros(ceil(TRIAL_LEN * 30000), 1);
            end
            
            Event_Time = cell(Nseq, 1);
            if params.use_photodiodo
                Event_Time_pdiode = cell(Nseq, 1);
            end
            Event_Value = cell(Nseq, 1);
            detecctions = cell(Nseq, length(channels));
            init_times = cell(Nseq, 1);
        end
        
        % Average timing components (from parameters defined in Section 6)
        avg_blank = min_blank + max_rand_blank / 2;
        avg_lines_on = min_lines_onoff + max_rand_lines_onoff / 2;
        avg_lines_off = avg_blank - avg_lines_on;
        
        % Calculate estimated duration for this block using actual ISI from scr_config
        seq_stim_time = sum(ISI) * seq_length;           % picture presentation time per sequence
        seq_blank_time = avg_blank * (NISI + 1);          % blank periods (one before each ISI + final)
        seq_lines_time = avg_lines_on + avg_lines_off;    % lines on/off time
        seq_overhead = 3;                                 % inter-sequence processing/break
        estimated_duration = ((seq_stim_time + seq_blank_time + seq_lines_time + seq_overhead) * Nseq) / 60;
        
        % On first block, print full SCREENING OVERVIEW for all blocks
        if n_scr == 1
            % Use the actual ISI from scr_config to project all blocks
            actual_ISI = mean(ISI);
            actual_min_seq_length = ceil(30 / actual_ISI);
            
            fprintf('\n============ SCREENING OVERVIEW ============\n');
            fprintf('Total screening blocks: %d\n', n_blks);
            fprintf('Total images available: %d\n', total_figures);
            fprintf('Timing params: ISI=%.3fs, avg_blank=%.2fs, avg_lines_on=%.2fs\n', ...
                actual_ISI, avg_blank, avg_lines_on);
            total_estimated_time = 0;
            for blk = 1:n_blks
                blk_npics = NPICS(min(blk, length(NPICS)));
                blk_nrep = NREP(min(blk, length(NREP)));
                blk_p2remove = P2REMOVE(min(blk, length(P2REMOVE)));
                
                % Estimate Nseq and seq_length (same logic as shuffle functions)
                if blk_npics < actual_min_seq_length
                    blk_nrepxseq = ceil(actual_min_seq_length / blk_npics);
                    blk_seq_length = blk_npics * blk_nrepxseq;
                    blk_nseq = ceil(blk_nrep / blk_nrepxseq);
                else
                    blk_nseqxrep = floor(blk_npics / actual_min_seq_length);
                    blk_seq_length = floor(blk_npics / blk_nseqxrep);
                    blk_nseq = blk_nseqxrep * blk_nrep;
                end
                
                blk_stim_time = actual_ISI * blk_seq_length;
                blk_est_duration = ((blk_stim_time + avg_blank * 2 + avg_lines_on + avg_lines_off + seq_overhead) * blk_nseq) / 60;
                total_estimated_time = total_estimated_time + blk_est_duration;
                
                fprintf('  Block %d: NPICS=%d, NREP=%d, P2REMOVE=%d, ~Nseq=%d, seqlen=%d, est=%.1f min\n', ...
                    blk, blk_npics, blk_nrep, blk_p2remove, blk_nseq, blk_seq_length, blk_est_duration);
            end
            fprintf('----------------------------------------------\n');
            fprintf('Total estimated duration: %.1f min (%.1f hours)\n', total_estimated_time, total_estimated_time/60);
            fprintf('==============================================\n\n');
        end
        
        fprintf('\nSubscr %d: Nseq=%d, Npics=%d, seqlen=%d, ISI=%.2fs, est=%.1f min\n', ...
            n_scr, Nseq, NPICS(n_scr), seq_length, mean(ISI), estimated_duration);
        
        % Begin sequence presentation
        if ~params.auto_resp && n_scr == 1
            print_message(message_begin{ind_lang}, black, window);
            pressed = false;
            firstPress = zeros(1, 256);
            while ~(pressed && any(firstPress([startKey exitKey])))
                [pressed, firstPress, ~, ~] = multiKbQueueCheck(dev_used);
            end
            if pressed && firstPress(exitKey) > 0
                abort = 1;
                break;
            end
        elseif ~params.auto_resp
            print_message(message_continue{ind_lang}, black, window);
            [~, ~, pressed, firstPress] = get_response(dev_used, params.device_resp, [exitKey continueKey], 0.2, params.auto_resp, gamepad_ix);
            if pressed && firstPress(exitKey) > 0
                abort = 1;
                break;
            end
        end
        
        for d = dev_used
            KbQueueFlush(d);
        end
        
        Priority(params.ptb_priority_high);
        HideCursor;
        iind = 1;
        
        %% ===============================================================
        %  SEQUENCE LOOP
        %  ===============================================================
        
        for irep = 1:Nseq
            fprintf('%d, ', irep);
            
            if params.is_online
                % === WORKER-BASED COLLECTION MODE ===
                if use_background_collection && ~isempty(bg_collector)
                    % Start trial collection in worker (worker handles clear_buffer internally)
                    bg_collector.start_trial();
                    trial_data_loss = false;

                    % Initialize local references (will be populated from collector at end)
                    trial_datacounter = zeros(length(channels), 1);
                    trial_streams_timestamps = {};
                    trial_Event_Time = [];
                    trial_Event_Value = [];
                    trial_Event_Time_pdiode = [];
                else
                    % === ORIGINAL INLINE MODE ===
                    device_com('clear_buffer');
                    pause(0.2)
                    
                    % Initialize per-trial data accumulators
                    trial_datacounter = zeros(length(channels), 1);
                    for ii = 1:length(channels)
                        data{ii} = zeros(ceil(TRIAL_LEN * 30000), 1);
                    end
                    trial_streams_timestamps = {};
                    trial_Event_Time = [];
                    trial_Event_Value = [];
                    trial_Event_Time_pdiode = [];
                    trial_data_loss = false;
                    
                    % Immediately get first stream to anchor timestamp
                    [data, trial_datacounter, trial_streams_timestamps, trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, ~] = ...
                        collect_stream_data(data, trial_datacounter, trial_streams_timestamps, ...
                        trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, params.use_photodiodo);
                end
            end
            
            ich_blank = 1;
            ich_pic = 1;
            WaitSecs(0.150);
            for d = dev_used
                KbQueueFlush(d);
            end
            
            % Blank screen
            Screen('FillRect', window, bgnd_color);
            Screen('FillRect', window, black, flickerSquare);
            times(k) = Screen('Flip', window);
            if params.use_daq
                dig_out.send(blank_on);
            end
            inds_start_seq(irep) = k;
            tprev = times(k);
            
            color_up = scr_config.color_start.up{irep};
            color_down = scr_config.color_start.down{irep};
            color_oval = colorOval(randsample([1 2], 1), :);
            
            % Draw lines
            Screen('FillRect', window, bgnd_color);
            Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
            Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
            if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
            end
            Screen('FillRect', window, black, flickerSquare);
            
            if params.use_daq && experiment.with_reset
                WaitSecs('UntilTime', times(k) + wait_reset);
                dig_out.send(value_reset);
            end
            k = k + 1;
            
            times(k) = Screen('Flip', window, times(k-1) + randTime_lines_on(1, irep));
            if params.use_daq
                dig_out.send(lines_onoff);
            end
            
            % Handle line changes during blank
            if lines_change{irep}{1}{ich_blank, 1} == 1
                color_up = lines_change{irep}{1}{ich_blank, 3};
                color_down = lines_change{irep}{1}{ich_blank, 4};
                Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                    if isequal(color_oval, colorOval(1, :))
                        color_oval = colorOval(2, :);
                    else
                        color_oval = colorOval(1, :);
                    end
                    Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                end
                
                if params.use_daq && experiment.with_reset
                    WaitSecs('UntilTime', times(k) + wait_reset);
                    dig_out.send(value_reset);
                end
                k = k + 1;
                times(k) = Screen('Flip', window, times(k-1) + lines_change{irep}{1}{ich_blank, 2});
                if params.use_daq
                    dig_out.send(lines_flip_blank);
                end
                ich_blank = ich_blank + 1;
            end
            k = k + 1;
            
            % ISI loop - present pictures
            for iISI = 1:NISI
                which_ISI = order_ISI(iISI, irep);
                
                % First picture in ISI
                Screen('DrawTexture', window, tex(order_pic(1, which_ISI, irep)), [], destRect{order_pic(1, which_ISI, irep)}, 0);
                Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                    Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                end
                Screen('FillRect', window, white, flickerSquare);
                
                [times(k), t_stimon(iind), t_fliptime(iind)] = Screen('Flip', window, tprev + randTime_blank(iISI, irep) - slack, 1);
                
                if params.use_daq
                    dig_out.send(pic_onoff(2, ceil(3 * irep / Nseq)));
                    t_DAQpic(iind) = GetSecs;
                end
                inds_pics(iind) = k;
                tprev = times(k);
                iind = iind + 1;
                
                Screen('FillRect', window, black, flickerSquare);
                Screen('Flip', window, times(k) + flicker_duration - slack);
                
                if params.use_daq && experiment.with_reset
                    WaitSecs('UntilTime', times(k) + wait_reset);
                    dig_out.send(value_reset);
                end
                k = k + 1;
                
                % Quick inline data collection to prevent buffer overflow (only in non-background mode)
                if params.is_online && ~use_background_collection
                    [data, trial_datacounter, trial_streams_timestamps, trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, loss] = ...
                        collect_stream_data(data, trial_datacounter, trial_streams_timestamps, ...
                        trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, params.use_photodiodo);
                    trial_data_loss = trial_data_loss || loss;
                end
                
                % Handle line changes during pic
                if which_ISI == lines_change{irep}{2}{ich_pic, 1} && lines_change{irep}{2}{ich_pic, 5} == 1
                    Screen('DrawTexture', window, tex(order_pic(1, which_ISI, irep)), [], destRect{order_pic(1, which_ISI, irep)}, 0);
                    color_up = lines_change{irep}{2}{ich_pic, 3};
                    color_down = lines_change{irep}{2}{ich_pic, 4};
                    Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                    Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                    Screen('FillRect', window, black, flickerSquare);
                    
                    if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                        if isequal(color_oval, colorOval(1, :))
                            color_oval = colorOval(2, :);
                        else
                            color_oval = colorOval(1, :);
                        end
                        Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                    end
                    times(k) = Screen('Flip', window, tprev + lines_change{irep}{2}{ich_pic, 2} - slack);
                    if params.use_daq
                        dig_out.send(lines_flip_pic);
                    end
                    ich_pic = ich_pic + 1;
                    k = k + 1;
                end
                
                % FreqTag stim off
                if contains(subtask, 'FreqTag')
                    Screen('FillRect', window, bgnd_color, destRect{order_pic(1, which_ISI, irep)});
                    Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                    Screen('FillRect', window, black, flickerSquare);
                    times(k) = Screen('Flip', window, tprev + 0.8 * ISI(which_ISI) - slack);
                    if params.use_daq
                        dig_out.send(stim_off);
                    end
                    k = k + 1;
                end
                
                % Remaining pictures in sequence
                for ipic = 2:seq_length
                    Screen('DrawTexture', window, tex(order_pic(ipic, which_ISI, irep)), [], destRect{order_pic(ipic, which_ISI, irep)}, 0);
                    Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                    Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                    if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                        Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                    end
                    Screen('FillRect', window, white, flickerSquare);
                    
                    [times(k), t_stimon(iind), t_fliptime(iind)] = Screen('Flip', window, tprev + ISI(which_ISI) - slack, 1);
                    if params.use_daq
                        dig_out.send(pic_onoff(mod(ipic, 2) + 1, ceil(3 * irep / Nseq)));
                        t_DAQpic(iind) = GetSecs;
                    end
                    inds_pics(iind) = k;
                    tprev = times(k);
                    iind = iind + 1;
                    
                    Screen('FillRect', window, black, flickerSquare);
                    Screen('Flip', window, times(k) + flicker_duration - slack);
                    
                    if params.use_daq && experiment.with_reset
                        WaitSecs('UntilTime', times(k) + wait_reset);
                        dig_out.send(value_reset);
                    end
                    k = k + 1;
                    
                    % Quick inline data collection to prevent buffer overflow (only in non-background mode)
                    if params.is_online && ~use_background_collection
                        [data, trial_datacounter, trial_streams_timestamps, trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, loss] = ...
                            collect_stream_data(data, trial_datacounter, trial_streams_timestamps, ...
                            trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, params.use_photodiodo);
                        trial_data_loss = trial_data_loss || loss;
                    end
                    
                    % Line changes during pic
                    if which_ISI == lines_change{irep}{2}{ich_pic, 1} && lines_change{irep}{2}{ich_pic, 5} == ipic
                        Screen('DrawTexture', window, tex(order_pic(ipic, which_ISI, irep)), [], destRect{order_pic(ipic, which_ISI, irep)}, 0);
                        color_up = lines_change{irep}{2}{ich_pic, 3};
                        color_down = lines_change{irep}{2}{ich_pic, 4};
                        Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                        Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                        Screen('FillRect', window, black, flickerSquare);
                        if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                            if isequal(color_oval, colorOval(1, :))
                                color_oval = colorOval(2, :);
                            else
                                color_oval = colorOval(1, :);
                            end
                            Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                        end
                        times(k) = Screen('Flip', window, tprev + lines_change{irep}{2}{ich_pic, 2} - slack);
                        if params.use_daq
                            dig_out.send(lines_flip_pic);
                        end
                        ich_pic = ich_pic + 1;
                        k = k + 1;
                    end
                    
                    % FreqTag stim off
                    if contains(subtask, 'FreqTag')
                        Screen('FillRect', window, bgnd_color, destRect{order_pic(ipic, which_ISI, irep)});
                        Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                        Screen('FillRect', window, black, flickerSquare);
                        times(k) = Screen('Flip', window, tprev + 0.8 * ISI(which_ISI) - slack);
                        if params.use_daq
                            dig_out.send(stim_off);
                        end
                        k = k + 1;
                    end
                end
                
                % Blank between ISIs
                Screen('FillRect', window, bgnd_color);
                Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                Screen('FillRect', window, black, flickerSquare);
                if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                    Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                end
                times(k) = Screen('Flip', window, tprev + ISI(which_ISI) - slack);
                if params.use_daq
                    dig_out.send(lines_onoff);
                end
                tprev = times(k);
                
                if params.use_daq && experiment.with_reset
                    WaitSecs('UntilTime', times(k) + wait_reset);
                    dig_out.send(value_reset);
                end
                k = k + 1;
                
                % Line changes during blank
                if lines_change{irep}{1}{ich_blank, 1} == 1 + iISI
                    color_up = lines_change{irep}{1}{ich_blank, 3};
                    color_down = lines_change{irep}{1}{ich_blank, 4};
                    Screen('DrawLine', window, color_up, destRect{1}(1), destRect{1}(2) - lines_offset, destRect{1}(3), destRect{1}(2) - lines_offset, size_line);
                    Screen('DrawLine', window, color_down, destRect{1}(1), destRect{1}(4) + lines_offset, destRect{1}(3), destRect{1}(4) + lines_offset, size_line);
                    Screen('FillRect', window, black, flickerSquare);
                    if strcmp(subtask, 'CategLocaliz') || contains(subtask, 'FreqTag')
                        if isequal(color_oval, colorOval(1, :))
                            color_oval = colorOval(2, :);
                        else
                            color_oval = colorOval(1, :);
                        end
                        Screen('FillOval', window, color_oval, [xcenter - size_point/2, ycenter - size_point/2, xcenter + size_point/2, ycenter + size_point/2]);
                    end
                    times(k) = Screen('Flip', window, times(k-1) + lines_change{irep}{1}{ich_blank, 2});
                    if params.use_daq
                        dig_out.send(lines_flip_blank);
                    end
                    ich_blank = ich_blank + 1;
                    k = k + 1;
                end
                
                % === INCREMENTAL DATA COLLECTION DURING TRIAL === (only in non-background mode)
                if params.is_online && ~use_background_collection
                    [data, trial_datacounter, trial_streams_timestamps, trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, loss] = ...
                        collect_stream_data(data, trial_datacounter, trial_streams_timestamps, ...
                        trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, params.use_photodiodo);
                    trial_data_loss = trial_data_loss || loss;
                end
            end
            
            % End of sequence
            Screen('FillRect', window, bgnd_color);
            Screen('FillRect', window, black, flickerSquare);
            times(k) = Screen('Flip', window, tprev + randTime_lines_off(1, irep) - slack);
            if params.use_daq
                dig_out.send(blank_on);
            end
            k = k + 1;
            
            WaitSecs(tprev + randTime_blank(NISI + 1, irep) - GetSecs);
            
            % Check for abort
            [pressed, firstPress, ~, ~] = multiKbQueueCheck(dev_used);
            if pressed && firstPress(exitKey) > 0
                abort = 1;
                disp('Abort key pressed.')
                break
            end
            
            % Check if this is the final trial of the final subscreening
            is_final_trial = (irep == Nseq) && (n_scr == length(NPICS));
            
            % Show appropriate message while processing
            if is_final_trial
                print_message(message_final{ind_lang}, black, window);
            else
                print_message(message_wait{ind_lang}, black, window);
            end
            
            % === ONLINE PROCESSING FOR THIS TRIAL ===
            if params.is_online
                % === GET TRIAL DATA ===
                if use_background_collection && ~isempty(bg_collector)
                    % Stop worker collection and get accumulated data
                    % get_trial_data() sends stop command and returns data
                    [data, trial_datacounter, events] = bg_collector.get_trial_data();
                    trial_Event_Time = events.time;
                    trial_Event_Value = events.value;
                    trial_Event_Time_pdiode = events.pdiode;
                    trial_streams_timestamps = events.timestamps;
                    trial_data_loss = events.data_loss;
                    
                    % Print collection stats for first trial
                    if n_scr == 1 && irep == 1 && isfield(events, 'collection_count')
                        fprintf('\n--- Worker collection stats (trial 1) ---\n');
                        fprintf('  Collections: %d\n', events.collection_count);
                        fprintf('  Data samples per channel: %d (%.2f sec)\n', ...
                            trial_datacounter(1), trial_datacounter(1)/30000);
                        fprintf('------------------------------------------\n');
                    end
                else
                    % Get any remaining neural data from buffer (original mode)
                    [data, trial_datacounter, trial_streams_timestamps, trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, loss] = ...
                        collect_stream_data(data, trial_datacounter, trial_streams_timestamps, ...
                        trial_Event_Time, trial_Event_Value, trial_Event_Time_pdiode, params.use_photodiodo);
                    trial_data_loss = trial_data_loss || loss;
                end
                
                % Report data loss if detected
                if trial_data_loss
                    warning('DATA LOSS detected in trial %d', irep);
                    outputinfo{end+1} = sprintf('Data loss in trial %d', irep);
                end
                
                % Use accumulated data counter
                datacounter = trial_datacounter;
                
                % Store accumulated events
                Event_Time{irep} = trial_Event_Time;
                Event_Value{irep} = trial_Event_Value;
                if params.use_photodiodo
                    Event_Time_pdiode{irep} = trial_Event_Time_pdiode;
                end
                
                % Debug output for first trial of first subscreening
                %if n_scr == 1 && irep == 1
                %fprintf('\n---Trial %d data collection summary ---\n', irep);
                %fprintf('  DAQ events: %d\n', numel(trial_Event_Time));
                %if params.use_photodiodo
                %    fprintf('  Photodiode events: %d\n', numel(trial_Event_Time_pdiode));
                %end
                %fprintf('  Data samples per channel: %d (%.2f sec)\n', ...
                %    trial_datacounter(1), trial_datacounter(1)/30000);
                %fprintf('-------------------------------------------\n');
                %end
                
                init_times{irep} = trial_streams_timestamps;
                
                % === ASYNC vs SYNC PROCESSING ===
                if use_async && ~isempty(async_processor)
                    % ASYNC MODE: Submit trial for background processing
                    % This allows PTB to continue immediately with next trial
                    try
                        trial_id = (n_scr - 1) * 1000 + irep;  % Unique ID per trial
                        async_processor.submit_trial(data, datacounter, trial_id);
                        %fprintf('Trial %d submitted async. ', irep);
                        
                        % Trigger sorting after ntrial2sort trials (async)
                        if DO_SORTING && n_scr == 1 && irep == ntrial2sort && ~mu_only
                            % Wait for previous trials to complete before sorting
                            fprintf('\nWaiting for trials 1-%d to complete for sorting...\n', ntrial2sort);
                            async_processor.wait_all(120);  % Wait up to 2 min
                            
                            % Add all spikes to sorter
                            for tid = 1:ntrial2sort
                                trial_id_sort = (n_scr - 1) * 1000 + tid;
                                async_processor.add_spikes_to_sorter(trial_id_sort);
                                async_spikes_added(end+1) = trial_id_sort;
                            end
                            
                            sorter.do_remaining_sorting();
                            fprintf('Spike Sorting Done.\n');
                            sorting_done = true;
                        end
                    catch ME_async
                        warning('ASYNC PROCESSING FAILED: %s. Screening and collection continue.', ME_async.message);
                        fprintf('\n*** ASYNC LOST - Trial %d not processed. Continuing with screening. ***\n', irep);
                        % Disable async for remaining trials to prevent repeated errors
                        use_async = false;
                        fprintf('*** ASYNC DISABLED for remaining trials. ***\n');
                    end
                else
                    % SYNC MODE: Original blocking processing
                    if DO_SORTING
                        sorter.retrieve_sorting_results();
                        spikes = cell(length(channels), 1);
                        f(1:length(channels)) = parallel.FevalFuture;
                        for i = 1:length(channels)
                            f(i) = parfeval(@get_spikes_online, 2, ...
                                data{i}(1:datacounter(i)), ch_filters{i}.det, par, det_conf(i));
                        end
                        for i = 1:length(channels)
                            wait(f(i));
                            [spikes{i}, detecctions{irep, i}] = fetchOutputs(f(i));
                            if numel(spikes{i}) < 1
                                fprintf('No spikes in channel %d. See if micros are connected. \n', channels(i))
                            end
                        end
                        clear f
                        
                        if b_remove_collisions
                            trial_det = detecctions(irep, :);
                            try
                                [spikes, detecctions(irep, :)] = remove_collisions(data, trial_det, ...
                                    chan_label, spikes, b_make_coll_plots);
                            catch ME_coll
                                if ~b_collision_warning_shown
                                    
                                    warning('Error in remove_collisions: %s', ME_coll.message);
                                    b_collision_warning_shown = true;
                                end
                            end
                        end
                        
                        sorter.add_spikes(spikes);
                        
                        if n_scr == 1 && irep == ntrial2sort && ~mu_only
                            sorter.do_remaining_sorting();
                            fprintf('Spike Sorting Done.\n');
                            sorting_done = true;
                        end
                    else
                        parfor i = 1:length(channels)
                            detecctions{irep, i} = detect_mu_online(data{i}(1:datacounter(i)), ch_filters{i}.det, par, det_conf(i));
                        end
                    end
                    
                    fprintf('Trial %d processed. ', irep);
                end
            end
            
            times(k) = GetSecs;
            if params.use_daq
                dig_out.send(continue_msg_on);
            end
            k = k + 1;
            
            % Check if this is the final trial of the final subscreening
            is_final_trial = (irep == Nseq) && (n_scr == length(NPICS));
            
            if is_final_trial
                % Final trial - message already shown before processing, just exit loop
                Screen('FillRect', window, bgnd_color);
                Screen('FillRect', window, black, flickerSquare);
                drawnow;
                break;
            elseif irep == Nseq
                % Last trial of this subscreening but not final - just continue to next subscr
                % message_wait was already shown before processing
                drawnow;
            else
                % Not final trial - processing done, show ready to continue
                print_message(message_continue{ind_lang}, black, window);
                
                Screen('FillRect', window, bgnd_color);
                Screen('FillRect', window, black, flickerSquare);
                
                for d = dev_used
                    KbQueueFlush(d);
                end
                [~, ~, pressed, firstPress] = get_response(dev_used, params.device_resp, [exitKey continueKey], 0.2, params.auto_resp, gamepad_ix);
                times(k) = GetSecs;
                if params.use_daq
                    dig_out.send(trial_on);
                end
                k = k + 1;
                for d = dev_used
                    KbQueueFlush(d);
                end
                if pressed && firstPress(exitKey) > 0
                    abort = 1;
                    break
                end
            end
            
            drawnow;
        end % End sequence loop
        
        fprintf('\n');
        
        %% ===============================================================
        %  SECTION 9: POST-SUBSCREENING PROCESSING
        %  ===============================================================
        
        % Save screening end data
        scr_end = struct;
        times(isnan(times)) = [];
        inds_pics(inds_pics == 0) = [];
        inds_start_seq(inds_start_seq == 0) = [];
        
        scr_end.times = times;
        scr_end.t_stimon = t_stimon;
        scr_end.t_fliptime = t_fliptime;
        scr_end.t_DAQpic = t_DAQpic;
        scr_end.inds_pics = inds_pics;
        scr_end.inds_start_seq = inds_start_seq;
        scr_end.abort = abort;
        scr_end.last_rep = irep - abort;
        scr_end.times_break = times_break;
        scr_end.time_wait = time_wait;
        scr_end_cell{end+1} = scr_end;
        
        % Online post-processing (also collect async results on abort)
        if params.is_online
            available_trials = scr_end.last_rep;
            if ~abort && available_trials ~= scr_config.Nseq
                warning('Some trials lost: %d of %d', available_trials, scr_config.Nseq);
            end

          if available_trials > 0
            % === COLLECT ASYNC RESULTS ===
            if use_async && ~isempty(async_processor)
                fprintf('\n--- Collecting async processing results for subscr %d ---\n', n_scr);


                async_processor.wait_all(180);  % Wait up to 3 min
                
                
                % Collect detections from async results
                detecctions = cell(available_trials, length(channels));
                all_spikes = cell(available_trials, 1);
                
                for trial_idx = 1:available_trials
                    trial_id = (n_scr - 1) * 1000 + trial_idx;
                    [det_result, spk_result, success] = async_processor.get_trial_results(trial_id, 'timeout', 30);
                    
                    if success
                        % Store detections (convert from row to proper format)
                        for ch_idx = 1:length(channels)
                            if ch_idx <= numel(det_result)
                                detecctions{trial_idx, ch_idx} = det_result{ch_idx};
                            end
                        end
                        all_spikes{trial_idx} = spk_result;
                        
                        % Add spikes to sorter if not already done at ntrial2sort checkpoint
                        if DO_SORTING && ~isempty(spk_result) && ~ismember(trial_id, async_spikes_added)
                            async_processor.add_spikes_to_sorter(trial_id);
                            async_spikes_added(end+1) = trial_id;
                        end
                    else
                        warning('Failed to get results for trial %d', trial_idx);
                    end
                end
                
                % Print async statistics
                %stats = async_processor.get_statistics();
                %fprintf('Async stats: %d submitted, %d completed, avg time %.1f sec\n', ...
                %    stats.total_submitted, stats.total_completed, stats.mean_processing_time);
                fprintf('\n--- Async collection complete ---\n\n');
            end
            
            % Process spike times
            %t_post = tic;
            %fprintf('\n--- Post-subscreening processing (subscr %d) ---\n', n_scr);
            valid_trials = [];
            pics_onset = [];
            blank_seq_beg_str = [experiment.blank_on, experiment.lines_onoff];
            blank_seq_end_str = [experiment.blank_on, experiment.continue_msg_on];
            
            wc_sp_index_cell{n_scr} = cell(size(detecctions, 2), 1);
            seq_beg_blanks = cell(available_trials, 1);
            seq_end_blanks = cell(available_trials, 1);
            daqtrials = 0;
            
            for ntrial = 1:available_trials
                seq_daq_events = Event_Value{ntrial};
                seq_daq_events_times = Event_Time{ntrial};
                
                for j = 1:size(detecctions, 2)
                    wc_sp_index_cell{n_scr}{j} = [wc_sp_index_cell{n_scr}{j}, ...
                        (double(detecctions{ntrial, j}) + init_times{ntrial}{1}(j)) / 30];
                end
                
                blank_on_seq_beg_idx = strfind(seq_daq_events', blank_seq_beg_str);
                blank_on_seq_end_idx = strfind(seq_daq_events', blank_seq_end_str);
                
                if ~isempty(blank_on_seq_beg_idx)
                    seq_beg_blanks{ntrial} = seq_daq_events_times(blank_on_seq_beg_idx:blank_on_seq_beg_idx+1);
                end
                if ~isempty(blank_on_seq_end_idx)
                    seq_end_blanks{ntrial} = seq_daq_events_times(blank_on_seq_end_idx:blank_on_seq_end_idx+1);
                end
                
                % Try photodiode first, fall back to DAQ if it fails
                if params.use_photodiodo && ~isempty(Event_Time_pdiode{ntrial})
                    [complete_times, text_out] = fix_photodiode_times(Event_Time_pdiode{ntrial}, ntrial, scr_config);
                    if isempty(complete_times)
                        % Photodiode failed - save error and fall back to DAQ
                        daqtrials = daqtrials + 1;
                        Event_Time_pdiode_trial = Event_Time_pdiode{ntrial};
                        save(fullfile(result_folder, sprintf('pdiode_error_subscr%d_ntrial%d', n_scr, ntrial)), ...
                            'Event_Time_pdiode_trial', 'ntrial', 'scr_config');
                        [complete_times, text_out] = fix_onset_times(seq_daq_events_times, seq_daq_events, ntrial, experiment, scr_config);
                    end
                else
                    [complete_times, text_out] = fix_onset_times(seq_daq_events_times, seq_daq_events, ntrial, experiment, scr_config);
                end
                
                if ~isempty(text_out)
                    if ischar(text_out)
                        if ~strcmpi(text_out, 'no changes needed')
                            outputinfo{end+1} = text_out;
                        end
                    else
                        outputinfo = [outputinfo, text_out];
                    end
                end
                
                if ~isempty(complete_times)
                    pics_onset = [pics_onset, complete_times];
                    valid_trials(end+1) = ntrial;
                else
                    warning('Unable to detect events in trial %d, subscreening %d.', ntrial, n_scr);
                end
            end
            
            % Summary of timing source usage for this subscreening
            if params.use_photodiodo
                pdiode_trials = available_trials - daqtrials;
                %fprintf('Subscr %d timing: %d trials via photodiode, %d via DAQ fallback\n', ...
                %    n_scr, pdiode_trials, daqtrials);
            else
                fprintf('Subscr %d timing: %d trials via DAQ\n', n_scr, available_trials);
            end
            
            if daqtrials > 0
                fprintf('WARNING: Used DAQ fallback for %d trials (photodiode failed)\n', daqtrials);
                outputinfo{end+1} = sprintf('Using DAQ events in %d trials', daqtrials);
            end
            
            final_Nseq = length(valid_trials);
            pics_onset = reshape(pics_onset, seq_length, NISI, final_Nseq);
            pics_onset_cell{n_scr} = pics_onset;
            
            % Create stimulus structure
            stimulus = create_stimulus_online(scr_config.order_pic(:, :, valid_trials), NISI, ...
                scr_config.pics2use, experiment.ImageNames.name(scr_config.pics2use), ...
                scr_config.ISI, scr_config.order_ISI(:, valid_trials));
            stimulus_cell{n_scr} = stimulus;
            
            % Update grapes
            %t_grapes = tic;
            if b_use_blanks
                grapes = update_grapes_blanks(grapes, pics_onset, seq_beg_blanks, stimulus, ...
                    wc_sp_index_cell{n_scr}, channels, chan_label, 1, [], ...
                    scr_config.pics2use, n_scr, b_circshiftblanks, params.is_online);
            else
                grapes = update_grapes(grapes, pics_onset, stimulus, wc_sp_index_cell{n_scr}, ...
                    channels, chan_label, 1, [], scr_config.pics2use, n_scr, params.is_online);
            end
            %fprintf('  update_grapes (MU): %.1f sec\n', toc(t_grapes));
            
            % Add sorted spikes to grapes
            %t_sorting_grapes = tic;
            if DO_SORTING
                if sorter.sortings_state == 0
                    sorter.do_remaining_sorting();
                end
                done_chs_ix = [];
                while numel(done_chs_ix) < numel(channels)
                    [classes_out, ch_ix_out] = sorter.get_done_sorts(done_chs_ix);
                    for cii = 1:numel(classes_out)
                        if ~isempty(classes_out{cii})
                            classes_out{cii} = classes_out{cii}(end - numel(wc_sp_index_cell{n_scr}{ch_ix_out(cii)}) + 1:end);
                        end
                    end
                    if b_use_blanks
                        grapes = update_grapes_blanks(grapes, pics_onset, seq_beg_blanks, stimulus, ...
                            wc_sp_index_cell{n_scr}(ch_ix_out), channels(ch_ix_out), chan_label(ch_ix_out), ...
                            false, classes_out, scr_config.pics2use, n_scr, b_circshiftblanks, params.is_online);
                    else
                        grapes = update_grapes(grapes, pics_onset, stimulus, wc_sp_index_cell{n_scr}(ch_ix_out), ...
                            channels(ch_ix_out), chan_label(ch_ix_out), false, classes_out, ...
                            scr_config.pics2use, n_scr, params.is_online);
                    end
                    done_chs_ix = [done_chs_ix ch_ix_out];
                end
                %fprintf('  sorting + update_grapes (sorted): %.1f sec\n', toc(t_sorting_grapes));
            end
            
            clear detecctions;
            %fprintf('  spike processing + grapes total: %.1f sec\n', toc(t_post));
          end % available_trials > 0

            % === STIMULUS SELECTION ===
            %t_selection = tic;
            if ~abort && n_scr < numel(experiment.NPICS)
                %t_resp = tic;
                [datat, rank_config] = create_responses_data_parallel(grapes, scr_config.pics2use, ...
                    {'mu', 'class'}, ifr_calculator, resp_conf, not_online_channels, priority_channels);
                %fprintf('  create_responses_data_parallel: %.1f sec\n', toc(t_resp));
                
                % Update trial count in ImageNames
                [used_stims, ia, ~] = unique(datat.stim_number);
                experiment.ImageNames.stim_trial_count(used_stims) = datat.ntrials(ia);
                
                % Initialize CategLocaliz history on first screening
                if n_scr == 1 && contains(subtask, 'CategLocaliz')
                    [categ_localiz_history, img_info_table] = init_categ_localiz_history(...
                        experiment, used_stims, floc_task_group);
                end
                
                datat = sort_responses_table_online(datat, priority_chs_ranking);
                
                % Remove stimuli with max trials (use experiment.MAX_TRIALS if available)
                if isfield(experiment, 'MAX_TRIALS')
                    max_trials_threshold = experiment.MAX_TRIALS;
                else
                    max_trials_threshold = MAX_NTRIALS;
                end
                enough_trials = datat.ntrials >= max_trials_threshold;
                stim_rm_max_trials = unique(datat(enough_trials, :).stim_number);
                stim_rm_max_trials_cell{end+1} = stim_rm_max_trials;
                stim_rm = stim_rm_max_trials;
                
                datat = datat(~enough_trials, :);
                [stim_best, ia, ~] = unique(datat.stim_number, 'stable');
                datat_best = datat(ia, :);
                datat_best = datat_best(datat_best.min_spk_test == 1 & ~(datat_best.zscore < 4.5) & datat_best.good_lat == 1, :);
                num_datat_best = height(datat_best);
                
                % Reorder stim_best to have datat_best stim_numbers first
                stim_best_reordered = [stim_best(ismember(stim_best, datat_best.stim_number)); stim_best(~ismember(stim_best, datat_best.stim_number))];
                
                unused_pics = setdiff(unused_pics, stim_best);
                
                % Manual selection if configured
                if scr_config.manual_select
                    % Show wait message while user is selecting best stims
                    print_message(message_wait{ind_lang}, black, window);
                    drawnow;
                    
                    data2plot = create_best_stims_table(experiment, grapes, datat, ...
                        nwins_best_stims, true, priority_chs_ranking, ...
                        selected2notremove, selected2explore_cell, n_scr, true);
                    lbl = sprintf('EMU-%.3d_select_win', params.EMU_num);
                    
                    % Change to result folder so figures save there
                    cd(result_folder);
                    
                    [selected2explore, ~, selected2rm] = stimulus_selection_windows(...
                        data2plot, grapes, rank_config, n_scr, ifr_calculator, 6, ...
                        lbl, priority_chs_ranking, experiment, false, false, true, false);
                    
                    stim_rm = [stim_rm; selected2rm];
                else
                    selected2explore = [];
                    selected2rm = [];
                end
                
                % Update selection tracking
                if contains(subtask, 'DynamicScr') || contains(subtask, 'CategLocaliz')
                    selected2explore_cell{end+1} = selected2explore;
                    selected2rm_cell{end+1} = selected2rm;
                    extra_stim_rm = experiment.P2REMOVE(n_scr) - length(stim_rm);
                    selected2notremove = [selected2notremove; selected2explore];
                    selected2notremove = unique(selected2notremove, 'stable');
                    
                    % Remove max trial stims from selected2notremove
                    if numel(stim_rm_max_trials)
                        removed_imgs = experiment.ImageNames.name(stim_rm_max_trials(ismember(stim_rm_max_trials, selected2notremove)));
                        fprintf("After SCR %d: Removing %d max trial stims from selected2notremove (%d)(%s) \n", ...
                            n_scr, numel(removed_imgs), numel(selected2notremove), strjoin(removed_imgs, ', '));
                        selected2notremove = setdiff(selected2notremove, stim_rm_max_trials, 'stable');
                    end
                    
                    if numel(selected2notremove) > experiment.NPICS(n_scr+1)
                        fprintf("Skipping addition of same_units, same_categories. There's %d in selected2notremove, " + ...
                                "which is  more than Npics %d for screening %d \n", ...
                                numel(selected2notremove), experiment.NPICS(n_scr+1), n_scr+1);
                    end

                    extra_stims = setdiff(stim_best, [selected2notremove; stim_rm], 'stable');
                    if extra_stim_rm > 0
                        if numel(extra_stims) > extra_stim_rm
                            stim_rm = [stim_rm; extra_stims(end - extra_stim_rm + 1:end)];
                            extra_stims = extra_stims(1:end - extra_stim_rm);
                        elseif numel(extra_stims) > 0
                            stim_rm = [stim_rm; extra_stims];
                            extra_stims = [];
                        end
                    end
                    
                    stim_rm_cell{end+1} = stim_rm;
                    
                    % Calculate pics for next screening
                    if ~params.use_only_main_pics
                        new_pics2load = NPICS(2:end) - (NPICS(1:end-1) - P2REMOVE(1:end-1));
                    else
                        new_pics2load = zeros(size(NPICS(2:end)));
                    end
                    
                    tbl_unused_pics = experiment.ImageNames(unused_pics, :);
                    pic2add = [];
                    same_units = [];
                    same_categories = [];
                    scr_fetched_cell = {};
                    
                    % Get pics for next screening based on subtask
                    if experiment.MANUAL_SELECT(n_scr) && ~isempty(selected2explore)
                        if contains(subtask, 'DynamicScr')
                            % DynamicScr: use same_unit and same_category logic
                            if n_scr < 3 && numel(selected2notremove) < experiment.NPICS(n_scr + 1)
                                for c = selected2explore(:)'
                                    same_unit = find(cellfun(@(x) strcmp(x, experiment.ImageNames.concept_name{c}), tbl_unused_pics.concept_name));
                                    same_category = [];
                                    
                                    for i = 1:height(tbl_unused_pics)
                                        if tbl_unused_pics.concept_number(i) ~= 1 || any(i == same_unit)
                                            continue
                                        end
                                        this_categories = tbl_unused_pics.concept_categories{i};
                                        for xi = 1:numel(this_categories)
                                            share_category = any(strcmp(experiment.ImageNames.concept_categories{c}, this_categories{xi}));
                                            if share_category
                                                same_category(end+1) = i;
                                                break
                                            end
                                        end
                                    end
                                    
                                    % Convert indices to experiment.ImageNames indices
                                    if numel(same_unit) > 0
                                        su_idx_list = [];
                                        for su_idx = 1:length(same_unit)
                                            su_idx_list = [su_idx_list; find(cellfun(@(x) strcmp(x, tbl_unused_pics.name(same_unit(su_idx))), experiment.ImageNames.name))];
                                        end
                                        same_unit = su_idx_list;
                                    end
                                    if numel(same_category) > 0
                                        sc_idx_list = [];
                                        for sc_idx = 1:length(same_category)
                                            sc_idx_list = [sc_idx_list; find(cellfun(@(x) strcmp(x, tbl_unused_pics.name(same_category(sc_idx))), experiment.ImageNames.name))];
                                        end
                                        same_category = sc_idx_list;
                                    end
                                    
                                    same_units = [same_units; same_unit(:)];
                                    same_categories = [same_categories; same_category(:)];
                                    fprintf('\n%d pictures same_unit to %s\n', numel(same_unit), experiment.ImageNames.concept_name{c})
                                    for pic_idx = same_unit
                                        fprintf('%s ', experiment.ImageNames.name{pic_idx})
                                    end
                                    fprintf('\n%d pictures same_category to %s\n', numel(same_category), experiment.ImageNames.concept_name{c})
                                    for pic_idx = same_category
                                        fprintf('%s ', experiment.ImageNames.name{pic_idx})
                                    end
                                end
                                
                                fprintf('%d pictures same_unit in total\n', numel(same_units))
                                fprintf('%d pictures same_category  in total\n', numel(same_categories))

                                % Add pics
                                num_pics_to_add = min(experiment.NPICS(n_scr + 1) - numel(selected2notremove), new_pics2load(n_scr));
                                fprintf("%d pics to be added in SCR: %d \n", num_pics_to_add, n_scr+1)
                                fprintf("==================================\n")
                                added_counter = 0;
                                
                                for rp = same_units'
                                    if all(rp ~= pic2add)
                                        if added_counter == num_pics_to_add
                                            break
                                        end
                                        added_counter = added_counter + 1;
                                        pic2add(end+1) = rp;
                                    end
                                end
                                same_units = same_units(ismember(same_units, pic2add));
                                
                                for rp = same_categories'
                                    if all(rp ~= pic2add)
                                        if added_counter == num_pics_to_add
                                            break
                                        end
                                        added_counter = added_counter + 1;
                                        pic2add(end+1) = rp;
                                    end
                                end
                                same_categories = same_categories(ismember(same_categories, pic2add));
                                
                                fprintf('%d pictures same_unit added\n', numel(same_units))
                                fprintf('%d pictures same_category added\n', numel(same_categories))
                                fprintf('%d pictures added IN TOTAL\n', numel(pic2add))
                                fprintf('%d pictures selected\n', numel(selected2explore))
                                fprintf("==================================\n")
                            end
                        elseif contains(subtask, 'CategLocaliz')
                            % CategLocaliz: use specialized functions
                            [scr_fetched_cell, experiment, categ_localiz_history] = get_categ_localiz_pics(...
                                n_scr, experiment, img_info_table, categ_localiz_history, selected2explore);
                            grapes.ImageNames = experiment.ImageNames;
                            fetched_pics_cell{end+1} = scr_fetched_cell;
                            
                            % Compute all_fetched_pics from fetched_pics_cell
                            all_fetched_pics = [];
                            for i = 1:numel(scr_fetched_cell)
                                for j = 1:numel(fetched_pics_cell)
                                    rule_info = fetched_pics_cell{j}{i};
                                    categ_info_list = rule_info.categ_info;
                                    for k = 1:numel(categ_info_list)
                                        categ_info = categ_info_list{k};
                                        all_fetched_pics = [all_fetched_pics; categ_info.pic_ids];
                                    end
                                end
                            end
                            
                            % Compute curr_fetched_pics_list from current screening
                            curr_fetched_pics_list = [];
                            for i = 1:numel(scr_fetched_cell)
                                rule_info = scr_fetched_cell{i};
                                categ_info_list = rule_info.categ_info;
                                for k = 1:numel(categ_info_list)
                                    categ_info = categ_info_list{k};
                                    curr_fetched_pics_list = [curr_fetched_pics_list; categ_info.pic_ids];
                                end
                            end
                            
                            % Update selected2notremove for CategLocaliz
                            all_max_trial_stims = cell2mat(stim_rm_max_trials_cell');
                            all_selected2explore = cell2mat(selected2explore_cell');
                            all_selected2rm = cell2mat(selected2rm_cell');
                            selected2notremove = [all_selected2explore; all_fetched_pics];
                            selected2notremove = setdiff(selected2notremove, [all_max_trial_stims; all_selected2rm]);
                            if numel(all_max_trial_stims)
                                selected2notremove = setdiff(selected2notremove, all_max_trial_stims, 'stable');
                            end
                            selected2notremove = unique(selected2notremove, 'stable');
                            
                            % Compute extra_used_stims and stim_keep for keep_endangered_pics
                            scr_used_stims = stim_best_reordered(ismember(stim_best_reordered, scr_config.pics2use'));
                            extra_used_stims = setdiff(scr_used_stims, [selected2notremove; stim_rm; stim_rm_max_trials], 'stable');
                            stim_keep = [];
                            stim_rm_categ = [];
                            stim_rm_count = P2REMOVE(n_scr) - length([stim_rm_max_trials; stim_rm]);
                            if stim_rm_count > 0 && stim_rm_count < numel(extra_used_stims)
                                stim_keep = extra_used_stims(1:end-stim_rm_count);
                                stim_rm_categ = extra_used_stims(numel(stim_keep)+1:end);
                            elseif stim_rm_count > 0 && stim_rm_count > numel(extra_used_stims) && numel(extra_used_stims) > 0
                                stim_rm_categ = extra_used_stims;
                            elseif stim_rm_count == 0
                                stim_keep = extra_used_stims;
                            end
                            
                            num_datat_best = min(numel(stim_keep), nwins_best_stims);
                            unused_pics_available = setdiff(unused_pics', selected2notremove, 'stable');
                            available_pics_categ = [selected2notremove; stim_keep; unused_pics_available];
                            
                            % Call keep_endangered_pics to show rule selection GUI
                            if n_scr < length(NPICS) && NPICS(n_scr + 1) > 0
                                exp_time_taken = toc(exp_start_time);
                                [next_scr_pics, pics_removed, categ_localiz_history] = keep_endangered_pics(...
                                    experiment, n_scr, available_pics_categ, unused_pics_available, ...
                                    stim_keep, num_datat_best, selected2notremove, ...
                                    selected2explore_cell, fetched_pics_cell, scr_config, ...
                                    exp_time_taken, true, categ_localiz_history);
                                
                                % Update NPICS for next screening based on selection
                                experiment.NPICS(n_scr + 1) = numel(next_scr_pics);
                                NPICS(n_scr + 1) = numel(next_scr_pics);
                                
                                if numel(pics_removed) > 0
                                    stim_rm_categ = extra_used_stims(ismember(extra_used_stims, pics_removed));
                                    unused_curr_fetched_pics = curr_fetched_pics_list(ismember(curr_fetched_pics_list, pics_removed));
                                    categ_localiz_history = update_categ_localiz_history(experiment, ...
                                        categ_localiz_history, unused_curr_fetched_pics);
                                end
                                
                                available_pics = next_scr_pics';
                            else
                                available_pics = available_pics_categ';
                            end
                            
                            stim_rm = [stim_rm; stim_rm_categ];
                        end
                    end
                    
                    same_units_cell{end+1} = same_units;
                    same_categories_cell{end+1} = same_categories;
                    
                    % Update available pics for next screening (for non-CategLocaliz)
                    if ~contains(subtask, 'CategLocaliz')
                        all_removed_stims = cell2mat(stim_rm_cell');
                        all_selected_to_exp = setdiff(cell2mat(selected2explore_cell'), all_removed_stims, 'stable');
                        all_same_units = setdiff(cell2mat(same_units_cell'), all_removed_stims, 'stable');
                        all_same_categories = setdiff(cell2mat(same_categories_cell'), all_removed_stims, 'stable');
                        
                        selected2notremove = unique([all_selected_to_exp; all_same_units; all_same_categories], 'stable');

                        fprintf("%d selected2notremove (%d selected_to_exp, %d same_units, %d same_categories) \n", ...
                            numel(selected2notremove), numel(all_selected_to_exp), ...
                            numel(all_same_units), numel(all_same_categories))
                        fprintf("==================================\n")
                        available_pics = [selected2notremove' extra_stims' setdiff(unused_pics, selected2notremove, 'stable')];
                    end
                end
            else
                stim_rm_cell{end+1} = [];
            end
        else
            stim_rm_cell{end+1} = [];
        end
        
        selected2notremove_cell{end+1} = selected2notremove;
        available_pics_cell{end+1} = available_pics;
        
        %if exist('t_selection', 'var')
        %    fprintf('  stimulus selection total: %.1f sec\n', toc(t_selection));
        %end
        
        % Save progress (all variables for compatibility with original)
        %t_save = tic;
        if ~strcmp(subtask, 'FirstTime')
            % Use -v7 (faster) for saves; -v7.3 only needed if any
            % single variable exceeds 2GB (these named variables don't)
            save(exp_prop_file, 'experiment', 'scr_config_cell', 'scr_end_cell', ...
                'available_pics_cell', 'stim_rm_cell', 'stim_rm_max_trials_cell', ...
                'selected2notremove_cell', 'priority_chs_ranking', 'selected2explore_cell', ...
                'same_units_cell', 'same_categories_cell');
            
            % Save workspace periodically
            %save(fullfile(experiment.folder_name, 'RSVP_SCR_workspace.mat'), '-regexp', '^(?!(M_PTB|f|backup_worker|bg_collector|async_processor|poolobj|collector_future|data_queue|collection_timer)$).');
        end
        %fprintf('  save: %.1f sec\n', toc(t_save));
        %fprintf('--- Post-subscreening total: %.1f sec ---\n\n', toc(t_post));
        
        ShowCursor;
        
        if abort
            % Stop background collector immediately on abort
            if exist('bg_collector', 'var') && ~isempty(bg_collector)
                try
                    bg_collector.stop();
                catch
                end
            end
            break
        end
        
    end % End subscreening loop
    
    %% ===================================================================
    %  SECTION 10: EXPERIMENT END AND CLEANUP
    %  ===================================================================
    
    %t_section10 = tic;
    %fprintf('\n=== SECTION 10: Experiment end processing ===\n');
    Priority(params.ptb_priority_normal);
    
    % Final save
    pics_used_ids = [];
    for ii = 1:length(scr_config_cell)
        pics_used_ids = [pics_used_ids scr_config_cell{ii}.pics2use];
    end
    pics_used_ids = unique(pics_used_ids);
    
    % Compute picsexplored_names for compatibility
    if ~isempty(selected2explore_cell)
        picsexplored_names = experiment.ImageNames.name(unique(cell2mat(selected2explore_cell')));
    else
        picsexplored_names = {};
    end
    
    if ~strcmp(subtask, 'FirstTime')
        save(exp_prop_file, 'experiment', 'scr_config_cell', 'scr_end_cell', ...
            'available_pics_cell', 'stim_rm_cell', 'pics_used_ids', 'abort', ...
            'stim_rm_max_trials_cell', 'selected2notremove_cell', 'priority_chs_ranking', ...
            'selected2explore_cell', 'same_units_cell', 'same_categories_cell', ...
            'picsexplored_names');
        
        % Save workspace before cleanup
        %save(fullfile(experiment.folder_name, 'RSVP_SCR_workspace.mat'), '-regexp', '^(?!(M_PTB|f|backup_worker|bg_collector|async_processor|poolobj|collector_future|data_queue|collection_timer)$).');
    end
    
catch ME
    % Error handling
    msgText = getReport(ME);
    disp(msgText);
    drawnow;
    
    % Update task history with error status
    try
        if exist('params', 'var') && isfield(params, 'task_history_csv')
            get_task_history(params.sub_ID, subtask, params.beh_rec_metadata, ...
                'force_emu', params.EMU_num, ...
                'force_run', params.run_num, ...
                'update_status', true, ...
                'status', 'error', ...
                'notes', sprintf('Error: %s', ME.message));
        end
    catch
    end
    
    try
        Screen('CloseAll');
        ShowCursor;
    catch
    end
    
    try
        for d = dev_used
            KbQueueStop(d);
            KbQueueRelease(d);
        end
    catch
    end
    
    Priority(params.ptb_priority_normal);
    
    % Close device_com only if NOT using background collection worker
    % (worker handles device_com internally)
    if params.is_online && params.disable_interactions && ~params.use_background_collection
        device_com('close');
    end
    
    if params.use_daq
        try
            dig_out.close();
        catch
        end
    end
    
    % Cleanup async processor on error
    if exist('async_processor', 'var') && ~isempty(async_processor)
        try
            async_processor.cleanup();
        catch
        end
    end
    
    % Cleanup background data collector worker on error
    if exist('bg_collector', 'var') && ~isempty(bg_collector)
        try
            bg_collector.shutdown();  % Shuts down worker and closes device_com
        catch
        end
    end
    
    save(fullfile(result_folder, 'error_workspace.mat'), '-regexp', '^(?!(M_PTB|f|backup_worker|bg_collector|async_processor|poolobj|collector_future|data_queue|collection_timer|sorter|dig_out|recording|custompath)$).');
    custompath.rm();
    diary off;
    rethrow(ME);
end

%% =======================================================================
%  SECTION 11: FINAL PROCESSING AND ANALYSIS
%  =======================================================================

task_duration = toc(init_time);
%fprintf('\n=== SECTION 11: Final processing ===\n');
%t_section11 = tic;
print_message('THAT WOULD BE ALL.\n THANK YOU !!!', black, window);
WaitSecs(3);

% Stop recording
if params.acq_network
    if abort || contains(subtask, 'DynamicScr') || contains(subtask, 'DynamicSeman') || contains(subtask, 'CategLocaliz')
        fprintf('\nPress any key to stop recording.\n');
        timeoutSecs = 540;
        startTime = GetSecs;
        while (GetSecs - startTime) < timeoutSecs
            [keyIsDown, ~, ~] = KbCheck;
            if keyIsDown
                break;
            end
            WaitSecs(0.01);
        end
    end
    recording.stop_and_close();
end

Screen('CloseAll');
ShowCursor;

for d = dev_used
    KbQueueStop(d);
    KbQueueRelease(d);
end

Priority(params.ptb_priority_normal);
fprintf('Task Duration: %.1f seconds.\n', task_duration);

%% =======================================================================
%  BACKUP AND DATA MOVEMENT (runs before analysis)
%  =======================================================================

% Copy ACQ (raw acquisition) files to experiment folder
%t_backup = tic;
if params.copy_backup && params.with_acq_folder && params.acq_network
    disp('Starting backup of raw acquisition files...');
    pause(3);  % Give time for recording files to be finalized
    backup_worker = parfeval(@backup_raw_data, 1, params, experiment.fname);
    
    if ~strcmp(backup_worker.State, 'finished')
        disp('Waiting for raw files backup...');
        wait(backup_worker);
    end
    
    if ~isempty(backup_worker.Error)
        warning('Backup worker error: %s', backup_worker.Error.message);
    else
        [bw_msg] = fetchOutputs(backup_worker);
        if ~isempty(bw_msg)
            warning('Error copying raw files: %s', bw_msg);
        else
            disp('Raw acquisition files backup done.');
        end
    end
end
%fprintf('  Backup raw files: %.1f sec\n', toc(t_backup));

% Copy script and online folder to experiment folder (for reproducibility)
if ~strcmp(subtask, 'FirstTime')
    rsvp_folder = fileparts(mfilename('fullpath'));
    
    try
        copyfile(fullfile(rsvp_folder, 'ABTPRSVP.m'), fullfile(experiment.folder_name, 'ABTPRSVP.m'));
    catch ME_copy
        warning('Could not copy ABTPRSVP.m: %s', ME_copy.message);
    end
    
    try
        online_src = fullfile(rsvp_folder);
        online_dst = fullfile(experiment.folder_name, 'online');
        if ~isfolder(online_dst)
            mkdir(online_dst);
        end
        copyfile(fullfile(online_src, '*.m'), online_dst);
    catch ME_copy
        warning('Could not copy online folder: %s', ME_copy.message);
    end
end

% Copy pics used
%t_copy_pics = tic;
if ~strcmp(subtask, 'FirstTime')
    pics_used_folder = fullfile(experiment.folder_name, 'pics_used');
    if ~isfolder(pics_used_folder)
        mkdir(pics_used_folder);
    end
    pics2backup = find(stim_trial_counter > 0);
    for i = 1:numel(pics2backup)
        if contains(subtask, 'CategLocaliz')
            src_file = fullfile(ImageNames.folder{pics2backup(i)}, ImageNames.name{pics2backup(i)});
        else
            src_file = fullfile(params.pics_root_beh, ImageNames.folder{pics2backup(i)}, ImageNames.name{pics2backup(i)});
        end
        if params.remove_pictures
            movefile(src_file, pics_used_folder);
        else
            copyfile(src_file, pics_used_folder);
        end
    end
end
%fprintf('  Copy pics: %.1f sec\n', toc(t_copy_pics));

fprintf('\n=== All data copied and saved. ===\n');

%% =======================================================================
%  UPDATE TASK HISTORY (after backup is complete)
%  =======================================================================
if ~strcmp(subtask, 'FirstTime')

    if isfield(params, 'task_history_csv')
        %fprintf('DEBUG: Updating task history for EMU-%d %s run-%d\n', params.EMU_num, subtask, params.run_num);
        total_trials = 0;
        for ii = 1:length(scr_end_cell)
            if ~isempty(scr_end_cell{ii}) && isstruct(scr_end_cell{ii})
                if isfield(scr_end_cell{ii}, 'last_rep')
                    total_trials = total_trials + scr_end_cell{ii}.last_rep;
                end
            elseif ~isempty(scr_end_cell{ii}) && isnumeric(scr_end_cell{ii})
                total_trials = total_trials + scr_end_cell{ii};
            end
        end
        completion_status = 'completed';
        if exist('abort', 'var') && abort
            completion_status = 'aborted';
        end
        get_task_history(params.sub_ID, subtask, params.beh_rec_metadata, ...
            'force_emu', params.EMU_num, ...
            'force_run', params.run_num, ...
            'update_status', true, ...
            'status', completion_status, ...
            'n_trials', total_trials, ...
            'experiment_folder', experiment.folder_name);
    end
end

%% =======================================================================
%  SEND TO TOWER (after task history is updated)
%  =======================================================================
% Offline processing can be done locally or on Tower (remote HPC)
% params.offline_processing = true: Send to Tower via SSH + SLURM
% params.offline_processing = false: Run locally (original behavior)

run_offline = (contains(subtask, 'DynamicScr') || ...
               contains(subtask, 'DynamicSeman')) && params.acq_network;

%t_tower = tic;
if run_offline
    run_folder = experiment.folder_name;  % Already in backup location

    if params.offline_processing
        %% Tower-based offline processing
        fprintf('\n=== TOWER OFFLINE PROCESSING ===\n');

        % Try to connect to Tower
        [tower_connected, tower_ip, ~] = connect_to_tower('verbose', true);

        if tower_connected
            fprintf('Connected to Tower at %s\n', tower_ip);

            % Build remote path: Exp/sub_ID/EMU/taskID
            remote_base_path = sprintf('/mnt/acq-hdd/%s/EMU', params.sub_ID);
            remote_exp_path = sprintf('%s/%s', remote_base_path, experiment.fname);

            % Submit processing job to Tower
            [job_id, submit_success] = submit_tower_processing(...
                tower_ip, ...
                run_folder, ...
                experiment.fname, ...
                'tower_exp_path', '/mnt/acq-hdd', ...
                'subject_id', params.sub_ID, ...
                'is_online', params.is_online, ...
                'copy2miniscrfolder', contains(subtask, 'DynamicScr'), ...
                'show_sel_count', contains(subtask, 'DynamicScr'), ...
                'show_best_stims_wins', contains(subtask, 'DynamicScr'), ...
                'max_spikes_plot', 500, ...
                'verbose', true);

            if submit_success
                fprintf('\n*** Processing job submitted to Tower ***\n');
                fprintf('Subject: %s\n', params.sub_ID);
                fprintf('Experiment: %s\n', experiment.fname);
                fprintf('Job ID: %d\n', job_id);
                fprintf('Tower IP: %s\n', tower_ip);
                fprintf('Remote path: %s\n', remote_exp_path);
                fprintf('\nTo check status, run:\n');
                fprintf('  status = check_tower_job_status(''%s'', ''%s'', ''subject_id'', ''%s'');\n', ...
                    tower_ip, experiment.fname, params.sub_ID);
                fprintf('\nOr SSH to Tower and check:\n');
                fprintf('  ssh user@%s\n', tower_ip);
                fprintf('  cat %s/processing_status.txt\n', remote_exp_path);

                % Save job info for later reference
                tower_job_info = struct();
                tower_job_info.job_id = job_id;
                tower_job_info.tower_ip = tower_ip;
                tower_job_info.subject_id = params.sub_ID;
                tower_job_info.experiment_name = experiment.fname;
                tower_job_info.submit_time = datestr(now);
                tower_job_info.remote_path = remote_exp_path;

                save(fullfile(run_folder, 'tower_job_info.mat'), 'tower_job_info');
            else
                warning('Failed to submit job to Tower.');
                params.offline_processing = false;  % Fall back to local
            end
        else
            warning('Could not connect to Tower.');
            params.offline_processing = false;  % Fall back to local
        end
    end
end
%fprintf('  Tower send: %.1f sec\n', toc(t_tower));

% Final analysis
%t_final_analysis = tic;
try
    if params.is_online && n_scr > 0
        all_picsused = unique(cell2mat(cellfun(@(x) x.pics2use, scr_config_cell, 'UniformOutput', false)));

        % Save grapes (always, even on abort)
        save(fullfile(result_folder, 'grapes_online.mat'), 'grapes');

        if ~abort
            fprintf('\n=== Final Analysis ===\n');

            % Create response data
            if ~sorting_done
                [data_final, rank_config] = create_responses_data_parallel(grapes, all_picsused, {'mu'}, ...
                    ifr_calculator, resp_conf, [], priority_channels);
            else
                [data_final, rank_config] = create_responses_data_parallel(grapes, all_picsused, {'mu', 'class'}, ...
                    ifr_calculator, resp_conf, [], priority_channels);
            end
            data_final = sort_responses_table(data_final);

            % Plot best stimuli
            if contains(subtask, 'DynamicScr')
                copy2miniscrfolder = true;
                show_sel_count = true;
                showwins = true;
            else
                copy2miniscrfolder = false;
                show_sel_count = false;
                showwins = false;
            end

            data_to_plot = create_best_stims_table(experiment, grapes, data_final, ...
                nwins_best_stims, true, priority_chs_ranking, [], [], n_scr, true);
            lbl = sprintf('EMU-%.3d_best_stim', params.EMU_num);

            % Load Pics.xlsx for face classification
            pics_xlsx_path = '/home/user/ReyLab/experimental_files/pics/Pics.xlsx';
            if exist(pics_xlsx_path, 'file')
                try
                    [~, ~, raw_data] = xlsread(pics_xlsx_path, 'All pics');
                    header_row = raw_data(1, :);
                    name_col_idx = find(strcmpi(header_row, 'Name'), 1);
                    folder_col_idx = find(strcmpi(header_row, 'Folder'), 1);
                    if ~isempty(name_col_idx) && ~isempty(folder_col_idx)
                        name_data = raw_data(2:end, name_col_idx);
                        folder_data = raw_data(2:end, folder_col_idx);
                        pics_lookup_table = table(name_data, folder_data, 'VariableNames', {'Name', 'Folder'});
                    else
                        pics_lookup_table = table();
                    end
                catch
                    pics_lookup_table = table();
                end
            else
                pics_lookup_table = table();
            end

            % Change to results folder for saving figures
            prev_folder = pwd;
            cd(result_folder);

            [selected2explore_final, s4miniscr_tbl, ~] = stimulus_selection_windows(...
                data_to_plot, grapes, rank_config, n_scr, ifr_calculator, nwins_best_stims, ...
                lbl, priority_chs_ranking, experiment, copy2miniscrfolder, ...
                show_sel_count, showwins, true, pics_lookup_table, 0, 0);

            if copy2miniscrfolder
                selected4miniscr_csv_file = fullfile(result_folder, 'selected4miniscr.csv');
                writetable(s4miniscr_tbl, selected4miniscr_csv_file, 'Delimiter', ',');
                disp('Miniscreening csv file written.');

                % Send selected4miniscr.csv to Tower results folder
                if exist('tower_job_info', 'var') && ~isempty(tower_job_info)
                    try
                        remote_results = sprintf('%s/results/', tower_job_info.remote_path);
                        scp_cmd = sprintf('ssh user@%s "mkdir -p %s" && scp "%s" user@%s:%s', ...
                            tower_job_info.tower_ip, remote_results, ...
                            selected4miniscr_csv_file, tower_job_info.tower_ip, remote_results);
                        [scp_status, scp_out] = system(scp_cmd);
                        if scp_status == 0
                            fprintf('selected4miniscr.csv sent to Tower: %s\n', remote_results);
                        else
                            warning('Failed to send selected4miniscr.csv to Tower: %s', scp_out);
                        end
                    catch ME_scp
                        warning('Error sending selected4miniscr.csv to Tower: %s', ME_scp.message);
                    end
                end
            end

            % Copy FreqTag images
            if showwins && ~isempty(selected2explore_final)
                freqtag_respfaces_folder = '/home/user/ReyLab/experimental_files/pics/pics_space/freq_tag/RespFaces';
                freqtag_respnonfaces_folder = '/home/user/ReyLab/experimental_files/pics/pics_space/freq_tag/RespNonFaces';
                try
                    copy_selected_to_freqtag(experiment, selected2explore_final, ...
                        freqtag_respfaces_folder, freqtag_respnonfaces_folder);
                catch ME_ft
                    warning('FreqTag copy failed: %s', ME_ft.message);
                end
            end


            % Sorting results
            if DO_SORTING
                disp('Saving sorting results...');
                create_sorting_figs(chan_label, sorter.spikes, sorter.classes, 'scr_online', conversion);
                templates_ms_path = fullfile(result_folder, 'templates_ms.mat');
                sorter.save_sorting_results(templates_ms_path);
                % Note: templates_ms is saved to results folder only, not to rec_metadata
                disp('Sorting results saved.');
            end

            % Return to previous folder
            cd(prev_folder);
        end % ~abort

        % Handle preprocessing info based on task type (always run)
        if par.preprocessing
            if is_screening_task
                % Screening tasks: MOVE from rec_metadata to results (clean rec_metadata)
                preprocessing_file = fullfile(params.processing_rec_metadata, 'pre_processing_info.mat');
                if exist(preprocessing_file, 'file')
                    if contains(subtask, 'Test')
                        % For Test subtasks, copy (don't move) so it stays for next runs
                        copy_file(preprocessing_file, result_folder, temp_folder);
                        disp('Copied pre_processing_info to results (Test mode - keeping in rec_metadata)');
                    else
                        % For production screening runs, move to results
                        move_file(preprocessing_file, result_folder, temp_folder);
                        disp('Moved pre_processing_info from rec_metadata to results');
                    end
                end
            else
                % Non-screening tasks: COPY from source to results (keep in both places)
                if exist('fallback_preprocessing', 'var') && exist(fallback_preprocessing, 'file')
                    copy_file(fallback_preprocessing, result_folder, temp_folder);
                    disp('Copied pre_processing_info from screening folder to results');
                end
            end
        end
        
        if ~isempty(MAPFILE) && exist(MAPFILE, 'file')
            copy_file(MAPFILE, result_folder, temp_folder);
        end
        
        % Handle templates based on task type
        if is_screening_task && ~contains(subtask, 'Test')
            % Screening tasks: Clean ALL template files from rec_metadata
            templates_to_clean = {
                fullfile(params.processing_rec_metadata, 'templates_ms.mat'), ...
                fullfile(params.processing_rec_metadata, 'templates_wc_offline.mat')
            };
            for tc = 1:length(templates_to_clean)
                if exist(templates_to_clean{tc}, 'file')
                    delete(templates_to_clean{tc});
                    fprintf('Deleted from rec_metadata: %s\n', templates_to_clean{tc});
                end
            end
            disp('Cleaned template files from rec_metadata');
        elseif exist('TEMPLATES_FILE', 'var') && exist(TEMPLATES_FILE, 'file')
            % Non-screening tasks or Test: COPY templates to results (keep originals)
            copy_file(TEMPLATES_FILE, result_folder, temp_folder);
            disp('Copied templates to results');
        end
        
        % Copy experiment properties to run folder
        copy_file(exp_prop_file, experiment.folder_name, temp_folder);
        
        % Close device (only when disable_interactions AND not using background collection)
        % Background collection worker handles device_com internally
        if params.is_online && params.disable_interactions && ~params.use_background_collection
            device_com('close');
        end
        
        % Show output info
        if exist('outputinfo', 'var') && ~isempty(outputinfo)
            fprintf('\n=== Online Report ===\n');
            for oi = 1:numel(outputinfo)
                fprintf('  %s\n', outputinfo{oi});
            end
            fprintf('=====================\n');
        else
            fprintf('\n=== Online Report: Everything OK ===\n');
        end
    end
    
catch ME_final
    warning('Error in final processing: %s', ME_final.message);
    disp(getReport(ME_final));
end
%fprintf('  Final analysis: %.1f sec\n', toc(t_final_analysis));

%% =======================================================================
%  SECTION 12: CLEANUP
%  =======================================================================

% Clean rec_metadata for screening tasks (DynamicScr/DynamicSeman)
% This ensures rec_metadata is cleared after each screening session
if params.is_online && (contains(subtask, 'DynamicScr') || contains(subtask, 'DynamicSeman')) && ~contains(subtask, 'Test')
    fprintf('\n=== Cleaning rec_metadata ===\n');
    files_to_clean = {
        fullfile(params.processing_rec_metadata, 'pre_processing_info.mat'), ...
        fullfile(params.processing_rec_metadata, 'templates_ms.mat'), ...
        fullfile(params.processing_rec_metadata, 'templates_wc_offline.mat')
    };
    for fc = 1:length(files_to_clean)
        if exist(files_to_clean{fc}, 'file')
            try
                delete(files_to_clean{fc});
                fprintf('Deleted: %s\n', files_to_clean{fc});
            catch ME_del
                warning('Failed to delete %s: %s', files_to_clean{fc}, ME_del.message);
            end
        end
    end
    fprintf('rec_metadata cleanup complete\n');
end

% Final workspace save (update after analysis)
% Save only the essential variables (matching online_loader_collide_3 style)
% instead of the entire workspace via -regexp, which was ~2GB due to
% transient buffers (Im, Event_Time/Value, randTime_*, t_stimon, etc.)
%t_final_save = tic;
if ~strcmp(subtask, 'FirstTime')
    ws_save_path = fullfile(experiment.folder_name, 'RSVP_SCR_workspace.mat');
    try
        save(ws_save_path, ...
            'experiment', 'scr_config_cell', 'scr_end_cell', ...
            'available_pics_cell', 'stim_rm_cell', 'stim_rm_max_trials_cell', ...
            'selected2notremove_cell', 'priority_chs_ranking', 'selected2explore_cell', ...
            'same_units_cell', 'same_categories_cell', ...
            'grapes', 'pics_onset_cell', 'stimulus_cell', 'wc_sp_index_cell', 'outputinfo');
    catch ME_ws_save
        warning('Workspace save failed: %s', ME_ws_save.message);
    end
end
%fprintf('  Final workspace save: %.1f sec\n', toc(t_final_save));
%fprintf('=== Section 11 grand total: %.1f sec ===\n\n', toc(t_section11));

disp([experiment.fname ':END']);

% Delete temp folder
if exist('temp_folder', 'var') && exist(temp_folder, 'dir')
    try
        rmdir(temp_folder, 's');
    catch
        warning('Could not delete temp folder: %s', temp_folder);
    end
end

% Close DAQ
if params.use_daq
    dig_out.close();
end

% Cleanup async processor
if exist('async_processor', 'var') && ~isempty(async_processor)
    % Print final async statistics
    if params.is_online
        fprintf('\n=== ASYNC PROCESSING CLEAN UP ===\n');
        %stats = async_processor.get_statistics();
        %fprintf('  Total trials submitted: %d\n', stats.total_submitted);
        %fprintf('  Total trials completed: %d\n', stats.total_completed);
        %fprintf('  Mean processing time: %.2f sec\n', stats.mean_processing_time);
        %fprintf('  Max processing time: %.2f sec\n', stats.max_processing_time);
        %fprintf('  Min processing time: %.2f sec\n', stats.min_processing_time);
        %fprintf('==========================================\n');
    end
    async_processor.cleanup();
    fprintf('Async processor cleanup complete\n');
    fprintf('==========================================\n');
end

% Cleanup background data collector worker
if exist('bg_collector', 'var') && ~isempty(bg_collector)
    if params.is_online
        fprintf('\n=== BACKGROUND COLLECTION WORKER CLEANUP ===\n');
        fprintf('Worker-based data collection completed.\n');
    end
    bg_collector.shutdown();  % Shuts down worker and closes device_com
    if params.is_online
        fprintf('Worker shutdown complete.\n');
        fprintf('=============================================\n');
    end
end

%% =======================================================================
%  SECTION 14: PLOT CHANEL RESPONSES
%  =======================================================================
% Plot channel responses (skip on abort - data_final not available)
if ~abort
    prev_folder = pwd;
    cd(result_folder);
    fprintf('\n=== PLOT CHANNEL RESPONSES ===\n');
    plot_channel_grapes('channels2plot', 'all', 'stim_list', 'all', 'order_by_rank', true, ...
        'data', data_final, 'grapes', grapes, 'n_scr', n_scr, 'nwins2plot', 2, ...
        'rank_config', rank_config, 'ifr_x', ifr_calculator.ejex, ...
        'save_fig', true, 'emu_num', params.EMU_num, ...
        'close_fig', true, 'order_offset', 0, ...
        'priority_chs_ranking', priority_chs_ranking, ...
        'parallel_plots', true, 'extra_lbl', '');


    cd(prev_folder);
end

% Remove paths
custompath.rm();

fprintf('\n=== ABTPRSVP Complete ===\n');

% End diary after everything is done (including plots)
diary off;

% Mark cleanup as done so onCleanup doesn't run again
cleanup_done = true;

%% =======================================================================
%  NESTED CLEANUP FUNCTION
%  =======================================================================
% This nested function is called by onCleanup to ensure resources are
% released even if the function exits due to error or Ctrl+C

    function do_cleanup()
        % Prevent double cleanup
        if cleanup_done
            return;
        end
        
        fprintf('\n=== EMERGENCY CLEANUP (onCleanup) ===\n');
        
        % Close screen
        try
            if ~isempty(window)
                Screen('CloseAll');
                ShowCursor;
            end
        catch
        end
        
        % Release keyboards
        try
            if ~isempty(dev_used)
                for d = dev_used
                    KbQueueStop(d);
                    KbQueueRelease(d);
                end
            end
        catch
        end
        
        % Reset priority
        try
            Priority(params.ptb_priority_normal);
        catch
        end
        
        % Close DAQ
        try
            if ~isempty(dig_out) && isvalid(dig_out)
                dig_out.close();
            end
        catch
        end
        
        % Cleanup async processor
        try
            if ~isempty(async_processor) && isvalid(async_processor)
                async_processor.cleanup();
            end
        catch
        end
        
        % Cleanup background data collector worker
        % This also closes device_com in the worker
        try
            if ~isempty(bg_collector) && isvalid(bg_collector)
                bg_collector.shutdown();
            end
        catch
        end
        
        % If not using background collection, close device_com on main thread
        try
            if params.is_online && ~params.use_background_collection
                device_com('close');
            end
        catch
        end
        
        % End diary
        try
            diary off;
        catch
        end
        
        fprintf('Emergency cleanup complete.\n');
        fprintf('=========================================\n');
    end

end % End main function

%% =======================================================================
%  HELPER FUNCTIONS
%  =======================================================================

function [data, datacounter, timestamps, ev_time, ev_value, ev_pdiode, data_loss] = ...
    collect_stream_data(data, datacounter, timestamps, ev_time, ev_value, ev_pdiode, use_photodiodo)
    % COLLECT_STREAM_DATA - Collect neural data from device buffer
    %   Reads stream data and accumulates it into the data cell array.
    %   Handles lost data by inserting zeros.
    %
    %   Inputs:
    %       data - cell array of channel data buffers (MODIFIED IN PLACE AND RETURNED)
    %       datacounter - vector of current data positions per channel
    %       timestamps - cell array of stream timestamps
    %       ev_time - accumulated event times (parallel port)
    %       ev_value - accumulated event values (parallel port)
    %       ev_pdiode - accumulated photodiode event times
    %       use_photodiodo - boolean flag for photodiode collection
    %
    %   Outputs: same as inputs, updated with new data (data is now returned!)
    
    data_loss = false;
    streams = device_com('get_stream');
    
    % Collect neural data from all channels
    for jj = 1:size(streams.data, 1)
        % Handle lost data before current segment
        if streams.lost_prev(jj) > 0
            data{jj}(datacounter(jj) + (1:streams.lost_prev(jj))) = 0;
            datacounter(jj) = datacounter(jj) + streams.lost_prev(jj);
            data_loss = true;
        end
        
        % Store current segment
        lseg = length(streams.data{jj});
        if lseg > 0
            data{jj}(datacounter(jj) + (1:lseg)) = streams.data{jj};
            datacounter(jj) = datacounter(jj) + lseg;
        end
        
        % Handle lost data after current segment
        if streams.lost_post(jj) > 0
            data{jj}(datacounter(jj) + (1:streams.lost_post(jj))) = 0;
            datacounter(jj) = datacounter(jj) + streams.lost_post(jj);
            data_loss = true;
        end
    end
    
    % Store timestamp
    timestamps{end+1} = streams.timestamp;
    
    % Collect parallel port events (DAQ)
    if isfield(streams, 'parallel') && ~isempty(streams.parallel.values)
        ev_time = [ev_time; double(streams.parallel.times) / 30];  % Convert to ms
        ev_value = [ev_value; streams.parallel.values];
    end
    
    % Collect photodiode events
    if use_photodiodo
        if isfield(streams, 'analog_ev_t') && ~isempty(streams.analog_ev_t)
            if ~isempty(streams.analog_ev_t{1})
                ev_pdiode = [ev_pdiode; double(streams.analog_ev_t{1}(:)) / 30];  % Convert to ms
            end
        end
    end
end

function print_message(message, bgnd, window)
    Screen('FillRect', window, bgnd);
    DrawFormattedText(window, message, 'center', 'center', [255 255 255]);
    Screen('Flip', window);
end

