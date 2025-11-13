function rsvpscr_EMU_online_2(subtask,varargin)
%Available subtasks: 'DynamicScr','DynamicScrTest','DynamicSeman','DynamicSemanTest','CategLocaliz','FirstTime','OnlineMiniScr','OnlineMiniScrMusic','Test'
% rsvpscr_EMU_online('DynamicScr', 'auto_resp', true)
% rsvpscr_EMU_online('FirstTime')
% rsvpscr_EMU_online('DynamicScrTest', 'location', 'MCW-BEH-RIP', 'online_notches', false, 'acq_network', false)
% rsvpscr_EMU_online_2('DynamicScrTest', 'location', 'MCW-ABTL-RIP', 'online_notches', false, 'acq_network', false)
% rsvpscr_EMU_online('OnlineMiniScr', 'location', 'MCW-BEH-RIP', 'online_notches', false, 'acq_network', false, 'auto_resp', true)
% rsvpscr_EMU_online('DynamicScr', 'location', 'MCW-BEH-RIP', 'auto_resp', true)
% rsvpscr_EMU_online_2('CategLocaliz', 'location', 'MCW-BEH-RIP', 'auto_resp', true,'acq_network',false, 'online_notches', false)
% rsvpscr_EMU_online_2('CategLocalizTest', 'location', 'MCW-BEH-RIP', 'auto_resp', true,'acq_network',true, 'online_notches', false)
% rsvpscr_EMU_online_2('CategLocalizTest', 'location', 'MCW-ABT-RIP', 'auto_resp', true,'acq_network',false, 'online_notches', false)
% rsvpscr_EMU_online_2('CategLocalizTest', 'location', 'MCW-ABT-RIP', 'auto_resp', true, 'online_notches', false)
% rsvpscr_EMU_online_2('CategLocaliz')
ipr = inputParser;
addParameter(ipr,'sub_ID', []); %string (current subject ID), if empty (default) the code will be look inside the file current_subject_id.txt rec_metadata (behaviour pc)
addParameter(ipr,'run_num', []); %integer (this particular subtask run number), if empty (default) the code will be look inside the acquisition disk to look for the previous files
addParameter(ipr,'EMU_num',[]); %integer (task number for this subject), if empty (default) the code will be look inside the acquisition disk to look for the previous files
addParameter(ipr,'location', 'MCW-FH-RIP'); %string (location used), if not included this will be the default
% addParameter(ipr,'location', 'MCW-BEH-RIP'); %string (location used), if not included this will be the default
addParameter(ipr,'is_online',true);% not implemented %if false, it will run without the online processing
addParameter(ipr,'auto_resp', false); %if true the subject inputs will be automatically press
addParameter(ipr,'ptb_debug',false); %if true PTB will be run in debug mode
addParameter(ipr,'Nrep',[]); %if not empty it overwrite the subtask default
addParameter(ipr,'which_nsp_micro',[]); %only used for BRK system  with multiple devices, if not empty overwrite the location default.
addParameter(ipr,'lang',[]); %string, if empty default in location used. Alternatives: 'english','spanish' and 'french'
addParameter(ipr,'device_resp',[]);  %string, if empty default in location used. Alternatives: 'keyboard','gamepad'
addParameter(ipr,'system',[]); %string, if empty default in location used. Alternatives: 'RIP','BRK'
addParameter(ipr,'acq_network',[]) % boolean, start and stop recording via network. if empty default in location used.
addParameter(ipr,'with_acq_folder',[]) %boolean, access to raw data from both matlabs. if empty default in location used.
addParameter(ipr,'templates_required',[]) %boolean, if not empty overwrite task default. If true it will require templates to run
addParameter(ipr,'use_daq',[]) %boolean, use or not a daq. if not empty overwrite the location default.
addParameter(ipr,'mapfile',[]); %(string) mapfile name inside metadata folder. if not empty overwrite the location default.
addParameter(ipr,'online_notches', []); %(boolean) load calculated notches. if not empty overwrite the location default.
addParameter(ipr,'disable_interactions',false); %(boolean) if true, the communication with the other matlab are disabled
addParameter(ipr,'use_photodiodo',[]); %(boolean)if not empty overwrite the location default.
addParameter(ipr,'debug',0); %(boolean) to activate debug mode.
parse(ipr,varargin{:})

addpath(fileparts(fileparts(fileparts(mfilename('fullpath')))));
custompath = reylab_custompath({'useful_functions/tasks_tools','tasks/.','JoyMEX'});
params = location_setup(ipr.Results.location);

inputs = fields(ipr.Results);
for i =1:numel(inputs)
    pstr = inputs{i};
    if any(strcmp(ipr.UsingDefaults,pstr)) && isempty(ipr.Results.(pstr)) && isfield(params, pstr) %parameter not passed and empty
        continue
    end
    params.(pstr) = ipr.Results.(pstr);
end

custompath.add(params.additional_paths,true)
params.do_sorting = true;
fail_safe_mode = ~params.is_online;
ini = [];
fin = [];
MAX_TRIALS = 12;

if strcmp(subtask,'DynamicScr')
    rel_path_pics = fullfile('pics_space','dynamic_scr');
    P2REMOVE = [80 90 60 60 0]; %old P2REMOVE = [50 50 50 60 60 0];
    %     MANUAL_SELECT = [true, true, true, false, false];
    MANUAL_SELECT = [true, true, true, false, false];
    NREP = [6 3 3 4 0]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     NREP = [2 1 1 1 1]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [180 180 180 120 0];
    MIN_SAFE_TRIALS = 15;
    MAX_TRIALS = 15;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    %if there are not enought pictures show the amount added from pic us
elseif strcmp(subtask,'DynamicScrTest')
    rel_path_pics = fullfile('pics_space','dynamic_scr');
    P2REMOVE = [80 90 60 60 0]; %old P2REMOVE = [50 50 50 60 60 0];
    %     MANUAL_SELECT = [true, true, true, false, false];
    MANUAL_SELECT = [true, true, true, false, false];
    NREP = [1 1 1 1 0]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     NREP = [2 1 1 1 1]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [180 180 180 120 0];
    MIN_SAFE_TRIALS = 5;
    MAX_TRIALS = 6;
    params.use_only_main_pics = false;
    params.auto_resp = true;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    %if there are not enought pictures show the amount added from pic us
elseif strcmp(subtask,'DynamicSeman')
    rel_path_pics = fullfile('pics_space','seman_pics');
    P2REMOVE = [0,0,0]; %old P2REMOVE = [50 50 50 60 60 0];
    MANUAL_SELECT = [false,false,false];
    NREP = [4,3,3]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [300,300,300];
    MIN_SAFE_TRIALS = 10;
    params.use_only_main_pics = true;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    %if there are not enought pictures show the amount added from pic us
elseif strcmp(subtask,'DynamicSemanTest')
    rel_path_pics = fullfile('pics_space','seman_test');
    %     P2REMOVE = [0]; %old P2REMOVE = [50 50 50 60 60 0];
    P2REMOVE = [0,0,0];
    %     MANUAL_SELECT = [false];
    MANUAL_SELECT = [false,false,false];
    %     NREP = [2]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NREP = [2,1,1]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     NPICS = [60];
    NPICS = [300,300,300];
    MIN_SAFE_TRIALS = 15;
    params.acq_network=0;
    params.use_only_main_pics = true;
    params.is_online = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    %if there are not enought pictures show the amount added from pic us
elseif strcmp(subtask,'CategLocalizTest')
    rel_path_pics = fullfile('pics_space','categ_localiz');
    P2REMOVE = [100,0,0]; %old P2REMOVE = [50 50 50 60 60 0];
    MANUAL_SELECT = [true,true,false];
    NREP = [1,1,3]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     NREP = [2,1,1]; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [180,180,0];
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
    NPICS = [180,180,0];
    %     NPICS = [120,120,120];
    MIN_SAFE_TRIALS = 10;
    MAX_TRIALS = 12;
    %params.acq_network=1;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    %if there are not enought pictures show the amount added from pic us
elseif strcmp(subtask,'FirstTime')
    rel_path_pics = 'picsfirst';
    params.is_online = false;
    fail_safe_mode = false;
    params.EMU_num = NaN;
    params.do_sorting = false;
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    NREP = 9; if ~isempty(params.Nrep), NREP= params.Nrep; end
    MIN_SAFE_TRIALS = max(NREP);
    params.acq_network=0;
    params.with_acq_folder=0;
    params.use_only_main_pics = true;
    if numel(NREP)>1
        error('cannot have multiple subscr in the mini_sr subtask')
    end
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    % elseif strcmp(subtask,'mini_scr')
elseif strcmp(subtask,'OnlineMiniScr')
    rel_path_pics = 'miniscr_pics';
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    %     NREP = 18; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     MIN_SAFE_TRIALS = max(NREP);
    params.use_only_main_pics = true;
    %     if numel(NREP)>1
    %         error('cannot have multiple subscr in the mini_sr subtask')
    %     end
    if isempty(params.templates_required), params.templates_required= true; end
    %     if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
    % move folder and delete templates, etc
% elseif strcmp(subtask,'DailyMiniScr')
%     rel_path_pics = 'miniscr_pics';
%     P2REMOVE = 0;
%     MANUAL_SELECT = false;
% %     NREP = 18; if ~isempty(params.Nrep), NREP= params.Nrep; end
% %     MIN_SAFE_TRIALS = max(NREP);
%     params.use_only_main_pics = true;
% %     if numel(NREP)>1
% %         error('cannot have multiple subscr in the mini_sr subtask')
% %     end
%     if isempty(params.templates_required), params.templates_required= true; end
%     %     if isempty(params.templates_required), params.templates_required= false; end
%     params.remove_pictures = false;
%     % move folder and delete templates, etc
elseif strcmp(subtask,'OnlineMiniScrMusic')
    rel_path_pics = 'miniscr_pics';
    P2REMOVE = 0;
    MANUAL_SELECT = false;
    %     NREP = 18; if ~isempty(params.Nrep), NREP= params.Nrep; end
    %     MIN_SAFE_TRIALS = max(NREP);
    params.use_only_main_pics = true;
    %     if numel(NREP)>1
    %         error('cannot have multiple subscr in the mini_sr subtask')
    %     end
    %     if isempty(params.templates_required), params.templates_required= true; end
    params.templates_required= false;
    params.online_notches= false;
    params.remove_pictures = false;
    % move folder and delete templates, etc
elseif strcmp(subtask,'Test')
    params.acq_network=0;
    params.with_acq_folder=1;
    rel_path_pics = fullfile('pics_space','custom_pics');
    P2REMOVE = [2 20 0];
    NREP = [2 2 6]*1; if ~isempty(params.Nrep), NREP= params.Nrep; end
    NPICS = [30 30 10];
    MIN_SAFE_TRIALS = 4;
    MANUAL_SELECT = [false, false, false];
    %     MANUAL_SELECT = [true, false, false];
    %     params.copy_backup=0;
    %     params.copy_backup=1;
    params.use_only_main_pics = false;
    if isempty(params.templates_required), params.templates_required= false; end
    params.remove_pictures = false;
else
    error('subtask no found. Available subtasks: DynamicScr, DynamicScrTest, DynamicSeman, DynamicSemanTest,CategLocaliz,FirstTime, OnlineMiniScr, DailyMiniScr and Test')%update this list if new option is added
end

min_seq_length = 60; % 30 secs is the minimum duration per trial (the actual length will be between min_seq_length and 2*min_seq_length)
n_blks = length(NPICS);

if isempty(params.sub_ID)
    params.sub_ID = strtrim(fileread(fullfile(params.beh_rec_metadata,'current_subject_id.txt')));
end
if isunix, system('nmcli radio wifi off'); end %disable wifi in ubuntu

if params.ptb_debug;  PsychDebugWindowConfiguration; end %enable extra debugs in ptb

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


if isempty(params.EMU_num) || isempty(params.run_num)
    allemus = dir(fullfile(params.acq_remote_folder_in_beh,['EMU-*' params.sub_ID '*']));
    allemus = {allemus.name};
    if exist(fullfile(params.acq_remote_folder_in_beh,'transferred'),'dir')
        transfemus = dir(fullfile(params.acq_remote_folder_in_beh,'transferred'));
        allemus = [allemus {transfemus.name}];
    end
    emulog_file = fullfile(params.acq_remote_folder_in_beh,['emulog_' params.sub_ID '.txt']);
    if exist(emulog_file,'file')
        lines = readlines(emulog_file,'EmptyLineRule','skip');
        allemus = [allemus lines'];
    end
    if isempty(params.EMU_num)
        if isempty(allemus)
            params.EMU_num = 1;
        else
            nums = zeros(1,length(allemus));
            for i = 1:length(allemus)
                num = regexp(allemus{i},['EMU-(\d+)_subj-', params.sub_ID, '*'],'tokens','once');
                if ~isempty(num)
                    nums(i) = str2num(num{1});
                end
            end
            params.EMU_num = max(nums) + 1;
        end
    end
    if isempty(params.run_num)
        if isempty(allemus)
            params.run_num = 1;
        else
            nums = zeros(1,length(allemus));
            for i = 1:length(allemus)
                num = regexp(allemus{i},['EMU-\d+_subj-', params.sub_ID, '_task-RSVP',subtask,'_run-(\d+)'],'tokens','once');
                if ~isempty(num)
                    nums(i) = str2num(num{1});
                end
            end
            params.run_num = max(nums) + 1;
        end
    end
end

if params.acq_network
    experiment.fname       = sprintf('EMU-%.3d_subj-%s_task-RSVP%s_run-%.2d',params.EMU_num,params.sub_ID,subtask,params.run_num);

else
    experiment.fname       = 'dyn_scr_test';
end

experiment.folder_name = [params.root_processing experiment.fname];

if ~isfolder(experiment.folder_name)
    mkdir(experiment.folder_name);
end

% experiment properties file
% experiment_prop_file = [experiment.folder_name filesep exp_prop_file];

with_reset = false;
abort = false;
reached_backup = false;


all_pics_path = [params.pics_root_beh filesep rel_path_pics];
experiment.task_pics_folder = all_pics_path;
custom_pics_path = [all_pics_path filesep 'custom_pics'];
% a = dir(Path_pics);
% filt_a = arrayfun(@(x) contains (lower(x.name), '.jp'),a);
% a = a(filt_a);

% Get custom pics
if isfolder(custom_pics_path)
    all_files = dir(fullfile(custom_pics_path, '**','*.*'));
    all_files = all_files(~[all_files.isdir]);  % remove directories
    custom_pics_list = all_files(endsWith({all_files.name}, {'.jpg','.jpeg'}, 'IgnoreCase', true));

    custom_pics_list = struct2table(custom_pics_list);
    custom_pics_list = custom_pics_list(:,1:2); % Keep only name, folderpath
    % add a column selectable to make a picture selectable as well as
    % colored differently based on rule that added the picture
    custom_pics_list.selectable = zeros(height(custom_pics_list), 1);
end


%     if isempty(custom_pics_list)
%         error(['No pictures for this session in ' custom_pics_path]);
%     end

if contains(subtask,'OnlineMiniScr')
    N = length(all_pics_list);
    [NREP, NSEQ, seq_length, estimated_duration] = calculate_miniscr_time(N);
    fprintf(['Estimated duration of the experiment with %d pics (%d trials each) ' ...
        'in %d sequences of %2.1f secs: %1.1f min\n'],N,NREP,NSEQ,seq_length*0.5,estimated_duration)
    if ~isempty(params.Nrep), NREP= params.Nrep; end
    MIN_SAFE_TRIALS = max(NREP);
    if numel(NREP)>1
        error('Cannot have multiple subscr in the mini_sr subtask')
    end
end

% if contains(subtask,'DailyMiniScr') 
%     N = length(a);
%     confirmed = false;
% 
%     fprintf('The number of pictures is: %d\n', N);
% 
%     while ~confirmed
% 
%         %Prompt user for NREP input
%         while true
% 
%             NREP = input('Enter the number of repetitions (NREP):');
%                
%             %Validate NREP
%             if isempty(NREP) || ~isnumber(NREP) || ~isscalar(NREP) || NREP <= 0 || mod(NREP,1) ~=0
%                 disp('Invalid entry. Please enter a positive integer.\n');
%             else
%                 break;
%             end
%         end
% 
%         %Estimate Timing
%         [NSEQ, seq_length, estimated_duration] = calculate_dailyscr_time(N, NREP);
% 
%         fprintf(['Estimated duration of the experiment with %d pics (%d trials each) ' ...
%              'in %d sequences of %2.1f secs: %1.1f min\n'],N,NREP,NSEQ,seq_length*0.5,estimated_duration);
% 
%         while true
% 
%             confirm = input('Is this okay? (Y/N):', 's');
%             if strcmpi(confirm, 'Y')
%                 confirmed = true;
%                 break;
%             elseif strcmpi(confirm, 'N')
%                 disp('Start again....');
%                 break;
%             else
%                 disp('Invalid entry. Please enter Y or N.\n');
%             end
%         end
% 
%     end   
%     if ~isempty(params.Nrep), NREP= params.Nrep; end
%     MIN_SAFE_TRIALS = max(NREP);
%     if numel(NREP)>1
%         error('Cannot have multiple subscr in the mini_sr subtask')
%     end
% end

if ~params.use_only_main_pics
    totalp2load = NPICS(1); % custom pics
    % Get all pics
    if isfolder(all_pics_path)
        all_files = dir(fullfile(all_pics_path, '**','*.*'));
        all_files = all_files(~[all_files.isdir]);  % remove directories
        all_pics_list = all_files(endsWith({all_files.name}, {'.jpg','.jpeg'}, 'IgnoreCase', true));
        all_pics_list = all_pics_list(randperm(length(all_pics_list))); %random order
        all_pics_list = struct2table(all_pics_list);
        all_pics_list = all_pics_list(:,1:2); % Keep only name, folderpath
        % add a column selectable to make a picture selectable as well as
        % colored differently based on rule that added the picture
        all_pics_list.selectable = zeros(height(all_pics_list), 1);
        if isfolder(custom_pics_path)
            % Move custom_pics to top of the table
            all_pics_list = vertcat(custom_pics_list, all_pics_list);
            [~, uniq_idxs] = unique(all_pics_list.name, 'stable');
            all_pics_list = all_pics_list(uniq_idxs, :);
        end
    end
    if height(all_pics_list)< totalp2load
        error('Not enough pictures in folders, add at least %d.',totalp2load-height(all_pics_list))
    end
    %ia = ia(1:totalp2load); %keep only the ones we will use
    all_pics_list.stim_trial_count = zeros(height(all_pics_list),1);
    ImageNames = all_pics_list;
    %     ImageNames = array2table([a(ia)',afolders(ia)],'VariableNames',{'name', 'folder'});

else
    custom_pics_list.stim_trial_count = zeros(height(custom_pics_list),1);
    ImageNames = custom_pics_list;
    totalp2load = height(custom_pics_list);
    if ~exist('NPICS','var') || n_blks == 1
        NPICS = height(custom_pics_list);
    end
end
if contains(subtask,'DynamicScr')
    ImageNames = load_dynamic_scr_img_info(ImageNames);
elseif contains(subtask,'CategLocaliz')
    ImageNames.concept_name = cellfun(@(x) regexpi(x,'^.*(?=(_\d*(?s)\D*$))','match','once'),ImageNames.name,'UniformOutput',false);

    without_numbers = cellfun('isempty',ImageNames.concept_name); %this could be done with a regular expression
    ImageNames.concept_name(without_numbers) = cellfun(@(x) regexpi(x,'^.*(?=((?s)\D*$))','match','once'),ImageNames.name(without_numbers),'UniformOutput',false);    
end
total_figures = height(ImageNames);

if params.acq_network
    recording = recording_handler(params, experiment.fname);
    params.recording_name = recording.rec_name;
end

data_transfer_copy = {'RSVP_SCR_workspace.mat';'rsvpscr_EMU_online.m';'online'};

[kbs,products_names] = GetKeyboardIndices;
dev_used = [];
for i =1:numel(params.keyboards)
    if isnumeric(params.keyboards{i})
        dev_used(end+1) = params.keyboards{i};
    else
        kbix = strcmp(params.keyboards{i}, products_names);
        if ~any(kbix)
            warning('Keyboard %s, not found', params.keyboards{i});
        else
            %             dev_used(end+1) = kbs(kbix);
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
Screen('Preference','VisualDebugLevel',3);


AssertOpenGL;    % Running on PTB-3? Abort otherwise.

exitKey = KbName('F2');
startKey = KbName('s'); %88 in Windows, 27 in MAC
continueKey= KbName('c');  %to continue if gamepad fails


message_begin = {'Ready to begin?';'Listo para empezar?';'Etes-vous pret pour commencer?'};
%     text_endtrial = {'Press the right key to begin the next trial';'Presione la flecha derecha para continuar'};
%     text_probe = {'Have you seen this picture in the last sequence?';'Viste esta imagen en la ultima secuencia?';'Avez vous vu cette image lors de la derniere sequence?'};
message_continue = {'Ready to continue?';'Listo para continuar?';'Etes-vous pret pour continuer?'};
message_wait = {'Take a short break.\nWe will resume shortly';'Take a short break.\nWe will resume shortly';'Take a short break.\nWe will resume shortly'};

msgs_Mat.exper_saved = 'experiment_properties has been saved';
msgs_Mat.rec_started = 'recording has started';
msgs_Mat.scr_conf_updated = 'scr_conf has been updated';
msgs_Mat.scr_conf_read = 'scr_conf has been read';
msgs_Mat.exper_finished = 'experiment finished';
msgs_Mat.ready_begin = 'ready to begin';
msgs_Mat.trial_begin = 'trial starting';
msgs_Mat.trial_end = 'trial ending';
%     msgs_Mat.start_processing = 'trial processing has started';
msgs_Mat.process_ready = 'ready to begin the next trial';
msgs_Mat.process_end = 'online processing completed';
msgs_Mat.exper_aborted = 'experiment was aborted';
msgs_Mat.rec_finished = 'recording has finished';

msgs_Mat.error = 'there was an error. stop waiting';

pre_times=NaN(1,0);

wait_reset = 0.1;  % IT MUST BE SHORTER THAN THE SHORTEST ISI
value_reset = 0;
min_blank=1.25;
max_rand_blank = 0.5;
min_lines_onoff=0.5;
max_rand_lines_onoff = 0.2;
size_line = 5;
size_point = 10;
colorOval = [[255,0,0];[255 255 0]];
gamepadname = 'Logitech'; %To be updated if we use a different gamepad in the future 


pic_onoff = [[1 4 16];[2 8 32]];  % first pic with row 2
bits_for_break = [];
blank_on = 11;
lines_onoff = 13;
continue_msg_on = 15;
lines_flip_blank = 103;
lines_flip_pic = 22;
trial_on = 26;
data_signature_on = 64;
data_signature_off = 128;

msgs = {'blank on';'lines on';'pic change';...
    'lines change blank';'lines change pic';'lines off';'trial ended'};
msgs_colors = {255;65280;16711680;16777215;65535;16711935;16776960};

rng('shuffle', 'twister')

experiment.pwd=pwd;
experiment.params=params;
experiment.subtask = subtask;
% experiment.date=date;
experiment.date=datetime(now,'ConvertFrom','datenum');
experiment.pic=pic_onoff;
experiment.blank_on=blank_on;
experiment.lines_onoff=lines_onoff;
experiment.continue_msg_on=continue_msg_on;
experiment.lines_flip_blank=lines_flip_blank;
experiment.lines_flip_pic=lines_flip_pic;
experiment.trial_on=trial_on;
experiment.bits_for_break=bits_for_break;
experiment.data_signature=[data_signature_on data_signature_off];
experiment.value_reset=value_reset;
experiment.wait_reset=wait_reset;
experiment.ImageNames=ImageNames;
experiment.msgs=msgs;
experiment.deviceresp=params.device_resp;
experiment.msgs_Mat=msgs_Mat;
experiment.with_reset=with_reset;
experiment.P2REMOVE = P2REMOVE;
experiment.NREP = NREP;
experiment.NPICS = NPICS;
experiment.MANUAL_SELECT = MANUAL_SELECT;
experiment.MAX_TRIALS = MAX_TRIALS;
experiment.N_BLKS = n_blks;

exp_prop_file = 'experiment_properties_online3.mat';
%save(exp_prop_file,'experiment','-v7');
access_exp_prop_file(exp_prop_file,'experiment',experiment);

if params.is_online
    answer = questdlg('Start the online function on a separate Matlab and wait for instructions to continue','Online started?','Continue','Continue');
    if isempty(answer)
        error('Question dialog window closed.')
    end

    M_PTB = matconnect(params.proccesing_machine);
end

Screen('Preference', 'SkipSyncTests', double(IsWin));

if strcmp(params.lang,'english')
    ind_lang=1;
elseif strcmp(params.lang,'spanish')
    ind_lang=2;
elseif strcmp(params.lang,'french')
    ind_lang=3;
end

t_stimon = [];
times = [];
inds_pics = [];
inds_start_seq = [];
t_fliptime = [];
t_DAQpic = [];
times_break = [];
lines_offset = 50;

gamepad_ix = [];
if strcmp(params.device_resp,'gamepad')
    if IsWin
        clear JoyMEX;
        JoyMEX('init',0);
    elseif IsLinux
        numGamepads = Gamepad('GetNumGamepads');
        if (numGamepads == 0)
            error('Gamepad not connected');
        else
            [~, gamepad_name] = GetGamepadIndices;
            idx = find(contains(gamepad_name, gamepadname, 'IgnoreCase', true), 1);
            gamepad_name = gamepad_name{idx};
            gamepad_ix = Gamepad('GetGamepadIndicesFromNames',gamepad_name);
        end
    else
        %         % Initialization of the gamepad, for collecting aswers
        %         numGamepads = Gamepad('GetNumGamepads');
        %         if (numGamepads == 0)
        %             error('Gamepad not connected');
        %         else
        %             [~, gamepad_name] = GetGamepadIndices;
        %             gamepad_index = Gamepad('GetGamepadIndicesFromNames',gamepad_name);
        %             gp_numButtons = Gamepad('GetNumButtons', gamepad_index);
        %             % gamepad button map:
        %             % A = 1, B = 2, X = 3, Y = 4, upper-left = 5, upper-right = 6
        %         end
        error('gamepad not coded for this OS');
    end
end
scr_end_cell = {};
%rsvpscr diary
rsvpscr_diary_txt_filename = [datestr(now,'mm-dd-yy') '_' datestr(now,'HH_MM_SSAM') '_RSVPSCR_diary.txt'];
rsvpscr_diary_txt_file = [params.processing_rec_metadata filesep ...
    experiment.fname '_'...
    rsvpscr_diary_txt_filename];

rsvpscr_diary_txt_file = fullfile(rsvpscr_diary_txt_file(~isspace(rsvpscr_diary_txt_file)));

diary(rsvpscr_diary_txt_file)
try
    screens=Screen('Screens');
    whichScreen=max(screens);
    % whichScreen=1; % uncomment when using rdp
    % if IsWin
    %     whichScreen = 0; % Temp
    % end
    if params.use_daq;   dig_out  = TTL_device(params.ttl_device); end

    % Open screen.  Do this before opening the
    % offscreen windows so you can align offscreen
    % window memory to onscreen for faster copying.
    if isfield(params,'windowRect')
        [window,windowRect]=Screen(whichScreen,'OpenWindow',0, params.windowRect);
    else
        [window,windowRect]=Screen(whichScreen,'OpenWindow',0);
    end
    %flush diary buffer
    %fprintf("\n")
    %disp('For diary: Screen opened and textures created');
    drawnow;

    tex_all=zeros(1,total_figures);
    imageRect = cell(total_figures,1);
    destRect_all = cell(total_figures,1);
    for i=1:total_figures
        %             pic_path = fullfile(params.pics_root_beh, ImageNames.folder{i},ImageNames.name{i});
        pic_path = fullfile(ImageNames.folder{i},ImageNames.name{i});
        Im = imread(pic_path);
        nRows=size(Im,1); nCols=size(Im,2);
        if contains(subtask,'CategLocaliz') || strcmp(subtask,'DynamicSeman')
            if ~(nRows==320 && nCols==320)
                error('picture %s with wrong dimensions', pic_path)
            end
        else
            if ~(nRows==160 && nCols==160)
                error('picture %s with wrong dimensions', pic_path)
            end
        end
        imageRect{i}=SetRect(0,0,nCols,nRows);
        destRect_all{i}=CenterRect(imageRect{i},windowRect);
        tex_all(i)=Screen('MakeTexture',window,Im);
    end
    %window and monitor properties
    xcenter=windowRect(3)/2;
    ycenter=windowRect(4)/2;
    %     Priority(9); %Enable realtime-scheduling in MAC
    Priority(params.ptb_priority_high); %high priority in Windows
    ifi = Screen('GetFlipInterval', window, 200);
    slack=ifi/4;
    Priority(params.ptb_priority_normal); %normal priority
    frame_rate=1/ifi;
    white=WhiteIndex(window);
    black=BlackIndex(window);
    Screen('TextSize',window, 32);
    flickerSquare = flickerSquareLoc(windowRect,24,2,'BottomLeft');

    experiment.xcenter=xcenter;
    experiment.ycenter=ycenter;
    experiment.frame_duration=ifi;
    experiment.frame_rate=frame_rate;
    experiment.flickerSquare=flickerSquare;

    keysOfInterest=zeros(1,256);
    firstPress=zeros(1,256);
    keysOfInterest([exitKey startKey continueKey])=1;

    %save(exp_prop_file,'experiment', '-v7.3');
    access_exp_prop_file(exp_prop_file,'experiment',experiment);

    if params.is_online
        M_PTB.send(msgs_Mat.exper_saved);
    end

    answer = questdlg('Tell the subject to get ready to begin. Press OK to continue','Subject ready?','OK','OK');
    k=1;

    if params.acq_network
        try
            recording.start();
            pre_times(k)=GetSecs; k=k+1;
        catch ME
            errMsg = getReport(ME);
            disp(errMsg);
            error('Failed to start recording');
        end

    end
    WaitSecs(6);
    %flush diary buffer 
    %fprintf("\n");
    %disp('For diary: Recording started');
    drawnow;

    if params.is_online
        M_PTB.send(msgs_Mat.rec_started);
    end

    if params.use_daq
        dig_out.send(data_signature_on);
        WaitSecs(0.05);
        dig_out.send(data_signature_off);
        WaitSecs(0.45);
        dig_out.send( data_signature_on);
        WaitSecs(0.05);
        dig_out.send( data_signature_off);
        WaitSecs(0.45);
        dig_out.send( data_signature_on);
        WaitSecs(0.05);
        dig_out.send( data_signature_off);
        pre_times(k)=GetSecs; k=k+1;
    end

    for d=dev_used
        KbQueueCreate(d,keysOfInterest);
        KbQueueStart(d);
    end

    pressed=0;
    save('RSVP_SCR_workspace','-regexp', '^(?!(M_PTB)$).');
    if params.is_online
        msg_received = M_PTB.waitmessage(5);
        if ~strcmp(msg_received,msgs_Mat.ready_begin)
            if isempty(msg_received)
                error('Connection timed out');
            else
                error(['Inconsistency with message received:' msg_received]);
            end
        end
    end
    scr_config_cell = {};
    pics2use = 1:height(custom_pics_list);
    stim_rm_cell = {};
    scr_id = 0;
    safe_not_done = true;

    if contains(subtask,'DynamicSeman')
        lists_path = fullfile(params.pics_root_beh,'pics_space','seman_lists');
        lists = dir(fullfile(lists_path, '*.*'));
        lists = lists(~[lists.isdir]);
        if length(lists) > 100
            for m = 1:(length(lists)-100)
                lists(1) = [];
            end
        end
        if isempty(lists)
            error('No files found in the folder.');
        end
        %          numFiles = numel(lists);

        %         randomIndex = randi(numel(lists));

        %         if numFiles < numToSelect
        %             error('There are not enough files in the folder to select %d files.', numToSelect);
        %         end

        selectedLists={};
        for i = 1:length(lists)

            if params.run_num == 1

                for y = 1:length(lists)
                    if ~contains(lists(y).name, 'seeg_stimlist-A-10-')
                        first_lists{y} = lists(y).name;
                    else
                        last_list{y} = lists(y).name;
                    end
                end
                sorted_lists = [first_lists, last_list];
                sorted_lists = sorted_lists(~cellfun('isempty', sorted_lists));

                if contains(lists(i).name, 'seeg_stimlist-A-')
                    selectedLists{i} = lists(i).name;
                end
            else
                for y = 1:length(lists)
                    if ~contains(lists(y).name, 'seeg_stimlist-B-10-')
                        first_lists{y} = lists(y).name;
                    else
                        last_list{y} = lists(y).name;
                    end
                end
                sorted_lists = [first_lists, last_list];
                sorted_lists = sorted_lists(~cellfun('isempty', sorted_lists));
                if contains(lists(i).name, 'seeg_stimlist-B-')
                    selectedLists{i} = lists(i).name;
                end
            end
        end




        %         randomIndices = randperm(numFiles, numToSelect);
        %
        %         selectedLists = lists(randomIndices);

    end

    while (scr_id < n_blks) && safe_not_done %|| fail_safe_mode
        scr_id = scr_id + 1;
        if fail_safe_mode
            safe_not_done = false;
            pics_shown = ImageNames.stim_trial_count(pics2use)>0;
            pics2use = pics2use(pics_shown); %only keep using the pics already shown
            tokeep = ImageNames.stim_trial_count(pics2use)<MIN_SAFE_TRIALS;
            if ~any(tokeep)
                break
            end
            max_t_counter = max(ImageNames.stim_trial_count(pics2use(tokeep)));
            Nrep = MIN_SAFE_TRIALS-max_t_counter; %could be changed
            Npics = sum(tokeep);

            if Npics<min_seq_length
                Npics=min_seq_length;
            end

            Nseqxrep = floor(Npics/min_seq_length);
            seq_length = floor(Npics/Nseqxrep);
            while (seq_length*Nseqxrep<Npics)
                Npics = Npics+1;
                Nseqxrep = floor(Npics/min_seq_length);
                seq_length = floor(Npics/Nseqxrep);
            end
            if Npics>numel(pics2use) %it needs more pictures than the available
                Npics = sum(tokeep);
                Nrepxseq = ceil(min_seq_length/Npics);
                Nseq = Nrep/Nrepxseq;
                if mod(Nseq,1)~=0
                    Nrep = ceil(Nseq)*Nrepxseq;
                end
            else
                [~, ssorted] = sort(ImageNames.stim_trial_count(pics2use));
                pics2use = pics2use(ssorted(1:Npics));
            end
            
            NREP(scr_id) = Nrep;
            if isempty(ini)
                ini = 1;
            else
                ini = ini + NREP(scr_id-1);
            end
            fin = ini + NREP(scr_id);
            NPICS(scr_id) = Npics;
        else
            if isempty(ini)
                ini = 1;
            else
                ini = ini + NREP(scr_id-1);
            end
            fin = ini + NREP(scr_id)-1;
        end



        %load texture info for the pics2use
        tex = tex_all(pics2use);
        destRect = destRect_all(pics2use);

        if contains(subtask,'DynamicSeman')
            scr_config = shuffle_rsvpSCR_online3(NREP(scr_id),NPICS(scr_id),subtask,ImageNames,sorted_lists,scr_id,ini,fin);
        else
            [experiment, scr_config] = shuffle_rsvp_dynamic_2(experiment, pics2use, scr_id);
            %save(exp_prop_file,'experiment', '-v7.3');
            access_exp_prop_file(exp_prop_file,'experiment',experiment);

        end

        %         scr_config = shuffle_rsvpSCR_online3(NREP(n_scr),NPICS(n_scr),selectedLists);
        scr_config.pics2use = pics2use;
        scr_config.fail_safe_mode = fail_safe_mode;
        scr_config.manual_select = MANUAL_SELECT(scr_id);
        scr_config_cell{end+1} = scr_config;

        %save(exp_prop_file,'scr_config_cell','-append');
        access_exp_prop_file(exp_prop_file,'scr_config_cell',scr_config_cell);

        pause(3)
        if params.is_online
            M_PTB.send(msgs_Mat.scr_conf_updated);
        end
        order_pic=scr_config.order_pic;
        order_ISI=scr_config.order_ISI;
        ISI=scr_config.ISI;
        seq_length=scr_config.seq_length;
        Nseq=scr_config.Nseq;
        lines_change=scr_config.lines_change;


        NISI=numel(ISI);
        times=NaN*ones(1,Nseq*(NISI+1+2+NISI*seq_length+6+1));
        times(1:length(pre_times)) = pre_times;
        k=numel(pre_times)+1;
        t_stimon = NaN*ones(1,Nseq*seq_length);
        t_fliptime = NaN*ones(1,Nseq*seq_length);
        t_DAQpic = NaN*ones(1,Nseq*seq_length);
        time_wait = NaN*ones(1,Nseq);
        inds_pics = zeros(1,seq_length*NISI*Nseq);
        inds_start_seq = zeros(1,Nseq);
        times_break = [];

        randTime_blank = min_blank + max_rand_blank*rand(NISI+1,Nseq);
        randTime_lines_on = min_lines_onoff + max_rand_lines_onoff*rand(1,Nseq);
        randTime_lines_off = randTime_blank(NISI+1,:) - (min_lines_onoff + max_rand_lines_onoff*rand(1,Nseq));
        if ~fail_safe_mode && params.is_online
            msg_received = M_PTB.waitmessage(5);
            if ~strcmp(msg_received,msgs_Mat.scr_conf_read)
                if isempty(msg_received)
                    warning('Connection timed out');
                else
                    warning('Inconsistency with message received');
                end
                fail_safe_mode = true;
                warning('fail_safe_mode: activated')
            end
        end

        while ~abort
            irep = 1;
            if params.auto_resp
                init_time = tic();
            else
                if scr_id==1
                    init_time = tic();
                    print_message(message_begin{ind_lang},black,window);
                    while ~(pressed && any(firstPress([startKey exitKey])))
                        [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);
                    end
                else
                    print_message(message_continue{ind_lang},black,window);
                    [~,~,pressed,firstPress] = get_response(dev_used,params.device_resp,[exitKey continueKey],0.2,params.auto_resp,gamepad_ix);
                    if pressed && firstPress(exitKey)>0, abort = true; break; end
                end

                if pressed && firstPress(exitKey)>0, abort = true; break; end
                for d= dev_used; KbQueueFlush(d);   end
            end
            Priority(params.ptb_priority_high);

            iind=1;
            HideCursor;

            estimated_duration = ((0.5*seq_length + 3 + 5)* Nseq)/60;

            fprintf('\n');
            fprintf(['\n Estimated duration of the subscreening: %1.1f min (%2.1f secs per sequence)'],estimated_duration, seq_length*0.5);

            fprintf('\n Subscr %d. Current sequence (Nseq = %d, Npics = %d, seqlen = %d): ',scr_id,Nseq,NPICS(scr_id),seq_length);

            for irep=1:Nseq
                if params.is_online
                    M_PTB.send(msgs_Mat.trial_begin);
                end
                fprintf('%d, ',irep)
                ich_blank=1;
                ich_pic=1;
                WaitSecs(0.150);
                for d= dev_used; KbQueueFlush(d);   end

                Screen('FillRect',window,black);
                times(k)=Screen('Flip',window);
                if params.use_daq
                    dig_out.send( blank_on);
                end
                inds_start_seq(irep)=k;
                tprev = times(k);

                color_up = scr_config.color_start.up{irep};
                color_down = scr_config.color_start.down{irep};
                color_oval = colorOval(randsample([1 2],1),:);

                Screen('FillRect',  window,black);
                Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                if contains(subtask,'CategLocaliz')
                    Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                end

                if params.use_daq && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    dig_out.send(value_reset);
                end
                k=k+1;

                times(k) = Screen('Flip',window,times(k-1)+randTime_lines_on(1,irep));

                if params.use_daq
                    dig_out.send( lines_onoff);
                end

                if lines_change{irep}{1}{ich_blank,1}==1
                    color_up = lines_change{irep}{1}{ich_blank,3};
                    color_down = lines_change{irep}{1}{ich_blank,4};
                    Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    if contains(subtask,'CategLocaliz')
                        if color_oval == colorOval(1,:)
                            color_oval = colorOval(2,:);
                        else
                            color_oval = colorOval(1,:);
                        end
                        Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                    end

                    if params.use_daq && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        dig_out.send(value_reset);
                    end
                    k=k+1;
                    times(k) = Screen('Flip',window,times(k-1)+lines_change{irep}{1}{ich_blank,2});
                    if params.use_daq
                        dig_out.send( lines_flip_blank);
                    end
                    if params.use_daq && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        dig_out.send(value_reset);
                    end
                    ich_blank = ich_blank +1;
                end
                k=k+1;

                for iISI=1:NISI
                    which_ISI = order_ISI(iISI,irep);

                    Screen('DrawTexture', window,tex(order_pic(1,which_ISI,irep)),[],destRect{order_pic(1,which_ISI,irep)},0);
                    Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    if contains(subtask,'CategLocaliz')
                        Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                    end
                    Screen('FillRect',window,white,flickerSquare);

                    [times(k),t_stimon(iind),t_fliptime(iind)]=Screen('Flip',window,tprev+randTime_blank(iISI,irep)-slack,1);

                    if params.use_daq
                        dig_out.send( pic_onoff(2,ceil(3*irep/Nseq)));
                        t_DAQpic(iind) = GetSecs;
                    end
                    inds_pics(iind)=k;
                    tprev = times(k);
                    iind=iind+1;
                    Screen('FillRect',window,black,flickerSquare);
                    Screen('Flip',window,times(k)+3*ifi-slack);

                    if params.use_daq && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        dig_out.send(value_reset);
                    end
                    k=k+1;

                    if which_ISI==lines_change{irep}{2}{ich_pic,1} && lines_change{irep}{2}{ich_pic,5}==1
                        Screen('DrawTexture', window,tex(order_pic(1,which_ISI,irep)),[],destRect{order_pic(1,which_ISI,irep)},0);
                        color_up = lines_change{irep}{2}{ich_pic,3};
                        color_down = lines_change{irep}{2}{ich_pic,4};
                        Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                        Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                        if contains(subtask,'CategLocaliz')
                            if color_oval == colorOval(1,:)
                                color_oval = colorOval(2,:);
                            else
                                color_oval = colorOval(1,:);
                            end
                            Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                        end
                        times(k) = Screen('Flip',window,tprev+lines_change{irep}{2}{ich_pic,2}-slack);
                        if params.use_daq
                            dig_out.send( lines_flip_pic);
                        end
                        ich_pic = ich_pic +1;
                        if params.use_daq && with_reset
                            WaitSecs('UntilTime', times(k)+wait_reset);
                            dig_out.send(value_reset);

                        end
                        k=k+1;
                    end

                    for ipic=2:seq_length
                        Screen('DrawTexture', window,tex(order_pic(ipic,which_ISI,irep)),[],destRect{order_pic(ipic,which_ISI,irep)},0);
                        Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                        Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                        if contains(subtask,'CategLocaliz')
                            Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                        end
                        Screen('FillRect',window,white,flickerSquare);

                        [times(k), t_stimon(iind), t_fliptime(iind)] = Screen('Flip',window,tprev+ISI(which_ISI)-slack,1);
                        if params.use_daq
                            dig_out.send( pic_onoff(mod(ipic,2)+1,ceil(3*irep/Nseq)));
                            t_DAQpic(iind) = GetSecs;
                        end
                        inds_pics(iind)=k;
                        tprev = times(k);
                        iind=iind+1;
                        Screen('FillRect',window,black,flickerSquare);
                        Screen('Flip',window,times(k)+3*ifi-slack);

                        if params.use_daq && with_reset
                            WaitSecs('UntilTime', times(k)+wait_reset);
                            dig_out.send( value_reset);
                        end
                        k=k+1;

                        if which_ISI==lines_change{irep}{2}{ich_pic,1} && lines_change{irep}{2}{ich_pic,5}==ipic
                            Screen('DrawTexture', window,tex(order_pic(ipic,which_ISI,irep)),[],destRect{order_pic(ipic,which_ISI,irep)},0);
                            color_up = lines_change{irep}{2}{ich_pic,3};
                            color_down = lines_change{irep}{2}{ich_pic,4};
                            Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                            Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                            if contains(subtask,'CategLocaliz')
                                if color_oval == colorOval(1,:)
                                    color_oval = colorOval(2,:);
                                else
                                    color_oval = colorOval(1,:);
                                end
                                Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                            end
                            times(k) = Screen('Flip',window,tprev+lines_change{irep}{2}{ich_pic,2}-slack);
                            if params.use_daq
                                dig_out.send( lines_flip_pic);
                            end
                            ich_pic = ich_pic +1;
                            if params.use_daq && with_reset
                                WaitSecs('UntilTime', times(k)+wait_reset);
                                dig_out.send( value_reset);
                            end
                            k=k+1;
                        end
                    end

                    Screen('FillRect',  window,black);
                    Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    if contains(subtask,'CategLocaliz')
                        Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                    end
                    times(k) = Screen('Flip',window,tprev+ISI(which_ISI)-slack);
                    if params.use_daq
                        dig_out.send(lines_onoff);
                    end

                    tprev = times(k);

                    if params.use_daq && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        dig_out.send(value_reset);
                    end
                    k=k+1;

                    if lines_change{irep}{1}{ich_blank,1}==1+iISI
                        color_up = lines_change{irep}{1}{ich_blank,3};
                        color_down = lines_change{irep}{1}{ich_blank,4};
                        Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                        Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                        if contains(subtask,'CategLocaliz')
                            if color_oval == colorOval(1,:)
                                color_oval = colorOval(2,:);
                            else
                                color_oval = colorOval(1,:);
                            end
                            Screen('FillOval', window, color_oval, [xcenter - size_point / 2, ycenter - size_point / 2, xcenter + size_point / 2, ycenter + size_point / 2]);
                        end
                        times(k) = Screen('Flip',window,times(k-1)+lines_change{irep}{1}{ich_blank,2});
                        if params.use_daq
                            dig_out.send( lines_flip_blank);
                        end
                        ich_blank = ich_blank +1;
                        if params.use_daq && with_reset
                            WaitSecs('UntilTime', times(k)+wait_reset);
                            dig_out.send(value_reset);
                        end
                        k=k+1;
                    end
                end

                Screen('FillRect',  window,black);
                times(k) = Screen('Flip',window,tprev+randTime_lines_off(1,irep)-slack);
                if params.use_daq
                    dig_out.send(blank_on);
                end
                if params.use_daq && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    dig_out.send(value_reset);
                end
                k=k+1;

                WaitSecs(tprev+randTime_blank(NISI+1,irep)-GetSecs);

                % CHECK KBQUEUE (see how to collect all the spacebar presses)
                [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);

                if pressed && firstPress(exitKey)>0
                    abort = true;
                    disp('Abort key pressed.')
                    break
                end

                print_message(message_wait{ind_lang},black,window);
                if ~fail_safe_mode && params.is_online
                    if  (scr_id == n_blks && irep==Nseq && iISI==NISI)
                        M_PTB.send(msgs_Mat.exper_finished);
                    else
                        M_PTB.send(msgs_Mat.trial_end);
                    end
                    [msg_received, time_wait(irep)]= M_PTB.waitmessage_or_key('F2',dev_used);

                    if ~strcmp(msg_received,msgs_Mat.process_ready)
                        if isempty(msg_received)
                            warning('Keypressed to stop waiting');
                        elseif strcmp(msg_received,msgs_Mat.error)
                            warning('error in online Matlab');
                        else
                            warning('Inconsistency in online Matlab');
                        end
                        fail_safe_mode = true;
                        warning('fail_safe_mode: activated');
                    end
                end
                times(k)=GetSecs;
                if params.use_daq
                    dig_out.send(continue_msg_on);
                end
                if params.use_daq && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    dig_out.send(value_reset);
                end
                k=k+1;
                print_message(message_continue{ind_lang},black,window);

                Screen('FillRect',window,black);
                for d= dev_used; KbQueueFlush(d);   end
                [~,~,pressed,firstPress] = get_response(dev_used,params.device_resp,[exitKey continueKey],0.2,params.auto_resp,gamepad_ix);
                times(k)=GetSecs;
                if params.use_daq
                    dig_out.send(trial_on);
                end
                if params.use_daq && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    dig_out.send(value_reset);
                end
                k=k+1;
                for d= dev_used; KbQueueFlush(d);   end
                if pressed && firstPress(exitKey)>0, abort = true; break; end

             %flush diary buffer
             %fprintf("\n");
             %frintf('For diary: Finished subscreen %d\n', n_scr);
             drawnow;

            end

            scr_end = struct;
            times(isnan(times))=[];
            inds_pics(inds_pics==0)=[];

            inds_start_seq(inds_start_seq==0)=[];
            scr_end.times=times;
            scr_end.t_stimon=t_stimon;
            scr_end.t_fliptime=t_fliptime;
            scr_end.t_DAQpic=t_DAQpic;
            scr_end.inds_pics=inds_pics;
            scr_end.inds_start_seq=inds_start_seq;
            scr_end.answer=answer;
            scr_end.abort=abort;
            scr_end.last_rep=irep-abort;
            scr_end.times_break=times_break;
            scr_end.time_wait=time_wait;
            scr_end_cell{end+1} = scr_end;

            pics_used_ids = [];
            for ii=1:length(scr_config_cell)
                pics_used_ids = [pics_used_ids scr_config_cell{ii}.pics2use];
            end
            pics_used_ids = unique(pics_used_ids);

            %save(exp_prop_file,'pics_used_ids','scr_end_cell','-append');
            access_exp_prop_file(exp_prop_file,'pics_used_ids',pics_used_ids);
            access_exp_prop_file(exp_prop_file,'scr_end_cell',scr_end_cell);

            break % break while loop if all sequences are done
        end % while ~abort

        %save(exp_prop_file,'abort','-append');
        access_exp_prop_file(exp_prop_file,'abort',abort);
        if abort
            break
        end
        ShowCursor;
        print_message(message_wait{ind_lang},black,window);
        if ~fail_safe_mode && params.is_online
            msg_received = M_PTB.waitmessage_or_key('F2',dev_used);
            if isempty(msg_received) || ~strcmp(msg_received,msgs_Mat.scr_conf_updated)
                fail_safe_mode = true;
                warning('Error detected in online processing');
            end
            load(exp_prop_file,'available_pics_cell');
            load(exp_prop_file,'experiment');
            if exist('experiment', 'var')
                NPICS = experiment.NPICS;
            end
            if exist('available_pics_cell','var')
                pics2use =  available_pics_cell{end};
            end
        end
    end %end subscr loop (FC)

    if params.is_online
        if abort
            M_PTB.send(msgs_Mat.exper_aborted);
        else
            M_PTB.send(msgs_Mat.exper_finished);
        end
    end
    diary off
catch ME
    msgText = getReport(ME)
    %flush buffer diary
    %fprintf("\n");
    %disp(msgText);
    drawnow;

    if ~fail_safe_mode && params.is_online
        M_PTB.send(msgs_Mat.error);
    end
    if ~reached_backup
        if params.acq_network
            recording.stop_and_close()
        end
        Screen('CloseAll');
        ShowCursor;
    end

    times(isnan(times))=[];
    inds_pics(inds_pics==0)=[];
    inds_start_seq(inds_start_seq==0)=[];
    times(isnan(times))=[];
    inds_pics(inds_pics==0)=[];
    inds_start_seq(inds_start_seq==0)=[];
    try
        multiKbQueueCheck(dev_used);
        for d = dev_used
            KbQueueStop(d);
            KbQueueRelease(d);
        end
    catch
        disp('no keyboard queue active')
    end
    %     Screen('CloseAll')
    Priority(params.ptb_priority_normal);
    experiment.ME=ME;

    scr_end = struct;
    scr_end.times=times;
    scr_end.t_stimon=t_stimon;
    scr_end.t_fliptime=t_fliptime;
    scr_end.t_DAQpic=t_DAQpic;
    scr_end.inds_pics=inds_pics;
    scr_end.inds_start_seq=inds_start_seq;
    scr_end.answer=answer;
    scr_end.abort=abort;
    scr_end.times_break=times_break;
    scr_end_cell{end+1} = scr_end;

    pics_used_ids = [];
    for ii=1:length(scr_config_cell)
        pics_used_ids = [pics_used_ids scr_config_cell{ii}.pics2use];
    end
    pics_used_ids = unique(pics_used_ids);

    %save(exp_prop_file,'pics_used_ids','scr_end_cell','-append');
    access_exp_prop_file(exp_prop_file,'pics_used_ids',pics_used_ids);
    access_exp_prop_file(exp_prop_file,'scr_end_cell',scr_end_cell);
    save('RSVP_SCR_workspace','-regexp', '^(?!(M_PTB)$).');
    save('experiment_properties_error','experiment');
    if params.use_daq
        dig_out.close()
    end
    if params.is_online
        M_PTB.close()
    end
    custompath.rm()
    msgbox('There was an error in the script. Review the data saved and back it up manually if necessary');
    diary off
    rethrow(ME)
end
task_duration = toc(init_time);
print_message('THAT WOULD BE ALL.\n THANK YOU !!!',black,window)
ttt=GetSecs;

% if params.acq_network
%     recording.stop_and_close()
% end

if params.acq_network 
    if contains(subtask, 'DynamicScr') || contains(subtask,'DynamicSeman') || contains(subtask,'CategLocaliz')
        fprintf("\n");
        fprintf("\n");
        fprintf('Press any key to stop recording.\n')

        % Define timeout in seconds
        timeoutSecs = 540;
        startTime = GetSecs;
        
        % Wait for either key press or timeout
        while (GetSecs - startTime) < timeoutSecs
            [keyIsDown, ~, ~] = KbCheck;
            if keyIsDown
                break; % Exit loop if any key is pressed
            end
            WaitSecs(0.01); % Small pause to prevent CPU overload
        end
        
        recording.stop_and_close();
    else
        recording.stop_and_close();
    end
end

Screen('CloseAll');
ShowCursor;

for d = dev_used
    KbQueueStop(d);
    KbQueueRelease(d);
end
Priority(params.ptb_priority_normal);
fprintf('.\nTask Duration: %.1f seconds.\n',task_duration);

save('RSVP_SCR_workspace','-regexp', '^(?!(M_PTB)$).');
WaitSecs(10-(GetSecs-ttt));

reached_backup = true;
if params.is_online
    M_PTB.send(msgs_Mat.rec_finished)
end
if params.use_daq
    dig_out.close()
end
if params.is_online
    M_PTB.close()
end
custompath.rm()

if ~strcmp(subtask,'FirstTime')
    rsvp_folder = fileparts(mfilename('fullpath'));
    status = []; msg_copy={};
    for i=1:length(data_transfer_copy)
        [status(end+1),msg_copy{end+1}] = copyfile(fullfile(rsvp_folder,data_transfer_copy{i}), fullfile(experiment.folder_name,data_transfer_copy{i})); %overwrites if already exists
    end
    pics_used_folder = fullfile(experiment.folder_name,'pics_used');
    if ~isfolder(pics_used_folder)
        mkdir(pics_used_folder)
    end
    pics2backup =  find(ImageNames.stim_trial_count>0);
    for i=1:numel(pics2backup)
        if params.remove_pictures
            [status(end+1),msg_copy{end+1}] = movefile(fullfile(ImageNames.folder{pics2backup(i)}, ImageNames.name{pics2backup(i)}),pics_used_folder);
        else
            [status(end+1),msg_copy{end+1}] = copyfile(fullfile(ImageNames.folder{pics2backup(i)}, ImageNames.name{pics2backup(i)}),pics_used_folder);
        end
    end
    if any(~status)
        for jj=find(status)
            disp(msg_copy{jj})
        end
        error('Check the data copied to the Blackrock PC as there was an error')
    end
end
end


function print_message(message,black,window)
Screen('FillRect',  window,black);
DrawFormattedText(window, message, 'center', 'center', [255 255 255]);
Screen('Flip',window);
end