function record(name,varargin)
% record('Recall')
% record('MusicAttention')
% record('MusicPassive')
% record('MusicFamiliar')
% record('MusicUnfamiliar')
% record('PTSD_interview')
% record('AudioBook')
% record('Conversation')
% record('Podcast')
% record('Breathwork_5_5')
% record('Breathwork_4_4_4_4')
% record('Bowman')

%this code should be run in the behaviour computer
close all
clc

ipr = inputParser;
addOptional(ipr,'sub_ID', []); %if empty it will check current_subject_id.txt to get subject ID
addParameter(ipr,'run_num', []); %integer (this particular subtask run number), if empty (default) the code will be look inside the acquisition disk to look for the previous files
addOptional(ipr,'location', 'MCW-FH-RIP'); % in this task, same if using 'MCW-BEH-RIP'
% addOptional(ipr,'mode', 'real'); % in this task, same if using 'MCW-BEH-RIP'
addParameter(ipr,'EMU_num',[]) %numeric value, if empty check files in acq folder and logs to automatically find the next one.
addParameter(ipr,'with_acq_folder',[]) % after finishing move the files inside folder, requires access to folder with raw files
parse(ipr,varargin{:})

% addpath(fileparts(fileparts(fileparts(mfilename('fullpath')))));
% custompath = reylab_custompath({'useful_functions/tasks_tools','tasks/.'});
% params = location_setup(ipr.Results.location);

% rec_length = 7200; % time in sec 

addpath(fullfile(fileparts(mfilename('fullpath')),'../..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','tasks/.'});

params = location_setup(ipr.Results.location);
inputs = fields(ipr.Results);

for i =1:numel(inputs)
    pstr = inputs{i};
    if any(strcmp(ipr.UsingDefaults,pstr)) && isempty(ipr.Results.(pstr)) && isfield(params, pstr) %parameter not passed and empty
        continue
    end
    params.(pstr) = ipr.Results.(pstr);
end

if isempty(params.sub_ID)
    params.sub_ID = strtrim(fileread(fullfile(params.beh_rec_metadata,'current_subject_id.txt')));
end

if strcmp(params.system, 'RIP')
    custompath.add(params.xippmex_path,true)
elseif strcmp(params.system, 'BRK')
    custompath.add(params.cbmex_path,true)
else
    custompath.rm()
    error('unknown system');
end

if params.with_acq_folder || isempty(params.EMU_num)  
    if ~params.acq_is_beh && ~test_remote_folder(params.acq_remote_folder_in_beh)
        custompath.rm()
        error('remote acq folder not detected and needed to find EMU_num or to move files inside folder at the end')
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
    if isempty(params.run_num)
        if isempty(allemus)
            params.run_num = 1;
        else
            nums = zeros(1,length(allemus));
            for i = 1:length(allemus)
                num = regexp(allemus{i},['EMU-\d+_subj-', params.sub_ID, '_task-',name,'_run-(\d+)'],'tokens','once');
                if ~isempty(num)
                    nums(i) = str2num(num{1});
                end
            end
            params.run_num = max(nums) + 1;        
        end
    end
end

if params.with_acq_folder
    folder_name = sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d',params.EMU_num,params.sub_ID,name,params.run_num);
    folder_name = fullfile(params.acq_remote_folder_in_beh,folder_name);

    prev_exps = dir([folder_name '*']);
    if ~isempty(prev_exps)
        error('Filename or folder already exists. Please check everything and run again')
    end
end

% run_num=1;
% run_num=params.run_num;
recording = recording_handler(params, '');

fig  = figure('Name', 'Recording','Toolbar','none','MenuBar','none','position',[300 400 340 50]);
set(fig,'CloseRequestFcn','')
uicontrol('Parent',fig, 'style','push', 'units','pix',...
                 'position',[10 5 130 30], 'fontsize',11,...
                 'string','Stop Recording', 'callback',{@(x,y) delete(gcf)});
uicontrol('Parent',fig, 'style','text', 'units','pix',...
                 'position',[150 5 150 30],...
                 'fontsize',9,'string','Pressing the button will stop the gaps task');

% n_attempts=0;

loop_rec(fig,recording,name,params,folder_name)
custompath.rm()
end

function loop_rec(fig,recording,name,params,folder_name)
 
%     MEnew = ME;
    try
        recording.stop_with_connection_check()
    catch
        disp('recording not stopped (or already stopped)')
    end
%     run_num = run_num + 1;
%     started = false;
    abort = false;
%     while ~abort
%         if ~isgraphics(fig)
%             abort = true;  
%             break;
%         end
        recording.rec_name = sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d_%s',params.EMU_num,params.sub_ID,name,params.run_num,params.system);
        pause(0.3)
        recording.start()
%         t0 = tic;
%         logfile(run_num).file = recording.rec_name;
%         logfile(run_num).start = datetime(now,'ConvertFrom','datenum'); 
%         logfile(run_num).ME = ME; 
%         fprintf('\nfile: %s, date: %s\n',logfile(run_num).file,logfile(run_num).start)
%         started = true;
%         prev_time = toc(t0);
%         while (toc(t0)) < rec_length
    if strcmp(name,'MusicAttention') || strcmp(name,'MusicPassive') || strcmp(name,'AudioBook') || strcmp(name,'Podcast') 
        input('\nPress ''Enter'' for sound check...\n');
        start_beeps = '/home/user/share/codes_emu/tasks/record/start-beeps.mp3';
        playAudio(start_beeps);
    end
        while ~abort
            pause(1)
            if ~isgraphics(fig)
                abort = true;  
                break;
            end
        end
        recording.stop_with_connection_check()
%         run_num = run_num + 1;
%         started = false;
%     end 
    recording.close()

if params.with_acq_folder
    files = sprintf('EMU-%.3d_subj-%s_task-%s_run-*',params.EMU_num,params.sub_ID,name);
    mkdir(folder_name)
    pause(5)
    movefile(fullfile(params.acq_remote_folder_in_beh,files),fullfile(folder_name));
   
end
end

function playAudio(path)
% Read the audio file
[audio, sampling_rate] = audioread(path);

% Play the audio
sound(audio, sampling_rate);
end
