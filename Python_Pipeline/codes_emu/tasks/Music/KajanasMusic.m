function KajanasMusic(subtask,test,varargin)
% Subtask: 'MusicAttention', 'MusicPassive'
% KajanasMusic('MusicAttention')
% KajanasMusic('MusicPassive')

% TEST 
% KajanasMusic('MusicAttention',1)
% KajanasMusic('MusicPassive',1)


%this code should be run in the behaviour computer
close all
clc

if ~exist('test')
    test = false;
end

ipr = inputParser;
addOptional(ipr,'location', 'MCW-BEH-RIP');
parse(ipr,varargin{:})

addpath(fullfile(fileparts(mfilename('fullpath')),'../..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','tasks/.'});
params = location_setup(ipr.Results.location);

params.sub_ID= [];
params.EMU_num= [];
params.run_num = [];
params.acq_network = [];

%%

if strcmp(params.system, 'RIP')
    custompath.add(params.xippmex_path,true)
elseif strcmp(params.system, 'BRK')
    custompath.add(params.cbmex_path,true)
else
    custompath.rm()
    error('unknown system');
end

if isempty(params.sub_ID)
    params.sub_ID = strtrim(fileread(fullfile(params.beh_rec_metadata,'current_subject_id.txt')));
end
if isunix, system('nmcli radio wifi off'); end %disable wifi in ubuntu


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
                num = regexp(allemus{i},['EMU-\d+_subj-', params.sub_ID, '_task-',subtask,'_run-(\d+)'],'tokens','once');
                if ~isempty(num)
                    nums(i) = str2num(num{1});
                end
            end
            params.run_num = max(nums) + 1;
        end
    end
end

if params.with_acq_folder
    folder_name = sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d',params.EMU_num,params.sub_ID,subtask,params.run_num);
    folder_name = fullfile(params.acq_remote_folder_in_beh,folder_name);

    prev_exps = dir([folder_name '*']);
    if ~isempty(prev_exps)
        error('Filename or folder already exists. Please check everything and run again')
    end
end
%%

addpath(fileparts(fileparts(fileparts(mfilename('fullpath')))));
custompath = reylab_custompath({'useful_functions','tasks/.'});
if test == true 
    params.acq_network = 0;
else 
    params.acq_network=1;
end
params.system = 'RIP';
params.with_acq_folder = 1;
params.use_daq = 1;
params.ttl_device = 'LJ';
params.acq_remote_folder_in_processing = '/media/acq';
params.acq_remote_folder_in_beh = '/media/acq';
params.beh_rec_metadata = '/home/user/share/experimental_files/rec_metadata'; %mapfile, preprocessing, templates, etc
params.keyboards = {'Microsoft Microsoft® 2.4GHz Transceiver v8.0'};
if isempty(params.sub_ID)
    params.sub_ID = strtrim(fileread(fullfile(params.beh_rec_metadata,'current_subject_id.txt')));
end
if isunix, system('nmcli radio wifi off'); end %disable wifi in ubuntu
if isempty(params.EMU_num) || isempty(params.run_num)
    allemus = dir(fullfile(params.acq_remote_folder_in_beh,['EMU-*' params.sub_ID '*']));
    allemus = {allemus.name};
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
                    nums(i) = str2double(num{1});
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
                num = regexp(allemus{i},['EMU-\d+_subj-', params.sub_ID, '_task-',subtask,'_run-(\d+)'],'tokens','once');
                if ~isempty(num)
                    nums(i) = str2double(num{1});
                end
            end
            params.run_num = max(nums) + 1;
        end
    end
end

abort = false;

if params.acq_network
    experiment.fname    =    sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d',params.EMU_num,params.sub_ID,subtask,params.run_num);
    experiment.folder_name = sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d',params.EMU_num,params.sub_ID,subtask,params.run_num);
    recording = recording_handler(params, experiment.fname);
    params.recording_name = recording.rec_name;
else
    experiment.folder_name = [subtask '_test'];
end

fig = msgbox('Preparing the songs. This takes approximately 2 min...');

[s,Fs]=audioread('/home/user/share/codes_emu/tasks/Music/Sugar_Cane.mp3');
[b,Fb]=audioread('/home/user/share/codes_emu/tasks/Music/Beethoven_12.mp3');
[f,Ff]=audioread('/home/user/share/codes_emu/tasks/Music/Beethoven_FurElise.mp3');
[m,Fm]=audioread('/home/user/share/codes_emu/tasks/Music/Motzart.mp3');

close(fig)

%START RECORDING
if params.acq_network
    recording.start();
end

fprintf('Recording has started \n\n');

WaitSecs(2);
%%
start_beeps = '/home/user/share/codes_emu/tasks/movement/start-beeps.mp3';

input('\nPress ''Enter'' for sound check...\n');
playAudio(start_beeps);

WaitSecs(5);

if strcmp(subtask,'MusicAttention')
    songs = {'Beethoven_12','Beethoven_FurElise','Sugar_Cane'};
    songs = reshape(repmat(songs,1,1),1,length(songs));
    songs = randsample(songs,length(songs));

    input('-Press ''Enter'' to start...\n')

    for i = 1:length(songs)
        if contains(songs{i},'12')
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(b,Fb);
            playblocking(PO)
            fprintf('\nQuestions for patient: \n-Do you recognize the piece? \n-How much did you like it? (1 to 10) \n-What mood was elicited? \n-Did it evoke any memories? \n\n')
            input('Press ''Enter'' to continue...\n')
        elseif contains(songs{i},'FurElise')
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(f,Ff);
            playblocking(PO)
            fprintf('\nQuestions for patient: \n-Do you recognize the piece? \n-How much did you like it? (1 to 10) \n-What mood was elicited? \n-Did it evoke any memories? \n\n')
            input('Press ''Enter'' to continue...\n')
        else 
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(s,Fs);
            playblocking(PO)
            fprintf('\nQuestions for patient: \n-Do you recognize the piece? \n-How much did you like it? (1 to 10) \n-What mood was elicited? \n-Did it evoke any memories? \n\n')
            input('Press ''Enter'' to continue...\n')
        end
        disp('-----------------------');
    end

else 
    songs = {'Beethoven_12','Beethoven_FurElise','Sugar_Cane','Motzart'};
    songs = reshape(repmat(songs,1,1),1,length(songs));
    songs = randsample(songs,length(songs));

    input('Press ''Enter'' to start...\n')

    for i = 1:length(songs)
        if contains(songs{i},'12')
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(b,Fb);
            playblocking(PO)
        elseif contains(songs{i},'FurElise')
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(f,Ff);
            playblocking(PO)
        elseif contains(songs{i},'Sugar')
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(s,Fs);
            playblocking(PO)
        else
            disp(['>> Song: ' songs{i}]);
            PO=audioplayer(m,Fm);
            playblocking(PO)
        end
        disp('-----------------------');
    end

end

% STOP RECORDING
if params.acq_network
    recording.stop_and_close()
end

fprintf('Recording has finished \n');

if params.with_acq_folder
    files = sprintf('EMU-%.3d_subj-%s_task-%s_run-*',params.EMU_num,params.sub_ID,subtask);
    mkdir(folder_name)
    pause(3)
    movefile(fullfile(params.acq_remote_folder_in_beh,files),fullfile(folder_name))
%     [status,cmdout] = system(sprintf('cp %s %s',fullfile(pwd,[experiment.fname '.mat']),fullfile(folder_name,[experiment.fname '.mat'])));
%     [status,msg] = movefile(pwd,('Task information'),fullfile(folder_name));
%    if ~status
%         delete([experiment.fname '.mat'])
%     end
end

clear all
end

function playAudio(path)
    % Read the audio file
    [audio, sampling_rate] = audioread(path);

    % Play the audio
    sound(audio, sampling_rate);
end