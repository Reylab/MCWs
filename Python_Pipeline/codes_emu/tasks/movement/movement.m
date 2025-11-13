function movement(subtask,varargin)
% movement('MovementTest')
clc
if ~exist('subtask')
    subtask = 'Movement';
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
params.use_daq = 1;
params.ttl_device = 'LJ';
params.acq_remote_folder_in_processing = '/media/acq';
params.acq_remote_folder_in_beh = '/media/acq';
params.beh_rec_metadata = '/home/user/share/experimental_files/rec_metadata'; %mapfile, preprocessing, templates, etc
params.keyboards = {'Microsoft Microsoft® 2.4GHz Transceiver v8.0'};
if strcmp(subtask,'Movement')
    params.acq_network=1;
    params.with_acq_folder = 1;
    params.nrep = 2; % number of repetition per side
    params.nrep_sen = 2; % number of repetition per side (sensory test)
    params.in_ex = 5; % number of inhalation and exhalation
elseif strcmp(subtask,'MovementTest')
    params.acq_network=0;
    params.with_acq_folder=0;
    params.nrep = 1; % number of repetition per side
    params.nrep_sen = 1; % number of repetition per side (sensory test)
    params.in_ex = 1; % number of inhalation and exhalation
else
    error('Subtask no found. Available subtasks: DefinitionNaming, DefinitionNamingTest')
end

daq_signatures.data_signature_on = 60;
daq_signatures.data_signature_off = 100;
daq_signatures.inhale_on = 33;
daq_signatures.exhale_off = 55;
daq_signatures.mov_on1 = 77;
daq_signatures.mov_off1 = 99;
daq_signatures.mov_on2 = 66;
daq_signatures.mov_off2 = 88;
daq_signatures.sensory_touch = 22;
daq_signatures.audio_right = 200;
daq_signatures.audio_left = 205;
daq_signatures.audio_both = 210;
daq_signatures.audio_straight = 215;
daq_signatures.audio_up = 220;
daq_signatures.audio_down = 225;
daq_signatures.audio_go = 230;
daq_signatures.reset = 0;
%%

if strcmp(params.system, 'RIP')
    custompath.add(params.xippmex_path,true)
elseif strcmp(params.system, 'BRK')
    custompath.add(params.cbmex_path,true)
else
    custompath.rm()
    error('unknown system');
end

if params.acq_network
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
end

abort = false;

if params.acq_network
    experiment.fname    =    sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d',params.EMU_num,params.sub_ID,subtask,params.run_num);
    experiment.folder_name = sprintf('EMU-%.3d_subj-%s_task-%s_run-%.2d',params.EMU_num,params.sub_ID,subtask,params.run_num);
    recording = recording_handler(params, experiment.fname);
    params.recording_name = recording.rec_name;
    run_folder = fullfile(params.backup_path,experiment.fname);
    map_path = dir(fullfile('/home/user/share/experimental_files/rec_metadata/','*.map'));
end

KbName('UnifyKeyNames')
exitKey = 'F2';

if params.use_daq;   dig_out  = TTL_device(params.ttl_device); end

times=NaN(1,0);
k=1;

%%
% Path to MP3 file
load('movements_info.mat');
r = '/home/user/share/codes_emu/tasks/movement/Right.mp3';
l = '/home/user/share/codes_emu/tasks/movement/Left.mp3';
b = '/home/user/share/codes_emu/tasks/movement/Both.mp3';
u = '/home/user/share/codes_emu/tasks/movement/Up.mp3';
d = '/home/user/share/codes_emu/tasks/movement/Down.mp3';
g = '/home/user/share/codes_emu/tasks/movement/Go.mp3';
s = '/home/user/share/codes_emu/tasks/movement/Straight.mp3';
start_beeps = '/home/user/share/codes_emu/tasks/movement/start-beeps.mp3';

% movements = {'Lift knee to chest (R/L) *', 'Lift and lower leg while straight(R/L) *', 'Point and flex feet (R/L/B) *', ...
%     'Raise arm straight up (R/L/B) *', 'Do bicep curls (R/L/B) *', ...
%     'Grab something from tray table and put it back (R/L) *', ...
%     'Squeeze foam ball or make a fist (R/L) *','* Inhalation and exhalation *','* Sensory test *'};


% movements = {'Lift and lower leg while straight(R/L) *', 'Do bicep curls (R/L/B) *',...
%     'Move head side to side (R/L) *','Move head up and down (U/D) *','Close eyes (R/L/B) *','Big smile (show teeth) (R/L/B) *',...
%     'Pout *','Tongue out (R/L/B) *','* Inhalation and exhalation *'};
global mov;
global m;
% mov=[];

all_movements = {'Lift knee to chest (R/L)', 'Lift and lower leg while straight (R/L)', 'Point and flex feet (R/L/B)',...
    'Raise arm straight up (R/L/B)', 'Do bicep curls (R/L/B)','Grab something from tray table and put it back (R/L)', ...
    'Squeeze foam ball or make a fist (R/L)','Move head side to side (R/L)','Move head up and down (U/D)',...
    'Close eyes (R/L/B)','Big smile (show teeth) (R/L/B)','Eyebrows up (R/L/B)','Pout (G)','Tongue out (R/L/S)'};

fig  = uifigure('Name', 'Movements','position',[400 400 350 400]);
set(fig,'CloseRequestFcn','')
c = uilistbox('Parent',fig,'Multiselect','on','Position',[20 80 310 260],'FontSize',12,'Items',all_movements,'ValueChangedFcn',@selection);
c1 = uicontrol(fig, 'Style','text','Position',[20 360 320 30],'String','Select all the movements to include in the task. Hold down ctrl key + left click to select multiple options.');
f = uicontrol(fig,'Style','checkbox','Position',[25 42 200 25],'String','Inhalation and exhalation','HitTest','on','callback',@add);
f1 = uicontrol(fig,'Style','checkbox','Position',[25 15 200 25],'String','Sensory touch','HitTest','on','callback',@add);
f2 = uicontrol(fig, 'Style','push', 'units','pix','position',[220 30 100 25], 'fontsize',11,'string','continue...','callback',{@(x,y) delete(fig)});

waitfor(fig)
mov = [mov m];
clear m;

diary('Task_movement_diary.txt');

%START RECORDING
if params.acq_network
    recording.start();
    fprintf('\n * Recording has started * \n');
else
    fprintf('\n * Test has started * \n');
end

WaitSecs(2);

if params.use_daq
    dig_out.send(daq_signatures.data_signature_on);
    WaitSecs(0.05);
    dig_out.send(daq_signatures.data_signature_off);
    WaitSecs(0.45);
    dig_out.send(daq_signatures.data_signature_on);
    WaitSecs(0.05);
    dig_out.send(daq_signatures.data_signature_off);
    WaitSecs(0.45);
    dig_out.send(daq_signatures.data_signature_on);
    WaitSecs(0.05);
    dig_out.send(daq_signatures.data_signature_off);
    times(k)=GetSecs; k=k+1;
end

WaitSecs(2);

fprintf('Selection: \n')
disp(mov);

mov_info = [];
mov_info_table = [];
% daq_vector = [];

input('\nPress ''Enter'' for sound check...\n');
playAudio(start_beeps);

WaitSecs(5);

for i = 1:length(mov)
    if contains(mov{i},'Inhalation')
        fprintf('-Inhalation and exhalation:\n');
        clear sides
        for l=1:params.in_ex
            pause; % Wait for a key press
            fprintf('%d: Inhale...',l);
            if params.use_daq
                dig_out.send(daq_signatures.inhale_on);
                mov_info = horzcat(mov_info, daq_signatures.inhale_on);
                mov_info = [mov{i} {'Inhale'} mov_info];
                mov_info_table = [mov_info_table;mov_info];
                mov_info = [];
            end

            pause; % Wait for a key press
            fprintf('exhale. \n');
            if params.use_daq
                dig_out.send(daq_signatures.exhale_off);
                mov_info = horzcat(mov_info, daq_signatures.exhale_off);
                mov_info = [mov{i} {'Exhale'} mov_info];
                mov_info_table = [mov_info_table;mov_info];
                mov_info = [];
            end

            %             pause; % Wait for a key press
            %             fprintf('off\n');
            %             if params.use_daq
            %                 dig_out.send(4);
            %             end
        end
        if i<length(mov)
            disp('--------');
            input('Press ''Enter'' to continue to the next movement...\n');
        end

    else
        if contains(mov{i},'Sensory')
            fprintf('-Sensory touch:\n');
            sides = {'Eyebrown - Left','Eyebrown - Right','Eyebrown - Middle',...
                'Cheek - Left','Cheek - Right','Lip - Left','Lip - Right','Lip - Middle',...
                'LowerLip - Left','LowerLip - Right','Chin - Middle','Nose - Left',...
                'Nose - Right','Nose - Middle'};
            params.nrep = params.nrep_sen;
        else
            disp(['>> Movement: ' mov{i}]);
            % Ask to press 'spacebar' to show sides
            disp('Press ''spacebar'' to continue...');
            pause; % Wait for a key press
            % Choose sides for the current movement
            if endsWith(mov{i},'(R/L)')
                sides = {'Right', 'Left'};
            elseif endsWith(mov{i},'(R/L/B)')
                sides = {'Right','Left','Both'};
            elseif endsWith(mov{i},'(R/L/S)')
                sides = {'Right','Left','Straight'};
            elseif endsWith(mov{i},'(U/D)')
                sides = {'Up','Down'};
            elseif endsWith(mov{i},'(G)')
                sides = {'Go'};
            elseif endsWith(mov{i},'(L)')
                sides = {'Left'};
            elseif endsWith(mov{i},'(R)')
                sides = {'Right'};
            elseif endsWith(mov{i},'(B)')
                sides = {'Both'};
            end
        end
        sides = reshape(repmat(sides,params.nrep,1),1,params.nrep*length(sides));
        sides = randsample(sides,length(sides));
    end

    if ~contains(mov{i},'Sensory') && ~contains(mov{i},'Inhalation')
        % Show sides randomly
        for j = 1:length(sides)
            disp([sides{j}]);
            if  strcmp(sides{j}, 'Right')
                dig_out.send(daq_signatures.audio_right);
                playAudio(r);
                mov_info = horzcat(mov_info, daq_signatures.audio_right);
            elseif strcmp(sides{j}, 'Left')
                dig_out.send(daq_signatures.audio_left);
                playAudio(l);
                mov_info = horzcat(mov_info, daq_signatures.audio_left);
            elseif strcmp(sides{j}, 'Both')
                dig_out.send(daq_signatures.audio_both);
                playAudio(b);
                mov_info = horzcat(mov_info, daq_signatures.audio_both);
            elseif strcmp(sides{j}, 'Straight')
                dig_out.send(daq_signatures.audio_straight);
                playAudio(s);
                mov_info = horzcat(mov_info, daq_signatures.audio_straight);
            elseif strcmp(sides{j}, 'Up')
                dig_out.send(daq_signatures.audio_up);
                playAudio(u);
                mov_info = horzcat(mov_info, daq_signatures.audio_up);
            elseif strcmp(sides{j}, 'Down')
                dig_out.send(daq_signatures.audio_down);
                playAudio(d);
                mov_info = horzcat(mov_info, daq_signatures.audio_down);
            else
                dig_out.send(daq_signatures.audio_go);
                playAudio(g);
                mov_info = horzcat(mov_info, daq_signatures.audio_go);
            end

            for k = 1:length(movements_info)
                if contains(mov{i},movements_info(k).Movements)
%                     fprintf('Times:');
%                     disp(movements_info(k).Times);
                    if movements_info(k).Times == 4
                        pause; % Wait for a key press
                        fprintf('On(1)...');
                        if params.use_daq
                            dig_out.send(daq_signatures.mov_on1);
                            mov_info = horzcat(mov_info, daq_signatures.mov_on1);
                        end
                        pause;
                        fprintf('off(1)...');
                        if params.use_daq
                            dig_out.send(daq_signatures.mov_off1);
                            mov_info = horzcat(mov_info, daq_signatures.mov_off1);
                        end
                        pause; % Wait for a key press
                        fprintf('on(2)...');
                        if params.use_daq
                            dig_out.send(daq_signatures.mov_on2);
                            mov_info = horzcat(mov_info, daq_signatures.mov_on2);
                        end
                        pause; % Wait for a key press
                        fprintf('off(2). \n');
                        if params.use_daq
                            dig_out.send(daq_signatures.mov_off2);
                            mov_info = horzcat(mov_info, daq_signatures.mov_off2);
                        end

                        mov_info = [mov{i} {sides{j}} mov_info];
                mov_info_table = [mov_info_table;mov_info];
                mov_info = [];
                    else
                        pause; % Wait for a key press
                        fprintf('On...');
                        if params.use_daq
                            dig_out.send(daq_signatures.mov_on1);
                            mov_info = horzcat(mov_info, daq_signatures.mov_on1);
                        end
                        pause;
                        fprintf('off. \n');
                        if params.use_daq
                            dig_out.send(daq_signatures.mov_off1);
                            mov_info = horzcat(mov_info, daq_signatures.mov_off1);
                        end
                        mov_info = [mov{i} {sides{j}} mov_info];
                mov_info_table = [mov_info_table;mov_info];
                mov_info = [];
                    end

                    % Ask to press 'spacebar' for the next side
            if j < length(sides)
                disp('Press ''spacebar'' to continue...');
                pause; % Wait for a key press
            elseif i < length(mov)
                % Wait for Enter key press, except for the last movement
                disp('--------');
                input('Press ''Enter'' to continue to the next movement...\n');
            end
                end
            end
            
        end

        

        %                     % Wait for Enter key press, except for the last movement
        %                     if i < length(mov)
        %                         pause;
        %                         fprintf('2...');
        %                         if params.use_daq
        %                             dig_out.send(77);
        %                         end
        %                         pause; % Wait for a key press
        %                         fprintf('off \n');
        %                         if params.use_daq
        %                             dig_out.send(daq_signatures.mov_on2);
        %                         end
        %
        %                         disp('--------');
        %                         input('Press ''Enter'' to continue to the next movement...\n');
        %                     end

    elseif ~contains(mov{i},'Inhalation')
        for j = 1:length(sides)
            part1 = sprintf('%d/%d: ',j,length(sides));
            part2 = sides{j};
            msg = sprintf('%s %s\n', part1, part2);
            fig = figure('Position',[600 400 600 280]);
            fig = uicontrol('Style','text','string',msg,'FontSize',35,'Position',[50 0 500 200]);
            fprintf('%d/%d: ',j,length(sides));
            disp([sides{j}]);
            if j < length(sides)
                pause; % Wait for a key press
                close(gcf)
                fprintf('T \n');
                if params.use_daq
                    dig_out.send(daq_signatures.sensory_touch);
                    mov_info = horzcat(mov_info, daq_signatures.sensory_touch);
                    WaitSecs(0.05);
                    dig_out.send(daq_signatures.reset);
                    mov_info = horzcat(mov_info, daq_signatures.reset);
                end
                disp('Press ''spacebar'' to continue...');
                pause; % Wait for a key press
            else
                pause; % Wait for a key press
                close(gcf)
                fprintf('T \n');
                if params.use_daq
                    dig_out.send(daq_signatures.sensory_touch);
                    mov_info = horzcat(mov_info, daq_signatures.sensory_touch);
                    WaitSecs(0.05);
                    dig_out.send(daq_signatures.reset);
                    mov_info = horzcat(mov_info, daq_signatures.reset);
                end
            end
            split_side = split(sides{j});
            sensory_t_name = sprintf('Sensory touch: %s',string(split_side(1)));
             mov_info = [sensory_t_name {split_side(3)} mov_info];
                mov_info_table = [mov_info_table;mov_info];
                mov_info = [];
        end
    end
end

diary off;

% STOP RECORDING
pause;
if params.acq_network
    input('\nPress Enter to finish the recording...');
    recording.stop_and_close()
    fprintf('Recording has finished \n');
else
    input('\nPress Enter to finish the test...');
    fprintf('Test has finished \n');
end

if params.use_daq
    dig_out.close()
end

if params.with_acq_folder
    save experiment_properties.mat mov_info_table daq_signatures params movements_info
    files = sprintf('EMU-%.3d_subj-%s_task-%s_run-*',params.EMU_num,params.sub_ID,subtask);
    mkdir(folder_name)
    mkdir(run_folder)
    pause(3)
    move_file('Task_movement_diary.txt', run_folder);
    movefile(fullfile(params.acq_remote_folder_in_beh,files),fullfile(folder_name))
    [status,cmdout] = system(sprintf('cp %s %s',fullfile(pwd,[experiment.fname '.mat']),fullfile(folder_name,[experiment.fname '.mat'])));
    %     [status,msg] = movefile(pwd,('Task information'),fullfile(folder_name));
%     [~,msg]  = unix(sprintf('mv %s %s','Task_information.txt',fullfile(folder_name)));
    [status,cmdout] = move_file('experiment_properties.mat',run_folder);
    [status,cmdout] = copy_file('movement.m',run_folder);
    cd(map_path.folder)
    [status,cmdout] = copy_file(map_path.name,run_folder);
%         if ~status
%         delete([experiment.fname '.mat'])
%     end
else 
    delete Task_movement_diary.txt
end

clear all
end

function selection(c,event)
global mov;
mov = c.Value;
end

function add(f,event)
global m;
if f.Value == 1
    m = [m {f.String}];
else
    for l=1:length(m)
        if contains(f.String, m{l})
            m{l} = [];
        end
    end
    m = m(~cellfun(@isempty, m));
end
end

function playAudio(path)
% Read the audio file
[audio, sampling_rate] = audioread(path);

% Play the audio
sound(audio, sampling_rate);
end
