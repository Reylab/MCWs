function visnaming01_HGR(subtask,varargin)
% visnaming01_HGR('PictureNaming')
% visnaming01_HGR('PictureNamingTest')

% addpath(fileparts(fileparts(fileparts(mfilename('fullpath')))));
% custompath = reylab_custompath({'useful_functions/tasks_tools','tasks/.'});
% params = location_setup('MCW-FH-RIP');
clc

if ~exist('subtask')
    subtask = 'PictureNaming';
end

ipr = inputParser;
addOptional(ipr,'location', 'MCW-FH-RIP'); % in this task, same if using 'MCW-BEH-RIP'
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define parameters of the experiment
% nblocks=1;              %total naming blocks;
fixMarkSize=8;          % size of square fixation mark in pixel
% nTrials=130;            % number of trials in each block 
% nTrials=60;            % number of trials in each block 
% nTrials=2;
% stimDuration=0.2;       % duration of stimulus in seconds
stimDuration=1;       % duration of  stimulus in seconds
controlFraction=0.4;    % Fraction of trials where a control stimulus appears
beepOnStim=0;           % Set to 1 if you want a beep during stim
beepfreq=500;           % freq of beep in Hz
minterval=1;            % stimulus interval is minterval+2*rand()to minimize cue anticipation
trig_1=1;               %'1' will set parallel port pin 1 to +5v
trig_2=2;               %'2' will set parallel port pin 2 to +5v
data_signature_on = 64;
data_signature_off = 128;
% background=0;           % 0=black, 1=gray, 2=white background
background=0;           % 0=black, 2=white background
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Program runs trials of language and control task
% Language task trials: visual naming of line drawings
% Control task trials: scrambled lines--response is "junk" or "noise"
% Stimulus onset markers are sent to the parallel port on each trial

% Requires 
% PTB3 installed. 
% io32 driver (use io64 driver for 64 bit Win OSes)
% function myrandint.m by David Fass (or use comm-toolbox randint.m)

%History
%04/06/2016  mr     wrote it, tested on MATLAB R2007b, 32-bit Win Xp
%06/15/2018  mr     tested to work on Matlab R2010b, 64-bit Win10 w/ io64

%%
if strcmp(subtask,'PictureNaming')
    params.acq_network=1;
    params.with_acq_folder = 1;
    nblocks=1; %number of blocks of "test" and "control" condtions, 7 for exp
    nTrials=60; %number of trials in each block. 10 for experiment
elseif strcmp(subtask,'PictureNamingTest')
    params.acq_network=0;
    params.with_acq_folder=0;
    nblocks=1; %number of blocks of "test" and "control" condtions, 7 for exp
    nTrials=5; %number of trials in each block. 10 for experiment
else
    error('Subtask no found. Available subtasks: PictureNaming, PictureNamingTest')
end 

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
     run_folder = fullfile(params.root_beh,experiment.fname);
end

% [kbs,products_names] = GetKeyboardIndices;
% dev_used = [];
% for i =1:numel(params.keyboards)
%     if isnumeric(params.keyboards{i})
%         dev_used(end+1) = params.keyboards{i};
%     else
%         kbix = strcmp(params.keyboards{i}, products_names);
%         if ~any(kbix)
%             warning('Keyboard %s, not found', params.keyboards{i});
%         else
% %             dev_used(end+1) = kbs(kbix);
%             dev_used = [dev_used kbs(kbix)];
%         end
%     end
% end
% if isempty(dev_used)
%    error('Keyboards not found') 
% end

KbName('UnifyKeyNames');
exitKey = 'F2';

% exitKey = KbName('F2');
% startKey = KbName('s'); %88 in Windows, 27 in MAC
% continueKey= KbName('c');  %to continue 
% keysOfInterest=zeros(1,256);
% firstPress=zeros(1,256);
% keysOfInterest([exitKey continueKey startKey])=1;

% message_begin = {'Ready to begin?';'Listo para empezar?';'Etes-vous pret pour commencer?'};
% message_begin = 'Ready to begin?';

Screen('Preference', 'SkipSyncTests', double(IsWin));

% echo off;
    
if params.use_daq;   dig_out  = TTL_device(params.ttl_device); end

% % %Initialize low latency parallel port driver
% % ioObj=io64;%create a parallel port handle
% % status=io64(ioObj);%if this returns '0' the port driver is loaded & ready 
% % lptaddress=hex2dec('EFF8');%'378' is the default address of LPT1 in hex

% %Generate auditory waveforms for stimulus beeps
% beep=MakeBeep(beepfreq,stimDuration);
% %get paths to stimulus image files in 3 folders
fs = 44100;
t = 0:1/fs:stimDuration;
beep_signal = sin(2*pi*beepfreq*t);

%test images
cd test_images;% cd to where the test image files are
testfilelist=dir();%get the list of image files
% testimgfilepath=genpath(cd);
testimgfilepath=pwd;
% testimgfilepath(length(testimgfilepath))='\';%fix the semicolon at the end of the path string
cd ..;

%control images
cd control_images;% cd to where the control image files are
controlfilelist=dir();%get the list of image files
% controlimgfilepath=genpath(cd);
controlimgfilepath=pwd;
% controlimgfilepath(length(controlimgfilepath))='\';%fix the semicolon at the end of the path string
cd ..;

% Setup some PTB3 environment variables
oldVisualDebugLevel = Screen('Preference', 'VisualDebugLevel', 3);
oldSupressAllWarnings = Screen('Preference', 'SuppressAllWarnings', 1);

% Find out how many screens and use largest screen number.
whichScreen = max(Screen('Screens'));
% Hides the mouse cursor
HideCursor;%This is important for Screen timing functions to work
[window, screenRect]= Screen('OpenWindow', whichScreen,0);

%determine screen center coordinates
xcenter = floor((screenRect(3)-screenRect(1))/2);
ycenter = floor((screenRect(4)-screenRect(2))/2);
flickerSquare = flickerSquareLoc(screenRect,24,2,'BottomLeft');

Priority(1); 
ifi = Screen('GetFlipInterval', window, 200);
%     slack=ifi/2;
slack=ifi/4;
Priority(0); %normal priority

% Set up colors
white = WhiteIndex(window); % pixel value for white
black = BlackIndex(window)+1; % pixel value for black
gray = (white+black)/2;
if round(gray)==white
	gray=black;
end
    
%Set background color
if(background==0)
    bgcolor=black;
    flickercolor = white;
elseif(background==1)
    bgcolor=gray;
elseif (background==2)
    bgcolor=white;
    flickercolor = black;
end
Screen(window, 'FillRect', bgcolor);

%Pick fixation color
if(bgcolor==black)
    fixcolor=white;
elseif(bgcolor==gray)
    fixcolor=black;
elseif (bgcolor==white)
    fixcolor=gray;
end

%Create a blank texture
blanktex=Screen(window, 'MakeTexture', bgcolor);
% Gray out the screen
Screen('DrawTexture', window, blanktex);
Screen(window, 'Flip');

if params.acq_network
    save([experiment.fname '.mat'])
end

times=NaN(1,0);
k=1;

%START RECORDING

if params.acq_network
    recording.start();
    fprintf('\n * Recording has started * \n\n');
else
    fprintf('\n * Test has started * \n\n');
end

WaitSecs(2);

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
    times(k)=GetSecs; k=k+1;
end

% % for d=dev_used
% %     KbQueueCreate(d,keysOfInterest);
% %     KbQueueStart(d);
% % end
% % pressed=0;

% %     print_message(message_begin,bgcolor,fixcolor,window);
% % %     while ~(pressed && any(firstPress([startKey exitKey])))
% %     while ~any(firstPress([startKey exitKey]))
% %         [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);
% %     end
% %     for d= dev_used; KbQueueFlush(d);   end

img_used = cell(nblocks,nTrials);
%loop over blocks of language & control task
for block=1:nblocks
    %fprintf('\nBlock # = %d.  \n',block);
    
%     %Provide some display-screen output
%     Screen('TextFont',window,'Courier New');
%     fontsize=24;
%     Screen('TextSize',window,fontsize);
%     %Screen('DrawText', window, 'Hit enter for each trial', 50,50);
%     messg=sprintf('Hit "Enter" to start trial block # %d',block);
%     DrawFormattedText(window, messg, 50, 50, fixcolor);
%     Screen(window, 'Flip');
    Screen('TextSize',window, 32);
    print_message('Ready to begin?',black,window)
    fprintf('Hit "Enter" to start trial block # %d \n',block)
            KbWait; % this is waiting for any key and not just enter

%      myKey=KbName;%Dont need KbWait if using KbName
% %         if (strcmp(myKey,'esc')) 
%         if (strcmp(myKey,exitKey)) 
%             break, 
%         end %Exit this block 
    
    %loop over trials to present cues
    trial=1;
    controltrials=0;%counter for control trials
    controlTrial=0;%flag for control trial
    fprintf('Total trials = %d\n',nTrials)
    while trial<=nTrials && ~abort    
        myKey=[];
        if mod(trial,5) == 0
            fprintf('trial %d\n',trial)
        end
        %Pick trial type
        if (rand<controlFraction && trial>1 && controltrials<=floor(controlFraction*nTrials) && controlTrial~=1)
            controlTrial=1;
            blockStim=controlfilelist;
            imgfilepath=controlimgfilepath;
            trig=trig_2;
        else
            controlTrial=0;
            blockStim=testfilelist;
            imgfilepath=testimgfilepath;
            trig=trig_1;
        end
       
        %fprintf(' Trial#=%d, controlTrial=%d \n',trial, controlTrial);
        %pause(0.2);
        % Draw a fixation point
        fms=fixMarkSize;
        fixationRect=CenterRect(SetRect(0,0,fms,fms),screenRect);
        Screen('FrameRect',window,fixcolor,fixationRect);
        Screen(window, 'Flip');

        % Wait for a variable interval between stimuli
        cueDelay=minterval+rand();
        pause(cueDelay);
        
        %Blank the fixation mark ahead of the stimulus
        if (1) 
            Screen('DrawTexture', window, blanktex);
            Screen(window, 'Flip');
        end
        pause(0.2); %shouldn't this be slightly random?
        
        % Pick and create offscreen image for this trial
        listlen=length(blockStim);%number of files in folder is 3 less that returned by the dir()
        randItem=myrandint(1,1,[3:listlen]);%start from 3
        filename=blockStim(randItem).name;% extract the randomly selected filename from file list structure
%         pathandname=strcat(imgfilepath,filename);% create string for path to this file
        pathandname=fullfile(imgfilepath,filename);% create string for path to this file
        img_used{block,trial} = pathandname;
        fileimage=imread(pathandname);% read the random image file selected
        trialimage=imcomplement(fileimage);
        w=Screen(window,'MakeTexture',trialimage);%create offscreen image
        
        %Display trial stimulus
        Screen('DrawTexture', window, w);% rapidly load image to the display screen
        Screen('FillRect',window,flickercolor,flickerSquare);

%         if (beepOnStim) Snd('Play',beep);end;
        if (beepOnStim)
            sound(beep_signal, fs); 
        end
        

        Priority(1);%raise to 2 for "real-time" priority
        times(k) = Screen('Flip',window);
        
        %ADD PULSE DAQ
        if params.use_daq
            dig_out.send(trig); 
        end
%         %Send a pulse lasting duration of stimulus to LPT1
%         io64(ioObj,lptaddress,trig); pause(stimDuration); io64(ioObj,lptaddress,0);
        
        %Blank out the images from screen
        Screen('DrawTexture', window, blanktex);
        Screen('FillRect',window,bgcolor,flickerSquare);
        Screen(window, 'Flip',times(k)+stimDuration-slack);
        Priority(0);
        if params.use_daq
            dig_out.send(0); 
        end
        k=k+1;
        %increment counters
        trial=trial+1;
        if(controlTrial==1)
           controltrials=controltrials+1;
        end  
%             KbWait;  
            myKey=KbName;%Dont need KbWait if using KbName
%         if (strcmp(myKey,'esc')) 
        if (strcmp(myKey,exitKey)) 
            break, 
        end %Exit this block 
%         if (strcmp(myKey,exitKey)), break, end

% %             while ~(pressed && any(firstPress([continueKey exitKey])))
%             while ~any(firstPress([continueKey exitKey]))
%                 [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);
%             end
%             for d= dev_used; KbQueueFlush(d);   end
% %         if pressed && firstPress(exitKey)>0, abort = true; break; end
%         if firstPress(exitKey)>0, abort = true; break; end
    end
    if params.acq_network
        save([experiment.fname '.mat'])
    end

% %     if pressed && firstPress(exitKey)>0, abort = true; break; end
%     if firstPress(exitKey)>0, abort = true; brseak; end
%         if (strcmp(myKey,'esc')) 
        if (strcmp(myKey,exitKey)) 
            break, 
        end %Exit this block pause(3);%pause between blocks
            pause(2);%pause between blocks
end
% clear mex;

Screen('TextSize',window, 32);
print_message('THAT WOULD BE ALL.\n THANK YOU !!!',black,window)

%STOP RECORDING
if params.acq_network
    recording.stop_and_close()
    fprintf('Recording has finished \n');
else
    fprintf('Test has finished \n');
end

pause(3);
Screen('CloseAll');
% Restore the mouse cursor.
ShowCursor;
% try
%     multiKbQueueCheck(dev_used);
%     for d = dev_used
%         KbQueueStop(d);
%         KbQueueRelease(d);
%     end
% catch
%     disp('no keyboard queue active')
% end
if params.use_daq
    dig_out.close()
end

% Restore preferences
Screen('Preference', 'VisualDebugLevel', oldVisualDebugLevel);
Screen('Preference', 'SuppressAllWarnings', oldSupressAllWarnings);

if params.with_acq_folder
    mat_file = sprintf('%s.mat', experiment.fname);
    files = sprintf('EMU-%.3d_subj-%s_task-%s_run-*',params.EMU_num,params.sub_ID,subtask);
    mkdir(folder_name)
    mkdir(run_folder)
    pause(3)
    movefile(fullfile(params.acq_remote_folder_in_beh,files),fullfile(folder_name))
    copy_file('visnaming01_HGR.m',run_folder);
    move_file(mat_file,run_folder);
%     [status,msg] = copyfile(fullfile(pwd,[experiment.fname '.mat']), fullfile(folder_name,[experiment.fname '.mat']));
    [status,cmdout] = system(sprintf('cp %s %s',fullfile(pwd,[experiment.fname '.mat']),fullfile(folder_name,[experiment.fname '.mat'])));
    if ~status
        delete([experiment.fname '.mat'])
    end
end

custompath.rm()

% function print_message(message,bgcolor,fixcolor,window)
% Screen('FillRect',  window,bgcolor);
% if fixcolor==255
%     col = [255 255 255];
% elseif fixcolor==1    
%     col = [0 0 0];
% end
% Screen('TextSize',window, 32);
% DrawFormattedText(window, message, 'center', 'center', col);
% Screen('Flip',window);
% end


end

function print_message(message,black,window)
Screen('FillRect',window,black);
DrawFormattedText(window, message, 'center', 'center', [255 255 255]);
Screen('Flip',window);
end