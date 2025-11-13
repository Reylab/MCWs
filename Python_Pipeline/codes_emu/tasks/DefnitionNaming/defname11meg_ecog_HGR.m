function defname11meg_ecog_HGR(subtask,varargin)
% defname11meg_ecog_HGR('DefinitionNamingTest')
% defname11meg_ecog_HGR('DefinitionNaming')

%Program presents stimuli for auditory definition naming & control task
%Sound files provided by Dr. Jeffrey Binder, Medical College of Wisconsin
%Written by Manoj Raghavan
clc

if ~exist('subtask')
    subtask = 'DefinitionNaming';
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

%HISTORY

%10/13/07 MR    Wrote the program on Mac OS X matlab. Runs prespecified blocks of "test" and
%               "control" conditions for definition naming
%11/17/07 MR    All audio output now uses low-latency "psychportaudio"
%11/18/07 MR    Stim onset marker & stim combined into seperate channels of
%               one stereo sound file so they can be extracted seperately
%               from the stereo sound output. The stimulus sound goes to external
%               speaker for mono output.The stimulus onset marker goes to EEG synchronous with stimulus onset.
%07/06/09 MR    Now running on Windows XP with ASIO4all driver and
%               PsychPortAudio plugin for windows
%07/07/09 MR    Added a parallel port low-latency stim marker output using
%               the io32 low level driver
%11/05/09 MR    Cleaned up routines for integrating with stim marking on
%               MEG system. Control task switched to spectrally matched
%               noise of similar duration to speech stimuli
%03/14/12 MR    Program now presents stimuli with or without re-using any stim from
%               designated folder based on parameter 'noreplacement'. Manual triggering of trial reintroduced with a
%               delay of trialdelay+rand() after the experimenter triggers
%               each trial. Modified code to guarantee equal number of
%               test and control trials. Removed old redundant code meant to provide stim
%               marker for EEG through one of the audio channels
%05/07/15 MR    Fixed all trial-type tracking errors when using multiple blocks
%12/01/16 MR    Program now uses modulated noise as control stimuli
%06/12/18 MR    Works with Win 10 64 bit using io64 driver & expresscard parallel port
%06/12/18 MR    Presenting control stim made optional

%ENVIRONMENT NOTES: 
%Use io64 instead of i032 on 64-bit Windows 7 system
%Replace wavread with audioread for Matlab R2014b onwards.

%REQUIREMENTS
%Requires PTB3 installed
%Low-latency(&jitter) sound output requires psychportaudio plugin and ASIO4ALL
%Low-latency parallel port output requires io32/io64 driver (on Windows)

%%
%DEFINE PARAMETERS FOR EXPERIMENT
% controlstim=1;%if 1 present noise control stimuli
controlstim=0;%if 1 present noise control stimuli
% nblocks = 1;%number of blocks of "test" and "control" condtions, 7 for exp
% Note: 110 pairs if control stimuli being presented, ~19 mins
%If using no controls: can use 1 block of 80 trials
% ntrials = 60 ;%number of trials in each block. 10 for experiment
% ntrials = 2 ;
trialdelay = .5;% delay to trial after experimenter hits key = [trialdelay+rand()]
noreplacement=1;%if 1, do not reuse a stim thats already been presented, 0 for pre-march-2012 random presentation
data_signature_on = 64;
data_signature_off = 128;

%DEFINE PARAMETERS OF SOUND STIMULUS FILES
samplerate=44100;%sample rate of sound files
%%

if strcmp(subtask,'DefinitionNaming')
    params.acq_network=1;
    params.with_acq_folder = 1;
    nblocks = 1;%number of blocks of "test" and "control" condtions, 7 for exp
    ntrials = 60;%number of trials in each block. 10 for experiment
elseif strcmp(subtask,'DefinitionNamingTest')
    params.acq_network=0;
    params.with_acq_folder=0;
    controlstim=0;%if 1 present noise control stimuli
    nblocks = 1;%number of blocks of "test" and "control" condtions, 7 for exp
    ntrials = 5;%number of trials in each block. 10 for experiment
else
    error('Subtask no found. Available subtasks: DefinitionNaming, DefinitionNamingTest')
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

% exitKey = KbName('F2');
exitKey = 'F2';
% startKey = KbName('s'); %88 in Windows, 27 in MAC
% continueKey= KbName('c');  %to continue 
% keysOfInterest=zeros(1,256);
% firstPress=zeros(1,256);
% keysOfInterest([exitKey continueKey startKey])=1;
% 
% % message_begin = {'Ready to begin?';'Listo para empezar?';'Etes-vous pret pour commencer?'};
% message_begin = {'Ready to begin?'};

Screen('Preference', 'SkipSyncTests', double(IsWin));

%RANDOMISE THE RAND FUNCTION STATE FOR EACH RUN
% rand('state',sum(100*clock));
rng('shuffle', 'twister')

%Suppress PTB warnings
%oldLevel = Screen('Preference', 'Verbosity',0);% 0 to suppress all warnings
oldEnableFlag = Screen('Preference', 'SuppressAllWarnings', 1);

%INITIALIZE PSYCHPORTAUDIO FOR LOWEST LATENCY
InitializePsychSound(1);
% Open the default audio device [], with default mode [] (==Only playback),
% and a required latencyclass of 1 == low-latency mode, as well as
% a frequency of 'samplerate' and nrchannels=2 (stereo) sound channels.
% This returns a handle to the audio device:

audiodevices = PsychPortAudio('GetDevices',8); % 8=Linux/ALSA.
outdevice = strcmp('front',{audiodevices.DeviceName});
pahandle = PsychPortAudio('Open',audiodevices(outdevice).DeviceIndex,[],3,samplerate,2);
% pahandle = PsychPortAudio('Open', [], [], 1, samplerate, 2);

% %INITIALIZE THE LOW_LATENCY PARALLEL PORT DRIVER
% ioObj=io64;%create a parallel port handle
% status=io64(ioObj);%if this returns '0' the port driver is loaded & ready 
% %TO get the right address below look in device manager -> parallel
% %port->resources-start address
% address=hex2dec('EFF8');%'378' is usually default address of LPT1 in hex
if params.use_daq;   dig_out  = TTL_device(params.ttl_device); end


% %SET PARALLEL PORT OUTPUTS TO 0
% io64(ioObj,address,0);%initialize the parallel port outputs to 0 volts

%locate the stimulus sound files
cd commonnames;% change directory to where the test sound files are
testfilelist=dir('*.wav');%get the ONLY the list of files of type .wav
% soundfilepath=genpath(cd);
soundfilepath=pwd;
% soundfilepath(length(soundfilepath))='\';%fix the semicolon at the end of the path string. Use '\' in Windows
cd ..;
nstimfiles=length(testfilelist);
%generate a permutated list based on number of stim files
testsequence=randperm(nstimfiles);

cd commonnames_modul_noise
prtestfilelist=dir('*.wav');%get ONLY the list of files of type .wav 
% prsoundfilepath=genpath(cd);
prsoundfilepath=pwd;
% prsoundfilepath(length(prsoundfilepath))='\';%fix the semicolon at the end of the path string. Use '\' in Windows
cd ..;
prnstimfiles=length(prtestfilelist);
%generate a permutated list based on number of stim files
controlsequence=randperm(prnstimfiles);
%save(['run_outputs\',date,'.mat'],'controlsequence')

% run a check to see if the number of trials requested exceeds stimuli
% available if "noreplacement"==1
if (noreplacement==1 && nstimfiles<ntrials)
    %close soundport and clear mex before providing an error massage & quitting
    PsychPortAudio('Close', pahandle);
%     clear mex;
    error('Too many trials to present stimuli without replacement..fix this.');
end

% if (0)
%     %close soundport and clear mex before providing an error massage & quitting
%     PsychPortAudio('Close', pahandle);
%     clear mex;
%     error('Breaking for debugging.');
% end;

%initialize trial types
ntest = 1;
npr = 1;
%if no control sounds required prevent by setting npr to ntrials
if(controlstim==0)
    npr=ntrials+1;
end
trial = 1;

if params.acq_network
    save([experiment.fname '.mat'])
end
times=NaN(1,0);
k=1;

%START RECORDING
if params.acq_network
    recording.start();
    fprintf('\n * Recording has started *\n');
else
    fprintf('\n * Test has started *');
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

% for d=dev_used
%     KbQueueCreate(d,keysOfInterest);
%     KbQueueStart(d);
% end
% pressed=0;

input('\nPress ''Enter'' for sound check...\n');
trialsound=audioread('/home/user/share/codes_emu/tasks/DefnitionNaming/start-beeps.wav');
wavedata = repmat(trialsound,1,2);
soundstim=wavedata';
PsychPortAudio('FillBuffer', pahandle, soundstim);
PsychPortAudio('Start', pahandle, 1, 0, 2);
PsychPortAudio('Stop', pahandle,1);

% Some instructions for screen output    
fprintf('\nStarting run of %d blocks, with %d test and control trials each per block. \n\n', nblocks,ntrials);
fprintf('Hit any key to start each trial when trial # is prompted...\n');
fprintf ('Hit F2 at any trial prompt to quit the program gracefully.\n');

% while ~(pressed && any(firstPress([startKey exitKey])))
%     [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);
% end
aud_used = cell(nblocks,ntrials);        

for block=1:nblocks
    fprintf ('\nBlock %d:\n',block);
    %initialize block trial types #
    blocktrial =1;
    blocktest=1;
    blockpr=1;
    %if no control sounds required prevent by setting blockpr to ntrials
    if(controlstim==0)
        blockpr=ntrials+1;
    end

    while blocktest <= ntrials || blockpr <= ntrials
        if(rand(1)>0.5) 
            cond=1;
        else
            cond=0;
        end
        myKey=[];
        % if you already have enough trials in one stim category force it
        % to run the other stim category
        if(blocktest>ntrials) 
            cond=0;
        end
        
        if(blockpr>ntrials) 
            cond=1;
        end   
        
        if (blocktest <= ntrials && cond==1)
            trialtype='test';
            if noreplacement==0
                %Load a random stimulus sound file
                randint=myrandint(1,1,1:nstimfiles);
            elseif noreplacement==1
                %Load a file based on testsequence generated earlier
                randint=testsequence(ntest);% if you use blocktest instead of ntest sequence is repeated each block
            end
            filename=testfilelist(randint).name;% extract the randomly selected filename from file list structure
%             pathandname=strcat(soundfilepath,filename);% create the path string to this file
            pathandname=fullfile(soundfilepath,filename);% create string for path to this file
%             trialsound=wavread(pathandname);% read the .wave file selected
            trialsound=audioread(pathandname);% read the .wave file selected
            ntest = ntest + 1;
            blocktest=blocktest+1;
            trigger_out=1;% '255' will set all pins to 1 (+5v) on the parallel port--chose others as needed
     
        elseif (blockpr <= ntrials && cond==0)
            trialtype='noise';
            if noreplacement==0
                %Load a random control sound file
                randint=myrandint(1,1,1:prnstimfiles);
            elseif noreplacement==1
                % Load a file based on permuted testsequence generated earlier
                randint=controlsequence(npr);% if you use blockpr instead of npr sequence is repeated each block
            end
            filename=prtestfilelist(randint).name;% extract the randomly selected filename from file list structure
%             pathandname=strcat(prsoundfilepath,filename);% create the path string to this file
            pathandname=fullfile(prsoundfilepath,filename);% create the path string to this file
%             trialsound=wavread(pathandname);% read the .wave file selected
            trialsound=audioread(pathandname);% read the .wave file selected
            npr = npr + 1;
            blockpr=blockpr+1;
            trigger_out=2;% '255' will set all pins to 1 (+5v) on the parallel port--choose others as needed     
        end
        aud_used{block,blocktrial} = pathandname;

        % Get the length of the sound in samples for later use
        stimlength=length(trialsound);
        % Make it stereo
        wavedata = repmat(trialsound,1,2);
        soundstim=wavedata';%need sound data as a row vector for psychportaudio call
        
        %Fill the audio playback buffer with the audio data
        PsychPortAudio('FillBuffer', pahandle, soundstim);
        soundlength=stimlength/samplerate;
        fprintf('Hit any key to present global trial #%d\n',trial);
        myKey=KbName;%Dont need KbWait if using KbName
%         if (strcmp(myKey,'esc')) 
        if (strcmp(myKey,exitKey)) 
            break, 
        end %Exit this block 
        vardelay=trialdelay+rand();
        pause(vardelay);
        Priority(1);%raise priority for stimulus presentation
%         %send correct trigger marker to LPT1
%         io64(ioObj,address,trigger_out); %pause(0.002); io32(ioObj,address,0);
        PsychPortAudio('Start', pahandle, 1, 0, 2);
        if params.use_daq
            dig_out.send(trigger_out); 
        end
%         pause(soundlength);
%         %turn trigger off at end of sound
%         io64(ioObj,address,0);
%         PsychPortAudio('Stop', pahandle);
        [times(k),endDura] = PsychPortAudio('Stop', pahandle,1);  
        if params.use_daq
            dig_out.send(0); 
        end
        times(k+1) = times(k)+endDura;
        k=k+2;
        if cond==1
            trialtypenumber=ntest-1;%gets incremented in loop above
        elseif cond==0
            trialtypenumber=npr-1;%gets incremented in loop above
        end
        % Provide some useful console output
        fprintf('Blocktrial=%d, %sTrial#=%d, stim-length=%fs, stim-delay=%fs\n',blocktrial,trialtype,trialtypenumber,soundlength,vardelay);
        Priority(0);%drop priority back to normal
        blocktrial = blocktrial+1;
        trial = trial + 1;
    end   
    if params.acq_network
        save([experiment.fname '.mat'])
    end
%             if (strcmp(myKey,'esc')), break, end %Exit this block 
            if (strcmp(myKey,exitKey)), break, end %Exit this block 
end
fprintf('\n');
if params.acq_network
    fprintf('Hit any key to finish the recording.')
else 
    fprintf('Hit any key to finish the test.')
end
fprintf('\n');
myKey=KbName;

%STOP RECORDING
if params.acq_network
    recording.stop_and_close()
    fprintf('Recording has finished \n');
else 
    fprintf('Test has finished \n');
end

Screen('CloseAll');
% Restore the mouse cursor.
ShowCursor;
% Close the audio device:
PsychPortAudio('Close', pahandle);
if params.use_daq
    dig_out.close()
end

Screen('Preference','SuppressAllWarnings',oldEnableFlag);

if params.with_acq_folder
    mat_file = sprintf('%s.mat', experiment.fname);
    files = sprintf('EMU-%.3d_subj-%s_task-%s_run-*',params.EMU_num,params.sub_ID,subtask);
    mkdir(folder_name)
    mkdir(run_folder)
    pause(3)
    movefile(fullfile(params.acq_remote_folder_in_beh,files),fullfile(folder_name))
    copy_file('defname11meg_ecog_HGR.m',run_folder);
    move_file(mat_file,run_folder);
    [status,cmdout] = system(sprintf('cp %s %s',fullfile(pwd,[experiment.fname '.mat']),fullfile(folder_name,[experiment.fname '.mat'])));
    if ~status
        delete([experiment.fname '.mat'])
    end
end
custompath.rm()
% clear mex;
end