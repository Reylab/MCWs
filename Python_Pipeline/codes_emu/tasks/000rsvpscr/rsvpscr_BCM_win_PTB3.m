function rsvpscr_BCM_win_PTB3(sub_ID,EMU_num,run_num,which_system,is_online,overw_Nrep,which_nsp_comment,lang,device_resp,Path_pics)
% cbmex comments with max 127 characters
% rsvpscr_BCM_win_PTB3('eee',1,1,'RIP',1,[],1,[],'keyboard')
auto_resp = 0
% auto_resp = 1
% if ~exist('which_system','var') || isempty(which_system), which_system='RIP'; end  % which_system = 'BRK'; which_system = 'RIP';
if ~exist('which_system','var') || isempty(which_system), which_system='BRK'; end  % which_system = 'BRK'; which_system = 'RIP';
if ~exist('lang','var') || isempty(lang), lang='english'; end
if ~exist('Path_pics','var') || isempty(Path_pics),    Path_pics=[pwd '_10pic']; end
% if ~exist('Path_pics','var') || isempty(Path_pics),    Path_pics=[pwd '_context']; end
% if ~exist('Path_pics','var') || isempty(Path_pics),    Path_pics=[pwd '_pos_pic']; end
% if ~exist('Path_pics','var') || isempty(Path_pics),    Path_pics=[pwd '_pic']; end
if ~exist('is_online','var') || isempty(is_online),    is_online=false;  end
if ~exist('device_resp','var') || isempty(device_resp),    device_resp='gamepad';  end
if ~exist('which_nsp_comment','var') || isempty(which_nsp_comment),    which_nsp_comment=2;  end
% if ~exist('which_nsp_comment','var') || isempty(which_nsp_comment),    which_nsp_comment=1;  end
if strcmp(sub_ID,'test')
    withevents=1; recording_on=0; Nrep = 2; device_resp='keyboard'; with_local_net = true;
PsychDebugWindowConfiguration; Path_pics=[pwd '_picfirst'];
elseif strcmp(sub_ID,'first_time')
    withevents=0; recording_on=0; Nrep = 1; with_local_net = false; %PsychDebugWindowConfiguration;
    Path_pics=[pwd '_picfirst'];
    device_resp='gamepad';
%     device_resp='keyboard';
else
    withevents=3; recording_on=1; Nrep = 15; with_local_net = true; %withevents=1 only comments; withevents=2 only DAQ;  withevents=3 DAQ and comments
%     withevents=3; recording_on=1; Nrep = 15; with_local_net = false; %withevents=1 only comments; withevents=2 only DAQ;  withevents=3 DAQ and comments
end
if exist('overw_Nrep','var') && ~isempty(overw_Nrep),    Nrep=overw_Nrep;  end
with_analysis = true;
if is_online, with_analysis = false; end
with_reset = false;
abort = false;
reached_backup = false;
% if ~exist('order_pics_RSVP_SCR.mat','file') &&~exist('variables_RSVP_SCR.mat','file')
    shuffle_rsvpSCR(Nrep,Path_pics)
% else
%     N = load('order_pics_RSVP_SCR.mat','Nrep');
%     if N.Nrep ~= Nrep
%         shuffle_rsvpSCR(Nrep)
%     end
% end
addpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX'))
addpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\BCM-EMU\useful_functions\tasks_tools'))

if recording_on    % won't work if auto increment is enabled!!!!
    experiment.fname = sprintf('EMU-%.3d_subj-%s_task-RSVPscr_run-%.2d',EMU_num,sub_ID,run_num);
    experiment.folder_name = sprintf('EMU-%.3d_task-RSVPscr_run-%.2d',EMU_num,run_num);
    if with_local_net
        if exist(sprintf('Z:\\%s.ns5',experiment.fname),'file')
            error('Filename %s already exists. Please check everything and run the task again',experiment.fname)
        end
%         if exist(sprintf('Z:\\%s',experiment.folder_name),'dir')
%             error('Folder %s already exists. Please check everything and run the task again',experiment.folder_name)
%         else
            mkdir(sprintf('Z:\\%s',experiment.folder_name))
%         end
    end
end

if is_online
%     data_transfer_copy = {'experiment_properties.mat';'RSVP_SCR_workspace.mat';'rsvpscr_BCM_win_PTB3.m';'shuffle_rsvpSCR.m';'create_lines_change_RSVP_SCR.m';'flickerSquareLoc.m';'get_response.m';'matconnect.m';'online';'results'};
    data_transfer_copy = {'experiment_properties.mat';'RSVP_SCR_workspace.mat';'rsvpscr_BCM_win_PTB3.m';'shuffle_rsvpSCR.m';'create_lines_change_RSVP_SCR.m';'flickerSquareLoc.m';'get_response.m';'online';'results'};
else
    data_transfer_copy = {'experiment_properties.mat';'RSVP_SCR_workspace.mat';'rsvpscr_BCM_win_PTB3.m';'shuffle_rsvpSCR.m';'create_lines_change_RSVP_SCR.m';'flickerSquareLoc.m';'get_response.m'};
end
% data_transfer_move = {'experiment_properties.mat';'RSVP_SCR_workspace.mat'};

load(fullfile(pwd,'order_pics_RSVP_SCR.mat'))
load(fullfile(pwd,'variables_RSVP_SCR.mat'))
% lines_offset = 11;
lines_offset = 50;
NISI=numel(ISI);
times=NaN*ones(1,Nseq*(NISI+1+2+NISI*seq_length+6+1));
t_stimon = NaN*ones(1,Nseq*seq_length);
t_fliptime = NaN*ones(1,Nseq*seq_length);
t_DAQpic = NaN*ones(1,Nseq*seq_length);
time_wait = NaN*ones(1,Nseq);
inds_pics = zeros(1,seq_length*NISI*Nseq);
inds_start_seq = zeros(1,Nseq);
times_break = [];
cant_breaks=0;
answer = NaN;
% deviceIndex=-1;
% deviceIndex=0;
kID = GetKeyboardIndices;
if kID==0
    %     dev_used=deviceIndex(end);
    dev_used=0;
end
KbName('UnifyKeyNames');
Screen('Preference','VisualDebugLevel',3);
AssertOpenGL;    % Running on PTB-3? Abort otherwise.

exitKey = KbName('F2');
startKey = KbName('s'); %88 in Windows, 27 in MAC
%     spaceKey= KbName('space');  %for color changes (not really saved)
breakKey= KbName('p');  %to pause
continueKey= KbName('n');  %to continue if gamepad fails

if is_online
    answer = questdlg('Start the online function on a separate Matlab and wait for instructions to continue','Online started?','Continue','Continue');
    %     disp('Start the online function on a separate Matlab. Press OK key to continue')
    %     [~, ~, keyCode] = KbCheck(dev_used);
    %     while ~keyCode(startKey), [~, ~, keyCode] = KbCheck(dev_used); end
    M_PTB = matconnect(1);
    M_PTB.send(num2str(which_nsp_comment));
end

Screen('Preference', 'SkipSyncTests', 1);

if strcmp(lang,'english')
    ind_lang=1;
elseif strcmp(lang,'spanish')
    ind_lang=2;
elseif strcmp(lang,'french')
    ind_lang=3;
end

% cl=clock;
% prf=sprintf('-%s-%d-%d-%d-%s',date,cl(4),cl(5),round(cl(6)));

try
    screens=Screen('Screens');
    whichScreen=max(screens);
%     	whichScreen=1;
    
    wait_reset = 0.1;  % IT MUST BE SHORTER THAN THE SHORTEST ISI
    value_reset = 0;
    
    %     min_cross=0.2;
    min_blank=1.25;
    max_rand_blank = 0.5;
    min_lines_onoff=0.5;
    max_rand_lines_onoff = 0.2;
    %     max_rand_cross = 0.1;
    %     tend_trial = 0.1;
    %     dura_base=1.0;
    size_line = 5;
    
    if strcmp(device_resp,'gamepad')
        %         % STEP 1.5
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
        gp_state_1 = false; %this will be left key
        gp_state_2 = false; %this will be button B (red)
        
        clear JoyMEX;
        JoyMEX('init',0);
    end
    
    a=dir(sprintf('%s/*',Path_pics));
    b = zeros(length(a),1);
    b = b>0;
    for i = 1:length(a)
        b(i) = contains (lower(a(i).name), '.jp');
    end
    a = a(b);
    if isempty(a)
        error(['No pictures for this session in ' Path_pics]);
    end
    Npic=length(a);
    %     if mod(Npic,2)==1
    %         error(['There is an odd number of pictures in ' Path_pics]);
    %     end
    ImageNames = cell(Npic,1);
    for i=1:Npic,ImageNames{i}=a(i).name;end
    
    message_begin = {'Ready to begin?';'Listo para empezar?';'Etes-vous pret pour commencer?'};
    %     text_endtrial = {'Press the right key to begin the next trial';'Presione la flecha derecha para continuar'};
    %     text_probe = {'Have you seen this picture in the last sequence?';'Viste esta imagen en la ultima secuencia?';'Avez vous vu cette image lors de la derniere sequence?'};
    message_continue = {'Ready to continue?';'Listo para continuar?';'Etes-vous pret pour continuer?'};
    message_wait = {'Take a short break.\nWe will resume shortly';'Take a short break.\nWe will resume shortly';'Take a short break.\nWe will resume shortly'};
    
    msgs_Mat.exper_saved = 'experiment_properties has been saved';
    msgs_Mat.rec_started = 'BRK recording has started';
%     msgs_Mat.sign_request = 'ready to receive signature';
%     msgs_Mat.sign_detected = 'signature successfully detected';
    msgs_Mat.ready_begin = 'ready to begin';
    msgs_Mat.trial_begin = 'trial starting';
    msgs_Mat.trial_end = 'trial ending';
    %     msgs_Mat.start_processing = 'trial processing has started';
    msgs_Mat.process_ready = 'ready to begin the next trial';
    msgs_Mat.process_end = 'online processing completed';
    msgs_Mat.exper_aborted = 'experiment was aborted';
    msgs_Mat.error = 'there was an error. stop waiting';
    
    % session number
%     cl=clock;
%     prf=sprintf('-%s-%d-%d-%d',date,cl(4),cl(5),round(cl(6)));
    
    if ismember(withevents,[2 3])
        [DAQ_param_1,DAQ_param_2] = initialize_TTL(which_system);
%         ljasm = NET.addAssembly('LJUDDotNet'); %NECESSARY??
%         ljudObj = LabJack.LabJackUD.LJUD; 
%         
%         [ljerror, ljhandle] = ljudObj.OpenLabJackS('LJ_dtU3', 'LJ_ctUSB', '0', true, 0);% Open the first found LabJack U3.
%         ljudObj.ePutS(ljhandle, 'LJ_ioPIN_CONFIGURATION_RESET', 0, 0, 0);
%         ljudObj.ePutS(ljhandle, 'LJ_ioPUT_ANALOG_ENABLE_PORT', 0, 0, 8);  %enables the fist 8 bits as digital outputs
    end
    
    % WARNING !!!!this version will not work to sync EEG KCL way as there are
    % several places with issues to maintain the odd-even sequence
    
    %     pic_onoff = [[1 5 17];[3 9 33]];  % first pic with row 2
    pic_onoff = [[1 4 16];[2 8 32]];  % first pic with row 2
    %     bits_for_break = [65 128];
    bits_for_break = [];
    %     lines_onoff = 77;
    lines_onoff = 13;
    %     blank_on = 69;
    blank_on = 11;
    lines_flip_blank = 103;
    %     lines_flip_pic = 133;
    lines_flip_pic = 22;
    %     trial_on = 113;
    trial_on = 26;
    %     data_signature_on = 85;
    %     data_signature_off = 84;
    data_signature_on = 64;
    data_signature_off = 128;
    
    msgs = {'blank on';'lines on';'pic change';...
        'lines change blank';'lines change pic';'lines off';'trial ended'};
    msgs_colors = {255;65280;16711680;16777215;65535;16711935;16776960};
    
    %     rand('twister',sum(100*clock));
    rng('shuffle', 'twister')
    
    randTime_blank = min_blank + max_rand_blank*rand(NISI+1,Nseq);
    randTime_lines_on = min_lines_onoff + max_rand_lines_onoff*rand(1,Nseq);
    randTime_lines_off = randTime_blank(NISI+1,:) - (min_lines_onoff + max_rand_lines_onoff*rand(1,Nseq));
    
    experiment.pwd=pwd;
    experiment.date=date;
    experiment.pic=pic_onoff;
    experiment.blank_on=blank_on;
    experiment.lines_onoff=lines_onoff;
    experiment.lines_flip_blank=lines_flip_blank;
    experiment.lines_flip_pic=lines_flip_pic;
    experiment.trial_on=trial_on;
    experiment.bits_for_break=bits_for_break;
    experiment.data_signature=[data_signature_on data_signature_off];
    experiment.value_reset=value_reset;
    experiment.wait_reset=wait_reset;
    experiment.order_pic=order_pic;
    experiment.order_ISI=order_ISI;
    experiment.ISI=ISI;
    experiment.ImageNames=ImageNames;
    experiment.Nrep=Nrep;
    experiment.seq_length=seq_length;
    experiment.Nseq=Nseq;
    experiment.lines_change=lines_change;
    experiment.msgs=msgs;
    experiment.deviceresp=device_resp;
    experiment.is_online=is_online;
    experiment.msgs_Mat=msgs_Mat;
    experiment.nchanges_blank=nchanges_blank;
    experiment.nchanges_pic=nchanges_pic;
    experiment.with_reset=with_reset;
    experiment.Path_pics=Path_pics;
    
    save('experiment_properties','experiment');
    
    if is_online
        M_PTB.send(msgs_Mat.exper_saved);
    end
    
    % Open screen.  Do this before opening the
    % offscreen windows so you can align offscreen
    % window memory to onscreen for faster copying.
    
    [window,windowRect]=Screen(whichScreen,'OpenWindow',0);
    %     [window,windowRect]=Screen('OpenWindow',whichScreen,0,[0 0 1024 768]);
    
    %window and monitor properties
    xcenter=windowRect(3)/2;
    ycenter=windowRect(4)/2;
    %     Priority(9); %Enable realtime-scheduling in MAC
    Priority(1); %high priority in Windows
    ifi = Screen('GetFlipInterval', window, 200);
%     slack=ifi/2;
    slack=ifi/4;
    Priority(0); %normal priority
    frame_rate=1/ifi;
    white=WhiteIndex(window);
    black=BlackIndex(window);
    %     Screen('TextSize',window, 72);
    Screen('TextSize',window, 32);
    flickerSquare = flickerSquareLoc(windowRect,24,2,'BottomLeft');
    
    experiment.xcenter=xcenter;
    experiment.ycenter=ycenter;
    experiment.frame_duration=ifi;
    experiment.frame_rate=frame_rate;
    experiment.flickerSquare=flickerSquare;
    
    tex=zeros(1,Npic);
    imageRect = cell(Npic,1);
    destRect = cell(Npic,1);
    for i=1:Npic
        Im=imread(sprintf('%s/%s',Path_pics,ImageNames{i}));
        nRows=size(Im,1); nCols=size(Im,2);
        imageRect{i}=SetRect(0,0,nCols,nRows);
        destRect{i}=CenterRect(imageRect{i},windowRect);
        tex(i)=Screen('MakeTexture',window,Im);
    end
    
    keysOfInterest=zeros(1,256);
    firstPress=zeros(1,256);
    %     keyCode=zeros(1,256);
%     scanlist=zeros(1,256);
    keysOfInterest([exitKey breakKey startKey continueKey])=1;
%     scanlist([exitKey breakKey startKey continueKey])=1;
    save('RSVP_SCR_workspace');
    
%     if is_online
        answer = questdlg('Tell the subject to get ready to begin. Press OK to continue','Subject ready?','OK','OK');
%     end
    k=1;
    
    if recording_on
        [onlineNSP,experiment.blackrockData,experiment.videoData] = StartBlackrockAquisition_noAutoInc(experiment.fname,0);
%                 [onlineNSP,experiment.blackrockData,experiment.videoData] = StartBlackrockAquisition_old(experiment.fname,0);
        times(k)=GetSecs; k=k+1;
    elseif withevents==1
        cbmex('open')
        onlineNSP=which_nsp_comment;
    end
    
    WaitSecs(4);
    
    if is_online
        M_PTB.send(msgs_Mat.rec_started);
%         msg_received = M_PTB.waitmessage();
%         if strcmp(msg_received,msgs_Mat.error)
%             is_online = false; experiment.is_online = 'error in online Matlab';
%         elseif ~strcmp(msg_received,msgs_Mat.sign_request)
%             warning('Inconsistency with messages sent')
%         end
    end
   
    
    if ismember(withevents,[2 3])
        send_TTL(which_system,DAQ_param_1,DAQ_param_2, data_signature_on); 
%         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, data_signature_on, 8, 0);
%         ljudObj.GoOne(ljhandle);
%         %         err_off=DaqDOut(dio,0,data_signature_on);                           %just starts
        WaitSecs(0.05);
        send_TTL(which_system,DAQ_param_1,DAQ_param_2, data_signature_off); 
%         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, data_signature_off, 8, 0);
%         ljudObj.GoOne(ljhandle);
%         %         err_on=DaqDOut(dio,0,data_signature_off);            %pulse is on
        WaitSecs(0.45);
        send_TTL(which_system,DAQ_param_1,DAQ_param_2, data_signature_on); 
%         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, data_signature_on, 8, 0);
%         ljudObj.GoOne(ljhandle);
%         %         err_off=DaqDOut(dio,0,data_signature_on);
        WaitSecs(0.05);
        send_TTL(which_system,DAQ_param_1,DAQ_param_2, data_signature_off); 
%         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, data_signature_off, 8, 0);
%         ljudObj.GoOne(ljhandle);
%         %         err_on=DaqDOut(dio,0,data_signature_off);
        WaitSecs(0.45);
        send_TTL(which_system,DAQ_param_1,DAQ_param_2, data_signature_on); 
%         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, data_signature_on, 8, 0);
%         ljudObj.GoOne(ljhandle);
%         %         err_off=DaqDOut(dio,0,data_signature_on);
        WaitSecs(0.05);
        send_TTL(which_system,DAQ_param_1,DAQ_param_2, data_signature_off); 
%         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, data_signature_off, 8, 0);
%         ljudObj.GoOne(ljhandle);
        times(k)=GetSecs; k=k+1;
        %         err_on=DaqDOut(dio,0,data_signature_off);
    end
    
%     if is_online
%         msg_received = M_PTB.waitmessage(3);
%         if isempty(msg_received)
%             abort=true;
%         elseif strcmp(msg_received,msgs_Mat.error)
%             is_online = false; experiment.is_online = 'error in online Matlab';
%         elseif ~strcmp(msg_received,msgs_Mat.sign_detected)
%             warning('Inconsistency with messages sent')
%         end
%     end
    %         ListenChar(2);
    KbQueueCreate(dev_used,keysOfInterest);
    pressed=0;             KbQueueStart(dev_used);
    
    while ~abort
        %     FlushEvents('keyDown');
        
        if is_online
            msg_received = M_PTB.waitmessage(5);
            if isempty(msg_received)
                abort=true; break;
            elseif strcmp(msg_received,msgs_Mat.error)
                is_online = false; experiment.is_online = 'error in online Matlab';
                warning(experiment.is_online);
            elseif ~strcmp(msg_received,msgs_Mat.ready_begin)
                warning('Inconsistency with messages sent')
            end
        end

        print_message(message_begin{ind_lang},black,window);
        %press ESC to start session
        while ~(pressed && any(firstPress([startKey exitKey])))
            [pressed,firstPress,~,~]=KbQueueCheck(dev_used);
        end
        %         [~, ~, keyCode] = KbCheck(dev_used);
        %         while ~keyCode([startKey exitKey]), [~, ~, keyCode] = KbCheck(dev_used); end
        
        if pressed && firstPress(exitKey)>0, abort = true; break; end
        %         if keyCode(exitKey), abort = true; break; end
        %     ListenChar(0);
        %     FlushEvents('keyDown');
        %         KbQueueCreate(dev_used,keysOfInterest);
        %                     KbQueueStop(dev_used);
        KbQueueFlush(dev_used);
        
        %     Priority(9)
        Priority(1);
        
        iind=1;
        HideCursor;
        
        
        %     KbQueueStart(dev_used);
%         fprintf('Current sequence (total = %d): ',Nseq)
        
        %     for irep=1:Nrep
        for irep=1:Nseq
            if is_online
                M_PTB.send(msgs_Mat.trial_begin);
            end
            fprintf('Current sequence (total = %d): %d\n',Nseq,irep)
%             fprintf('%d, ',irep)
            ich_blank=1;
            ich_pic=1;
            WaitSecs(0.150);
            %     KbQueueCreate(dev_used,keysOfInterest);
            %             KbQueueStart(dev_used);
            KbQueueFlush(dev_used);
            
            Screen('FillRect',window,black);
            times(k)=Screen('Flip',window);
            if ismember(withevents,[2 3])
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, blank_on); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, blank_on, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             err=DaqDOut(dio,0,blank_on);
            end
            if ismember(withevents,[1 3])
                cbmex('comment', msgs_colors{1}, 0, sprintf('%s seq%d',msgs{1},irep),'instance',which_nsp_comment-1);
            end
            inds_start_seq(irep)=k;
            tprev = times(k);
            
            color_up = color_start.up{irep};
            color_down = color_start.down{irep};
            
            Screen('FillRect',  window,black);
            Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
            Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
            
            if ismember(withevents,[2 3]) && with_reset
                WaitSecs('UntilTime', times(k)+wait_reset);
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, value_reset, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             fff=DaqDOut(dio,0,value_reset);
            end
            k=k+1;
            
            times(k) = Screen('Flip',window,times(k-1)+randTime_lines_on(1,irep));
            
            if ismember(withevents,[2 3])
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, lines_onoff); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, lines_onoff, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             err=DaqDOut(dio,0,lines_onoff);
            end
            if ismember(withevents,[1 3])
                cbmex('comment', msgs_colors{2}, 0, sprintf('%s seq%d',msgs{2},irep),'instance',which_nsp_comment-1);
            end
            %         tprev = times(k);
            
            if lines_change{irep}{1}{ich_blank,1}==1
                color_up = lines_change{irep}{1}{ich_blank,3};
                color_down = lines_change{irep}{1}{ich_blank,4};
                Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                
                if ismember(withevents,[2 3]) && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, value_reset, 8, 0);
%                     ljudObj.GoOne(ljhandle);
%                     %                 fff=DaqDOut(dio,0,value_reset);
                end
                k=k+1;
                times(k) = Screen('Flip',window,times(k-1)+lines_change{irep}{1}{ich_blank,2});
                if ismember(withevents,[2 3])
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, lines_flip_blank); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, lines_flip_blank, 8, 0);
%                     ljudObj.GoOne(ljhandle);
%                     %                 err=DaqDOut(dio,0,lines_flip_blank);
                end
                if ismember(withevents,[1 3])
                    cbmex('comment', msgs_colors{4}, 0, sprintf('%s seq%d',msgs{4},irep),'instance',which_nsp_comment-1);
                end
                if ismember(withevents,[2 3]) && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, value_reset, 8, 0);
%                     ljudObj.GoOne(ljhandle);
%                     %                 fff=DaqDOut(dio,0,value_reset);
                end                
                ich_blank = ich_blank +1;
            end
            k=k+1;
            %%% randTime_blank > randTime_lines_on + lines_change{irep}{1}{ich_blank,2}
            
            for iISI=1:NISI
                which_ISI = order_ISI(iISI,irep);
                
                Screen('DrawTexture', window,tex(order_pic(1,which_ISI,irep)),[],destRect{order_pic(1,which_ISI,irep)},0);
                Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                Screen('FillRect',window,white,flickerSquare);
                
                [times(k),t_stimon(iind),t_fliptime(iind)]=Screen('Flip',window,tprev+randTime_blank(iISI,irep)-slack,1);
                
                if ismember(withevents,[2 3])
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, pic_onoff(2,ceil(3*irep/Nseq))); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0, pic_onoff(2,ceil(3*irep/Nseq)), 8, 0);
%                     ljudObj.GoOne(ljhandle);
                    t_DAQpic(iind) = GetSecs;
                    %                 err=DaqDOut(dio,0,pic_onoff(2,ceil(3*irep/Nseq)));
                end
                if ismember(withevents,[1 3])
                    cbmex('comment', msgs_colors{3}, 0, sprintf('%s seq%d. pic%d',msgs{3},irep,order_pic(1,which_ISI,irep)),'instance',which_nsp_comment-1);
                end
                inds_pics(iind)=k;
                tprev = times(k);
                iind=iind+1;
                
                %             Screen('DrawTexture', window,tex(order_pic(1,which_ISI,irep)),[],destRect{order_pic(1,which_ISI,irep)},0);
                %             Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                %             Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                Screen('FillRect',window,black,flickerSquare);
                Screen('Flip',window,times(k)+3*ifi-slack);
                
                if ismember(withevents,[2 3]) && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                     ljudObj.GoOne(ljhandle);
%                     %                 fff=DaqDOut(dio,0,value_reset);
                end
                k=k+1;
                
                if which_ISI==lines_change{irep}{2}{ich_pic,1} && lines_change{irep}{2}{ich_pic,5}==1
                    Screen('DrawTexture', window,tex(order_pic(1,which_ISI,irep)),[],destRect{order_pic(1,which_ISI,irep)},0);
                    color_up = lines_change{irep}{2}{ich_pic,3};
                    color_down = lines_change{irep}{2}{ich_pic,4};
                    Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    %                     Screen('FillRect',window,black,flickerSquare);
                    times(k) = Screen('Flip',window,tprev+lines_change{irep}{2}{ich_pic,2}-slack);
                    if ismember(withevents,[2 3])
                        send_TTL(which_system,DAQ_param_1,DAQ_param_2, lines_flip_pic); 
%                         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,lines_flip_pic, 8, 0);
%                         ljudObj.GoOne(ljhandle);
%                         %                     err=DaqDOut(dio,0,lines_flip_pic);
                    end
                    if ismember(withevents,[1 3])
                        cbmex('comment', msgs_colors{5}, 0, sprintf('%s seq%d',msgs{5},irep),'instance',which_nsp_comment-1);
                    end
                    ich_pic = ich_pic +1;
                    if ismember(withevents,[2 3]) && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                         ljudObj.GoOne(ljhandle);
%                         %                     fff=DaqDOut(dio,0,value_reset);
                    end
                    k=k+1;
                end
                
                for ipic=2:seq_length
                    Screen('DrawTexture', window,tex(order_pic(ipic,which_ISI,irep)),[],destRect{order_pic(ipic,which_ISI,irep)},0);
                    Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    Screen('FillRect',window,white,flickerSquare);
                    
                    [times(k), t_stimon(iind), t_fliptime(iind)] = Screen('Flip',window,tprev+ISI(which_ISI)-slack,1);
                    if ismember(withevents,[2 3])
                        send_TTL(which_system,DAQ_param_1,DAQ_param_2, pic_onoff(mod(ipic,2)+1,ceil(3*irep/Nseq))); 
%                         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,pic_onoff(mod(ipic,2)+1,ceil(3*irep/Nseq)), 8, 0);
%                         ljudObj.GoOne(ljhandle);
                        t_DAQpic(iind) = GetSecs;
                        %                     err=DaqDOut(dio,0,pic_onoff(mod(ipic,2)+1,ceil(3*irep/Nseq)));                     %PICTURE ONSET
                    end
                    if ismember(withevents,[1 3])
                        cbmex('comment', msgs_colors{3}, 0, sprintf('%s seq%d. pic%d',msgs{3},irep,order_pic(ipic,which_ISI,irep)),'instance',which_nsp_comment-1);
                    end
                    inds_pics(iind)=k;
                    tprev = times(k);
                    iind=iind+1;
                    
                    %                 Screen('DrawTexture', window,tex(order_pic(ipic,which_ISI,irep)),[],destRect{order_pic(ipic,which_ISI,irep)},0);
                    %                 Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    %                 Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    Screen('FillRect',window,black,flickerSquare);
                    Screen('Flip',window,times(k)+3*ifi-slack);
                    
                    if ismember(withevents,[2 3]) && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                         ljudObj.GoOne(ljhandle);
%                         %                     fff=DaqDOut(dio,0,value_reset);
                    end
                    k=k+1;
                    
                    if which_ISI==lines_change{irep}{2}{ich_pic,1} && lines_change{irep}{2}{ich_pic,5}==ipic
                        Screen('DrawTexture', window,tex(order_pic(ipic,which_ISI,irep)),[],destRect{order_pic(ipic,which_ISI,irep)},0);
                        color_up = lines_change{irep}{2}{ich_pic,3};
                        color_down = lines_change{irep}{2}{ich_pic,4};
                        Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                        Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                        %                         Screen('FillRect',window,black,flickerSquare);
                        times(k) = Screen('Flip',window,tprev+lines_change{irep}{2}{ich_pic,2}-slack);
                        if ismember(withevents,[2 3])
                            send_TTL(which_system,DAQ_param_1,DAQ_param_2, lines_flip_pic); 
%                             ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,lines_flip_pic, 8, 0);
%                             ljudObj.GoOne(ljhandle);
%                             %                         err=DaqDOut(dio,0,lines_flip_pic);
                        end
                        if ismember(withevents,[1 3])
                            cbmex('comment', msgs_colors{5}, 0, sprintf('%s seq%d',msgs{5},irep),'instance',which_nsp_comment-1);
                        end
                        ich_pic = ich_pic +1;
                        if ismember(withevents,[2 3]) && with_reset
                            WaitSecs('UntilTime', times(k)+wait_reset);
                            send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                             ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                             ljudObj.GoOne(ljhandle);
%                             %                         fff=DaqDOut(dio,0,value_reset);
                        end
                        k=k+1;
                    end
                end
                
                Screen('FillRect',  window,black);
                Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                times(k) = Screen('Flip',window,tprev+ISI(which_ISI)-slack);
                if ismember(withevents,[2 3])
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, blank_on); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,blank_on, 8, 0);
%                     ljudObj.GoOne(ljhandle);
%                     %                 err=DaqDOut(dio,0,blank_on);
                end
                if ismember(withevents,[1 3])
                    cbmex('comment', msgs_colors{1}, 0, sprintf('%s seq%d',msgs{1},irep),'instance',which_nsp_comment-1);
                end
                tprev = times(k);
                
                if ismember(withevents,[2 3]) && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                     ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                     ljudObj.GoOne(ljhandle);
%                     %                 fff=DaqDOut(dio,0,value_reset);
                end
                k=k+1;
                
                if lines_change{irep}{1}{ich_blank,1}==1+iISI
                    color_up = lines_change{irep}{1}{ich_blank,3};
                    color_down = lines_change{irep}{1}{ich_blank,4};
                    Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    times(k) = Screen('Flip',window,times(k-1)+lines_change{irep}{1}{ich_blank,2});
                    if ismember(withevents,[2 3])
                        send_TTL(which_system,DAQ_param_1,DAQ_param_2, lines_flip_blank); 
%                         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,lines_flip_blank, 8, 0);
%                         ljudObj.GoOne(ljhandle);
%                         %                     err=DaqDOut(dio,0,lines_flip_blank);
                    end
                    if ismember(withevents,[1 3])
                        cbmex('comment', msgs_colors{4}, 0, sprintf('%s seq%d',msgs{4},irep),'instance',which_nsp_comment-1);
                    end
                    ich_blank = ich_blank +1;
                    if ismember(withevents,[2 3]) && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                         ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                         ljudObj.GoOne(ljhandle);
%                         %                     fff=DaqDOut(dio,0,value_reset);
                    end
                    k=k+1;
                end
            end
            
            Screen('FillRect',  window,black);
            times(k) = Screen('Flip',window,tprev+randTime_lines_off(1,irep)-slack);
            if ismember(withevents,[2 3])
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, lines_onoff); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,lines_onoff, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             err=DaqDOut(dio,0,lines_onoff);
            end
            if ismember(withevents,[1 3])
                cbmex('comment', msgs_colors{6}, 0, sprintf('%s seq%d',msgs{6},irep),'instance',which_nsp_comment-1);
            end
            %             tprev = times(k);
            if ismember(withevents,[2 3]) && with_reset
                WaitSecs('UntilTime', times(k)+wait_reset);
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             fff=DaqDOut(dio,0,value_reset);
            end
            k=k+1;
            
            WaitSecs(tprev+randTime_blank(NISI+1,irep)-GetSecs);
            
            % CHECK KBQUEUE (see how to collect all the spacebar presses)
            [pressed,firstPress,~,~]=KbQueueCheck(dev_used);
            
            if pressed && firstPress(exitKey)>0
                abort = true;
                break
            end
            
            %             KbQueueStop(dev_used);
            %         KbQueueRelease(dev_used);
            
            if is_online
                M_PTB.send(msgs_Mat.trial_end);            
                print_message(message_wait{ind_lang},black,window);
                [msg_received, time_wait(irep)]= M_PTB.waitmessage(60);
                if isempty(msg_received)
                    is_online = false; experiment.is_online = 'online Matlab went offline';
                    warning(experiment.is_online);
                elseif strcmp(msg_received,msgs_Mat.error)
                    is_online = false; experiment.is_online = 'error in online Matlab';
                    warning(experiment.is_online);
                elseif ~strcmp(msg_received,msgs_Mat.process_ready)
                    disp('Inconsistency with messages sent')
                end
            end
            
            print_message(message_continue{ind_lang},black,window);
            
            Screen('FillRect',window,black);
            %         FlushEvents('keyDown');
            %             [~, ~, keyCode] = KbCheck(dev_used,scanlist);
            KbQueueFlush(dev_used);
            %                     KbQueueStart(dev_used);
            
            if auto_resp
                [~,~,pressed,firstPress] = get_response(dev_used,device_resp,[exitKey continueKey],0.2,1);
            else
                [~,~,pressed,firstPress] = get_response(dev_used,device_resp,[exitKey continueKey],0.2);
            end
%             [pressed,firstPress,~,~]=KbQueueCheck(dev_used);            
%             if strcmp(device_resp,'gamepad')
%                 %             gp_state_1 = Gamepad('GetButton', gamepad_index, 1);
%                 %             gp_state_2 = Gamepad('GetButton', gamepad_index, 2);
%                 %             while (~sum([gp_state_1, gp_state_2]))&& ~sum(keyCode([startKey,exitKey]))
%                 %                 gp_state_1 = Gamepad('GetButton', gamepad_index, 1);
%                 %                 gp_state_2 = Gamepad('GetButton', gamepad_index, 2);
%                 %                 [~, ~, keyCode] = KbCheck(dev_used,scanlist);
%                 %             end;
%                 %             while ~any([(pressed && firstPress(exitKey)>0) gp_state_1 gp_state_2])
%                 while ~any([(pressed && any(firstPress([exitKey continueKey]))) gp_state_1 gp_state_2])
%                     %                 while ~any([keyCode([exitKey continueKey]) gp_state_1 gp_state_2])
%                     [pressed,firstPress,~,~]=KbQueueCheck(dev_used);
%                     %                     [~, ~, keyCode] = KbCheck;
%                     [a,ab] = JoyMEX(0);
%                     gp_state_1 =  a(1)==-1;
%                     gp_state_2 =  ab(2);
%                 end
%                 %             fprintf('Button Far Left = %d, Button Far Right = %d\n',gp_state_1,gp_state_2)
%                 pause(0.2)
%                 gp_state_1=false; gp_state_2=false;
%             else
%                 while ~(pressed && any(firstPress([continueKey,exitKey])))
%                     %                 while ~sum(keyCode([startKey,exitKey]))
%                     [pressed,firstPress,~,~]=KbQueueCheck(dev_used);
%                     %                     [~, ~, keyCode] = KbCheck(dev_used,scanlist);
%                 end
%             end
            times(k)=GetSecs;
            if ismember(withevents,[2 3])
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, trial_on); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,trial_on, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             err=DaqDOut(dio,0,trial_on);
            end
            if ismember(withevents,[1 3])
                cbmex('comment', msgs_colors{7}, 0, sprintf('%s seq%d',msgs{7},irep),'instance',which_nsp_comment-1);
            end
            if ismember(withevents,[2 3]) && with_reset
                WaitSecs('UntilTime', times(k)+wait_reset);
                send_TTL(which_system,DAQ_param_1,DAQ_param_2, value_reset); 
%                 ljudObj.AddRequestS(ljhandle, 'LJ_ioPUT_DIGITAL_PORT', 0,value_reset, 8, 0);
%                 ljudObj.GoOne(ljhandle);
%                 %             fff=DaqDOut(dio,0,value_reset);
            end
            k=k+1;
            KbQueueFlush(dev_used);
            %             KbQueueStop(dev_used);
            %         ListenChar(0);
            if pressed && firstPress(exitKey)>0, abort = true; break; end
            %             if keyCode(exitKey), abort = true; break; end
        end
        break
    end
    %         ListenChar(0);
    print_message('THAT WOULD BE ALL.\n THANK YOU !!!',black,window)
    ttt=GetSecs;
    
    if is_online && abort
        M_PTB.send(msgs_Mat.exper_aborted);
        warning(msgs_Mat.exper_aborted);
    end
    
    save('RSVP_SCR_workspace');
        
    times(isnan(times))=[];
    inds_pics(inds_pics==0)=[];
    inds_start_seq(inds_start_seq==0)=[];
    KbQueueStop(dev_used);
    KbQueueRelease(dev_used);
    Priority(0);
    %     save(['timesanswer' prf], 'times', 'times_break','answer');
    %     save('timesanswer','times', 'times_break','answer');
    experiment.times=times;
    experiment.t_stimon=t_stimon;
    experiment.t_fliptime=t_fliptime;
    experiment.t_DAQpic=t_DAQpic;
    experiment.inds_pics=inds_pics;
    experiment.inds_start_seq=inds_start_seq;
    experiment.answer=answer;
    experiment.abort=abort;
    experiment.times_break=times_break;
    experiment.time_wait=time_wait;
    experiment.cant_breaks=cant_breaks;
    save('experiment_properties','experiment');            
    
    if recording_on && exist('onlineNSP','var')
        StopBlackrockAquisition(experiment.fname,onlineNSP);
    end
    
    WaitSecs(10-(GetSecs-ttt));

    Screen('CloseAll');
    ShowCursor;
    rmpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX'))

    if with_local_net
        reached_backup = true;
        if with_analysis
%             [status, msg_copy]= copyfile(sprintf('Z:\\%s.ns5',experiment.fname), fullfile(pwd,[experiment.folder_name '.ns5']));
%             [status, msg_copy]= copyfile(sprintf('Z:\\%s.nev',experiment.fname), fullfile(pwd,[experiment.folder_name '.nev']));
            [status, msg_copy]= copyfile(sprintf('Z:\\%s*.ns5',experiment.fname), pwd);
            [status, msg_copy]= copyfile(sprintf('Z:\\%s*.nev',experiment.fname), pwd);
        end
        
%         A=dir(sprintf('Z:\\%s*',experiment.fname));
        if k==1
            rmdir(sprintf('Z:\\%s',experiment.folder_name))
            delete(sprintf('Z:\\%s*',experiment.fname))
        else
%             [status, msg_copy]= copyfile(sprintf('Z:\\%s*',experiment.fname), sprintf('Z:\\%s',experiment.folder_name));
            [status, msg_copy]= movefile(sprintf('Z:\\%s*',experiment.fname), sprintf('Z:\\%s',experiment.folder_name));
            if ~status
                disp(msg_copy)
                error('Check the data copied within the Blackrock PC as there was an error')
            end
            status = []; msg_copy={};
            
            if is_online
                disp('Waiting for online processing to be completed')
                msg_received = M_PTB.waitmessage(600);
                if strcmp(msg_received,msgs_Mat.error)
                    is_online = false; experiment.is_online = 'error in online Matlab';
                    warning(experiment.is_online);
                elseif ~strcmp(msg_received,msgs_Mat.process_end)
                    warning('Inconsistency with messages sent')
                end
            end
            
            for i=1:length(data_transfer_copy)
                [status(end+1), msg_copy{end+1}] = copyfile(data_transfer_copy{i}, sprintf('Z:\\%s\\%s',experiment.folder_name,data_transfer_copy{i})); %overwrites if already exists
            end
            [~,noMatch] = regexp(pwd,filesep,'match','split');
            
            [status(end+1),msg_copy{end+1}] = copyfile(Path_pics, sprintf('Z:\\%s\\%s',experiment.folder_name,[noMatch{end} '_pic'])); %overwrites if already exists
%             [status(end+1),msg_copy{end+1}] = copyfile([pwd '_pic'], sprintf('Z:\\%s\\%s',experiment.folder_name,[noMatch{end} '_pic'])); %overwrites if already exists
            if exist('data_transfer_move','var')
                for i=1:length(data_transfer_move)
                    [status(end+1),msg_copy{end+1}] = movefile(data_transfer_move{i}, sprintf('Z:\\%s\\%s',experiment.folder_name,data_transfer_move{i})); %overwrites if already exists
                end
            end
            if any(~status)
                for jj=find(status)
                    disp(msg_copy{jj})
                end
                error('Check the data copied to the Blackrock PC as there was an error')
            end
            answer = questdlg('Do you want to backup the data now? If so, verify that the server link is up and running','Backup?','Yes','No','Yes');
            if strcmp(answer,'Yes')
                bat_name = 'Z:\backup.bat';
                fid = fopen(bat_name,'w');
                %% CHECK PATH NAMES AND INCLUDE SUBJECT NAME THERE. INCLUDE _BACKUP IN THE FOLDER NAME FOR HDD D:
                fprintf(fid,'Xcopy /E "C:\\Users\\User\\Desktop\\DATA\\%s" "S:\\ECoG_Data\\%sDatafile\\DATA\\%s\\" \n',experiment.folder_name,sub_ID,experiment.folder_name);
%                 fprintf(fid,'robocopy "C:\\Users\\User\\Desktop\\DATA\\%s" "D:\\DATA BACKUP\\DATA\\%sDatafile_BACKUP\\%s" /E /XC /XN /XO /MOVE\n',experiment.folder_name,sub_ID,experiment.folder_name);
                fprintf(fid,'robocopy "C:\\Users\\User\\Desktop\\DATA\\%s" "D:\\DATA BACKUP\\DATA\\%sDatafile_BACKUP\\%s" /E /XC /XN /XO \n',experiment.folder_name,sub_ID,experiment.folder_name);
                %         fprintf(fid,'move /-Y "C:\\Users\\User\\Desktop\\DATA\\%s" "D:"',experiment.fname);
                fclose(fid);
                msgbox('Go to Blackrock PC and drag "backup.bat" in DATA folder to a command prompt to finish the back up');
            end
        end
    end
catch ME
%     rethrow(ME)
    if is_online
        M_PTB.send(msgs_Mat.error);
    end   
    
    times(isnan(times))=[];
    inds_pics(inds_pics==0)=[];
    inds_start_seq(inds_start_seq==0)=[];
    try
        [pressed,firstPress,~,~]=KbQueueCheck(dev_used);
        KbQueueStop(dev_used);
        KbQueueRelease(dev_used);
    catch
        disp('no keyboard queue active')
    end
%     Screen('CloseAll')
    Priority(0);
    %     save(['timesanswer' prf], 'times', 'times_break','answer');
    %     save('timesanswer','times', 'times_break','answer');
    experiment.times=times;
    experiment.t_stimon=t_stimon;
    experiment.t_fliptime=t_fliptime;
    experiment.t_DAQpic=t_DAQpic;
    experiment.inds_pics=inds_pics;
    experiment.inds_start_seq=inds_start_seq;
    experiment.answer=answer;
    %     experiment.tstart=tstart;
    experiment.times_break=times_break;
    experiment.cant_breaks=cant_breaks;
    experiment.ME=ME;
    save('RSVP_SCR_workspace');
    save('experiment_properties','experiment');
    if ~reached_backup
        print_message('THAT WOULD BE ALL./n THANK YOU !!!',black,window)
        Screen('CloseAll');
        ShowCursor;

        if recording_on && exist('onlineNSP','var')
            StopBlackrockAquisition(experiment.fname,onlineNSP);
        end

        rmpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX'))
        rmpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\BCM-EMU\useful_functions\tasks_tools'))

     end
%     ShowCursor;
    msgbox('There was an error in the script. Review the data saved and back it up manually if necessary');
    rethrow(ME)
end

function print_message(message,black,window)
Screen('FillRect',  window,black);
DrawFormattedText(window, message, 'center', 'center', [255 255 255]);
Screen('Flip',window);