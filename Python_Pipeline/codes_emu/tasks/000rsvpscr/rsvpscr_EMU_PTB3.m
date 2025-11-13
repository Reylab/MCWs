function rsvpscr_EMU_PTB3(varargin)
% function rsvpscr_EMU_PTB3(sub_ID,EMU_num,run_num,subtask,varargin)
close all
ipr = inputParser;
addOptional(ipr,'location', 'MCW-BEH-RIP');
addParameter(ipr,'is_online',false);
addParameter(ipr,'auto_resp', false);
addParameter(ipr,'ptb_debug',false);
addParameter(ipr,'Nrep',[]);
addParameter(ipr,'which_nsp_comment',[]);
addParameter(ipr,'lang',[]);
addParameter(ipr,'device_resp',[]);
addParameter(ipr,'system',[]);
addParameter(ipr,'acq_network',[]) % start and stop recordings remotely
addParameter(ipr,'use_BRK_comment',[]) %if true, will force to send comments
addParameter(ipr,'use_daq',true) 

addParameter(ipr,'sub_ID','none') 
addParameter(ipr,'EMU_num',111) 
addParameter(ipr,'run_num',1) 
addParameter(ipr,'subtask','first_time') 

parse(ipr,varargin{:})

addpath(fileparts(fileparts(fileparts(mfilename('fullpath')))));
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

sub_ID = params.sub_ID;
EMU_num = params.EMU_num;
run_num = params.run_num;
subtask = params.subtask;

MOVE_PICS = false; %if true, the pictures will be moved instead of only copy to other folder
% cbmex comments with max 127 characters
switch subtask
    case 'scr'
        %requires custom
        if isempty(params.Nrep), params.Nrep=15; end
        Path_pics = 'custom_pics';
        params.use_daq=1; 
        params.backup_network=1;
        params.use_BRK_comment=0;
        params.acq_network = 1;
        params.is_online = 1;
        device_resp='gamepad';
        
    case 'miniscr'
        if isempty(params.Nrep), params.Nrep=18; end
        Path_pics = 'miniscr_pics';
        MOVE_PICS = true;
        params.use_daq=1; 
        params.backup_network=1;
        params.use_BRK_comment=0;
        params.acq_network = 1;
        params.is_online = 1;
        device_resp='gamepad';
    case 'test' %to check
        params.use_daq=0; 
        params.backup_network=0;
        params.use_BRK_comment=0;
        params.acq_network = 0;
        if isempty(params.Nrep), params.Nrep=9; end
        %device_resp='gamepad';
        params.is_online = 0;
        device_resp='gamepad';
        Path_pics='picsfirst';
    case 'online_test' %to check
        params.use_daq=1; 
        params.backup_network=1;
        params.use_BRK_comment=0;
        params.acq_network = 1;
        params.is_online = 1;
        params.auto_resp = 1;
        if isempty(params.Nrep), params.Nrep=12; end
        device_resp='keyboard';
        %device_resp='gamepad';
        Path_pics='custom_pics';
    case 'first_time'
        params.backup_network=0;
        params.use_BRK_comment=0;
        params.acq_network = 0;
        params.is_online = 0;
        if isempty(params.Nrep), params.Nrep=12; end %this assumes that a folder "picsfirst", containing 10 random pictures, has been created in pics_root
        Path_pics='picsfirst';
        device_resp='gamepad';      
    otherwise
    	error('invalid subtask')
end

if ~exist('device_resp','var')
    device_resp = params.device_resp;
end
Nrep = params.Nrep;
abs_path_pics = [params.pics_root_beh filesep Path_pics];
if ~exist(abs_path_pics,'dir')
    error('Folder: %s not found.', abs_path_pics );
end

if isunix
    system('nmcli radio wifi off'); %disable wifi
end


custompath.add(params.additional_paths,true)
if EMU_num==0 &&  isnumeric(sub_ID)  &&sub_ID==0 && contains(subtask,'test')
    sub_ID = 'test';
    run_num = randi(98,1,1)+1;
end
if params.backup_network
    if params.is_online 
        data_transfer_copy = {'experiment_properties.mat';'RSVP_SCR_workspace.mat';'rsvpscr_EMU_PTB3.m';'shuffle_rsvpSCR.m';'create_lines_change_RSVP_SCR.m';'online'};
    else
        data_transfer_copy = {'experiment_properties.mat';'RSVP_SCR_workspace.mat';'rsvpscr_EMU_PTB3.m';'shuffle_rsvpSCR.m';'create_lines_change_RSVP_SCR.m'};
    end
end

if ~params.acq_network
    if params.backup_network
        error('backup_network=True incompatible with acq_network=False')
    end
end


if params.ptb_debug;  PsychDebugWindowConfiguration; end

ttl_device_name = params.ttl_device;

shuffle_rsvpSCR(params.Nrep,abs_path_pics)

experiment.fname = sprintf('EMU-%.3d_subj-%s_task-RSVPscr_run-%.2d',EMU_num,sub_ID,run_num);
experiment.folder_name = sprintf('EMU-%.3d_task-RSVPscr_run-%.2d',EMU_num,run_num);
if params.backup_network
%     if exist(sprintf('%s%c%s.ns5',params.remote_disk_root,filesep,experiment.fname),'file')
%         error('Filename %s already exists. Please check everything and run the task again',experiment.fname)
%     end
%     mkdir(sprintf('%s%c%s.ns5',params.remote_disk_root,filesep,experiment.fname))
end
if params.acq_network 
   recording = recording_handler(params, experiment.fname);
end


with_reset = false;
abort = false;
reached_backup = false;


load(fullfile(pwd,'order_pics_RSVP_SCR.mat'))
load(fullfile(pwd,'variables_RSVP_SCR.mat'))

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

[kbs,products_names] = GetKeyboardIndices;
dev_used = [];
for i =1:numel(params.keyboards)
    if isnumeric(params.keyboards{i})
        dev_used(end+1) = params.keyboard{i};
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
if isempty(dev_used)
   error('Keyboards not found') 
end

KbName('UnifyKeyNames');
Screen('Preference','VisualDebugLevel',3);
AssertOpenGL;    % Running on PTB-3? Abort otherwise.

exitKey = KbName('F2');
startKey = KbName('s'); %88 in Windows, 27 in MAC
%     spaceKey= KbName('space');  %for color changes (not really saved)
breakKey= KbName('p');  %to pause
continueKey= KbName('c');  %to continue if gamepad fails

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
    pic_onoff = [[1 4 16];[2 8 32]];  % first pic with row 2
    bits_for_break = []; %to check 
    lines_onoff = 13;
    blank_on = 11;
    lines_flip_blank = 103;
    lines_flip_pic = 22;
    trial_on = 26;

    data_signature_on = 64;
    data_signature_off = 128;

    wait_reset = 0.1;  % IT MUST BE SHORTER THAN THE SHORTEST ISI
    value_reset = 0;
    
    msgs = {'blank on';'lines on';'pic change';...
        'lines change blank';'lines change pic';'lines off';'trial ended'};
    msgs_colors = {255;65280;16711680;16777215;65535;16711935;16776960};


    experiment.pwd=pwd;
    experiment.date=date;
    experiment.pic=pic_onoff;
    experiment.blank_on=blank_on;
    experiment.lines_onoff=lines_onoff;
    experiment.lines_flip_blank=lines_flip_blank;
    experiment.lines_flip_pic=lines_flip_pic;
    experiment.trial_on=trial_on;
    experiment.data_signature=[data_signature_on data_signature_off];
    experiment.order_pic=order_pic;
    experiment.order_ISI=order_ISI;
    experiment.ISI=ISI;
    experiment.Nrep=Nrep;
    experiment.seq_length=seq_length;
    experiment.Nseq=Nseq;
    experiment.msgs=msgs;
    experiment.deviceresp=device_resp;
    experiment.is_online=params.is_online;
    experiment.msgs_Mat=msgs_Mat;
    experiment.nchanges_blank=nchanges_blank;
    experiment.nchanges_pic=nchanges_pic;
    experiment.with_reset=with_reset;
    experiment.Path_pics=Path_pics;
    experiment.value_reset=value_reset;
    experiment.wait_reset=wait_reset;    
    experiment.lines_change=lines_change;
    experiment.params = params;
    
    save('experiment_properties','experiment');

if params.is_online
    answer = questdlg('Start the online function on a separate Matlab and wait for instructions to continue','Online started?','Continue','Continue');
    %     disp('Start the online function on a separate Matlab. Press OK key to continue')
    %    [~, ~, keyCode] = KbCheck(dev_used);
    %    while ~keyCode(startKey), [~, ~, keyCode] = KbCheck(dev_used); end
    if isempty(answer)
        error('message window closed')
    end
    M_PTB = matconnect(params.proccesing_machine);
    M_PTB.send(num2str(params.which_nsp_comment));
end

Screen('Preference', 'SkipSyncTests', double(IsWin));

if strcmp(params.lang,'english')
    ind_lang=1;
elseif strcmp(params.lang,'spanish')
    ind_lang=2;
elseif strcmp(params.lang,'french')
    ind_lang=3;
end
screen_closed = 0;
try
    screens=Screen('Screens');
    whichScreen=max(screens);
%     	whichScreen=1;
    
    %     min_cross=0.2;
    min_blank=1.25;
    max_rand_blank = 0.5;
    min_lines_onoff=0.5;
    max_rand_lines_onoff = 0.2;
    %     max_rand_cross = 0.1;
    %     tend_trial = 0.1;
    %     dura_base=1.0;
    size_line = 5;
    
    gamepad_ix = [];
    if strcmp(device_resp,'gamepad')
        if IsWin
            clear JoyMEX;
            JoyMEX('init',0);
        elseif IsLinux
            numGamepads = Gamepad('GetNumGamepads');
            if (numGamepads == 0)
                error('Gamepad not connected');
            else
                [~, gamepad_name] = GetGamepadIndices;
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
    
    a=dir(sprintf('%s/*',abs_path_pics));
    b = zeros(length(a),1);
    b = b>0;
    for i = 1:length(a)
        b(i) = contains (lower(a(i).name), '.jp');
    end
    a = a(b);
    if isempty(a)
        error(['No pictures for this session in ' abs_path_pics]);
    end
    Npic=length(a);
    ImageNames = cell(Npic,1);
    for i=1:Npic,ImageNames{i}=a(i).name;end
    

    if params.use_daq;   dig_out  = TTL_device(ttl_device_name); end
    

    
    rng('shuffle', 'twister')
    
    randTime_blank = min_blank + max_rand_blank*rand(NISI+1,Nseq);
    randTime_lines_on = min_lines_onoff + max_rand_lines_onoff*rand(1,Nseq);
    randTime_lines_off = randTime_blank(NISI+1,:) - (min_lines_onoff + max_rand_lines_onoff*rand(1,Nseq));
    
    experiment.ImageNames=ImageNames;

        
    save('experiment_properties','experiment');
    
    if params.is_online; M_PTB.send(msgs_Mat.exper_saved);  end
    
    % Open screen.  Do this before opening the
    % offscreen windows so you can align offscreen
    % window memory to onscreen for faster copying.
    
    if isfield(params,'windowRect')
        [window,windowRect]=Screen(whichScreen,'OpenWindow',0, params.windowRect);
    else
        [window,windowRect]=Screen(whichScreen,'OpenWindow',0);
    end

%         [window,windowRect]=Screen('OpenWindow',whichScreen,0,[0 0 1024 768]);
    
    %window and monitor properties
    xcenter=windowRect(3)/2;
    ycenter=windowRect(4)/2;
    Priority(1); 
    ifi = Screen('GetFlipInterval', window, 200);
%     slack=ifi/2;
    slack=ifi/4;
    Priority(params.ptb_priority_normal); %normal priority
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
        Im=imread(sprintf('%s/%s',abs_path_pics,ImageNames{i}));
        nRows=size(Im,1); nCols=size(Im,2);
         if ~(nRows==160 && nCols==160)
             error('picture %s with wrong dimentions', ImageNames{i})
         end
        imageRect{i}=SetRect(0,0,nCols,nRows);
        destRect{i}=CenterRect(imageRect{i},windowRect);
        tex(i)=Screen('MakeTexture',window,Im);
    end
    
    keysOfInterest=zeros(1,256);
    firstPress=zeros(1,256);
    keysOfInterest([exitKey breakKey startKey continueKey])=1;
    save('RSVP_SCR_workspace','-regexp', '^(?!(M_PTB)$).');
    
    answer = questdlg('Tell the subject to get ready to begin. Press OK to continue','Subject ready?','OK','OK');
    k=1;
    
    if params.acq_network
        recording.start();
        times(k)=GetSecs; k=k+1;
    elseif params.use_BRK_comment
        cbmex('open')
    end
    
    WaitSecs(4);
    
    if params.is_online; M_PTB.send(msgs_Mat.rec_started);  end
   
    
    if params.use_daq
        dig_out.send(data_signature_on);
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
        times(k)=GetSecs; k=k+1;
    end
    
    for d=dev_used
        KbQueueCreate(d,keysOfInterest);
        KbQueueStart(d);
    end
    pressed=0;     
    while ~abort
        if params.is_online
            msg_received = M_PTB.waitmessage(5);
            if isempty(msg_received)
                abort=true; break;
            elseif strcmp(msg_received,msgs_Mat.error)
               params.is_online = false; experiment.is_online = 'error in online Matlab';
                warning(experiment.is_online);
            elseif ~strcmp(msg_received,msgs_Mat.ready_begin)
                warning('Inconsistency with messages sent')
            end
        end

        print_message(message_begin{ind_lang},black,window);
        %press ESC to start session
        while ~(pressed && any(firstPress([startKey exitKey])))
            [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);
        end
%                 [~, ~, keyCode] = KbCheck(dev_used);
%                  while ~keyCode([startKey exitKey]), [~, ~, keyCode] = KbCheck(dev_used); end
%         
%         if pressed && firstPress(exitKey)>0, abort = true; break; end
        %         if keyCode(exitKey), abort = true; break; end
        for d= dev_used; KbQueueFlush(d);   end
        Priority(params.ptb_priority_high);
        
        iind=1;
        HideCursor;
        
        
        for irep=1:Nseq
            if params.is_online
                M_PTB.send(msgs_Mat.trial_begin);
            end
            fprintf('Current sequence (total = %d): %d\n',Nseq,irep)
            ich_blank=1;
            ich_pic=1;
            WaitSecs(0.150);

            for d= dev_used; KbQueueFlush(d);   end
            
            Screen('FillRect',window,black);
            times(k)=Screen('Flip',window);
            if params.use_daq
                dig_out.send(blank_on); 
            end
            if params.use_BRK_comment
                cbmex('comment', msgs_colors{1}, 0, sprintf('%s seq%d',msgs{1},irep),'instance',params.which_nsp_comment-1);
            end
            inds_start_seq(irep)=k;
            tprev = times(k);
            
            color_up = color_start.up{irep};
            color_down = color_start.down{irep};
            
            Screen('FillRect',  window,black);
            Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
            Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
            
            if params.use_daq && with_reset
                WaitSecs('UntilTime', times(k)+wait_reset);
                dig_out.send(value_reset); 
            end
            k=k+1;
            
            times(k) = Screen('Flip',window,times(k-1)+randTime_lines_on(1,irep));
            
            if params.use_daq
                dig_out.send(lines_onoff); 
            end
            if params.use_BRK_comment
                cbmex('comment', msgs_colors{2}, 0, sprintf('%s seq%d',msgs{2},irep),'instance',params.which_nsp_comment-1);
            end
            %         tprev = times(k);
            
            if lines_change{irep}{1}{ich_blank,1}==1
                color_up = lines_change{irep}{1}{ich_blank,3};
                color_down = lines_change{irep}{1}{ich_blank,4};
                Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                
                if params.use_daq && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    dig_out.send(value_reset); 
                end
                k=k+1;
                times(k) = Screen('Flip',window,times(k-1)+lines_change{irep}{1}{ich_blank,2});
                if params.use_daq
                    dig_out.send(lines_flip_blank); 
                end
                if params.use_BRK_comment
                    cbmex('comment', msgs_colors{4}, 0, sprintf('%s seq%d',msgs{4},irep),'instance',params.which_nsp_comment-1);
                end
                if params.use_daq && with_reset
                    WaitSecs('UntilTime', times(k)+wait_reset);
                    dig_out.send(value_reset); 
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
                
                if params.use_daq
                    dig_out.send(pic_onoff(2,ceil(3*irep/Nseq))); 
                    t_DAQpic(iind) = GetSecs;
                end
                if params.use_BRK_comment
                    cbmex('comment', msgs_colors{3}, 0, sprintf('%s seq%d. pic%d',msgs{3},irep,order_pic(1,which_ISI,irep)),'instance',params.which_nsp_comment-1);
                end
                inds_pics(iind)=k;
                tprev = times(k);
                iind=iind+1;
                %             Screen('DrawTexture', window,tex(order_pic(1,which_ISI,irep)),[],destRect{order_pic(1,which_ISI,irep)},0);
                %             Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                %             Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
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
                    %                     Screen('FillRect',window,black,flickerSquare);
                    times(k) = Screen('Flip',window,tprev+lines_change{irep}{2}{ich_pic,2}-slack);
                    if params.use_daq
                        dig_out.send(lines_flip_pic); 
                    end
                    if params.use_BRK_comment
                        cbmex('comment', msgs_colors{5}, 0, sprintf('%s seq%d',msgs{5},irep),'instance',params.which_nsp_comment-1);
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
                    Screen('FillRect',window,white,flickerSquare);
                    
                    [times(k), t_stimon(iind), t_fliptime(iind)] = Screen('Flip',window,tprev+ISI(which_ISI)-slack,1);
                    if params.use_daq
                        dig_out.send(pic_onoff(mod(ipic,2)+1,ceil(3*irep/Nseq))); 
                        t_DAQpic(iind) = GetSecs;
                    end
                    if params.use_BRK_comment
                        cbmex('comment', msgs_colors{3}, 0, sprintf('%s seq%d. pic%d',msgs{3},irep,order_pic(ipic,which_ISI,irep)),'instance',params.which_nsp_comment-1);
                    end
                    inds_pics(iind)=k;
                    tprev = times(k);
                    iind=iind+1;
                    
                    %                 Screen('DrawTexture', window,tex(order_pic(ipic,which_ISI,irep)),[],destRect{order_pic(ipic,which_ISI,irep)},0);
                    %                 Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                    %                 Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                    Screen('FillRect',window,black,flickerSquare);
                    Screen('Flip',window,times(k)+3*ifi-slack);
                    
                    if params.use_daq && with_reset
                        WaitSecs('UntilTime', times(k)+wait_reset);
                        dig_out.send(value_reset); 
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
                        if params.use_daq; dig_out.send(lines_flip_pic);   end
                        if params.use_BRK_comment
                            cbmex('comment', msgs_colors{5}, 0, sprintf('%s seq%d',msgs{5},irep),'instance',params.which_nsp_comment-1);
                        end
                        ich_pic = ich_pic +1;
                        if params.use_daq && with_reset
                            WaitSecs('UntilTime', times(k)+wait_reset);
                            dig_out.send(value_reset); 
                        end
                        k=k+1;
                    end
                end
                
                Screen('FillRect',  window,black);
                Screen('DrawLine',window,color_up,destRect{1}(1),destRect{1}(2)-lines_offset,destRect{1}(3),destRect{1}(2)-lines_offset,size_line);
                Screen('DrawLine',window,color_down,destRect{1}(1),destRect{1}(4)+lines_offset,destRect{1}(3),destRect{1}(4)+lines_offset,size_line);
                times(k) = Screen('Flip',window,tprev+ISI(which_ISI)-slack);
                if params.use_daq
                    dig_out.send(blank_on); 
                end
                if params.use_BRK_comment
                    cbmex('comment', msgs_colors{1}, 0, sprintf('%s seq%d',msgs{1},irep),'instance',params.which_nsp_comment-1);
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
                    times(k) = Screen('Flip',window,times(k-1)+lines_change{irep}{1}{ich_blank,2});
                    if params.use_daq
                        dig_out.send(lines_flip_blank); 
                    end
                    if params.use_BRK_comment
                        cbmex('comment', msgs_colors{4}, 0, sprintf('%s seq%d',msgs{4},irep),'instance',params.which_nsp_comment-1);
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
                dig_out.send(lines_onoff); 
            end
            if params.use_BRK_comment
                cbmex('comment', msgs_colors{6}, 0, sprintf('%s seq%d',msgs{6},irep),'instance',params.which_nsp_comment-1);
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
                break
            end
            
            if params.is_online
                M_PTB.send(msgs_Mat.trial_end);            
                print_message(message_wait{ind_lang},black,window);
                [msg_received, time_wait(irep)]= M_PTB.waitmessage(60);
                if isempty(msg_received)
                    params.is_online = false; experiment.is_online = 'online Matlab went offline';
                    warning(experiment.is_online);
                elseif strcmp(msg_received,msgs_Mat.error)
                    params.is_online = false; experiment.is_online = 'error in online Matlab';
                    warning(experiment.is_online);
                elseif ~strcmp(msg_received,msgs_Mat.process_ready)
                    disp('Inconsistency with messages sent')
                end
            end
            
            print_message(message_continue{ind_lang},black,window);
            
            Screen('FillRect',window,black);
            %         FlushEvents('keyDown');
            %             [~, ~, keyCode] = KbCheck(dev_used,scanlist);
            for d= dev_used; KbQueueFlush(d);   end
            
            [~,~,pressed,firstPress] = get_response(dev_used,device_resp,[exitKey continueKey],0.2,params.auto_resp,gamepad_ix);

            times(k)=GetSecs;
            if params.use_daq
                dig_out.send(trial_on); 
            end
            if params.use_BRK_comment
                cbmex('comment', msgs_colors{7}, 0, sprintf('%s seq%d',msgs{7},irep),'instance',params.which_nsp_comment-1);
            end
            if params.use_daq && with_reset
                WaitSecs('UntilTime', times(k)+wait_reset);
                dig_out.send(value_reset); 
            end
            k=k+1;
            for d= dev_used; KbQueueFlush(d);   end
            if pressed && firstPress(exitKey)>0, abort = true; break; end
        end
        break
    end
    %         ListenChar(0);
    print_message('THAT WOULD BE ALL.\n THANK YOU !!!',black,window)
    ttt=GetSecs;
    
    if params.is_online && abort
        M_PTB.send(msgs_Mat.exper_aborted);
        warning(msgs_Mat.exper_aborted);
    end
    
    save('RSVP_SCR_workspace','-regexp', '^(?!(M_PTB)$).');
        
    times(isnan(times))=[];
    inds_pics(inds_pics==0)=[];
    inds_start_seq(inds_start_seq==0)=[];
    
    
    for d = dev_used
        KbQueueStop(d);
        KbQueueRelease(d);
    end
    

    Priority(params.ptb_priority_normal);

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
    if params.acq_network
        recording.stop_and_close()
    elseif params.use_BRK_comment
        cbmex('close')
    end
    WaitSecs(10-(GetSecs-ttt));

    Screen('CloseAll');
    screen_closed = 1;
    ShowCursor;

    % try to backup data
    if params.backup_network
        reached_backup = true;

%         if k==1
%             rmdir(sprintf('%s%c%s',params.remote_disk_root,filesep,experiment.folder_name))
%             delete(sprintf('%s%c%s*',params.remote_disk_root,filesep,experiment.fname))
%         else
%             [status, msg_copy]= movefile(sprintf('%s%s*',experiment.fname), sprintf('Z:\\%s',experiment.folder_name));
%             if ~status
%                 disp(msg_copy)
%                 error('Check the data copied within the Blackrock PC as there was an error')
%             end
%             status = []; msg_copy={};
% 
%             if params.is_online
%                 disp('Waiting for online processing to be completed')
%                 msg_received = M_PTB.waitmessage(600);
%                 if strcmp(msg_received,msgs_Mat.error)
%                     params.is_online = false; experiment.is_online = 'error in online Matlab';
%                     warning(experiment.is_online);
%                 elseif ~strcmp(msg_received,msgs_Mat.process_end)
%                     warning('Inconsistency with messages sent')
%                 end
%             end

%             for i=1:length(data_transfer_copy)
%                 [status(end+1), msg_copy{end+1}] = copyfile(data_transfer_copy{i}, sprintf('Z:\\%s\\%s',experiment.folder_name,data_transfer_copy{i})); %overwrites if already exists
%             end
%             [~,noMatch] = regexp(pwd,filesep,'match','split');
% 
%             [status(end+1),msg_copy{end+1}] = copyfile(abs_path_pics, sprintf('Z:\\%s\\%s',params.remote_disk_root =,experiment.folder_name,[noMatch{end} '_pic'])); %overwrites if already exists
%             if exist('data_transfer_move','var')
%                 for i=1:length(data_transfer_move)
%                     [status(end+1),msg_copy{end+1}] = movefile(data_transfer_move{i}, sprintf('Z:\\%s\\%s',experiment.folder_name,data_transfer_move{i})); %overwrites if already exists
%                 end
%             end
%             if any(~status)
%                 for jj=find(status)
%                     disp(msg_copy{jj})
%                 end
%                 error('Check the data copied to the Blackrock PC as there was an error')
%             end
%             answer = questdlg('Do you want to backup the data now? If so, verify that the server link is up and running','Backup?','Yes','No','Yes');
%             if strcmp(answer,'Yes')
%                 bat_name = 'Z:\backup.bat';
%                 fid = fopen(bat_name,'w');
%                 %% CHECK PATH NAMES AND INCLUDE SUBJECT NAME THERE. INCLUDE _BACKUP IN THE FOLDER NAME FOR HDD D:
%                 fprintf(fid,'Xcopy /E "C:\\Users\\User\\Desktop\\DATA\\%s" "S:\\ECoG_Data\\%sDatafile\\DATA\\%s\\" \n',experiment.folder_name,sub_ID,experiment.folder_name);
%                 fprintf(fid,'robocopy "C:\\Users\\User\\Desktop\\DATA\\%s" "D:\\DATA BACKUP\\DATA\\%sDatafile_BACKUP\\%s" /E /XC /XN /XO \n',experiment.folder_name,sub_ID,experiment.folder_name);
%                 fclose(fid);
%                 msgbox('Go to Blackrock PC and drag "backup.bat" in DATA folder to a command prompt to finish the back up');
%             end
%         end

    end
    if params.use_daq
        dig_out.close();
    end
    custompath.rm()
    if params.backup_network
        rsvp_folder = fileparts(mfilename('fullpath'));
        status = []; msg_copy={};
        for i=1:length(data_transfer_copy)
            [status(end+1),msg_copy{end+1}] = copyfile(fullfile(rsvp_folder,data_transfer_copy{i}), fullfile(params.root_beh,experiment.folder_name,data_transfer_copy{i})); %overwrites if already exists
        end
        if MOVE_PICS
            movefile(abs_path_pics,fullfile(params.root_beh,experiment.folder_name,'pics_used'));
        else
            copyfile(abs_path_pics,fullfile(params.root_beh,experiment.folder_name,'pics_used'));
        end
        if any(~status)
            for jj=find(status)
                disp(msg_copy{jj})
            end
            error('Check the data copied as there was an error')
        end
    end
catch ME
    if params.is_online
        M_PTB.send(msgs_Mat.error);
    end   
    
    times(isnan(times))=[];
    inds_pics(inds_pics==0)=[];
    inds_start_seq(inds_start_seq==0)=[];
    try
        multiKbQueueCheck(dev_used);
        for d=dev_used
            KbQueueStop(d);
            KbQueueRelease(d);
        end
    catch
        disp('no keyboard queue active')
    end
%     Screen('CloseAll')
    Priority(params.ptb_priority_normal);
    experiment.times=times;
    experiment.t_stimon=t_stimon;
    experiment.t_fliptime=t_fliptime;
    experiment.t_DAQpic=t_DAQpic;
    experiment.inds_pics=inds_pics;
    experiment.inds_start_seq=inds_start_seq;
    experiment.answer=answer;
    experiment.times_break=times_break;
    experiment.cant_breaks=cant_breaks;
    experiment.ME=ME;
    save('RSVP_SCR_workspace','-regexp', '^(?!(M_PTB)$).');
    save('experiment_properties','experiment');
    if ~reached_backup
        if screen_closed == 0
            print_message('THAT WOULD BE ALL./n THANK YOU !!!',black,window)
            Screen('CloseAll');
            ShowCursor;
        end
        if params.acq_network 
            recording.stop_and_close()
        elseif params.use_BRK_comment
            cbmex('close')
        end
     end
%     ShowCursor;
    if params.use_daq
        dig_out.close()
    end
    custompath.rm()
    msgbox('There was an error in the script. Review the data saved and back it up manually if necessary');

    rethrow(ME)
end



function print_message(message,black,window)
    Screen('FillRect',  window,black);
    DrawFormattedText(window, message, 'center', 'center', [255 255 255]);
    Screen('Flip',window);
