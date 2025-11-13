%Script for running subjects in the voluntary OBA project
%Written by David Hughes
%Last updated on 7/15/22
%RISE algorithm adapted from code written by Esterman for 2010 paper
%This version doesn't have any QUEST code in it. This familiarizes subjects
%with the task and provides feedback on target trials and false alarms. It
%does not provide feedback on correct rejections.x

subName = params.sub_ID;

%Hide cursor during image presentation
HideCursor;
fdbck_flg = 0;
cue_dur = frame_dur/rate/2;
num_frames = frame_dur/rate;

f_pressed = 0;
h_pressed = 0;
bw_frames = round(6/ifi);
%Create keyboard queue for only response keys
if button_config < 3
    KbQueueCreate(kb_dev_id(1), key_list_resp);                  % Create Queue
    KbQueueStart(kb_dev_id(1));
end
clear trial_frames;
streams = [ones([1 24]), 2*ones([1 24]), ones([1 24]), 2*ones([1 24]),...
           ones([1 24]), 2*ones([1 24])];
%% Stim presentation and response collection
for trial = 1:num_trials
    if streams(trial) == 1
        targ = targ_face;
    else
        targ = targ_house;
    end
    %grab trial type and set rectangle color for this trial
    
    %Set rectangle color. Trial type key:
        %1=Target, 2=Shift, 3=Hold, 4=Standard
    trl_type = trial_list{trial, 2};  
    if trial == 1
        vbl = Screen('Flip', window);
    end
    for f = 1:frame_dur/rate
        %Face House Stim
        lstim = lstimuli(trial);
        Screen('PutImage', window, targ, train_rec);
        Screen('DrawTexture', window, lstim,[], c_rec_rect);
        if f == 1
            Screen('FillRect', window, white, timer_rec);
        else
            Screen('FillRect', window, black, timer_rec);
        end
        
        if(f < (frame_dur/rate/2)) && fdbck_flg
        %Fixation Dot
            DrawFormattedText(window, '$', 'center', 'center', white);
        else
            Screen('DrawDots', window, [xCenter yCenter], dotSizePix,...
                white, [], 2);
        end
  
        [vbl, onset1] = Screen('Flip', window, vbl + wait_time);
        
        if(f == 1)
            resp_start(trial) = onset1;
            trial_start(trial) = onset1 - expStart;
            dig_out.send(blank_on);
        elseif(f==2)
            dig_out.send(value_reset);
        end
        
        gp_state_f = Gamepad('GetButton', game_pad_index, f_key); %face response
        gp_state_h = Gamepad('GetButton', game_pad_index, h_key); %house response
 
        if gp_state_f && ~f_pressed
            f_pressed = 1;
            f_time = onset1;
        elseif gp_state_h && ~h_pressed
            h_pressed = 1;
            h_time = onset1;
        end
    end
    fdbck_flg =0;
    Screen('Close', lstim);
    
    trial_end(trial) = onset1 - expStart;
    %Record the trial data into out data matrix
        %Each row is a trial, c1 is the trial number, c2 is trial type,
        %c3 is the attended stream, c4 is the face file for the trial,
        %c5 is the house trial for the trial, c6 is the response the
        %subject made, c7 is the RT
    respMat{trial, 1} = trial;
    respMat{trial, 2} = trl_type;
    respMat{trial, 3} = streams(trial);

    if trl_type == 1
        targ_flag = 1;
        targ_t_num = trial;
    end
    
    if targ_flag
        if (trial-targ_t_num) > 2
            num_misses = num_misses + 1;
            targ_flag = 0;
        end
        resp_to_check = targ_t_num;
    else
        resp_to_check = trial;
    end

    %Check for response
    response = 0;
    rt = 0;
    if f_pressed
        response = 1;
        rt       = f_time - resp_start(resp_to_check);
        f_pressed = 0;
        gp_state_f = 0;
        
        if ~targ_flag
            num_false_alarms = num_false_alarms + 1;
        elseif targ_flag
            if (mod(streams(trial), 2) == 1)
                num_hits = num_hits + 1;
            end
            fdbck_flg = 1;
            targ_flag = 0;
        end
    elseif h_pressed
        response = 2;
        rt       = h_time - resp_start(resp_to_check);
        h_pressed = 0;
        gp_state_h = 0;
        
        if ~targ_flag
            num_false_alarms = num_false_alarms + 1;
        elseif targ_flag
            if (mod(streams(trial), 2) == 0)
                num_hits = num_hits + 1;
            end
            fdbck_flg = 1;
            targ_flag = 0;
        end
    end
     
    respMat{trial, 6} = trial_start(trial);
    respMat{trial, 7} = trial_end(trial);
    if mod(trial, 24) == 0
         for frame = 1:bw_frames
            Screen('DrawDots', window, [xCenter yCenter], dotSizePix, white, [], 2);
            Screen('FillRect', window, black, timer_rec);
            [VBLTimestamp, StimulusOnsetTime1] = Screen('Flip', window);
         end
    end
end
if button_config < 3
    KbQueueFlush(kb_dev_id);
    KbQueueStop(kb_dev_id(1));
    KbQueueRelease(kb_dev_id(1));
end
ShowCursor;

save_file_beh = ['TRAINING_' subName '_results' num2str(blk_num) '.mat'];

%save initial accuracy estimates - entry 1 is num correct for std trials,
%entry 2 is num correct for shift trials, entry 3 is misses, entry 4 is
%false alarms, and 5 is the overall accuracy
targ_acc = (num_hits)/(num_hits+num_misses)*100;
acc_est = [num_cor_rej, num_hits, num_misses, num_false_alarms, targ_acc];
save(save_file_beh, 'respMat', 'acc_est', 'targs', 'trial_list');
%copyfile save_file_beh 
clear acc_est trial_list expStart targ_viewing_time;