%Script for running subjects in the voluntary OBA project
%This is the CUED MRI version
%Written by David Hughes
%Last updated on 5/2/23
%RISE algorithm adapted from code written by Mike Esterman for 2010 paper

subName = params.sub_ID;
%% Experiment time
cue_dur = frame_dur/rate/2;
num_frames = frame_dur/rate;

%Create keyboard queue for only response keys
if button_config < 3
    KbQueueCreate(kb_dev_id(1), key_list_resp);                  % Create Queue
    KbQueueStart(kb_dev_id(1));
end  
clear trial_frames;
f_pressed = 0;
h_pressed = 0;

%% Stim presentation and response collection
for trial = 1:num_trials

    %grab trial type and set rectangle color for this trial
    trl_type = trial_list{trial, 2};
    
    if trl_type == 2 || trl_type == 3 || trl_type == 5 || trl_type ==6
        if trial_list{trial, 6} == 1
            %Attend to left face
            cue_text = '#';
        elseif trial_list{trial, 6} == 2
            %Attend to left house
            cue_text = '%';
        elseif trial_list{trial, 6} == 3
            %Attend to right face
            cue_text = '?';
        elseif trial_list{trial, 6} == 4
            %Attend to right house
            cue_text = '@';
        end
        cue_color = black;
    else
        cue_text = '';
        cue_color = white;
    end
    
    %% This is the start of stimulus presentation
    if trial == 1
        vbl = Screen('Flip', window);
    end
    
    for f = 1:frame_dur/rate
        %Face House Stim
        lstim = lstimuli(f + (frame_dur/rate)*(trial-1));
        rstim = rstimuli(f + (frame_dur/rate)*(trial-1));
        Screen('DrawTexture', window, lstim, [], left_rec);
        Screen('DrawTexture', window, rstim, [], right_rec);
        if f == 1
            Screen('FillRect', window, white, timer_rec);
        else
            Screen('FillRect', window, black, timer_rec);
        end
   
        
        if(f < (frame_dur/rate/2)) && ~(strcmpi(cue_text, ''))
        %Fixation Dot
            DrawFormattedText(window, cue_text, 'center', 'center', cue_color);
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
        Screen('Close', lstim);
        Screen('Close', rstim);
        
        if gp_state_f && ~f_pressed
            f_pressed = 1;
            f_time = onset1;
        elseif gp_state_h && ~h_pressed
            h_pressed = 1;
            h_time = onset1;
        end
    end
    
    trial_end(trial) = onset1 - expStart;
     %Record the trial data into out data matrix
        %Each row is a trial, c1 is the trial number, c2 is trial type,
        %c3 is the attended stream, c4 is the face file for the trial,
        %c5 is the house trial for the trial, c6 is the response the
        %subject made, c7 is the RT
    respMat{trial, 1} = trial;
    respMat{trial, 2} = trl_type;
    respMat{trial, 3} = strm_trck;
    
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
            if (mod(strm_trck, 2) == 1)
                num_hits = num_hits + 1;
            end
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
            if (mod(strm_trck, 2) == 0)
                num_hits = num_hits + 1;
            end
            targ_flag = 0;
        end
    end
    
    respMat{resp_to_check, 4} = response;
    respMat{resp_to_check, 5} = rt;
        
   if trl_type == 2
        if mod(strm_trck, 2) == 1
            strm_trck = strm_trck + 1;
        else
            strm_trck = strm_trck - 1;
        end
    elseif trl_type == 5
        if strm_trck == 1
            strm_trck = 3;
        elseif strm_trck == 2
            strm_trck = 4;
        elseif strm_trck == 3
            strm_trck = 1;
        elseif strm_trck == 4
            strm_trck = 2;
        end
   elseif trl_type == 6
       if strm_trck == 1
           strm_trck = 4;
       elseif strm_trck == 2
           strm_trck = 3;
       elseif strm_trck == 3
           strm_trck = 2;
       elseif strm_trck == 4
           strm_trck = 1;
       end
    end
    
    prev_trial = trl_type;
    respMat{trial, 6} = trial_start(trial);
    respMat{trial, 7} = trial_end(trial);
end

if button_config < 3
    KbQueueFlush(kb_dev_id);
    KbQueueStop(kb_dev_id(1));
    KbQueueRelease(kb_dev_id(1));
end

% ShowCursor;

save_file_beh = ['iEEG_' subName '_results_coh' num2str(coh_thresh) '_blk' num2str(blk_num) '.mat'];

%save initial accuracy estimates - entry 1 is num correct for std trials,
%entry 2 is num correct for shift trials, entry 3 is misses, entry 4 is
%false alarms, and 5 is the overall accuracy
targ_acc = (num_hits)/(num_hits+num_misses)*100;
acc_est = [num_cor_rej, num_hits, num_misses, num_false_alarms, targ_acc];
if exist(save_file_beh) == 0
    save(save_file_beh, 'respMat', 'acc_est', 'targs', 'trial_list', 'expStart',...
        'targ_viewing_time');
else
    new_save_file = ['iEEG_' subName '_results_blk' num2str(blk_num) '_CHECK.mat'];
    disp(['Tried saving as: ' save_file_beh '; saved as: ' new_save_file]);
    save(new_save_file, 'respMat', 'acc_est', 'targs', 'trial_list', 'expStart',...
         'targ_viewing_time');
end
%copyfile save_file_beh 
clear acc_est trial_list targ_viewing_time;