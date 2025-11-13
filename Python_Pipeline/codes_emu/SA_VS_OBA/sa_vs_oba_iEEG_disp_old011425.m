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
    
    %Set rectangle color. Trial type key:
     if trl_type == 4 || trl_type == 1
        %standard color (white)
        rect_color = white;
    elseif trl_type == 3
        %hold
        rect_color = hold;
    elseif trl_type == 2
        %shift object
        rect_color = shift_obj;
    elseif trl_type == 5
        %shift location
        rect_color = shift_loc;
     end
    
    %% This is the start of stimulus presentation

    %First frame
    lstim = lstimuli(1 + (num_frames)*(trial-1));
    rstim = rstimuli(1 + (num_frames)*(trial-1));
    
    Screen('DrawTexture', window, lstim, [], left_rec);
    Screen('DrawTexture', window, rstim, [], right_rec);
    Screen('FillRect', window, white, timer_rec);
    Screen('DrawDots', window, [xCenter yCenter], dotSizePix, rect_color, [], 2);
    
    if trial == 1
        vbl = Screen('Flip', window);
    end
    
    [vbl, onset1] = Screen('Flip', window, vbl + wait_time);
    
    dig_out.send(blank_on);
    
    
    gp_state_f = Gamepad('GetButton', game_pad_index, f_key); %face response
    gp_state_h = Gamepad('GetButton', game_pad_index, h_key); %house response
    if gp_state_f && ~f_pressed
        f_pressed = 1;
        f_time = onset1;
    elseif gp_state_h && ~h_pressed
        h_pressed = 1;
        h_time = onset1;
    end
    
    resp_start(trial) = onset1;
    trial_start(trial) = onset1 - expStart;
    
    Screen('Close', lstim);
    Screen('Close', rstim);
    
    %Cue presentation period
    for f = 2:(cue_dur-1)
        %Face House Stim
        lstim = lstimuli(f + (num_frames)*(trial-1));
        rstim = rstimuli(f + (num_frames)*(trial-1));
        
        Screen('DrawTexture', window, lstim, [], left_rec);
        Screen('DrawTexture', window, rstim, [], right_rec);
        Screen('FillRect', window, black, timer_rec);
        Screen('DrawDots', window, [xCenter yCenter], dotSizePix, rect_color, [], 2);
    
        [vbl, onset1] = Screen('Flip', window, vbl + wait_time);
        if f == 2
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
        
        Screen('Close', lstim);
        Screen('Close', rstim);
        
    end
    
    %Full coherence period
    for f = cue_dur:num_frames
        %Face House Stim
        lstim = lstimuli(f + (num_frames)*(trial-1));
        rstim = rstimuli(f + (num_frames)*(trial-1));
        
        Screen('DrawTexture', window, lstim, [], left_rec);
        Screen('DrawTexture', window, rstim, [], right_rec);
        Screen('FillRect', window, black, timer_rec);
        Screen('DrawDots', window, [xCenter yCenter], dotSizePix, white, [], 2);
       
        [vbl, onset1] = Screen('Flip', window, vbl + wait_time);
        
        gp_state_f = Gamepad('GetButton', game_pad_index, f_key); %face response
        gp_state_h = Gamepad('GetButton', game_pad_index, h_key);
        if gp_state_f && ~f_pressed
            f_pressed = 1;
            f_time = onset1;
        elseif gp_state_h && ~h_pressed
            h_pressed = 1;
            h_time = onset1;
        end
        
        Screen('Close', lstim);
        Screen('Close', rstim);
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
    %Check for response
    response = 0;
    rt = 0;
    if ~targ_flag
        if trl_type ~= 1
            %not a target trial so check for a response
            if f_pressed
                num_false_alarms = num_false_alarms + 1;
                
                response = 1;
                rt       = f_time - resp_start(resp_to_check);
                
                f_pressed = 0;
                gp_state_f = 0;
            elseif h_pressed
                num_false_alarms = num_false_alarms + 1;
                
                response = 2;
                rt       = h_time - resp_start(resp_to_check);
                
                h_pressed = 0;
                gp_state_h = 0;
            else
                num_cor_rej = num_cor_rej + 1;
            end
            resp_to_check = trial;
        else
            targ_flag = 1;
            targ_t_num = trial;
            time_elapsed = time_elapsed+frame_dur;
        end
        respMat{resp_to_check, 4} = response;
        respMat{resp_to_check, 5} = rt;
    else
        %target trial so wait for time elapsed
        if time_elapsed >= resp_time && targ_flag
            %Check for a button press and figures out which button
            %was pressed if there was one
            if f_pressed
                response = 1;
                rt       = f_time - resp_start(resp_to_check);
                
                f_pressed = 0;
                gp_state_f = 0;
                
                if (mod(strm_trck, 2) == 1)
                    num_hits = num_hits + 1;
                end
            elseif h_pressed
                response = 2;
                rt       = h_time - resp_start(resp_to_check);
              
                h_pressed = 0;
                gp_state_h = 0;
                
                if (mod(strm_trck, 2) == 0)
                    num_hits = num_hits + 1;
                end
            else
                num_misses = num_misses + 1;
            end
            
            resp_to_check = resp_to_check + 1;
            respMat{resp_to_check, 4} = response;
            respMat{resp_to_check, 5} = rt;
            time_elapsed = 0;
            targ_flag = 0;
        else
            time_elapsed = time_elapsed+frame_dur;
            respMat{trial, 4} = response;
            respMat{trial, 5} = rt;
            targ_flag = 1;
        end
    end
    
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

ShowCursor;

save_file = ['iEEG_' subName '_results_blk_' num2str(blk_num) '.mat'];

%save initial accuracy estimates - entry 1 is num correct for std trials,
%entry 2 is num correct for shift trials, entry 3 is misses, entry 4 is
%false alarms, and 5 is the overall accuracy
targ_acc = (num_hits)/(num_hits+num_misses)*100;
acc_est = [num_cor_rej, num_hits, num_misses, num_false_alarms, targ_acc];
if exist(save_file) == 0
    save(save_file, 'respMat', 'acc_est', 'targs', 'trial_list', 'expStart',...
        'targ_viewing_time');
else
    new_save_file = ['iEEG_' subName '_results_blk_CHECK.mat'];
    disp(['Tried saving as: ' save_file '; saved as: ' new_save_file]);
    save(new_save_file, 'respMat', 'acc_est', 'targs', 'trial_list', 'expStart',...
         'targ_viewing_time');
end

clear acc_est trial_list expStart targ_viewing_time;