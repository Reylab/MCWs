function extract_blank_on_events_ripple(filename_NEV,which_dig)

    if ~exist('which_dig','var')|| isempty(which_dig),  which_dig = 'Parallel Input'; end % 'Parallel Input' or 'SMA 1'
    
    [ns_status, hFile] = ns_OpenFile(['./' filename_NEV], 'single');
    
    entity_ix = find(strcmp({hFile.Entity.Reason},which_dig));
    nevents = hFile.Entity(entity_ix).Count;
    events = zeros(nevents,1); %in seconds
    val = zeros(nevents,1);
    for ei = 1:nevents
        if strcmp(which_dig,'Parallel Input')
            [~, events(ei), val(ei), ~] = ns_GetEventData(hFile,entity_ix,ei);
        elseif strcmp(which_dig,'SMA 1')
            [~, events(ei)] = ns_GetEventData(hFile,entity_ix,ei);
            val(ei) = NaN;
        end
    end
    events   = events*1000;
    
    %% set paradigm pulses characteristics and create stimulus structure
    matlab_times=0;
    seq_start = 1;
    load 'experiment_properties_online3.mat'
    Nscr = numel(scr_config_cell);
    n_scr_ended = min(Nscr, numel(scr_end_cell));

    if Nscr>n_scr_ended
        warning('some subscreening not ended, using just the completed screenings');
        Nscr = n_scr_ended;
    end
    seq_beg_blanks_cell = cell(Nscr, 1);
    seq_end_blanks_cell = cell(Nscr, 1);
    for i =1:Nscr
        Nseq = numel(scr_end_cell{i}.inds_start_seq) - scr_end_cell{i}.abort;
        %% look for initial signature
        if i==1
            if strcmp(which_dig,'Parallel Input')
                inds_signature = find(val == experiment.data_signature(1)); %it should be (1 3 5)
                fprintf('inds_signature = %s\n',num2str(inds_signature))
                t = diff(events); %time interval in msec
                fprintf(['t(1:7) = ' repmat('%2.f ',1,7) ' ms \n'],t(1:7))
                if isequal(inds_signature(:),[1 3 5]') ||isequal(inds_signature(:),[2 4 6]')
                    Event_Time=events(inds_signature(3)+2:end);
                    t_last_signat = events(inds_signature(3)+1);
                    Event_Value = val(inds_signature(3)+2:end);
                else
                    error('Check the signature')
                end
                if sum(unique(Event_Value) == 15) == 0 % No continue message daq signal until patient MCW-FH_016
                    blank_seq_beg_str = [experiment.blank_on, experiment.lines_onoff]; % experiment.blank_on = 11, experiment.lines_onoff = 13
                    inds_blank_on_seq_beg = strfind(Event_Value', blank_seq_beg_str);
                    % remove every 2nd blank_on event
                    inds_blank_on_seq_beg = inds_blank_on_seq_beg(1:2:end);

                    idx_blank_on = find(Event_Value == experiment.blank_on);
                    idx_lines_onoff = find(Event_Value == experiment.lines_onoff);
                    idx_trial_on = find(Event_Value == experiment.trial_on);

                else
                    blank_seq_beg_str = [experiment.blank_on, experiment.lines_onoff];
                    blank_seq_end_str = [experiment.blank_on, experiment.continue_msg_on];
                    inds_blank_on_seq_beg = strfind(Event_Value', blank_seq_beg_str);
                    inds_blank_on_seq_end = strfind(Event_Value', blank_seq_end_str);
                    seq_end_blanks = cell(numel(inds_blank_on_seq_end),1);
                    for idx=1:numel(inds_blank_on_seq_end)
                        blank_on_idx = inds_blank_on_seq_end(idx);
                        seq_end_blanks{idx} = Event_Time(blank_on_idx:blank_on_idx+1);
                    end

                    idx_blank_on = Event_Value == experiment.blank_on; % experiment.blank_on = 11
                    idx_lines_onoff = Event_Value == experiment.lines_onoff; % experiment.lines_onoff = 13
                    idx_trial_on = Event_Value == experiment.trial_on; % experiment.trial_on = 26
                    idx_continue_msg_on = Event_Value == experiment.continue_msg_on; % experiment.continue_msg_on = 15
                    continue_msg_on_times = Event_Time(idx_continue_msg_on);
                    save('finalevents.mat', 'continue_msg_on_times','-append')
                    
                end
                
                seq_beg_blanks = cell(numel(inds_blank_on_seq_beg),1);
                for idx=1:numel(inds_blank_on_seq_beg)
                    blank_on_idx = inds_blank_on_seq_beg(idx);
                    seq_beg_blanks{idx} = Event_Time(blank_on_idx:blank_on_idx+1);
                end

                blank_on_times = Event_Time(idx_blank_on);
                lines_onoff_times = Event_Time(idx_lines_onoff);
                
                trial_on_times = Event_Time(idx_trial_on);
            end
        end
        seq_beg_blanks_cell{i} = seq_beg_blanks(seq_start:seq_start+Nseq-1);
        if exist("seq_end_blanks","var")
            seq_end_blanks_cell{i} = seq_end_blanks(seq_start:seq_start+Nseq-1);
        end
        seq_start = seq_start + Nseq;
    end
    save('finalevents.mat', 'seq_beg_blanks_cell', 'blank_on_times', ...
         'lines_onoff_times', 'trial_on_times', '-append')

    if exist("seq_end_blanks", "var") % Patient 16 onwards
        save('finalevents.mat', 'seq_end_blanks_cell', '-append')
    end