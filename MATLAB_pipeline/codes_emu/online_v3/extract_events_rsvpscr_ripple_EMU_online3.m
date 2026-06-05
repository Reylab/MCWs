function extract_events_rsvpscr_ripple_EMU_online3(filename_NEV,which_dig)

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

ff=figure(12321);
set(ff,'Units','normalized', 'OuterPosition',[0 0 1 1]);
%% set paradigm pulses characteristics and create stimulus structure
matlab_times=0;
prev_pictures=0;
load 'experiment_properties_online3.mat'
pics_onset_cell = cell(numel(scr_config_cell),1);
should_be_array = zeros(numel(scr_config_cell),1);
for i =1:numel(scr_config_cell)

    [seq_length,NISI,~] = size(scr_config_cell{i}.order_pic);
    Nseq = scr_config_cell{i}.Nseq;


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

            bits = [experiment.pic(:)' experiment.lines_onoff experiment.lines_flip_blank experiment.lines_flip_pic experiment.blank_on experiment.trial_on];

            %the signature is:
            %err_off=DaqDOut(dio,0,data_signature_on);    % off
            %WaitSecs(0.05);                        % wait 50ms
            %err_on=DaqDOut(dio,0,data_signature_off);      % on
            %WaitSecs(0.45);                        % wait 450ms
            %err_off=DaqDOut(dio,0,data_signature_on);    % off
            %WaitSecs(0.05);                        % wait 50ms
            %err_on=DaqDOut(dio,0,data_signature_off);      % on
            %WaitSecs(0.45);                        % wait 450ms
            %err_off=DaqDOut(dio,0,data_signature_on);    % off
            %WaitSecs(0.05);                        % wait 50ms
            %err_on=DaqDOut(dio,0,data_signature_off);      % on

            %therefore, in t1 you need to look for two indices that will be two
            %consecutive even or odd indices (typically 1 3 but also 2 4 or 3 5).  That
            %might happen because sometimes the DAQ shows some extra pulses at the
            %beginning. In these two examples sig2ind=2.
            %if you see something like 1 2 5 7, then sig2ind=4, with 1 2 3 6 8 it
            %should be 5 and so on so forth.

            if ~matlab_times
                npulses_noflip=(NISI+1)+2+(seq_length*NISI)+1;
                should_be_array(1) = npulses_noflip*Nseq + sum(scr_config_cell{i}.nchanges_blank(1:Nseq)+scr_config_cell{i}.nchanges_pic(1:Nseq));
                should_be_pics = seq_length*NISI*Nseq;
                for scci =2:numel(scr_config_cell)
                    npulses_noflip=(NISI+1)+2+(seq_length*NISI)+1;
                    [seq_length,NISI,~] = size(scr_config_cell{scci}.order_pic);
                    should_be_array(scci) = npulses_noflip*scr_config_cell{scci}.Nseq + sum(scr_config_cell{scci}.nchanges_blank(1:scr_config_cell{scci}.Nseq)+scr_config_cell{scci}.nchanges_pic(1:scr_config_cell{scci}.Nseq));
                    should_be_pics = should_be_pics + seq_length*NISI*scr_config_cell{scci}.Nseq;
                end
                should_be = sum(should_be_array);
                borders = [0; cumsum(should_be_array)];
                extras = find(~ismember(Event_Value,bits));
                fprintf('There should be %d pulses. There are %d spurious. %d Events\n',should_be,length(extras),length(Event_Time))
                if should_be+length(extras) == length(Event_Time)
                    Event_Time(extras) = [];
                    Event_Value(extras) = [];

                    inds_pics_onset = find(ismember(Event_Value,experiment.pic(:)));
                    num_pics_onset = numel(inds_pics_onset);
                    fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,num_pics_onset)
                    allpics_events = Event_Time(inds_pics_onset);
                    %pics onset should be a cell

                else
                    disp('Check what happened with spurious pulses. Will use matlab times instead')
                    matlab_times = 1;
                end
            end
            %going bavk for i=1
            %             [seq_length,NISI,~] = size(scr_config_cell{scci}.order_pic);
        elseif strcmp(which_dig,'SMA 1')
            allpics_events=events(1:2:end);
            Event_Value = val(1:2:end);
            should_be_pics = seq_length*NISI*Nseq;
            for scci =2:numel(scr_config_cell)
                [seq_length,NISI,~] = size(scr_config_cell{scci}.order_pic);
                should_be_pics = should_be_pics + seq_length*NISI*scr_config_cell{scci}.Nseq;
            end
            fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,numel(allpics_events))
        end
    end
    npics = seq_length * NISI*Nseq;
    pics_onset = reshape(allpics_events((1:npics)+prev_pictures),seq_length,NISI,Nseq);
    prev_pictures = prev_pictures + npics;

    %% with Ripple times
    clf(ff)        
    if ~matlab_times && strcmp(which_dig,'Parallel Input')

        ss_Event_Time = Event_Value(borders(i)+1: borders(i+1));
        TT=ss_Event_Time(ismember(ss_Event_Time,[experiment.pic(:);experiment.blank_on]));
        
        tdiff=diff(TT);
        blank_out = tdiff((seq_length+1)*NISI+1:(seq_length+1)*NISI+1:end);
        tdiff((seq_length+1)*NISI+1:(seq_length+1)*NISI+1:end)=[];

        blank_in = tdiff(1:seq_length+1:end);
        tdiff(1:seq_length+1:end)=[];

        dura_blank=blank_in;

        dura_pics = tdiff(:);
        tpic_teo=[];
        for ij=1:NISI
            tpic_teo = [tpic_teo ; repmat(scr_config_cell{i}.order_ISI(ij,:),seq_length,1)];
        end
        tpic_diff = dura_pics - 1000*scr_config_cell{i}.ISI(reshape(tpic_teo(:,1:Nseq),size(tpic_teo,1)*Nseq,1));

        subplot(121)
        plot(dura_blank,'m'); %should be 1500-1850 ms
        xlim([0 (NISI+1)*Nseq+1])
        h_legend=legend('blank duration (ms)');legend('boxoff')

        subplot(122)
        tpic_diff = diff(squeeze(pics_onset));
        plot(tpic_diff(:)-500,'b'); %should be 0 ms

        xlim([0 seq_length*Nseq+1])
        h_legend=legend('stimulus "error" (ms)','location','best');legend('boxoff')

        set(gcf,'PaperPositionMode','auto')
        drawnow
        title(['ttl_rsvp_subscr_' num2str(i)])

    elseif strcmp(which_dig,'SMA 1')        
        tpic_diff = diff(squeeze(pics_onset));
        plot(tpic_diff(:)-500,'b'); %should be 0 ms

        xlim([0 seq_length*Nseq+1])
        h_legend=legend('stimulus "error" (ms)','location','best');legend('boxoff')

        set(gcf,'PaperPositionMode','auto')
        drawnow
        title(['ttl_rsvp_subscr_' num2str(i)])
    end
    keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');

    if strcmp(keyval,'y')
        %% save figure and events with Ripple times
        print(['ttl_rsvp_subscr_' num2str(i) '.png'], '-dpng' );
        pics_onset_cell{i} = pics_onset;
    else
        matlab_times = 1;
        disp('Matlab times will be used now ...')
    end
end

%% recover time event from matlab times
if matlab_times
    error('matlab time not tested')
    %this need a for like the one for the hardware events
    should_be = npulses_noflip*Nseq + sum(scr_config_cell{i}.nchanges_blank(1:Nseq)+scr_config_cell{i}.nchanges_pic(1:Nseq));

    pulses_skipped = 2;
    times = experiment.times(pulses_skipped+1:end);
    if should_be==length(times)
        inds_pics_onset = experiment.inds_pics-pulses_skipped;
        inds_start_seq = experiment.inds_start_seq-pulses_skipped;
        Event_Time = (times-experiment.times(2))*1e3+t_last_signat; % assumes that the signature was recovered and Event_Time is in miliseconds
        Event_Value = [];
    else
        error('Check why the number of times is not right')
    end
    num_pics_onset=numel(inds_pics_onset);
    should_be_pics = seq_length*NISI*Nseq;
    fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,num_pics_onset)
    pics_onset = reshape(Event_Time(inds_pics_onset),seq_length,NISI,Nseq);

    tbase_end = Event_Time(inds_start_seq);


    AA=inds_pics_onset(diff(inds_pics_onset)>2)+1;
    TT=[Event_Time(sort([inds_pics_onset AA])) Event_Time(inds_pics_onset(end)+1)];
    figure
    set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
    subplot(122)

    tdiff=diff(TT);
    tdiff(seq_length+1:seq_length+1:end)=[];
    dura_pics = tdiff(:);
    tpic_teo=[];
    for ij=1:NISI
        tpic_teo = [tpic_teo ; repmat(scr_config_cell{i}.order_ISI(ij,:),seq_length,1)];
    end
    tpic_diff = dura_pics - 1000*scr_config_cell{i}.ISI(reshape(tpic_teo(:,1:Nseq),size(tpic_teo,1)*Nseq,1));

    plot(tpic_diff,'b'); %should be 0 ms
    xlim([0 (seq_length-1)*Nseq+1])
    h_legend=legend('stimulus "error" (ms)','location','best');legend('boxoff')

    set(gcf,'PaperPositionMode','auto')

    keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');

    if strcmp(keyval,'y')
        %% save figure and events with Matlab times
        print(['ttl_rsvp_subscr_' num2str(i) '.png'], '-dpng' );
        pics_onset_cell{i} = pics_onset;

    end
end
close all

pics_onset = pics_onset_cell;
save finalevents pics_onset