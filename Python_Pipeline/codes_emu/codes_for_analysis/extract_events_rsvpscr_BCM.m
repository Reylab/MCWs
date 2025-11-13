function extract_events_rsvpscr_BCM(NEV)
% this shouldn't be used as a function. Evaluate cell by cell
% HGR JAN14 signature was slightly changed, including 50 ms pause between
% previously consecutive pulses
%% set paradigm pulses characteristics and create stimulus structure
matlab_times=0;
photo_corrected = 1;
load 'experiment_properties.mat'
[seq_length,NISI,~] = size(experiment.order_pic);
Nseq = length(experiment.inds_pics)/seq_length;
create_stimulus_struct_rsvpscr(experiment,Nseq);
% load stimulus;
% Npic=length(experiment.ImageNames);
npulses_noflip=(NISI+1)+2+(seq_length*NISI)+1;%number of events per trial
if experiment.with_reset
    npulses_reset = npulses_noflip;
else
    npulses_reset=0;
end

%% read events from NEV structure
events   = NEV.Data.SerialDigitalIO.TimeStampSec*1000; %data in msec
events_TS   = NEV.Data.SerialDigitalIO.TimeStamp;
val        = NEV.Data.SerialDigitalIO.UnparsedData;
%% look for initial signature

inds_signature = find(val == experiment.data_signature(1)); %it should be (1 3 5)
fprintf('inds_signature = %s\n',num2str(inds_signature))
t = diff(events); %time interval in msec
fprintf('t(1:7) = %s\n',num2str(t(1:7)))

if isequal(inds_signature(:),[1 3 5]') ||isequal(inds_signature(:),[2 4 6]')
    Event_Time=events(inds_signature(3)+2:end);
    events_TS = events_TS(inds_signature(3)+2:end);
    t_last_signat = events(inds_signature(3)+1);
    Event_Value = val(inds_signature(3)+2:end);
else
    error('Check the signature')
end
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

%% check for spurious pulses and remove them

% Event_Time=Event_Time*1000; %in microsecs

bits = [experiment.pic(:)' experiment.lines_onoff experiment.lines_flip_blank experiment.lines_flip_pic experiment.blank_on experiment.trial_on];

if ~matlab_times
    if npulses_reset>0
        pulses_reset_shouldbe = npulses_noflip*Nseq + sum(experiment.nchanges_blank(1:Nseq)+experiment.nchanges_pic(1:Nseq));
        num_pulses_reset = sum(ismember(Event_Value,experiment.value_reset));
        fprintf('There should be %d reset pulses and there are %d\n',pulses_reset_shouldbe,num_pulses_reset)
        extras_reset = find(ismember(Event_Value,experiment.value_reset));
        if pulses_reset_shouldbe == num_pulses_reset
            Event_Time(extras_reset) = [];
            events_TS(extras_reset) = [];
            Event_Value(extras_reset) = [];
        else
            disp('Check what happened with pulses_reset. Will use matlab times instead')
            matlab_times = 1;
        end
    end
end

if ~matlab_times
    should_be = npulses_noflip*Nseq + sum(experiment.nchanges_blank(1:Nseq)+experiment.nchanges_pic(1:Nseq));
    extras = find(~ismember(Event_Value,bits));
    fprintf('There should be %d pulses. There are %d spurious. %d Events\n',should_be,length(extras),length(Event_Time))
    if should_be+length(extras) == length(Event_Time)
        Event_Time(extras) = [];
        events_TS(extras) = [];
        Event_Value(extras) = [];
    else
        disp('Check what happened with spurious pulses. Will use matlab times instead')
        matlab_times = 1;
    end
    if ~matlab_times
        inds_pics_onset = find(ismember(Event_Value,experiment.pic(:)));
        num_pics_onset=numel(inds_pics_onset);
        should_be_pics = seq_length*NISI*Nseq;
        fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,num_pics_onset)
        pics_onset = reshape(Event_Time(inds_pics_onset),seq_length,NISI,Nseq);
    end
    
%%
%     AA=find(Event_Value==11);
%     %%
%     i=38
%     find(ismember(Event_Value(AA(2*i-1):AA(2*i)),experiment.pic(:)))
%     %%
% 
%     for i=1:45
%         CC=find(ismember(Event_Value(AA(2*i-1):AA(2*i)),experiment.pic(:)));
%         fprintf('seq %d. num pics %d\n',i,numel(CC))
%     end
%%

    if photo_corrected
        channel = 257;
%         DAQ_pic_timestamp = Event_Time(inds_pics_onset)/1000*30000;
        DAQ_pic_timestamp = events_TS(inds_pics_onset);
        photo_pic_timestamp = zeros(size(DAQ_pic_timestamp));
        load('NSx.mat','NSx')
        posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
        f1 = fopen(sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext),'r','l');
        ch257 = fread(f1,'int16=>double')*NSx(posch).conversion;
        window_search = 0.75*experiment.ISI*30000;
        figure
        plot((1:length(ch257))/30,ch257)
        hold on
        for j=1:numel(DAQ_pic_timestamp)
            TS_DAQ = DAQ_pic_timestamp(j);
            [~,indM] = max(ch257(TS_DAQ:TS_DAQ+round(window_search)));
            photo_new=findchangepts(ch257(TS_DAQ:TS_DAQ+indM),'statistic','std');
            photo_pic_timestamp(j) = TS_DAQ+photo_new;
            line([photo_pic_timestamp(j)/30 photo_pic_timestamp(j)/30],ylim,'color','c')
        end        
        pics_onset = reshape(photo_pic_timestamp/30000*1000,seq_length,NISI,Nseq);
    end
end

%% with Blackrock times
if ~matlab_times
    tblank = Event_Time(ismember(Event_Value,experiment.blank_on));
    tbase_end = tblank(1:NISI+1:end);
    tlines_flip_blank = Event_Time(ismember(Event_Value,experiment.lines_flip_blank));
    tlines_flip_pic = Event_Time(ismember(Event_Value,experiment.lines_flip_pic));
    ttrial_on = Event_Time(ismember(Event_Value,experiment.trial_on));
    tlines_onoff = Event_Time(ismember(Event_Value,experiment.lines_onoff));
    TT=Event_Time(ismember(Event_Value,[experiment.pic(:);experiment.blank_on]));
    
    % produce figure to check pulses
    figure
    set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
    
%     tdiff=diff(TT)/1000;
    tdiff=diff(TT);
    blank_out = tdiff((seq_length+1)*NISI+1:(seq_length+1)*NISI+1:end);
    tdiff((seq_length+1)*NISI+1:(seq_length+1)*NISI+1:end)=[];
    
    blank_in = tdiff(1:seq_length+1:end);
    tdiff(1:seq_length+1:end)=[];
    
    dura_blank=blank_in;
    
    dura_pics = tdiff(:);
    tpic_teo=[];
    for i=1:NISI
        tpic_teo = [tpic_teo ; repmat(experiment.order_ISI(i,:),seq_length,1)];
    end
    tpic_diff = dura_pics - 1000*experiment.ISI(reshape(tpic_teo(:,1:Nseq),size(tpic_teo,1)*Nseq,1));
%     tpic_diff = dura_pics - experiment.ISI(tpic_teo(:));
    
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
    
    keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');
    
    if strcmp(keyval,'y')
        %% save figure and events with Blackrock times
        print -dpng ttl_rsvp.png
        save finalevents Event_Time Event_Value pics_onset ttrial_on tlines_onoff tlines_flip_pic tlines_flip_blank tblank dura_blank dura_pics tbase_end
    else
        matlab_times = 1;
        disp('Matlab times will be used now ...')
    end
end

%% recover time event from matlab times
if matlab_times
    should_be = npulses_noflip*Nseq + sum(experiment.nchanges_blank(1:Nseq)+experiment.nchanges_pic(1:Nseq));
%     times = experiment.times(2:end);
    pulses_skipped = 2;
    times = experiment.times(pulses_skipped+1:end);
    if should_be==length(times)
        inds_pics_onset = experiment.inds_pics-pulses_skipped;
        inds_start_seq = experiment.inds_start_seq-pulses_skipped;
%         Event_Time = (times-times(1))*1e6+Event_Time(1); % assumes that the signature was recovered and Event_Time is in microseconds
        Event_Time = (times-experiment.times(2))*1e3+t_last_signat; % assumes that the signature was recovered and Event_Time is in miliseconds
        Event_Value = [];
    else
        error('Check why the number of times is not right')
    end
    num_pics_onset=numel(inds_pics_onset);
    should_be_pics = seq_length*NISI*Nseq;
    fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,num_pics_onset)
    pics_onset = reshape(Event_Time(inds_pics_onset),seq_length,NISI,Nseq);
end

%% with Matlab times
if matlab_times
    tbase_end = Event_Time(inds_start_seq);
    
    AA=inds_pics_onset(diff(inds_pics_onset)>2)+1;
%     TT=[Event_Time(inds_pics_onset) Event_Time(inds_pics_onset(end)+1)-Event_Time(inds_pics_onset(end))];
%     TT=[Event_Time(sort([inds_pics_onset AA])) Event_Time(inds_pics_onset(end)+1)-Event_Time(inds_pics_onset(end))];
    TT=[Event_Time(sort([inds_pics_onset AA])) Event_Time(inds_pics_onset(end)+1)];
    figure
    set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
        subplot(122)

%     tdiff=diff(TT)/1000;
    tdiff=diff(TT);
    tdiff(seq_length+1:seq_length+1:end)=[];
    dura_pics = tdiff(:);
    tpic_teo=[];
    for i=1:NISI
        tpic_teo = [tpic_teo ; repmat(experiment.order_ISI(i,:),seq_length,1)];
    end
%     tpic_diff = dura_pics - 1000*experiment.ISI(tpic_teo(1:length(dura_pics))');
%     tpic_diff = dura_pics - experiment.ISI(tpic_teo(1:length(dura_pics))');
    tpic_diff = dura_pics - 1000*experiment.ISI(reshape(tpic_teo(:,1:Nseq),size(tpic_teo,1)*Nseq,1));

    plot(tpic_diff,'b'); %should be 0 ms
    xlim([0 (seq_length-1)*Nseq+1])
    h_legend=legend('stimulus "error" (ms)','location','best');legend('boxoff')
    
    set(gcf,'PaperPositionMode','auto')
    
    keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');
    
    if strcmp(keyval,'y')
        %% save figure and events with Matlab times
        print -dpng ttl_rsvp.png
        save finalevents Event_Time Event_Value pics_onset dura_pics tbase_end
    end
end
close all