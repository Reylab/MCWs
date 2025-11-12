function extract_events_rsvpscr_BCM_photo(NEV)
% this shouldn't be used as a function. Evaluate cell by cell
% HGR JAN14 signature was slightly changed, including 50 ms pause between
% previously consecutive pulses
%% set paradigm pulses characteristics and create stimulus structure
matlab_times=0;
channel_photo = 257;
thr = 2000;
square_duration = 50*30; % time in msec (3 ifi) but result in samples
% photo_corrected = 1;
load 'experiment_properties.mat'
[seq_length,NISI,~] = size(experiment.order_pic);
Nseq = length(experiment.inds_pics)/seq_length;
create_stimulus_struct_rsvpscr(experiment,Nseq);
% load stimulus;
% Npic=length(experiment.ImageNames);
% npulses_noflip=(NISI+1)+2+(seq_length*NISI)+1;%number of events per trial
% if experiment.with_reset
%     npulses_reset = npulses_noflip;
% else
%     npulses_reset=0;
% end

%% read events from NEV structure
events   = NEV.Data.SerialDigitalIO.TimeStampSec*1000; %data in msec
events_TS   = NEV.Data.SerialDigitalIO.TimeStamp;
val        = NEV.Data.SerialDigitalIO.UnparsedData;

lines_on = find(val==experiment.lines_onoff);
if numel(lines_on)~=Nseq*2
    blank_on = find(val==experiment.blank_on);
    if numel(blank_on)~=Nseq*2
        error('fail to have an onset per seq as ref')
    end
    ref_trial = double(events_TS(blank_on));
else
    ref_trial = double(events_TS(lines_on));
end

should_be_pics = seq_length*NISI*Nseq;  
load('NSx.mat','NSx')
posch = find(arrayfun(@(x) (x.electrode_ID==channel_photo),NSx));
f1 = fopen(sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext),'r','l');
ch_photo = fread(f1,'int16=>double')*NSx(posch).conversion;
        
smooth_ch_photo = smooth(ch_photo,100);
figure
plot((1:length(ch_photo))/30,ch_photo)
hold on
plot((1:length(ch_photo))/30,smooth_ch_photo,'r')
photo_pic_timestamp=[];
for ii=1:Nseq
    thr_inds = find(diff(smooth_ch_photo(ref_trial(2*ii-1):ref_trial(2*ii))>thr)==1)+ref_trial(ii); % < for birds
    if numel(thr_inds)~=seq_length
        error('seq %d with issues. increase smoothing or inspect visually',ii)
    end
    start_points = thr_inds-square_duration;

    for j=1:numel(start_points)
        photo_new=findchangepts(ch_photo(start_points(j):start_points(j)+square_duration),'statistic','std');
        t_photo(j) = start_points(j)+photo_new; % in samples
        line([t_photo(j)/30 t_photo(j)/30],ylim,'color','c')
    end
    photo_pic_timestamp = [photo_pic_timestamp t_photo];
end
fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,numel(photo_pic_timestamp))

pics_onset = reshape(photo_pic_timestamp/30000*1000,seq_length,NISI,Nseq);


% produce figure to check pulses
figure
set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
plot(diff(squeeze(pics_onset),2)-500,'b'); %should be 0 ms
xlim([0 seq_length*Nseq+1])
h_legend=legend('stimulus "error" (ms)','location','best');legend('boxoff')

set(gcf,'PaperPositionMode','auto')

keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');

if strcmp(keyval,'y')
    % save figure and events with Blackrock times
    print -dpng ttl_rsvp.png
    save finalevents Event_Time Event_Value pics_onset ttrial_on tlines_onoff tlines_flip_pic tlines_flip_blank tblank dura_blank dura_pics tbase_end
end