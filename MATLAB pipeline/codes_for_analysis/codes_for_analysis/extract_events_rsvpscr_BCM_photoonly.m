function extract_events_rsvpscr_BCM_photoonly(which_system_micro, concat_folder)
%% set paradigm pulses characteristics and create stimulus structure
if ~exist('concat_folder','var')
    concat_folder = [];
end

% matlab_times=0;
if strcmp(which_system_micro,'BRK')
    channel_photo = 257;
elseif strcmp(which_system_micro,'RIP')
    channel_photo = 10241;
end
thr = 2000;
square_duration = 50*30; % time in msec (3 ifi) but result in samples
% photo_corrected = 1;
load 'experiment_properties.mat'
window_search = round(0.75*experiment.ISI*30000);
[seq_length,NISI,~] = size(experiment.order_pic);
Nseq = length(experiment.inds_pics)/seq_length;
create_stimulus_struct_rsvpscr(experiment,Nseq);


%% 
should_be_pics = seq_length*NISI*Nseq;
if isempty(concat_folder)
    load('NSx.mat','NSx')
    posch = find(arrayfun(@(x) (x.electrode_ID==channel_photo),NSx));
    f1 = fopen(sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext),'r','l');
    ch_photo = fread(f1,'int16=>double')*NSx(posch).conversion;
else
    
    load([concat_folder filesep 'NSx.mat'],'NSx','files')
    [~,thisfolder]=fileparts(pwd);
    this_file = find(cellfun(@(x)~isempty(regexp(x,thisfolder,'match')),{files.name}));
    if length(this_file)~=1
        error('%d files found with the current folder name', length(this_file));
    end
    
    
    posch = find(arrayfun(@(x) (x.electrode_ID==channel_photo),NSx));
    f1 = fopen(sprintf('%s%s%s%s',concat_folder,filesep,NSx(posch).output_name,NSx(posch).ext),'r','l');
    ch_photo = fread(f1,'int16=>double')*NSx(posch).conversion;
    ch_photo = ch_photo(files((this_file)).first_sample:files((this_file)).first_sample+files((this_file)).lts-1);
end
smooth_ch_photo = smooth(ch_photo,150);
figure
plot((1:length(ch_photo))/30,ch_photo)
hold on
% plot((1:length(ch_photo))/30,smooth_ch_photo,'r')
start_points=find(diff(smooth_ch_photo>thr)==1)-square_duration;
if should_be_pics~=numel(start_points)
    error('check manually')
end
t_photo = zeros(size(start_points));
    for j=1:numel(start_points)
        [~,indM] = max(ch_photo(start_points(j):start_points(j)+window_search));
        photo_new=findchangepts(ch_photo(start_points(j):start_points(j)+indM),'statistic','std');
        t_photo(j) = start_points(j)+photo_new; % in samples
        line([t_photo(j)/30 t_photo(j)/30],ylim,'color','c')
    end
fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,numel(t_photo))

%%
pics_onset = reshape(t_photo/30000*1000,seq_length,NISI,Nseq);

% produce figure to check pulses
figure
set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
tdiff = diff(squeeze(pics_onset));
plot(tdiff(:)-500,'b'); %should be 0 ms
xlim([0 seq_length*Nseq+1])

h_legend=legend(sprintf('stimulus "error" (ms). %2.0f%% lower than 2ms',100*sum(abs(tdiff(:)-500)<2)/numel(tdiff)),'location','best');legend('boxoff')

set(gcf,'PaperPositionMode','auto')
drawnow
keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');

if strcmp(keyval,'y')
    % save figure and events with Blackrock times
    print -dpng ttl_rsvp.png
    save finalevents pics_onset
end