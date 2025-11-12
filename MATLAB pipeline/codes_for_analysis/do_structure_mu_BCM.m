function do_structure_mu_BCM(channels,skip,ons_ind,exp_type,phase,is_finalevents_us,time_pre,time_pos, concat_folder)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified tries to read them
% from tile or else goes through all of them.
if ~exist('is_finalevents_us','var') || isempty(is_finalevents_us), is_finalevents_us=false; end

% phase could be 'prescr' 'posscr' 'audio' 'video')
% for 'audio' or 'video' ons_ind=1 skip=1
% for exp_type = 'MOVIE', phase can be audio or video
if ~exist('concat_folder','var')
   concat_folder = [];
end
if ~exist('time_pre','var') || isempty(time_pre), time_pre=1e3; end
if ~exist('time_pos','var') || isempty(time_pos), time_pos=2e3; end

if isempty(concat_folder)
    load('NSx','NSx');
else
    load([concat_folder filesep 'NSx'],'NSx','files');
end
% if strcmp(exp_type,'STORY')
%     phase = 'audio';    
% end

if ~exist('phase','var') || isempty(phase) 
    grapes_name = 'grapes.mat';
    if exist(grapes_name,'file')>0; load(grapes_name); end
    load stimulus;
    load finalevents;
    phase = [];
else
    grapes_name = ['grapes_' phase '.mat'];
    if exist(grapes_name,'file')>0; load(grapes_name); end
    if strcmp(phase,'prescr') || strcmp(phase,'posscr')
        load stimulus;
        load finalevents;
    else
        load(['stimulus_' phase '.mat']);
        load(['finalevents_' phase '.mat']);
    end
    eval(['Event_Time = times_' phase ';']);
    if strcmp(phase,'audio')
        time_pre = 2e3;
        time_pos = 1e3;
    end
end

Nstim = length(stimulus); 

if strcmp(exp_type,'RSVPSCR')
    if is_finalevents_us
        onset_times = pics_onset/1000;
    else
        onset_times = pics_onset;
    end
    base_times = onset_times;           
else
    onset_times = Event_Time(ons_ind:skip:end);
	base_times=[];
end

ImageNames = cell(Nstim,1);
for j=1:Nstim                      %loop over stimuli
    ImageNames{j} = stimulus(j).name;
end
for i=1:length(channels)                              %loop over channels
    channel=channels(i);
    fprintf('%2d ',channel);
    
    posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
    filename=sprintf('%s_spikes.mat',NSx(posch).output_name);
    if ~isempty(concat_folder)
        filename = [concat_folder filesep filename];
    end
    if ~exist(filename,'file')
        disp([filename ' does not exist\n']);
        continue; 
    end
    load(filename,'index');

    %%
%     index = index*1000; %absolute times (in microsec)
    chfield = ['chan' num2str(channels(i))];
    if ~isempty(concat_folder)
        [~,thisfolder]=fileparts(pwd);
        this_file = find(cellfun(@(x)~isempty(regexp(x,thisfolder,'match')),{files.name}));
        if length(this_file)~=1
            error('%d files found with the current folder name', length(this_file));
        end
        

        start_ix_ms = (files(this_file).first_sample-1) / (NSx(posch).sr/1000);
        end_ix_ms =  (files(this_file).first_sample+files(this_file).lts) / (NSx(posch).sr/1000);

        index = index(index > start_ix_ms);
        index = index(index < end_ix_ms);
        index = index - start_ix_ms;
    end
    rasters.(chfield).details.mu = length(index);
%         eval(['chan' num2str(channels(i)) '.details.mu = ' num2str(length(index)) ';'])  
    rasters.(chfield).details.sr = NSx(posch).sr;
    rasters.(chfield).details.ch_label = NSx(posch).output_name;    

    for j=1:Nstim                       %loop over stimuli
        if strcmp(exp_type,'RSVP') || strcmp(exp_type,'RSVPNAN') || strcmp(exp_type,'RSVPSCR')
            times_stimulus = onset_times(sub2ind(size(onset_times),stimulus(j).terna_onset(:,1),stimulus(j).terna_onset(:,2),stimulus(j).terna_onset(:,3)));
            times_baseline = base_times(sub2ind(size(base_times),stimulus(j).terna_onset(:,1),stimulus(j).terna_onset(:,2),stimulus(j).terna_onset(:,3)));
        elseif strcmp(phase,'prescr') || strcmp(phase,'posscr')
            eval(['trial_stimulus = stimulus(j).trial_list_' phase ';']);
            times_stimulus = onset_times(trial_stimulus);
            times_baseline = times_stimulus;
        else
            trial_stimulus = stimulus(j).trial_list;
            times_stimulus = onset_times(trial_stimulus);
            times_baseline = times_stimulus;
        end
        spike_mat = zeros(length(times_stimulus),2)+10000;
        for k=1:length(times_stimulus)              %loop over trials
            ind_spikes_base = (index>=times_baseline(k)-time_pre) & (index<times_baseline(k));      %1sec. pre-baseline
            ind_spikes_stim = (index>=times_stimulus(k)) & (index<=times_stimulus(k)+time_pos);   %2sec. post-stimulus
%             spikes = [(index(ind_spikes_base)-times_baseline(k))  (index(ind_spikes_stim)-times_stimulus(k))] /1000;
            spikes = [(index(ind_spikes_base)-times_baseline(k))  (index(ind_spikes_stim)-times_stimulus(k))];
            if ~isempty(spikes)
                spike_mat(k,1:length(spikes)) = spikes;
            end
        end
        spike_mat(spike_mat==0) = 10000;
%         eval(['chan' num2str(channels(i)) ...
%                 '.mu.stim{' num2str(j) '} = spike_mat;']) 
        rasters.(chfield).mu.stim{j} = spike_mat;
    end
end
fprintf('\n');
clear channels channel

% save(grapes_name, 'chan*', 'ImageNames','onset_times', 'base_times', 'exp_type','time_pre','time_pos','phase');    
save(grapes_name, 'rasters', 'ImageNames','onset_times', 'base_times', 'exp_type','time_pre','time_pos','phase');    
