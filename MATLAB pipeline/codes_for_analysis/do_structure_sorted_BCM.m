function do_structure_sorted_BCM(channels,phase, concat_folder)
% function do_structure_sorted(channels)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified goes through all of
% them.
% MJI: 6/7/2010
if ~exist('concat_folder','var')
   concat_folder = [];
else
    load([concat_folder filesep 'NSx'],'files','NSx');
end
if ~exist('phase','var') || isempty(phase)
    grapes_name = 'grapes.mat';
    load stimulus;
else
    grapes_name = ['grapes_' phase '.mat'];
    if strcmp(phase,'prescr') || strcmp(phase,'posscr')
        load stimulus;
    else
        load(['stimulus_' phase '.mat']);
    end
end

if exist(grapes_name,'file')>0; load(grapes_name); end
% load('NSx','NSx');

for i=1:length(channels)                              %loop over channels
    channel=channels(i);
    fprintf('%3d ',channel);
    chfield = ['chan' num2str(channels(i))];   

%     posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
%     filename=sprintf('times_%s.mat',NSx(posch).output_name);
    if isempty(concat_folder)
        filename=sprintf('times_%s.mat',rasters.(chfield).details.ch_label);
        if ~exist(filename,'file'), continue; end
        load(filename,'cluster_class');
    
    else
        filename=sprintf('%s%stimes_%s.mat',concat_folder,filesep,rasters.(chfield).details.ch_label);
        if ~exist(filename,'file'), continue; end
        load(filename,'cluster_class');
        [~,thisfolder]=fileparts(pwd);
        this_file = find(cellfun(@(x)~isempty(regexp(x,thisfolder,'match')),{files.name}));
        if length(this_file)~=1
            error('%d files found with the current folder name', length(this_file));
        end
        
        posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
        start_ix_ms = (files(this_file).first_sample-1) / NSx(posch).sr*1000;
        end_ix_ms =  (files(this_file).first_sample+files(this_file).lts) / NSx(posch).sr*1000;
        cluster_class = cluster_class(cluster_class(:,2) > start_ix_ms,:);
        cluster_class = cluster_class(cluster_class(:,2) < end_ix_ms,:);
        cluster_class(:,2) = cluster_class(:,2) - start_ix_ms;
    end
    
    
    
    [q,p,v]=find(cluster_class(:,1));
    %             cluster_class(:,2)=cluster_class(:,2)*1000; %assumes spikes in msec
    
    % clear old classes
    fields = fieldnames(rasters.(chfield).details);
    fields_to_remove = fields(cellfun(@(x) all(x(1:2)=='cl'),fields));
    rasters.(chfield).details = rmfield(rasters.(chfield).details,fields_to_remove);
    rasters.(chfield) = rmfield(rasters.(chfield),fields_to_remove);
          
%     eval(['temp=chan' num2str(channels(i)) '.details.mu;']);
%     eval(['temp1=chan' num2str(channels(i)) '.mu;']);
%     eval(['clear chan' num2str(channels(i))]);
%     eval(['chan' num2str(channels(i)) '.details.mu = temp;']);
%     eval(['chan' num2str(channels(i)) '.mu = temp1;']);
    
    for clus = 1:max(v)                                 %loop over clusters in a channel
        clusfield = ['class' num2str(clus)];   
        sorted_times = cluster_class( cluster_class(:,1)==clus,2 )';
%         eval(['chan' num2str(channels(i)) '.details.class' num2str(clus) ...
%             ' = ' num2str(length(sorted_times)) ';'])
        rasters.(chfield).details.(clusfield) = length(sorted_times);

        for j=1:length(stimulus)                        %loop over stimuli
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
                ind_spikes_base = (sorted_times>=times_baseline(k)-time_pre) & (sorted_times<times_baseline(k));      %1sec. pre-baseline
                ind_spikes_stim = (sorted_times>=times_stimulus(k)) & (sorted_times<=times_stimulus(k)+time_pos);   %2sec. post-stimulus
                %                     spikes = [(sorted_times(ind_spikes_base)-times_baseline(k))  (sorted_times(ind_spikes_stim)-times_stimulus(k))] /1000;
                spikes = [(sorted_times(ind_spikes_base)-times_baseline(k))  (sorted_times(ind_spikes_stim)-times_stimulus(k))];
                if ~isempty(spikes)
                    spike_mat(k,1:length(spikes)) = spikes;
                end
            end
            spike_mat(spike_mat==0) = 10000;
            
%             eval(['chan' num2str(channels(i)) '.class' num2str(clus) ...
%                 '.stim{' num2str(j) '}= spike_mat;'])
            rasters.(chfield).(clusfield).stim{j} = spike_mat;
        end
    end
end
clear channels channel
fprintf('\n');

% save(grapes_name, 'chan*', 'onset_times', 'base_times', 'exp_type','time_pre','time_pos','phase');
save(grapes_name, 'rasters', 'ImageNames','onset_times', 'base_times', 'exp_type','time_pre','time_pos','phase');

