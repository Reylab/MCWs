function grapes = update_grapes(grapes,pics_onset,stimulus,sp_index,channels,chan_labels,domu,classes,stim_numbers,scri)
%stim_numbers use to write in grapes because stimulus could be just a
%partial part of the experiment
if ~exist('stim_numbers','var') || isempty(stim_numbers)
    stim_numbers = 1:length(stimulus);
end
sr = 30000;

if isempty(grapes)
    stim_numbers = 1:length(stimulus);
end
if ~isfield(grapes,'rasters')
    grapes.rasters = struct;
end
for chi=1:length(channels)
    chanN = ['chan' num2str(channels(chi))];
    if ~isfield(grapes.rasters,chanN)
        grapes.rasters.(chanN) = struct;
        grapes.rasters.(chanN).details = struct;
        if domu
            grapes.rasters.(chanN).mu = struct;
        end
        grapes.rasters.(chanN).details.sr = sr;
    end
    if domu
        if ~isfield(grapes.rasters.(chanN).details,'mu')
            grapes.rasters.(chanN).details.mu = length(sp_index{chi});
        else
            grapes.rasters.(chanN).details.mu = length(sp_index{chi})+grapes.rasters.(chanN).details.mu;
        end
    end
    if ~isfield(grapes.rasters.(chanN).details,'output_name')
        grapes.rasters.(chanN).details.output_name = chan_labels{chi};
        grapes.rasters.(chanN).details.ch_label = chan_labels{chi};
    end

    if ~domu && ~isempty(classes{chi})
        max_cl = max(classes{chi});  % it might have more classes from a previous run
        % if isfield(grapes.rasters.(chanN),'class1') %if any class exist, class1 must exist
        %     r_names = fieldnames(grapes.rasters.(chanN));
        %     r_names = r_names(cellfun(@(x)  startsWith( x , {'class'} ) ,r_names));
        %     max_cl_raster = max(cellfun(@(x)  str2double(x(6:end)), r_names));
            
        %     %             max_cl = max(max_cl, max_cl_raster);
        %     for pp= max_cl+1:max_cl_raster
        %         grapes.rasters.(chanN) = rmfield(grapes.rasters.(chanN), ['class' num2str(pp)]);
        %         grapes.rasters.(chanN).details = rmfield(grapes.rasters.(chanN).details, ['class' num2str(pp)]);
        %     end

        %     if scri==1    % this is because it will append the different sub-screenings
        %         for pp= 1:min([max_cl_raster max_cl])
        %             grapes.rasters.(chanN) = rmfield(grapes.rasters.(chanN), ['class' num2str(pp)]);
        %             grapes.rasters.(chanN).details = rmfield(grapes.rasters.(chanN).details, ['class' num2str(pp)]);
        %         end
        %     end
        % end
        ch_cls = 1:max_cl;
        for ci=1:numel(ch_cls)
            clname = ['class' num2str(ch_cls(ci))];
            spike_count = sum(classes{chi}==ci);
            if isfield(grapes.rasters.(chanN).details, clname)
                grapes.rasters.(chanN).details.(clname) = grapes.rasters.(chanN).details.(clname) + spike_count;
            else
                grapes.rasters.(chanN).details.(clname) = spike_count;
            end
            if ~isfield(grapes.rasters.(chanN),clname)
                grapes.rasters.(chanN).(clname) = struct;
                grapes.rasters.(chanN).(clname).stim = cell(1,max(stim_numbers));
            end
        end
        ch_cls2extend = ch_cls;        
    else
        if ~domu
            if isfield(grapes.rasters.(chanN),'class1')
                r_names = fieldnames(grapes.rasters.(chanN));
                r_names = r_names(cellfun(@(x)  startsWith( x , {'class'} ) ,r_names));
                ch_cls2extend = 1:max(cellfun(@(x)  str2double(x(6:end)), r_names));
            else
                ch_cls2extend =[];
            end
        end
        ch_cls = [];
    end
    if domu && ~isfield(grapes.rasters.(chanN).mu,'stim')
        grapes.rasters.(chanN).mu.stim = cell(1,max(stim_numbers));
    end
    
    for j=1:length(stimulus)                        %loop over stimuli
        snum = stim_numbers(j);
        if length(size(pics_onset))==3
            times_stimulus = pics_onset(sub2ind(size(pics_onset),stimulus(j).terna_onset(:,1),stimulus(j).terna_onset(:,2),stimulus(j).terna_onset(:,3)));
        else
            times_stimulus = pics_onset(stimulus(j).terna_onset(:,1),stimulus(j).terna_onset(:,3));
        end
        if domu
            if numel(grapes.rasters.(chanN).mu.stim)<snum || isempty(grapes.rasters.(chanN).mu.stim{snum})
                grapes.rasters.(chanN).mu.stim{snum} = cell(0);
            end
        else
            for ci=1:numel(ch_cls2extend)
                if numel(grapes.rasters.(chanN).(['class' num2str(ch_cls2extend(ci))]).stim)<snum || isempty(grapes.rasters.(chanN).(['class' num2str(ch_cls2extend(ci))]).stim{snum})
                    grapes.rasters.(chanN).(['class' num2str(ch_cls2extend(ci))]).stim{snum} = cell(0);
                end
            end
        end
        
        for k=1:length(times_stimulus)              %loop over trials
            ind_spikes = (sp_index{chi}>=times_stimulus(k)-grapes.time_pre) & (sp_index{chi}<=times_stimulus(k)+grapes.time_pos);
            if domu
                spikes = sp_index{chi}(ind_spikes)-times_stimulus(k);
                if isempty(spikes)
                    spikes = [9999];
                end
                %grapes.rasters.(chanN).mu.stim{snum}(end+1,1:length(spikes)) = spikes;
                grapes.rasters.(chanN).mu.stim{snum}{end+1} = spikes;
            else
                if isempty(ch_cls)
                    for ci=1:numel(ch_cls2extend)
                        grapes.rasters.(chanN).(['class' num2str(ch_cls2extend(ci))]).stim{snum}{end+1} = [9999];
                    end
                else
                    find_spikes = find(ind_spikes);
                    for ci=1:numel(ch_cls)
                        ix = find_spikes(classes{chi}(ind_spikes)==ch_cls(ci));
                        spikes = (sp_index{chi}(ix)-times_stimulus(k));
                        if isempty(spikes)
                            spikes = [9999];
                        end
                        grapes.rasters.(chanN).(['class' num2str(ch_cls(ci))]).stim{snum}{end+1} = spikes;
                    end
                end
            end
        end

   end
end




end