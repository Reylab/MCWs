function do_structure_sorted_BCM_online3(channels, use_blanks, circshiftblanks)
% function do_structure_sorted(channels)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified goes through all of
% them.
% MJI: 6/7/2010
begin_time = tic;
fprintf("do_structure_sorted_BCM_online3 (use_blanks:%s, circshiftblanks:%s): ", ...
                            mat2str(use_blanks), mat2str(circshiftblanks))
if ~exist('time_pre','var') || isempty(time_pre), time_pre=1e3; end
if ~exist('time_pos','var') || isempty(time_pos), time_pos=2e3; end

load('NSx','NSx');

if use_blanks && circshiftblanks
    grapes_name = 'grapes_blanks_circ.mat';
elseif use_blanks
    grapes_name = 'grapes_blanks.mat';
else
    grapes_name = 'grapes.mat';
end

if exist(grapes_name,'file')>0
    grapes = load(grapes_name);
else
    grapes = struct;
    %grapes.exp_type = exp_type; 
    grapes.time_pre = time_pre;
    grapes.time_pos = time_pos;
end
load stimulus;
load finalevents;

load('experiment_properties_online3.mat','experiment','scr_config_cell','scr_end_cell')
num_chan = numel(channels);
spikes = cell(num_chan,1);
classes = cell(num_chan,1);
output_names = cell(num_chan,1);
inds_notimes = [];
for i=1:num_chan                           %loop over channels
    channel=channels(i);
    posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
    filename=sprintf('times_%s.mat',NSx(posch).output_name);
    if ~exist(filename,'file') 
        disp([filename ' does not exist\n']);
        inds_notimes=[inds_notimes i]; 
        continue
    end
    load(filename,'cluster_class');
    non0 = cluster_class(:,1)>0;
    spikes{i} = cluster_class(non0,2)'; %I need row vectors latter
    classes{i} = cluster_class(non0,1)'; %I need row vectors latter
    output_names{i} =NSx(posch).output_name;
end
channels(inds_notimes) = [];
spikes(inds_notimes) = [];
classes(inds_notimes) = [];
output_names(inds_notimes) = [];

Nscr = numel(scr_config_cell);

n_scr_ended = numel(scr_end_cell);
if Nscr>n_scr_ended
    warning('some subscreening not ended, using just the completed screenings');
    Nscr = n_scr_ended;
end

for scri = 1:Nscr
    if use_blanks
        grapes = update_grapes_blanks(grapes,pics_onset{scri}, seq_beg_blanks_cell{scri}, ...
                                      stimulus{scri}, spikes, channels, output_names, 0, ...
                                      classes, scr_config_cell{scri}.pics2use, scri, circshiftblanks);
    else
        grapes = update_grapes(grapes,pics_onset{scri}, ...
                               stimulus{scri}, spikes, channels, output_names, 0, ...
                               classes, scr_config_cell{scri}.pics2use, scri);
    end

   fprintf('scr:%d ',scri);
end
fprintf('\n');
% grapes = update_grapes_with_blank_on_spikes(blank_on_onset, grapes, spikes, classes, channels);
grapes.ImageNames = experiment.ImageNames.name;
save(grapes_name,'-struct', 'grapes');    
% save(grapes_name,"-v7.3",'-struct', 'grapes');
tot_time = toc(begin_time);
fprintf("do_structure_sorted_BCM_online3 (use_blanks:%s, circshiftblanks:%s) done in (%0.2f seconds)\n", ...
                                                mat2str(use_blanks), mat2str(circshiftblanks), tot_time)
