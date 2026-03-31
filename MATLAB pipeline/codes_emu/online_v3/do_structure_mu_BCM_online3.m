function do_structure_mu_BCM_online3(channels,exp_type,use_blanks, circshiftblanks, is_online, time_pre,time_pos)
% Arranges the spike trains into matrices for each cluster and stimulus.
% Gets channel/s as input. If no channels are specified tries to read them
% from tile or else goes through all of them.
begin_time = tic;
fprintf("do_structure_mu_BCM_online3 (use_blanks:%s, circshiftblanks:%s): ", ...
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

load stimulus;
load finalevents;
if min(cell2mat({stimulus{1}.ISI}))<0.5
    time_pre=500; time_pos=750;
end

if exist([pwd filesep grapes_name],'file')>0
    grapes = load(grapes_name);
else
    grapes = struct;
    grapes.exp_type = exp_type; 
    grapes.time_pre = time_pre;
    grapes.time_pos = time_pos;
end


load('experiment_properties_online3.mat','experiment','scr_config_cell','scr_end_cell')
spikes = {};
output_names = {};
for i=1:length(channels)                              %loop over channels
    channel=channels(i);
    posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));
    filename=sprintf('%s_spikes.mat',NSx(posch).output_name);
    if ~exist(filename,'file') 
        disp([filename ' does not exist\n']);  % this cannot happen so it's fine to get an error afterwards when trying to load it
        channels(i)=[]; 
    end
    warning off    
    % try
    %     load(filename,'index_all');
    %     spikes{i} = index_all;
    % catch
    %     load(filename,'index');
    %     spikes{i} = index;
    % end
    load(filename,'index');
    spikes{i} = index;
    
    warning on
    output_names{i} =NSx(posch).output_name;    
end
        
Nscr = numel(scr_config_cell);
n_scr_ended = numel(scr_end_cell);

if Nscr>n_scr_ended
    warning('some subscreening not ended, using just the completed screenings');
    Nscr = n_scr_ended;
end
ISI_min = Inf;
for scri = 1:Nscr
    ISI_min = min(ISI_min,min(cell2mat({stimulus{scri}.ISI})));
    if use_blanks
        % check if seq_beg_blanks_cell exists, else show warning
        if ~exist('seq_beg_blanks_cell','var') || isempty(seq_beg_blanks_cell) || isempty(seq_beg_blanks_cell{scri})
            error('seq_beg_blanks_cell not found, run extract_blank_on_events_ripple.m from processing_steps to create it.');            
        end
        grapes = update_grapes_blanks(grapes, pics_onset{scri}, ...
                                      seq_beg_blanks_cell{scri}, ...
                                      stimulus{scri}, spikes, channels, ...
                                      output_names, 1, [], ...
                                      scr_config_cell{scri}.pics2use, scri, ...
                                      circshiftblanks, is_online);
    else
        grapes = update_grapes(grapes, pics_onset{scri}, ...
                               stimulus{scri}, spikes, channels, ...
                               output_names, 1, [], ...
                               scr_config_cell{scri}.pics2use, scri, is_online);
    end
   
   fprintf('scr:%d ',scri);
end
fprintf('\n');
grapes.ImageNames = experiment.ImageNames.name;
grapes.ISI_min = ISI_min; 
save(grapes_name,'-struct', 'grapes');   
tot_time = toc(begin_time);
fprintf("do_structure_mu_BCM_online3 (use_blanks:%s, circshiftblanks:%s) done in (%0.2f seconds)\n", ...
                                            mat2str(use_blanks), mat2str(circshiftblanks), tot_time)
