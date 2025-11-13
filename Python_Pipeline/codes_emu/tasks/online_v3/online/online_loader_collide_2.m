function online_loader_collide_2()
%online processing code for dynamic screening
online_dir = fileparts(mfilename('fullpath'));
[data_folder] = fileparts(online_dir);
exp_prop_file = [data_folder filesep 'experiment_properties_online3.mat' ];
load(exp_prop_file, 'experiment');
n_blks = numel(experiment.NPICS);

b_use_blanks = false;
b_circshiftblanks = true;
b_remove_collisions = true;
b_collision_warning_shown = false;
b_make_coll_plots = false; % Collision plots
b_exper_aborted  = false;
b_parallel = true;
params = experiment.params;
nwins_best_stims = 6; % No of windows for best_stims
floc_task_group = 'Other'; % all, RecDec or Other

blank_seq_beg_str = [experiment.blank_on, experiment.lines_onoff];
blank_seq_end_str = [experiment.blank_on, experiment.continue_msg_on];

if params.run_num == 1 && strcmp(experiment.subtask, 'DynamicScr')
    dresp = questdlg('Have you set priority channels?','Priority Channels','Yes','No','No');
    switch dresp
        case 'No'
            error('Priority channels not set.')
    end

end

temp_folder = [experiment.params.backup_path filesep 'temp'];
if ~exist(temp_folder, 'dir')
    mkdir(temp_folder)
end

if contains(experiment.subtask, 'DynamicScr')
    miniscr_folder = fullfile(experiment.params.pics_root_processing, 'miniscr_pics');
    try
        delete([miniscr_folder filesep '*']);
    catch
        warning('Deleting miniscr pics failed.');
    end
end

remove_channels_by_label = {'^[^[mc]](.*)$','^(micro(.*))$','(ref-\d*)$'};
% remove_channels_by_label = {'^[^[mc]](.*)$','^(micro*)$','^(micro_hr*)$','(ref-\d*)$'};

%list of channel numbers with negative or positive thresholds
chs_th_abs = [];
chs_th_pos = [];
remove_channels = [];
% remove_channels = [266:274];
priority_channels = [];
%priority_chs_ranking = [257:274 298:306 321:338];
priority_chs_ranking = [257:265 288:296 321:329];
% priority_chs_ranking = [289:305];
not_online_channels = [];
NOTCHES_USED = 25;
if strcmp(params.location,'MCW-FH-RIP')
    MAX_SORTING_CORES = 35;
elseif contains(params.location, 'ABT')
    MAX_SORTING_CORES = 12;
else
    MAX_SORTING_CORES = 5;
end

%% sorting parameters
sorting_done = false;
DO_SORTING = params.do_sorting;
SP2SORT = 500000;%spikes used to detect classes
nstd2plots = 3;
if contains(experiment.subtask,'Test')
    ntrial2sort = 2;% (17); (1);
    par.stdmax = 250 %(50);(250);
else
    ntrial2sort = 17;% (17); (1);
    %ntrial2sort = 25;% (17); (1); new task (will's task)
    par.stdmax = 50 %(50);(250);
end
mu_only = false;
if contains(experiment.subtask, 'OnlineMiniScr') && ~params.templates_required
    mu_only = true;
end

if contains(experiment.subtask, 'DailyMiniScr') && ~params.templates_required
    mu_only = true;
end

%deteccion parameters
par.only_det_filter = true; %false not fully implemented
par.stdmin = 5;
par.preprocessing = params.online_notches;
% par.preprocessing = false;
par.detect_order = 4;
par.sort_order = 2;
par.w_pre = 20;
par.w_post = 44;
par.int_factor = 5;
par.detect_fmin = 300;
par.detect_fmax = 3000;
par.sort_fmin = 300;
par.sort_fmax = 3000;
par.sr = 30000;
par.ref_ms = 1.5;
%%
WAIT_LOOP = 1; %seconds between buffer clearings, max 3seg

TIME_PRE=1e3;
TIME_POS=2e3;
RASTER_SIMILARITY_THR = 0.85;
MAX_NTRIALS = 15;

address = {'192.168.137.3','192.168.137.178'}; %index using which_nsp-1
PHOTODIODE_LEVEL = '1900mV';
NSP_TYPE = 265;

if strcmp(params.system,'BRK')
    if NSP_TYPE == 265
        PHOTODIODE = 257;
    else
        PHOTODIODE = 129;
    end
    MAPFILE = [];
else
    PHOTODIODE = 1; %which sma
    mapfiles = dir([params.processing_rec_metadata filesep params.mapfile]);
    if numel(mapfiles)>1
        error('multiple mapfiles found')
    elseif numel(mapfiles)==1
        MAPFILE = [params.processing_rec_metadata filesep mapfiles(1).name];
    else
        error('mapfile not found')
    end
end
%grapes and plot parameters
MAX_RASTERS_PER_STIM = 3;
resp_conf = struct;
resp_conf.from_onset = 1;
resp_conf.smooth_bin=1500;
resp_conf.min_spk_median=1;
resp_conf.tmin_median=200;
resp_conf.tmax_median=700;
resp_conf.psign_thr = 0.05;
resp_conf.t_down=20;
resp_conf.over_threshold_time = 75;
resp_conf.below_threshold_time = 100;
resp_conf.nstd = 3;
resp_conf.win_cent=1;

resp_conf.sigma_gauss = 10;
resp_conf.alpha_gauss = 3.035;
resp_conf.ifr_resolution_ms = 1;
resp_conf.sr = par.sr;
resp_conf.TIME_PRE = TIME_PRE;
resp_conf.TIME_POS = TIME_POS;
resp_conf.tmin_base = -900; % ms
resp_conf.tmax_base = -100; % ms
resp_conf.FR_resol = 10; % ms
resp_conf.min_ifr_thr = 4; % Hz

addpath(fullfile(fileparts(mfilename('fullpath')),'../../..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','useful_functions/task_tools','wave_clus_reylab','wave_split','codes_for_analysis','mex','tasks/online_v3/online'});

if params.copy_backup && params.with_acq_folder && ~params.acq_is_processing && ~test_remote_folder(params.acq_remote_folder_in_processing)
    error('remote acq folder not detected')
end


TEMPLATES_MS = [params.processing_rec_metadata filesep 'templates_ms.mat'];
TEMPLATES_WC = [params.processing_rec_metadata filesep 'templates_wc_offline.mat'];
if contains(experiment.subtask, 'DynamicScr') && ~params.templates_required
    if isfile(TEMPLATES_WC)
        delete(TEMPLATES_WC);
    elseif isfile(TEMPLATES_MS)
        delete(TEMPLATES_MS);
    end
elseif params.templates_required
    if isfile(TEMPLATES_WC)
        TEMPLATES_FILE = TEMPLATES_WC;
        disp('TEMPLATES_WC available.')
    elseif isfile(TEMPLATES_MS)
        TEMPLATES_FILE = TEMPLATES_MS;
        disp('TEMPLATES_WC not found, using TEMPLATES_MS.')
    else
        error('Templates required but not found')
    end
end

ifr_calculator= IFRCalculator(resp_conf.alpha_gauss,resp_conf.sigma_gauss,resp_conf.ifr_resolution_ms,resp_conf.sr,TIME_PRE,TIME_POS);

%%
if ~isempty(intersect(chs_th_pos,chs_th_abs))
    error('repeated channel in both deteccion signs')
end


if par.preprocessing
    if exist([params.processing_rec_metadata filesep 'pre_processing_info.mat'],'file')
        load([params.processing_rec_metadata filesep 'pre_processing_info.mat'],'process_info')
    else
        error('pre_processing_info.mat not found')
    end
end
online_loader_diary_txt_filename = [datestr(now,'mm-dd-yy') '_' datestr(now,'HH_MM_SSAM') '_diary.txt'];
online_loader_diary_txt_file = [params.processing_rec_metadata filesep ...
    experiment.fname '_'...
    online_loader_diary_txt_filename];

online_loader_diary_txt_file = fullfile(online_loader_diary_txt_file(~isspace(online_loader_diary_txt_file)));

diary(online_loader_diary_txt_file)
disp([experiment.fname ':BEGIN'])

if strcmp(params.system,'BRK')
    which_nsp=params.which_nsp_micro;
    inst_num = which_nsp -1;
    address = address{which_nsp};
    channel_id_offset = (inst_num>0)*(inst_num+1)*1000;
else
    channel_id_offset = 0;
    inst_num= 0;
    address = [];
end
if strcmp(params.system, 'BRK')
    ev_channels = PHOTODIODE;
elseif strcmp(params.system, 'RIP')
    ev_channels = 1;
else
    error('Unsupported device')
end

device_com('open',params.system,'address',address,'instance',inst_num,'mapfile',MAPFILE,'nsp_type',NSP_TYPE);

info = device_com('get_chs_info');
channels = [];
conversion = [];
chan_label = {};
for ci = 1:numel(info.ch)
    if any(info.ch(ci) == remove_channels) %removes the channel
        continue
    end
    rem_ch = false;
    for i=1:numel(remove_channels_by_label)
        if ~isempty(regexp(info.label{ci},remove_channels_by_label{i},'match'))
            rem_ch = true; break;
        end
    end
    if rem_ch
        continue
    end
    if info.ismicro(ci) %if micro
        channels(end+1) = info.ch(ci);
        chan_label{end+1} = info.label{ci};
        conversion(end+1) = info.conversion(ci);
    else
        error('channel with label: %s, is not a micro', info.label{ci})
    end
end
device_com('enable_chs', channels, true, ev_channels);

% Check if streams can be read without invalid timestamps exception
device_com('clear_buffer'); % Gets current timestamp from summit to be later used in get_stream
pause(0.2)


% reset the lastwarn message and id
lastwarn('', '');
% might throw a warning if Trellis isn't able to provide a stream for the
% requested timestamp
streams = device_com('get_stream');
% now if a warning was raised, warnMsg will not be empty.
[warnMsg, warnId] = lastwarn();
% popup a message to restart Trellis
if(~isempty(warnMsg))
    error(warnMsg, warnId);
end


addpath(genpath(fullfile(data_folder,'online')))
addpath(genpath(data_folder))

poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool;
end
mat_comm = matconnect(params.beh_machine);
if ~params.disable_interactions
    dresp = questdlg('Ready to begin, you can pres ok and go to the other matlab','Ready?','Ok','Ok');
    if isempty(dresp)
        error('Question dialog window closed.')
    end
    %     msg_received = mat_comm.waitmessage();
    %     if strcmp(msg_received,experiment.msgs_Mat.error)
    %         error('error in task Matlab');
    %     elseif ~strcmp(msg_received,experiment.msgs_Mat.ready_begin)
    %         error('Inconsistency with msg_receiveds sent')
    %     end
    % else
    %     msg_received = '';
end


%% calculating filters
[b_filter,a_filter] = ellip(par.detect_order,0.1,40,[par.detect_fmin par.detect_fmax]*2/par.sr);
[z_det,p_det,k_det] = tf2zpk(b_filter,a_filter);

if DO_SORTING
    if ~par.only_det_filter
        [b_sort_filter,a_sort_filter] = ellip(par.sort_order,0.1,40,[par.sort_fmin par.sort_fmax]*2/par.sr);
        [z_sort_det,p_sort_det,k_sort_det] = tf2zpk(b_sort_filter,a_sort_filter);
    end
end




outputinfo = {};
ch_filters = {};

new_pics2load = experiment.NPICS(2:end)-(experiment.NPICS(1:end-1)-experiment.P2REMOVE(1:end-1)); %pics to load after each sub_screening
totalp2load = experiment.NPICS(1) + sum(new_pics2load); % total amount of pics in the library
available_pics = 1:totalp2load;
available_pics_cell = {};
unused_pics = 1:height(experiment.ImageNames);
selected2explore = [];
selected2explore_cell = {};
selected2rm = [];
selected2rm_cell = [];
selected2notremove = [];
selected2notremove_cell = {};
stim_rm_max_trials = [];
stim_rm_max_trials_cell = {};
stim_rm = [];
stim_rm_cell = {};
same_units_cell = {};
same_categories_cell = {};
fetched_pics_cell = {};
categ_localiz_history = {};

for ci = 1:numel(info.ch)

    if par.preprocessing
        index = find([process_info(:).chID]==(info.ch(ci)+channel_id_offset));
        if isempty(index)
            preprocessing_info = [];
        else
            preprocessing_info = process_info(index);
        end
    end
    if par.preprocessing && ~isempty(preprocessing_info)
        if par.only_det_filter
            [sos,g] = calc_sosg(preprocessing_info.notches, z_det,p_det,k_det,NOTCHES_USED);
            ch_filters{end+1}.det = {sos, g};
        else
            ch_filters{end+1}.det = {b_filter, a_filter};
            if DO_SORTING
                ch_filters{end}.sort = {b_sort_filter,a_sort_filter};
            end
            [sos_notch,g_notch] = calc_sosg(preprocessing_info.notches, [],[],1,NOTCHES_USED);
            ch_filters{end}.notches = {sos_notch,g_notch};
        end
    else
        ch_filters{end+1}.det = {b_filter, a_filter};
        if DO_SORTING && ~par.only_det_filter
            ch_filters{end}.sort = {b_sort_filter,a_sort_filter};
        end
    end
end

if params.use_photodiodo && strcmp(params.system, 'BRK')
    pause(0.05)
    cbmex('config', PHOTODIODE, 'spkfilter', 0, 'spkthrlevel', PHOTODIODE_LEVEL, 'instance',inst_num)
    pause(0.05)
end

det_conf = -1*ones(size(channels));
det_conf(ismember(channels,chs_th_pos)) = 1;
det_conf(ismember(channels,chs_th_abs)) = 0;

if ~params.disable_interactions
    msg_received = mat_comm.waitmessage();
    if strcmp(experiment.msgs_Mat.exper_aborted,msg_received)
        b_exper_aborted = true;
    end
    if strcmp(msg_received,experiment.msgs_Mat.error)
        error('error in task Matlab');
    elseif ~strcmp(msg_received,experiment.msgs_Mat.exper_saved)
        disp('Inconsistency with msg_receiveds sent')
    end
end
load(exp_prop_file,'experiment');

if ~params.disable_interactions
    msg_received = mat_comm.waitmessage(); %wait for start recording
    if isempty(msg_received) || ~strcmp(msg_received,experiment.msgs_Mat.rec_started)
        error('Connection timed out or Inconsistency with msg_received received: %s',msg_received);
    end
    fprintf('msg_received read: %s\n',msg_received)
end

result_folder = [experiment.folder_name filesep  'results'];

[~,~,ms_id] = mkdir(result_folder);
delete([result_folder filesep '*'])
cd(result_folder) %try to remove this

selected4miniscr_csv_file = [result_folder filesep 'selected4miniscr.csv'];

if DO_SORTING
    sorter = online_sorter(channels,priority_channels,SP2SORT,true, MAX_SORTING_CORES);
    if exist('TEMPLATES_FILE', 'var')
        sorter.load_sorting_results(TEMPLATES_FILE)
        sorting_done = true;
    end
end
%creates grapes
grapes = struct;
grapes.exp_type = 'ONLINE_RSVPSCR';
grapes.time_pre = TIME_PRE;
grapes.time_pos = TIME_POS;

grapes.ImageNames = experiment.ImageNames;
% grapes.ImageNames.folder = fullfile(params.pics_root_processing,grapes.ImageNames.folder); %in grapes, the pics have the full path
nstim = height(grapes.ImageNames);



mat_comm.send(experiment.msgs_Mat.ready_begin)

available_trials_cell = cell(0,1);
wc_sp_index_cell = cell(0,1);
pics_onset_cell = cell(0,1);
stimulus_cell = cell(0,1);
exper_finished = false;
scr_id = 0;



try
    %flush diary buffer
    %fprintf("\n");
    %disp('For diary: Starting experiment loops');
    drawnow;

    exp_start_time = tic();
    while ~exper_finished && ~strcmp(experiment.msgs_Mat.exper_aborted,msg_received)
        daqtrials = 0;
        
        if ~params.disable_interactions
            msg_received = mat_comm.waitmessage();
            if strcmp(experiment.msgs_Mat.exper_aborted,msg_received)
                b_exper_aborted = true;
            end
            if strcmp(msg_received,experiment.msgs_Mat.exper_finished)
                exper_finished = true;
                break
            end
            if strcmp(msg_received,experiment.msgs_Mat.error) || ~strcmp(msg_received,experiment.msgs_Mat.scr_conf_updated)
                warning('error in task Matlab');
                break
            end
            fprintf('msg_received read: %s\n',msg_received)
        else
            if scr_id>0 && numel(scr_config_cell)==scr_id
                exper_finished = true;
                break
            end
        end
        scr_id = scr_id + 1;
        load(exp_prop_file,'scr_config_cell');
        scr_config = scr_config_cell{scr_id};

        mat_comm.send(experiment.msgs_Mat.scr_conf_read);
        TRIAL_LEN = (scr_config.ISI*scr_config.seq_length + 3 + 5)*1.2;
        [seq_length,NISI,~] = size(scr_config.order_pic);

        if params.debug
            TRIAL_LEN = 5 * TRIAL_LEN;
        end
        data_n = length(channels);

        data = cell(data_n,1);
        for i = 1:length(data)
            data{i} = zeros(ceil(TRIAL_LEN*30000),1);
        end

        %%
        Event_Time = cell(scr_config.Nseq,1);
        if params.use_photodiodo
            Event_Time_pdiode = cell(scr_config.Nseq,1);
        end
        Event_Value = cell(scr_config.Nseq,1);
        %blk_detecctions = cell(n_sp_cells,scr_config.Nseq);
        detecctions = cell(scr_config.Nseq, length(channels));
        num_art = cell(scr_config.Nseq, length(channels));
        init_times = cell(scr_config.Nseq,1);


        for seq_id = 1:scr_config.Nseq
            datacounter = zeros(length(channels),1);
            lines_onoff = 0;
            blank_on = 0;
            vec_tini_buffer = {};
            all_events = {};

            if ~params.disable_interactions
                msg_received = mat_comm.waitmessage(); %wait for trial to start
                if isempty(msg_received) || strcmp(experiment.msgs_Mat.exper_aborted,msg_received) || strcmp(experiment.msgs_Mat.error,msg_received)
                    error(['unexpected msg_received received: ' msg_received])
                end
                if strcmp(experiment.msgs_Mat.exper_aborted,msg_received)
                    b_exper_aborted = true;
                end
                device_com('clear_buffer'); %probably the main gives time
                %             device_com('get_stream'); %probably the main gives time
                pause(0.2)
            else
                if PHOTODIODE
                    ph_done_counter = 0;
                end
            end
            if params.debug
                all_streams = {};
            end
            start_trial = tic();
            data_loss_flag = false;
            channels_len = cell(length(channels),1);
            while toc(start_trial) < TRIAL_LEN
                start_loop = tic();
                streams = device_com('get_stream');
                if params.debug
                    all_streams{end+1} = streams;
                end
                if params.use_photodiodo
                    Event_Time_pdiode{seq_id} = [Event_Time_pdiode{seq_id}; double(streams.analog_ev_t{1}(:))/30]; %ms
                    if params.disable_interactions && isempty(Event_Time_pdiode{seq_id})
                        pause(0.5)
                        continue
                    end
                end

                for j=1:size(streams.data,1)
                    if streams.lost_prev(j)>0 %lost data before current data, adds zeros
                        streams.data{j}(datacounter(j)+(1:streams.lost_prev(j))) = 0;
                        datacounter(j) = datacounter(j) + streams.lost_prev(j);
                    end
                    lseg = length(streams.data{j});
                    channels_len{j} = [channels_len{j} lseg];
                    data{j}(datacounter(j)+(1:lseg)) = streams.data{j};
                    datacounter(j) = datacounter(j) + lseg;
                    if streams.lost_post(j)>0 %lost data after current data, adds zeros
                        streams.data{j}(datacounter(j)+(1:streams.lost_post(j))) = 0;
                        datacounter(j) = datacounter(j) + streams.lost_post(j);
                    end
                end
                if (sum(streams.lost_prev)+sum(streams.lost_post))>0
                    warning('DATA LOSS in loop')
                    data_loss_flag = true;
                end
                vec_tini_buffer{end+1} = streams.timestamp;

                if params.disable_interactions && (blank_on == 2 || lines_onoff==2)
                    if numel(Event_Time{seq_id})<10
                        blank_on = blank_on-(blank_on>1);
                        lines_onoff = lines_onoff-(lines_onoff>1);
                    else
                        break
                    end
                end
                if  ~isempty(streams.parallel.values)% DAQ
                    Event_Time{seq_id} = [Event_Time{seq_id}; double(streams.parallel.times)/30]; %ms
                    Event_Value{seq_id} = [Event_Value{seq_id}; streams.parallel.values];
                    blank_on = blank_on+any(streams.parallel.values==experiment.blank_on);
                    lines_onoff = lines_onoff+any(streams.parallel.values==experiment.lines_onoff);
                end
                if ~params.disable_interactions
                    msg_received = mat_comm.waitmessage_nofail(max(0.01,WAIT_LOOP - toc(start_loop))); %wait for trial to start
                    if strcmp(experiment.msgs_Mat.exper_finished,msg_received) || strcmp(experiment.msgs_Mat.trial_end,msg_received) || strcmp(experiment.msgs_Mat.exper_aborted,msg_received) || strcmp(experiment.msgs_Mat.error,msg_received)
                        if strcmp(experiment.msgs_Mat.exper_aborted,msg_received)
                            b_exper_aborted = true;
                        end
                        fprintf('Message received: %s--\n',msg_received)
                        break
                    end
                else
                    if params.use_photodiodo
                        ph_done_counter = ph_done_counter + is_trial_done_pd(Event_Time_pdiode{seq_id},scr_config);
                        if ph_done_counter==2
                            break
                        end
                    end
                    pause(max(0.01,WAIT_LOOP - toc(start_loop)))
                end

            end

            % if ~params.disable_interactions && (strcmp(experiment.msgs_Mat.exper_aborted,msg_received) || strcmp(experiment.msgs_Mat.error,msg_received))
            %     break
            % end
            init_times{seq_id} = vec_tini_buffer;
            if ~data_loss_flag && strcmp(params.system,'BRK')
                cl_buffer_time = cellfun(@(x) x(1), vec_tini_buffer);
                lens = channels_len{j}; %uses only data of the first channel
                fixed_initial_t = median(cl_buffer_time  - [0 cumsum(lens(1:end-1))]);
                times_diff = cl_buffer_time(1)-fixed_initial_t;
                %save(sprintf('seq%d_scr%d.mat',seq_id,n_scr),'vec_tini_buffer','channels_len')
                if abs(times_diff)>300
                    outputinfo{end+1} = sprintf('Sequence %d (subscr %d): fixed error of %d samples on first trialdata',seq_id,scr_id,ceil(times_diff));
                end
                init_times{seq_id}{1}(1:end) = fixed_initial_t;
            end

            elapsed_trial_time = toc(start_trial);

            if DO_SORTING
                sorter.retrieve_sorting_results();
                spikes = cell(length(channels),1);
                if b_parallel
                    f(1:length(channels)) = parallel.FevalFuture;
                    for i = 1:length(channels)
                        f(i) =  parfeval(@get_spikes_online,2,...
                            data{i}(1:datacounter(i)),ch_filters{i}.det, par,det_conf(i));
                    end
                    for i = 1:length(channels)
                        wait(f(i))
                        [spikes{i}, detecctions{seq_id,i}] = fetchOutputs(f(i));
                        if numel(spikes{i}) < 1
                            fprintf('No spikes in channel %d. See if micros are connected. \n', channels(i))
                        end
                    end
                    clear  f
                else
                    for i = 1:length(channels)
                        [spikes{i}, detecctions{seq_id,i}] =  get_spikes_online( ...
                            data{i}(1:datacounter(i)),ch_filters{i}.det, ...
                            par,det_conf(i));
                        if numel(spikes{i}) < 1
                            fprintf('No spikes in channel %d. See if micros are connected. \n', channels(i))
                        end
                    end
                end
                %if ~remove_collisions
                %    sorter.add_spikes(spikes);
                %end
            else
                parfor i = 1:length(channels)
                    detecctions{seq_id,i} = detect_mu_online(data{i}(1:datacounter(i)),ch_filters{i}.det, par,det_conf(i));
                end
            end
            fprintf('Detections for sequence: %d of %d, subscr: %d, done.\n',seq_id,scr_config.Nseq,scr_id)
            %flush diary buffer 
            %fprintf("\n");
            %fprintf('For diary: Finished trial %d of subscreen %d\n', ntrial, n_scr);
            drawnow;

            if b_remove_collisions
                trial_det = detecctions(seq_id,:);
                try
                    [spikes, detecctions(seq_id,:)] = remove_collisions(data, trial_det, ...
                        chan_label, spikes, b_make_coll_plots);
                catch ME
                    if ~b_collision_warning_shown
                        warning('\nError in scr:%d trial:%d remove_collisions.', scr_id, seq_id);
                        errMsg = getReport(ME);
                        disp(errMsg);
                    end
                    b_collision_warning_shown = true;
                end
            end
            %flush diary buffer 
            %fprintf("\n");
            %fprintf('For diary: Finished remove_collisions for trial %d, scr %d\n", ntrial, n_scr);
            drawnow;

            if DO_SORTING
                sorter.add_spikes(spikes);
            end

            if DO_SORTING && scr_id==1 && seq_id==ntrial2sort && ~mu_only || b_exper_aborted
                sorter.do_remaining_sorting();
                fprintf('Spike Sortings Done.\n')
                sorting_done = true;
            end

            if params.debug
                last_trial = struct;
                last_trial.data = cellfun(@(x) x(1:datacounter),data,'UniformOutput',false);
                last_trial.Event_Time = Event_Time{seq_id};
                last_trial.Event_Value = Event_Value{seq_id};
                last_trial.vec_tini_buffer = vec_tini_buffer;
                last_trial.all_events = all_events;
                last_trial.all_events = elapsed_trial_time;
                if params.use_photodiodo
                    last_trial.Event_Time_pdiode = Event_Time_pdiode;
                end
                save(['online_trial' num2str(seq_id) 'n_scr_' num2str(scr_id) ],'-struct','last_trial')

                clear last_trial
            end
            %after processing check if the trial was too long and breaks if it is
            if ~params.disable_interactions && ...
                    (strcmp(experiment.msgs_Mat.exper_aborted,msg_received) || ...
                    strcmp(experiment.msgs_Mat.error,msg_received) || ...
                    elapsed_trial_time > TRIAL_LEN)
                break
            end
            mat_comm.send(experiment.msgs_Mat.process_ready)

        end

        seqs_completed = seq_id;
        if seqs_completed ~= scr_config.Nseq
            warning('Experiment finished. Only %d/%d sequences completed in subscr:%d.', seqs_completed, scr_config.Nseq, scr_id)
            outputinfo{end+1} = sprintf('Only %d trials found in subscr: %d',seqs_completed,scr_id);
        end

        valid_seqs = [];
        pics_onset = [];

        wc_sp_index_cell{scr_id} = cell(size(detecctions,2),1);
        seq_beg_blanks = cell(seqs_completed,1);
        seq_end_blanks = cell(seqs_completed,1);
        for seq_id = 1:seqs_completed
            seq_daq_events = Event_Value{seq_id};
            seq_daq_events_times = Event_Time{seq_id};
            for j=1:size(detecctions,2)
                wc_sp_index_cell{scr_id}{j} =  [wc_sp_index_cell{scr_id}{j}, (double(detecctions{seq_id,j})+init_times{seq_id}{1}(j))/30]; %to ms
            end
            blank_on_seq_beg_idx = strfind(seq_daq_events', blank_seq_beg_str);
            blank_on_seq_end_idx = strfind(seq_daq_events', blank_seq_end_str);
            seq_beg_blanks{seq_id} = seq_daq_events_times(blank_on_seq_beg_idx:blank_on_seq_beg_idx+1);
            if ~isempty(blank_on_seq_end_idx)
                seq_end_blanks{seq_id} = seq_daq_events_times(blank_on_seq_end_idx:blank_on_seq_end_idx+1);
            end
            if params.use_photodiodo
                [complete_times, text_out] = fix_photodiode_times(Event_Time_pdiode{seq_id},seq_id,scr_config);
                if isempty(complete_times)
                    daqtrials = daqtrials+1;
                    Event_Time_pdiode_trial = Event_Time_pdiode{seq_id};
                    save(sprintf('pdiode_error_subscr%d_ntrial%d',scr_id,seq_id),'Event_Time_pdiode_trial','seq_id','scr_config')
                    [complete_times, text_out] = fix_onset_times(seq_daq_events_times,seq_daq_events ,seq_id,experiment,scr_config);
                end
            else
                [complete_times, text_out] = fix_onset_times(seq_daq_events_times,seq_daq_events ,seq_id,experiment,scr_config);
            end
            if ~isempty(text_out)
                outputinfo = [outputinfo, text_out];
            end
            if ~isempty(complete_times)
                pics_onset = [pics_onset , complete_times];
            else
                warning('Unable to detect events in trial %d, subscreening %d.',seq_id, scr_id)
                continue
            end
            valid_seqs(end+1) = seq_id;
        end
        if daqtrials>0
            outputinfo{end+1} = sprintf('Using DAC events in %d trials',daqtrials);
        end
        available_trials_cell{scr_id} = seq_id;

        final_Nseq = length(valid_seqs);
        pics_onset = reshape(pics_onset,seq_length,NISI,final_Nseq);

        pics_onset_cell{scr_id} = pics_onset;
        seq_beg_blanks_cell{scr_id} = seq_beg_blanks;
        seq_end_blanks_cell{scr_id} = seq_end_blanks;
        clear detecctions;
        %%

        stimulus = create_stimulus_online(scr_config.order_pic(:,:,valid_seqs),NISI,...
            scr_config.pics2use, experiment.ImageNames.name(scr_config.pics2use),...
            scr_config.ISI,scr_config.order_ISI(:,valid_seqs));

        stimulus_cell{scr_id} = stimulus;

        if b_use_blanks
            grapes = update_grapes_blanks(grapes, pics_onset, ...
                seq_beg_blanks, ...
                stimulus, wc_sp_index_cell{scr_id}, channels, ...
                chan_label, 1, [], ...
                scr_config.pics2use, scr_id, b_circshiftblanks);
        else
            grapes = update_grapes(grapes,pics_onset,stimulus,wc_sp_index_cell{scr_id},channels,chan_label,1,[],scr_config.pics2use,scr_id);
        end

        if DO_SORTING
            if sorter.sortings_state == 0
                sorter.do_remaining_sorting();
            end
            done_chs_ix = [];
            while numel(done_chs_ix) < numel(channels)
                [classes_out, ch_ix_out] = sorter.get_done_sorts(done_chs_ix);
                for cii = 1:numel(classes_out)
                    if ~isempty(classes_out{cii})
                        classes_out{cii} = classes_out{cii}(end-numel(wc_sp_index_cell{scr_id}{ch_ix_out(cii)})+1:end);
                    end
                end
                if b_use_blanks
                    grapes = update_grapes_blanks(grapes, pics_onset, ...
                                                  seq_beg_blanks, stimulus, ...
                                                  wc_sp_index_cell{scr_id}(ch_ix_out), ...
                                                  channels(ch_ix_out), chan_label(ch_ix_out), ...
                                                  false, classes_out, scr_config.pics2use, ...
                                                  scr_id, b_circshiftblanks);
                else
                    grapes = update_grapes(grapes,pics_onset,stimulus, ...
                                           wc_sp_index_cell{scr_id}(ch_ix_out), ...
                                           channels(ch_ix_out),chan_label(ch_ix_out), ...
                                           false,classes_out, scr_config.pics2use,scr_id);
                end
                done_chs_ix = [done_chs_ix ch_ix_out];
            end
        end

        all_selected_to_exp = [];
        all_same_units = [];
        all_same_categories  = [];
        if ~params.disable_interactions && ...
           ~strcmp(experiment.msgs_Mat.exper_finished,msg_received) && ...
            scr_id < n_blks

            [datat, rank_config] = create_responses_data_parallel( ...
                                   grapes, scr_config.pics2use,{'mu','class'}, ...
                                   ifr_calculator,resp_conf, ...
                                   not_online_channels,priority_channels);
            
            % update trial count
            [used_stims, ia, ~] = unique(datat.stim_number);
            experiment.ImageNames(used_stims,:).stim_trial_count = datat(ia,:).ntrials;
            if scr_id == 1 && contains(experiment.subtask, 'CategLocaliz')
                [categ_localiz_history, img_info_table] = init_categ_localiz_history( ...
                                                          experiment, ...
                                                          used_stims, floc_task_group);
            end

            datat = sort_responses_table_online(datat,priority_chs_ranking);
            %save sorted_datat?
            enough_trials = datat.ntrials >= MAX_NTRIALS;
            stim_rm_max_trials = unique(datat(enough_trials,:).stim_number);
            if scr_id > 1
                stim_rm_max_trials = setdiff(stim_rm_max_trials, cell2mat(stim_rm_max_trials_cell'));
            end
            stim_rm_max_trials_cell{end+1} = stim_rm_max_trials;

            datat = datat(~enough_trials, :);
            [stim_best,ia,~] = unique(datat.stim_number,'stable');
            datat_best = datat(ia,:);
            datat_best = datat_best(datat_best.min_spk_test==1 & ~(datat_best.zscore<4.5) & datat_best.good_lat==1,:);
            num_datat_best = height(datat_best);
    
            % Reorder stim_best to have datat_best stim_numbers first
            stim_best_reordered = [stim_best(ismember(stim_best, datat_best.stim_number)); stim_best(~ismember(stim_best, datat_best.stim_number))];
            
            unused_pics = setdiff(unused_pics, stim_best);

            if scr_config.manual_select && ~b_exper_aborted
                data2plot = create_best_stims_table(experiment, grapes, datat, ...
                                                    nwins_best_stims, true, priority_chs_ranking, ...
                                                    selected2notremove, selected2explore_cell, scr_id, true);
                lbl = sprintf('EMU-%.3d_select_win',params.EMU_num);
                [selected2explore, ~, selected2rm] = stimulus_selection_windows( ...
                                                     data2plot, grapes, rank_config, ...
                                                     scr_id, ifr_calculator, 6, ...
                                                     lbl, priority_chs_ranking, ...
                                                     experiment, false, false, true, false);  
                
                
                if 0 < numel(selected2explore)
                    str_pics_added = "";
                    for pic_id = selected2explore(:)'
                        str_pics_added = sprintf("%s %s", str_pics_added, experiment.ImageNames(pic_id,:).name{1});
                    end
                    fprintf('%d selected2explore: (%s) \n', numel(selected2explore), str_pics_added)
                end
                

                % Make images already shown on popup not show up again
                [best_stim_shown,~,~] = unique(data2plot.stim_number,'stable');
                experiment.ImageNames.selectable(best_stim_shown) = -1;
            else
                selected2explore = [];
                selected2rm = [];
            end
            

            selected2explore_cell{end+1}=selected2explore;
            selected2rm_cell{end+1} = selected2rm;            
            
            % remove max trial stims from selected2notremove
            all_max_trial_stims = cell2mat(stim_rm_max_trials_cell');
            all_selected2explore = cell2mat(selected2explore_cell');
            all_selected2rm = cell2mat(selected2rm_cell');
            
            % creating unordered selected2notremove just to know the count
            % will be properly ordered later.
            if scr_id == 1
                selected2notremove = unique(selected2explore, 'stable');
            else
                selected2notremove = [selected2notremove;unique(selected2explore, 'stable')];
            end

            tbl_unused_pics = experiment.ImageNames(unused_pics, :);

            scr_fetched_cell = {};
            % We only select same_units, same_category stimuli after the
            % first and second subscreening blocks of dynamicscr, third onwards we only
            % remove stimuli
            if experiment.MANUAL_SELECT(scr_id) && ~isempty(selected2explore)
                if contains(experiment.subtask, 'DynamicScr')
                    [scr_fetched_cell, experiment] = get_same_unit_categ_pics(scr_id, experiment, tbl_unused_pics, ...
                                                                              selected2explore, selected2notremove, ...
                                                                              new_pics2load);
                elseif contains(experiment.subtask, 'CategLocaliz')
                    [scr_fetched_cell, experiment, categ_localiz_history] = get_categ_localiz_pics( ...
                                                                            scr_id, experiment, img_info_table, ...
                                                                            categ_localiz_history, selected2explore);
                    
                end
            end
            
            grapes.ImageNames = experiment.ImageNames;
            fetched_pics_cell{end+1} = scr_fetched_cell;            

            all_fetched_pics = [];
            for i = 1:numel(scr_fetched_cell)
                for j = 1:numel(fetched_pics_cell)
                    % Loop through rules for each screening
                    rule_info = fetched_pics_cell{j}{i};
                    categ_info_list = rule_info.categ_info;
                    for k=1:numel(categ_info_list)
                        categ_info = categ_info_list{k};
                        all_fetched_pics = [all_fetched_pics; categ_info.pic_ids];
                    end
                end
            end
            curr_fetched_pics_list = [];
            for i = 1:numel(scr_fetched_cell)
                rule_info = scr_fetched_cell{i};
                categ_info_list = rule_info.categ_info;
                for k=1:numel(categ_info_list)
                    categ_info = categ_info_list{k};
                    curr_fetched_pics_list = [curr_fetched_pics_list; categ_info.pic_ids];
                end
            end
            selected2notremove = [all_selected2explore; all_fetched_pics];
            selected2notremove = setdiff(selected2notremove, [all_max_trial_stims; all_selected2rm]);
            if numel(all_max_trial_stims)                
                selected2notremove  = setdiff(selected2notremove, all_max_trial_stims, 'stable');
            end
            
            selected2notremove = unique(selected2notremove, 'stable');            
            
            scr_used_stims = stim_best_reordered(ismember(stim_best_reordered, scr_config.pics2use'));
            extra_used_stims = setdiff(scr_used_stims,[selected2notremove;selected2rm;stim_rm_max_trials],'stable');
            stim_keep = [];
            stim_rm = [];
            stim_rm_count = experiment.P2REMOVE(scr_id) - length([stim_rm_max_trials;selected2rm]);
            if stim_rm_count > 0 && ...
               stim_rm_count < numel(extra_used_stims)
                % remove some more to get to config pic count for this
                % screening
                stim_keep = extra_used_stims(1:end-stim_rm_count);
                stim_rm   = extra_used_stims(numel(stim_keep)+1:end);
            elseif stim_rm_count > 0 && ...
                   stim_rm_count > numel(extra_used_stims) && ...
                   0 < numel(extra_used_stims)                
                stim_rm = extra_used_stims;
            elseif stim_rm_count == 0
                stim_keep = extra_used_stims;
            end
            if numel(stim_keep) < num_datat_best
                num_datat_best = numel(stim_keep);
            end
            unused_pics_available = setdiff(unused_pics',selected2notremove,'stable');
            available_pics = [selected2notremove; stim_keep; unused_pics_available];
            next_scr_pics = [];
            
            if scr_id < numel(experiment.NPICS)        
                exp_time_taken = toc(exp_start_time);
                [next_scr_pics, pics_removed, categ_localiz_history] = keep_endangered_pics(experiment, scr_id, ...
                                                      available_pics, unused_pics_available, ...
                                                      stim_keep, num_datat_best, selected2notremove, ...
                                                      selected2explore_cell, fetched_pics_cell, ...
                                                      scr_config, exp_time_taken, true, categ_localiz_history);
                % Modifying NPICS previously set using config value.
                experiment.NPICS(scr_id + 1) = numel(next_scr_pics);
            
                if numel(pics_removed) > 0
                    stim_rm = extra_used_stims(ismember(extra_used_stims, pics_removed));
                    if contains(experiment.subtask, 'CategLocaliz')
                        unused_curr_fetched_pics = curr_fetched_pics_list(ismember(curr_fetched_pics_list, pics_removed));
                        categ_localiz_history = update_categ_localiz_history(experiment, ...
                                                categ_localiz_history, unused_curr_fetched_pics);
                    end
                end            
            end
            stim_rm_cell{end+1} = stim_rm;
            removed_stims = cell2mat(stim_rm_cell');
            % if scr_id+1 == n_blks && contains(experiment.subtask, 'CategLocaliz')
            %     next_scr_pics = [selected2notremove;setdiff(stim_best,[selected2notremove;removed_stims],'stable')];
            % end
        elseif params.disable_interactions
            stim_rm_cell{end+1} = [];
            removed_stims = cell2mat(stim_rm_cell');
            time_overhead = tic();
        else
            stim_rm_cell{end+1} = [];
            removed_stims = cell2mat(stim_rm_cell');
            time_overhead = tic();
        end

        
        
        selected2notremove_cell{end+1} = selected2notremove;
        available_pics_cell{end+1} = next_scr_pics';

%         save(exp_prop_file,'experiment','available_pics_cell','stim_rm_cell', ...
%             'stim_rm_max_trials_cell','selected2notremove_cell', 'priority_chs_ranking',...
%             'selected2explore_cell','fetched_pics_cell','-append');
        access_exp_prop_file(exp_prop_file, 'experiment', experiment);
        access_exp_prop_file(exp_prop_file, 'available_pics_cell', available_pics_cell);
        access_exp_prop_file(exp_prop_file, 'stim_rm_cell', stim_rm_cell);
        access_exp_prop_file(exp_prop_file, 'stim_rm_max_trials_cell', stim_rm_max_trials_cell);
        access_exp_prop_file(exp_prop_file, 'selected2notremove_cell', selected2notremove_cell);
        access_exp_prop_file(exp_prop_file, 'priority_chs_ranking', priority_chs_ranking);
        access_exp_prop_file(exp_prop_file, 'selected2explore_cell', selected2explore_cell);
        access_exp_prop_file(exp_prop_file, 'fetched_pics_cell', fetched_pics_cell);
        pause(3)
        mat_comm.send(experiment.msgs_Mat.scr_conf_updated);
    end

    %%
    %flush diary buffer 
    %fprintf("\n");
    %disp('For diary: Experiment finished');
    drawnow;

    if params.disable_interactions
        device_com('close')
    end
catch ME
    msgText = getReport(ME)
    %flush buffer diary
    %fprintf("\n");
    %disp(msgText);
    drawnow;

    mat_comm.send(experiment.msgs_Mat.error)
    if params.disable_interactions
        device_com('close')
    end
    save('run_error','ME')
    save('grapes_error','grapes')
    warning('Online processing stopped.')
    warning(ME.message)
end

try
   
    final_n_scr = scr_id;

    if final_n_scr == 0
        warning('None subscreening obtained.');
        return
    end
    all_picsused = unique(cell2mat(cellfun(@(x)x.pics2use, scr_config_cell, 'UniformOutput', false)));
    picsexplored_names=experiment.ImageNames.name(unique(cell2mat(selected2explore_cell')));
    %save(exp_prop_file,'picsexplored_names','-append');
    access_exp_prop_file(exp_prop_file, 'picsexplored_names', picsexplored_names);

    save(fullfile(result_folder, 'grapes_online.mat'),'grapes');

    if ~sorting_done
        [data, rank_config] = create_responses_data_parallel(grapes,all_picsused,{'mu'},ifr_calculator,resp_conf,[],priority_channels);
    else
        [data, rank_config] = create_responses_data_parallel(grapes,all_picsused,{'mu','class'},ifr_calculator,resp_conf,[],priority_channels);
    end
    data = sort_responses_table(data);
    
    %flush diary buffer
    drawnow;

    if contains(experiment.subtask, 'DynamicScr')
        copy2miniscrfolder = true; show_sel_count = true; showwins = true;
    else
        copy2miniscrfolder = false; show_sel_count = false; showwins = false;
    end

    % plot best stims for selection in mini screening
    data_to_plot = create_best_stims_table(experiment, grapes, data, ...
                                           nwins_best_stims, true, priority_chs_ranking, [], [], scr_id, true);
    lbl = sprintf('EMU-%.3d_best_stim',params.EMU_num);
    [~, s4miniscr_tbl, ~] = stimulus_selection_windows( data_to_plot, grapes, rank_config, ...
        scr_id, ifr_calculator, nwins_best_stims, ...
        lbl, priority_chs_ranking, ...
        experiment, copy2miniscrfolder, ...
        show_sel_count, showwins, false);
    if copy2miniscrfolder
        writetable(s4miniscr_tbl, selected4miniscr_csv_file,'Delimiter',',');
        disp('Miniscreening csv file written to disk.')
    end

    % Copy experiment properties to the run folder
    copy_file(exp_prop_file, experiment.folder_name, temp_folder);
    
    try
        plot_channel_grapes('channels2plot', 'all', 'stim_list', 'all', 'order_by_rank', true, ...
            'data', data, 'grapes', grapes, 'n_scr', final_n_scr, 'nwins2plot', 2, ...
            'rank_config', rank_config, 'ifr_x', ifr_calculator.ejex, ...
            'save_fig', true, 'emu_num', params.EMU_num, ...
            'close_fig', true, 'order_offset', 0, ...
            'priority_chs_ranking', priority_chs_ranking, ...
            'parallel_plots', true, 'extra_lbl', '');
    catch ME
        msgText = getReport(ME)
        save('plot_channel_grapes_error','ME')
        save('plot_channel_grapes_grapes','grapes')
    end

    %toc
    if exist('time_overhead','var')
        overhead_duration = toc(time_overhead);
        fprintf('\nTime Overhead: %.1f seconds.\n',overhead_duration);
    end
    %%

    %%
    fprintf('\n');
    fprintf('----------------------------------------------------------------------------------------------\n');
    if DO_SORTING
        disp('Sorting: BEGIN')
        create_sorting_figs(chan_label,sorter.spikes,sorter.classes,'scr_online',conversion)
        templates_ms_path = [result_folder filesep 'templates_ms.mat'];
        sorter.save_sorting_results(templates_ms_path)
        if isfile(templates_ms_path)
            copy_file(templates_ms_path, params.processing_rec_metadata, temp_folder);
        end
        disp('Sorting: END')
    end

    if params.online_notches && ~contains(experiment.subtask,'Test')
        move_file([params.processing_rec_metadata filesep 'pre_processing_info.mat'], result_folder, temp_folder);
        disp('Moved pre_processing_info to results')
    end
    if ~isempty(MAPFILE)
        copy_file(MAPFILE, result_folder, temp_folder);
    end

    if exist('TEMPLATES_FILE', 'var')
        move_file(TEMPLATES_FILE, result_folder, temp_folder);
    end

    if isempty(outputinfo)
        outputinfo = 'Everything: OK';
    end
    for i=1:length(outputinfo)
        fprintf("%s\n",outputinfo{i})
    end
    % questdlg(outputinfo,'Online Report','Ok','Ok');

    run_wc = 'n';
    if strcmp(params.location,'MCW-FH-RIP')
        run_wc = 'n';
        %else
        %run_wc = 'y';
        %run_wc = '';
        %while ~strcmp(run_wc,'y') && ~strcmp(run_wc,'n')
        %    run_wc = input("Do you want to run wave_clus?(y/n)\n","s");
        %end
    end

    if strcmp(run_wc,'y') || strcmp(run_wc,'yes')
        clear data;

        prev_folder = pwd;
        ws_folder = [result_folder filesep  'with_wc'];
        [~,~,ms_id] = mkdir(ws_folder);
        cd(ws_folder)
        if strcmp(ms_id,'MATLAB:MKDIR:DirectoryExists')
            delete('*.png');
            delete('*.mat');
        end
        disp('Starting waveclus processes...')
        save('online_data' ,'wc_sp_index_cell', 'stimulus_cell', 'pics_onset_cell',...
            'resp_conf', 'chan_label', 'channels')

        futures = repmat(parallel.FevalFuture,numel(chan_label),1);
        for i = 1:numel(chan_label)
            futures(i) = parfeval(@sort_and_plot_spikes,0,chan_label{i},sorter.spikes{i}*conversion(i),...
                cell2mat(cellfun(@(x)x{i},wc_sp_index_cell,'UniformOutput',false)),...
                par,ws_folder);
        end
        wait(futures);
        disp('Waveclus processes ended.')
        make_wc_rasters_online([], 1,experiment,scr_config_cell)
        cd(prev_folder)
    end

    pause(5)

    fprintf('\n');
    fprintf('----------------------------------------------------------------------------------------------\n');

    %move raw data in beh to a folder, copy to tower, move to transferred on
    %acq

    if params.copy_backup && params.with_acq_folder %&& ~strcmp(experiment.subtask,'dynamic_scr_test')
        msg_received = mat_comm.waitmessage_nofail(600); %wait for trial to start
        if ~strcmp(experiment.msgs_Mat.rec_finished,msg_received)
            warning('Message received: %s-- when waiting for recording stopped.\n',msg_received)
        elseif params.acq_network
            %start copyng process... better in parallel
            disp('Starting backup worker...')
            pause(3)
            backup_worker = parfeval(@backup_raw_data,1,params,experiment.fname);
        end
    end

    if params.copy_backup
        disp('Copying data from beh...')
        [~,msg] = copy_file(experiment.folder_name, fullfile(params.backup_path,params.sub_ID,'EMU'), temp_folder);
        if ~isempty(msg)
            warning('Error copying files: %s', msg)
        end

    end

    if params.with_acq_folder
        if exist('backup_worker','var') && ~strcmp(backup_worker.State, 'finished')
            disp('wait, copying raw files to backup...')
            wait(backup_worker)
        end
        if exist('backup_worker','var') && isfield(backup_worker,'Error') && ~isempty(backup_worker.Error.message)
            warning("backup worker error: %s\n", backup_worker.Error.message)
        elseif exist('backup_worker','var')
            [bw_msg] = fetchOutputs(backup_worker);
            if ~isempty(bw_msg)
                warning('Error copying files: %s', bw_msg)
            else
                disp('raw files backup done.')
            end
        end
    end

    mat_comm.close()
    custompath.rm()

    pause(5)

    fprintf('\n');
    fprintf('----------------------------------------------------------------------------------------------\n');

    if (contains(experiment.subtask,'DynamicScr') || ...
        contains(experiment.subtask, 'CategLocaliz') || ...
        contains(experiment.subtask, 'DynamicSeman')) && params.acq_network
        disp('Starting processing steps ...')
        % Offline analysis
        run_folder = fullfile(params.backup_path,params.sub_ID,'EMU',experiment.fname);
        cd(run_folder)

        if strcmp(params.location,'MCW-FH-RIP')
            addpath('/home/user/Documents/GitHub/codes_emu/codes_for_analysis');

        elseif strcmp(params.location,'MCW-BEH-RIP')
            addpath('/home/user/share/codes_emu/codes_for_analysis');
        elseif contains(params.location, 'MCW-ABTL-RIP')
            addpath(params.codes_for_analysis);
        end
        
        processing_steps_MCW('is_online', true, ...
            'copy2miniscrfolder', copy2miniscrfolder,'show_sel_count', show_sel_count, ...
            'show_best_stims_wins', showwins, 'max_spikes_plot', 500);
    end

    disp([experiment.fname ':END'])
    diary off
    if exist(online_loader_diary_txt_file, 'file')
        if ~exist('run_folder', 'var')
            run_folder = fullfile(params.backup_path,params.sub_ID,'EMU',experiment.fname);
        end
        addpath(fullfile(fileparts(mfilename('fullpath')),'../../..')) % always / for this trick
        custompath = reylab_custompath({'tasks/online_v3/online'});

        move_file(online_loader_diary_txt_file, run_folder, temp_folder);
        custompath.rm();
    else
        disp('Online diary not found to be moved to results folder.')
    end
    % Delete the temp_folder & its contents
    rmdir(temp_folder, 's')

catch ME
    msgText = getReport(ME)
    %flush buffer diary
    %fprintf("\n");
    %disp(msgText);
    drawnow;

    save('process_error','ME')
    save('grapes_error','grapes')
    mat_comm.send(experiment.msgs_Mat.process_end)
    warning('Process_end message sent for transfer data, but errors on final processing.')
    pwd

    custompath.rm()
    cd(online_dir)

    diary off
    if exist(online_loader_diary_txt_file, 'file')
        if exist('run_folder', 'var')
            addpath(fullfile(fileparts(mfilename('fullpath')),'../../..')) % always / for this trick
            custompath = reylab_custompath({'tasks/online_v3/online'});

            move_file(online_loader_diary_txt_file, run_folder, temp_folder);
            custompath.rm();
        end
    end

    rethrow(ME)
end
end

