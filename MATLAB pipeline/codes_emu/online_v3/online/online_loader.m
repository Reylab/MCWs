function online_loader()
%online processing code for dynamic screening
online_dir = fileparts(mfilename('fullpath'));
[data_folder] = fileparts(online_dir);
experiment_fname = [data_folder filesep 'experiment_properties_online3' ];
load(experiment_fname, 'experiment');

params = experiment.params;

remove_channels_by_label = {'^[^[mc]](.*)$','^(micro(.*))$','(ref-\d*)$'};
% remove_channels_by_label = {'^[^[mc]](.*)$','^(micro*)$','^(micro_hr*)$','(ref-\d*)$'};

%list of channel numbers with negative or positive thresholds
chs_th_abs = [];
chs_th_pos = [];
remove_channels = [];
priority_channels = [];
% priority_chs_ranking = [257:265];
priority_chs_ranking = [257:296 321:328 353:369];

not_online_channels = [];
NOTCHES_USED = 25;
if strcmp(params.location,'MCW-FH-RIP')
    MAX_SORTING_CORES = 35;
else
    MAX_SORTING_CORES = 5;
end

%% sorting parameters
sorting_done = false;
DO_SORTING = params.do_sorting;
SP2SORT = 500000;%spikes used to detect classes
nstd2plots = 3;
if contains(experiment.subtask,'test')
    ntrial2sort = 1;% (17); (1);
    par.stdmax = 250 %(50);(250);
else
    ntrial2sort = 17;% (17); (1);
    par.stdmax = 50 %(50);(250);
end
%deteccion parameters
par.only_det_filter = true; %false not fully implemented
par.stdmin = 5;
par.preprocessing = params.online_notches;
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

addpath(fullfile(fileparts(mfilename('fullpath')),'../../..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','wave_clus_reylab','wave_split','codes_for_analysis','mex','tasks/online_v3/online'});

if params.copy_backup && params.with_acq_folder && ~params.acq_is_processing && ~test_remote_folder(params.acq_remote_folder_in_processing)
    error('remote acq folder not detected')
end

TEMPLATE_FILE_MS = 'templates_ms.mat';
TEMPLATES_FILE = [params.processing_rec_metadata filesep  'templates.mat'];

if exist( TEMPLATES_FILE,'file')
    use_templates = true;
    disp('Templates loaded')
elseif params.templates_required
    error('Templates required but not found')
else
    use_templates = false;
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
%% calculating filters
[b_filter,a_filter] = ellip(par.detect_order,0.1,40,[par.detect_fmin par.detect_fmax]*2/par.sr);
[z_det,p_det,k_det] = tf2zpk(b_filter,a_filter);

if DO_SORTING
    if ~par.only_det_filter
        [b_sort_filter,a_sort_filter] = ellip(par.sort_order,0.1,40,[par.sort_fmin par.sort_fmax]*2/par.sr);
        [z_sort_det,p_sort_det,k_sort_det] = tf2zpk(b_sort_filter,a_sort_filter);
    end
end


channels = [];

outputinfo = {};
ch_filters = {};
conversion = [];
chan_label = {};

new_pics2load = experiment.NPICS(2:end)-(experiment.NPICS(1:end-1)-experiment.P2REMOVE(1:end-1)); %pics to load after each sub_screening
totalp2load = experiment.NPICS(1) + sum(new_pics2load); % total amount of pics in the library
available_pics = 1:totalp2load;
available_pics_cell = {};
selected2explore_cell = {};
selected2notremove = [];
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
    if strcmp(msg_received,experiment.msgs_Mat.error)
        error('error in task Matlab');
    elseif ~strcmp(msg_received,experiment.msgs_Mat.exper_saved)
        disp('Inconsistency with msg_receiveds sent')
    end
end
load(experiment_fname,'experiment');

if ~params.disable_interactions
    msg_received = mat_comm.waitmessage(); %wait for start recording
    if isempty(msg_received) || ~strcmp(msg_received,experiment.msgs_Mat.rec_started)
        error('Connection timed out or Inconsistency with msg_received received: %s',msg_received);
    end
    fprintf('msg_received read: %s\n',msg_received)
end

result_folder = [params.root_processing filesep experiment.folder_name filesep  'results'];

[~,~,ms_id] = mkdir(result_folder);
delete([result_folder filesep '*'])
cd(result_folder) %try to remove this

if DO_SORTING
    sorter = online_sorter(channels,priority_channels,SP2SORT,true, MAX_SORTING_CORES);
    if use_templates
        sorter.load_sorting_results(TEMPLATES_FILE)
    end
end
%creates grapes
grapes = struct;
grapes.exp_type = 'ONLINE_RSVPSCR';
grapes.time_pre = TIME_PRE;
grapes.time_pos = TIME_POS;

grapes.ImageNames = experiment.ImageNames;
grapes.ImageNames.folder = fullfile(params.pics_root_processing,grapes.ImageNames.folder); %in grapes, the pics have the full path
nstim = height(grapes.ImageNames);

device_com('enable_chs', channels, true, ev_channels);
mat_comm.send(experiment.msgs_Mat.ready_begin)
stim_rm_cell = {};
stim_rm_cell = {};
available_trials_cell = cell(0,1);
wc_sp_index_cell = cell(0,1);
pics_onset_cell = cell(0,1);
stimulus_cell = cell(0,1);
exper_finished = false;
n_scr = 0;
try

    while ~exper_finished && ~strcmp(experiment.msgs_Mat.exper_aborted,msg_received)
        daqtrials = 0;
        if ~params.disable_interactions
            msg_received = mat_comm.waitmessage();
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
            if n_scr>0 && numel(scr_config_cell)==n_scr
                exper_finished = true;
                break
            end
        end
        n_scr = n_scr + 1;
        load(experiment_fname,'scr_config_cell');
        scr_config = scr_config_cell{n_scr};

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
        init_times = cell(scr_config.Nseq,1);

        for ntrial = 1:scr_config.Nseq
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
                    Event_Time_pdiode{ntrial} = [Event_Time_pdiode{ntrial}; double(streams.analog_ev_t{1}(:))/30]; %ms
                    if params.disable_interactions && isempty(Event_Time_pdiode{ntrial})
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
                    if numel(Event_Time{ntrial})<10
                        blank_on = blank_on-(blank_on>1);
                        lines_onoff = lines_onoff-(lines_onoff>1);
                    else
                        break
                    end
                end
                if  ~isempty(streams.parallel.values)
                    Event_Time{ntrial} = [Event_Time{ntrial}; double(streams.parallel.times)/30]; %ms
                    Event_Value{ntrial} = [Event_Value{ntrial}; streams.parallel.values];
                    blank_on = blank_on+any(streams.parallel.values==experiment.blank_on);
                    lines_onoff = lines_onoff+any(streams.parallel.values==experiment.lines_onoff);
                end
                if ~params.disable_interactions
                    msg_received = mat_comm.waitmessage_nofail(max(0.01,WAIT_LOOP - toc(start_loop))); %wait for trial to start
                    if strcmp(experiment.msgs_Mat.exper_finished,msg_received) || strcmp(experiment.msgs_Mat.trial_end,msg_received) || strcmp(experiment.msgs_Mat.exper_aborted,msg_received) || strcmp(experiment.msgs_Mat.error,msg_received)
                        fprintf('Message received: %s--\n',msg_received)
                        break
                    end
                else
                    if params.use_photodiodo
                        ph_done_counter = ph_done_counter + is_trial_done_pd(Event_Time_pdiode{ntrial},scr_config);
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
            init_times{ntrial} = vec_tini_buffer;
            if ~data_loss_flag && strcmp(params.system,'BRK')
                cl_buffer_time = cellfun(@(x) x(1), vec_tini_buffer);
                lens = channels_len{j}; %uses only data of the first channel
                fixed_initial_t = median(cl_buffer_time  - [0 cumsum(lens(1:end-1))]);
                times_diff = cl_buffer_time(1)-fixed_initial_t;
                %save(sprintf('seq%d_scr%d.mat',ntrial,n_scr),'vec_tini_buffer','channels_len')
                if abs(times_diff)>300
                    outputinfo{end+1} = sprintf('Sequence %d (subscr %d): fixed error of %d samples on first trialdata',ntrial,n_scr,ceil(times_diff));
                end
                init_times{ntrial}{1}(1:end) = fixed_initial_t;
            end

            elapsed_trial_time = toc(start_trial);

            if DO_SORTING
                sorter.retrieve_sorting_results();
                spikes = cell(length(channels),1);
                f(1:length(channels)) = parallel.FevalFuture;
                for i = 1:length(channels)
                    f(i) =  parfeval(@get_spikes_online,2,...
                        data{i}(1:datacounter(i)),ch_filters{i}.det, par,det_conf(i));
                end
                for i = 1:length(channels)
                    wait(f(i))
                    [spikes{i}, detecctions{ntrial,i}] = fetchOutputs(f(i));
                end
                clear  f
                sorter.add_spikes(spikes);
            else
                parfor i = 1:length(channels)
                    detecctions{ntrial,i} = detect_mu_online(data{i}(1:datacounter(i)),ch_filters{i}.det, par,det_conf(i));
                end
            end
            fprintf('Detections for sequence: %d of %d, subscr: %d, done.\n',ntrial,scr_config.Nseq,n_scr)
            if DO_SORTING && n_scr==1 && ntrial==ntrial2sort
                sorter.do_remaining_sorting();
                fprintf('Spike Sortings Done.\n')
                sorting_done = true;
            end

            if params.debug
                last_trial = struct;
                last_trial.data = cellfun(@(x) x(1:datacounter),data,'UniformOutput',false);
                last_trial.Event_Time = Event_Time{ntrial};
                last_trial.Event_Value = Event_Value{ntrial};
                last_trial.vec_tini_buffer = vec_tini_buffer;
                last_trial.all_events = all_events;
                last_trial.all_events = elapsed_trial_time;
                if params.use_photodiodo
                    last_trial.Event_Time_pdiode = Event_Time_pdiode;
                end
                save(['online_trial' num2str(ntrial) 'n_scr_' num2str(n_scr) ],'-struct','last_trial')

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

        available_trials = ntrial;
        if available_trials ~= scr_config.Nseq
            warning('Experiment finished. Some trials lost')
            outputinfo{end+1} = sprintf('Only %d trials found in subscr: %d',available_trials,n_scr);
        end

        valid_trials = [];
        pics_onset = [];

        wc_sp_index_cell{n_scr} = cell(size(detecctions,2),1);

        for ntrial = 1:available_trials
            for j=1:size(detecctions,2)
                wc_sp_index_cell{n_scr}{j} =  [wc_sp_index_cell{n_scr}{j}, (double(detecctions{ntrial,j})+init_times{ntrial}{1}(j))/30]; %to ms
            end
            if params.use_photodiodo
                [complete_times, text_out] = fix_photodiode_times(Event_Time_pdiode{ntrial},ntrial,scr_config);
                if isempty(complete_times)
                    daqtrials = daqtrials+1;
                    Event_Time_pdiode_trial = Event_Time_pdiode{ntrial};
                    save(sprintf('pdiode_error_subscr%d_ntrial%d',n_scr,ntrial),'Event_Time_pdiode_trial','ntrial','scr_config')
                    [complete_times, text_out] = fix_onset_times(Event_Time{ntrial},Event_Value{ntrial} ,ntrial,experiment,scr_config);
                end
            else
                [complete_times, text_out] = fix_onset_times(Event_Time{ntrial},Event_Value{ntrial} ,ntrial,experiment,scr_config);
            end
            if ~isempty(text_out)
                outputinfo = [outputinfo, text_out];
            end
            if ~isempty(complete_times)
                pics_onset = [pics_onset , complete_times];
            else
                warning('Unable to detect events in trial %d, subscreening %d.',ntrial, n_scr)
                continue
            end
            valid_trials(end+1) = ntrial;
        end
        if daqtrials>0
            outputinfo{end+1} = sprintf('Using DAC events in %d trials',daqtrials);
        end
        available_trials_cell{n_scr} = ntrial;

        final_Nseq = length(valid_trials);
        pics_onset = reshape(pics_onset,seq_length,NISI,final_Nseq);

        pics_onset_cell{n_scr} = pics_onset;
        clear detecctions;
        %%

        stimulus = create_stimulus_online(scr_config.order_pic(:,:,valid_trials),NISI,...
            experiment.ImageNames.name(scr_config.pics2use),scr_config.ISI,scr_config.order_ISI(:,valid_trials));

        stimulus_cell{n_scr} = stimulus;

        grapes = update_grapes(grapes,pics_onset,stimulus,wc_sp_index_cell{n_scr},channels,chan_label,1,[],scr_config.pics2use,n_scr);

        if DO_SORTING
            if sorter.sortings_state == 0
                sorter.do_remaining_sorting();
            end
            done_chs_ix = [];
            while numel(done_chs_ix) < numel(channels)
                [classes_out, ch_ix_out] = sorter.get_done_sorts(done_chs_ix);
                for cii = 1:numel(classes_out)
                    if ~isempty(classes_out{cii})
                        classes_out{cii} = classes_out{cii}(end-numel(wc_sp_index_cell{n_scr}{ch_ix_out(cii)})+1:end);
                    end
                end
                grapes = update_grapes(grapes,pics_onset,stimulus,wc_sp_index_cell{n_scr}(ch_ix_out),channels(ch_ix_out),chan_label(ch_ix_out),false,classes_out, scr_config.pics2use,n_scr);
                done_chs_ix = [done_chs_ix ch_ix_out];
            end
        end


        if ~params.disable_interactions && ~strcmp(experiment.msgs_Mat.exper_finished,msg_received)
            [datat, rank_config] = create_responses_data_parallel(grapes, scr_config.pics2use,{'mu','class'},ifr_calculator,resp_conf,not_online_channels,priority_channels);

            datat = sort_responses_table(datat,priority_chs_ranking);
            %save sorted_datat?
            enough_trials = datat.ntrials >= MAX_NTRIALS;
            stim_rm = unique(datat(enough_trials,:).stim_number);

            selected2notremove = setdiff(selected2notremove,stim_rm);

            datat = datat(~enough_trials, :);
            [stim_best,istim_best,~] = unique(datat.stim_number,'stable');
            %data2plot = datat(istim_best,:);

            data2plot = [];
            nwin_choose = 2;

            datat.ismu = cellfun(@(x) strcmp(x,'mu'),datat.class);
            data2plot = datat(1,:);
            prim=1;

            if ~isempty(selected2explore_cell)
                while ismember(datat.stim_number(prim),cat(1,selected2explore_cell{1:n_scr-1}))
                    prim = prim+1;
                end
                data2plot = datat(prim,:);
            end

            for itable = prim+1:size(datat,1)
                if ~isempty(selected2explore_cell)
                    if ismember(datat.stim_number(itable),cat(1,selected2explore_cell{1:n_scr-1}))
                        continue
                    end
                end
                same_cltype = datat.ismu(itable) == data2plot.ismu;
                same_ch = cellfun(@(x) strcmp(x,datat.channel{itable}),data2plot.channel);
                same_stim = datat.stim_number(itable) == data2plot.stim_number;
                if any(~same_cltype & same_ch & same_stim)
                    continue
                end
                if sum(same_stim)>=MAX_RASTERS_PER_STIM
                    continue
                end
                ss_dc = find(same_stim & ~same_ch);
                for ss_dc_i = 1:numel(ss_dc)
                    rasters_similarty = calculate_raster_similarty(...
                        grapes.rasters.(data2plot.channel{ss_dc(ss_dc_i)}).(data2plot.class{ss_dc(ss_dc_i)}).stim{data2plot.stim_number(ss_dc(ss_dc_i))},...
                        grapes.rasters.(datat.channel{itable}).(datat.class{itable}).stim{datat.stim_number(itable)});
                    if rasters_similarty > RASTER_SIMILARITY_THR
                        continue
                    end
                end
                data2plot = [data2plot; datat(itable,:)];
                if size(data2plot,1)==(nwin_choose*20) %all the needed
                    break
                end
            end
            clear datat

            if scr_config.manual_select
                fc=loop_plot_responses_BCM_online(data2plot, grapes,n_scr,nwin_choose,rank_config,ifr_calculator.ejex,true,'select_win');
                sel_ix = select_stims(fc);
                sel_ix = sel_ix(1:min( numel(sel_ix), height(data2plot)));%remove selected out of options
                selected2rm = unique(data2plot(sel_ix==-1,:).stim_number,'stable');
                selected2explore = unique(data2plot(sel_ix==1,:).stim_number,'stable');
                stim_rm = [stim_rm; selected2rm];
            else
                selected2explore = [];
                selected2rm = [];
            end
            selected2explore_cell{end+1}=selected2explore;
            extra_stim_rm = experiment.P2REMOVE(n_scr) - length(stim_rm);
            selected2notremove = [selected2notremove;selected2explore]; %CHECK RO W OR COL
            
            pic2add = [];
            if n_scr < numel(experiment.P2REMOVE) && ~isempty(selected2explore)
                added_counter = 0 ;
                for c = selected2explore(:)'
%                     if added_counter == new_pics2load(n_scr)
                    if added_counter == min(experiment.NPICS(n_scr+1)-numel(selected2notremove),new_pics2load(n_scr))
                        break
                    end

                    same_unit = find(cellfun(@(x)strcmp(x,experiment.ImageNames.concept_name{c}),experiment.ImageNames.concept_name)); %invariance only
                    same_unit = same_unit(same_unit~=c);
                    this_concept_counter = 0;
                    added_counter = added_counter + this_concept_counter;
                    same_category = [];

                    for i = 1:height(experiment.ImageNames)
                        if c==i || experiment.ImageNames.concept_number(i) ~= 1 || any(i==same_unit)
                            continue
                        end
                        this_categories = experiment.ImageNames.concept_categories{i};
                        for xi = 1:numel(this_categories)
                            share_category = any(strcmp(experiment.ImageNames.concept_categories{c},this_categories{xi}));
                            if share_category
                                same_category(end+1) = i;
                                break
                            end
                        end
                    end

                    related_pics = [same_unit(:); same_category(:)];
%                     selected2notremove = [selected2notremove ; related_pics];
                    for rp = related_pics'
%                         if all(rp~=removed_stims) && all(rp~=stim_best) && all(rp~=pic2add) %not used
                        if all(rp~=stim_rm) && all(rp~=stim_best) && all(rp~=pic2add) %not used
                            added_counter = added_counter + 1;
                            this_concept_counter = this_concept_counter +1; %to report latter
                            pic2add(end+1) = rp;
%                             if added_counter == new_pics2load(n_scr)
                            if added_counter == min(experiment.NPICS(n_scr+1)-numel(selected2notremove),new_pics2load(n_scr))
                                break
                            end
                        end
                    end
                    fprintf('%d pictures added related to %s\n', this_concept_counter, experiment.ImageNames.concept_name{c})

                end
                fprintf('%d pictures selected\n', numel(selected2explore))
                fprintf('%d pictures added IN TOTAL\n', numel(pic2add))

                selected2notremove = [selected2notremove ; pic2add(:)];
                available_pics = [pic2add setdiff(available_pics,pic2add,'stable')];
                selected2notremove = unique(selected2notremove);
            end
            fprintf('%d pictures selected2notremove\n', numel(selected2notremove))          

            if extra_stim_rm>0
                extra_stims = setdiff(stim_best,[selected2notremove;stim_rm],'stable');
                if numel(extra_stims)>extra_stim_rm
                    stim_rm = [stim_rm; extra_stims(end-extra_stim_rm+1:end)];
                elseif numel(extra_stims)>0
                    stim_rm = [stim_rm; extra_stims];  
                    % extra_pics = extra_stim_rm - numel(extra_stims);
                    % extra_time = extra_pics * scr_config.Nrep * scr_config.ISI;
                    % b_extend_seq = extend_seq_dlg(extra_pics, extra_time);
                    % if b_extend_seq
                    %     disp('Extending sequence');
                    % else
                    %     disp('Reducing pics count to seq length')
                    %     extra_stims = setdiff(stim_best,stim_rm,'stable');% Is this needed?
                    %     if numel(extra_stims)>extra_stim_rm
                    %         stim_rm = [stim_rm; extra_stims(end-extra_stim_rm+1:end)];                    
                    %     end
                    % end
                end
            else
                stim_rm = stim_rm(1:(end+extra_stim_rm));
            end          
            stim_rm_cell{end+1} = stim_rm;
            removed_stims = cell2mat(stim_rm_cell');
        elseif params.disable_interactions
            stim_rm_cell{end+1} = [];
            removed_stims = cell2mat(stim_rm_cell');
            time_overhead = tic();
        else
            stim_rm_cell{end+1} = [];
            removed_stims = cell2mat(stim_rm_cell');
            time_overhead = tic();
        end
        available_pics_cell{end+1} = setdiff(available_pics,removed_stims,'stable');

        save(experiment_fname,'available_pics_cell','stim_rm_cell','selected2explore_cell','-append');
        pause(3)
        mat_comm.send(experiment.msgs_Mat.scr_conf_updated);
    end
    %%

    if params.disable_interactions
        device_com('close')
    end
catch ME
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
    final_n_scr = n_scr;

    if final_n_scr == 0
        warning('None subscreening obtained.');
        return
    end
    all_picsused = unique(cell2mat(cellfun(@(x)x.pics2use, scr_config_cell, 'UniformOutput', false)));
    picsexplored_names=experiment.ImageNames.name(unique(cell2mat(selected2explore_cell')));
    save(experiment_fname,'picsexplored_names','-append');

    save(fullfile(result_folder, 'grapes_online.mat'),'grapes');

    if ~sorting_done
        [data, rank_config] = create_responses_data_parallel(grapes,all_picsused,{'mu'},ifr_calculator,resp_conf,[],priority_channels);
    else
        [data, rank_config] = create_responses_data_parallel(grapes,all_picsused,{'mu','class'},ifr_calculator,resp_conf,[],priority_channels);
    end
    data = sort_responses_table(data);
    %save sorted_datat?
    [G, labels] = findgroups(data.channel);
    ismu = cellfun(@(x) strcmp(x,'mu'),data.class);
    futures(1:numel(labels)) = parallel.FevalFuture;
    channel_grapes = struct;
    channel_grapes.time_pre = grapes.time_pre;
    channel_grapes.time_pos = grapes.time_pos;
    channel_grapes.exp_type =  grapes.exp_type;
    channel_grapes.ImageNames =  grapes.ImageNames;
    channel_grapes.rasters = struct;
    for i=1:numel(labels)
        channel_grapes.rasters.(labels{i}).mu = grapes.rasters.(labels{i}).mu;
        futures = parfeval(@loop_plot_responses_BCM_online,0,data(G==i & ismu,:), channel_grapes,final_n_scr,1,rank_config,ifr_calculator.ejex,true,sprintf('final_ch_%s_mu',grapes.rasters.(labels{i}).details.ch_label),true);
        channel_grapes.rasters = struct;
    end
    wait(futures);
    futures = futures([]);
    %plotting single units
    for i=1:numel(labels)
        rasternames = fieldnames(grapes.rasters.(labels{i}));
        for rn = 1: numel(rasternames)
            rastername = rasternames{rn};
            if ~contains(rastername,'class')
                continue
            end
            isthisclass = cellfun(@(x) strcmp(x,rastername),data.class);
            channel_grapes.rasters.(labels{i}).(rastername) = grapes.rasters.(labels{i}).(rastername);
            futures(end+1) = parfeval(@loop_plot_responses_BCM_online,0,data(G==i & isthisclass ,:), channel_grapes,final_n_scr,1,rank_config,ifr_calculator.ejex,true,sprintf('final_ch_%s_%s',grapes.rasters.(labels{i}).details.ch_label,rastername),true);
            channel_grapes.rasters = struct;
        end
    end
    wait(futures);

    %toc
    if exist('time_overhead','var')
        overhead_duration = toc(time_overhead);
        fprintf('Time Overhead: %.1f seconds.\n',overhead_duration);
    end
    %%

    %%
    if DO_SORTING
        create_sorting_figs(chan_label,sorter.spikes,sorter.classes,'scr_online',conversion)
        sorter.save_sorting_results([result_folder filesep TEMPLATE_FILE_MS])
    end

    if params.online_notches && ~contains(experiment.subtask,'test')
        movefile([params.processing_rec_metadata filesep 'pre_processing_info.mat'],[result_folder filesep 'pre_processing_info.mat']  );
    end
    if ~isempty(MAPFILE)
        copyfile(MAPFILE, result_folder)
    end

    if use_templates
    	movefile(TEMPLATES_FILE,[result_folder filesep 'used_templates.mat']);
    end

    if isempty(outputinfo)
        outputinfo = 'Everything: OK';
    end
    questdlg(outputinfo,'Online Report','Ok','Ok');

    if strcmp(params.location,'MCW-FH-RIP')
        run_wc = 'n';
    else
        run_wc = 'y';
%         run_wc = '';
%         while ~strcmp(run_wc,'y') && ~strcmp(run_wc,'n')
%             run_wc = input("Do you want to run wave_clus?(y/n)\n","s");
%         end
    end
    %move raw data in beh to a folder, copy to tower, move to transferred on
    %acq
    if params.copy_backup && params.with_acq_folder && ~strcmp(experiment.subtask,'dynamic_scr_test')
        msg_received = mat_comm.waitmessage_nofail(60); %wait for trial to start
        if ~strcmp(experiment.msgs_Mat.rec_finished,msg_received)
            warning('Message received: %s-- when waiting for recording stopped.\n',msg_received)
        else
            %start copyng process... better in parallel
            disp('Starting backup worker...')
            pause(3)
            backup_worker = parfeval(@backup_raw_data,1,params,experiment.folder_name);
        end
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

%     if params.copy_backup
    if params.copy_backup && ~strcmp(experiment.subtask,'dynamic_scr_test')
        disp('Copying data from beh...')
        %     [status,msg] = copyfile([params.root_processing filesep experiment.folder_name], [params.backup_path filesep params.sub_ID filesep experiment.folder_name]);
        if ~isunix
            %     [~,msg] = copyfile(fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID,experiment.folder_name),'-a');
            [~,msg] = copyfile(fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID),'-a');
        else
            %     [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID,experiment.folder_name)));
            [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.root_processing,experiment.folder_name), fullfile(params.backup_path ,params.sub_ID)));
        end
        if ~isempty(msg)
            warning('Error copying files: %s', msg)
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
    end

    mat_comm.close()
    custompath.rm()
    
    if strcmp(params.location,'MCW-FH-RIP')
        cd(fullfile(params.backup_path,params.sub_ID,experiment.folder_name))
        addpath('/home/user/Documents/GitHub/codes_emu/codes_for_analysis');
        processing_steps_MCW
    end
    

    disp('Done')


catch ME
    mat_comm.send(experiment.msgs_Mat.process_end)
    warning('Process_end message sent for transfer data, but errors on final processing.')
    save('grapes_error','grapes')
    save('process_error','ME')
    mat_comm.close()
    custompath.rm()
    cd(online_dir)
    rethrow(ME)
end
end

function [msg] = backup_raw_data(params, folder_name)

    [~,msg] = mkdir(fullfile(params.acq_remote_folder_in_processing, folder_name));
    
    if ~isempty(msg), return; end
    if ~isunix
        [~,msg] = movefile(fullfile(params.acq_remote_folder_in_processing,[params.recording_name '*']),fullfile(params.acq_remote_folder_in_processing,folder_name));
    else
        [~,msg] = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,[params.recording_name '*']), fullfile(params.acq_remote_folder_in_processing,folder_name)));
    end
    if ~isempty(msg), return; end
    
    if ~isunix
        %             [~,msg] = copyfile(fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID,folder_name));
        [~,msg] = copyfile(fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID));
    else
        %             [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID,folder_name)));
        [~,msg] = unix(sprintf('cp -r %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name), fullfile(params.backup_path,params.sub_ID)));
    end
    if ~isempty(msg), return; end
    
    if ~isunix
        %             [~,msg] =  movefile(fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred',folder_name));
        [~,msg] =  movefile(fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred'));
    else
        %             [~,msg]  = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred',folder_name)));
        [~,msg]  = unix(sprintf('mv %s %s',fullfile(params.acq_remote_folder_in_processing,folder_name),fullfile(params.acq_remote_folder_in_processing,'transferred')));
    end
end