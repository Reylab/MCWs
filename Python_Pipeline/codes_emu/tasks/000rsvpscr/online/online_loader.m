function online_loader(varargin)


load(['..' filesep 'experiment_properties'],'experiment');
params = experiment.params;
ipr = inputParser;
addParameter(ipr,'mapfile',[]);
addParameter(ipr,'online_notches', []);
addParameter(ipr,'disable_interactions',false);
addParameter(ipr,'use_photodiodo',[]);
addParameter(ipr,'debug',0);
parse(ipr,varargin{:})


priority_channels = [];
chs_th_abs = [];
chs_th_pos = [];
remove_channels = [];
%maybe move to locations:
DO_SORTING = true;
EARLY_RESPONSE_AT = 0.66; %disable with value >1
DO_WAVECLUS = true;
TEMPLATES_FILE = [params.processing_rec_metadata filesep  'templates.mat'];
MAX_SORTING_CORES = 16;
use_templates = false;
addpath(fullfile(fileparts(mfilename('fullpath')),'../../..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','wave_clus_reylab','wave_split','codes_for_analysis','mex','tasks/.','tasks/000rsvpscr','tasks/online_v3/online'});

inputs = fields(ipr.Results);
for i =1:numel(inputs)
    pstr = inputs{i};
    if any(strcmp(ipr.UsingDefaults,pstr)) && isempty(ipr.Results.(pstr)) && isfield(params, pstr) %parameter not passed and empty
        continue
    end
    params.(pstr) = ipr.Results.(pstr);
end

remove_channels_by_label = {'^[^m](.*)$','^(micro \d*)$','(ref-\d*)$'};

%list of channel numbers with negative or positive thresholds


%%
%deteccion parameters
par.stdmin = 5;
par.stdmax = 50; %(50);(250);
par.only_det_filter = true;
par.preprocessing = params.online_notches;
par.detect_order = 2;
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
NOTCHS_USED = 5;
%for blackrock use:
address = {'192.168.137.3','192.168.137.178'}; %index using which_nsp-1
PHOTODIODE_LEVEL = '1900mV';

%% sorting parameters
SP2SORT = 10000;%spikes used to detect classes
sort_at_least_seq = 3;

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

%raster parameters
TIME_PRE = 1e3;
TIME_POS = 2e3;
nstd2plots = 3;

online_dir = pwd; %just to go back at the end
[data_folder] = fileparts(online_dir); %run the conde in online folder,it will add the needed folders
addpath(genpath(fullfile(data_folder,'online')))
addpath(genpath(data_folder))
try
load([data_folder filesep 'variables_RSVP_SCR.mat'],'Nseq')
catch
    load([data_folder filesep 'RSVP_SCR_workspace.mat'],'Nseq')
end
if Nseq < 6
    disp('Less than 6 sequences, early responses and sorting disabled.')
    EARLY_RESPONSE_AT = inf; %disable with value >1
    DO_WAVECLUS = false;
    
    if exist( TEMPLATES_FILE,'file')
        use_templates = true;
    else
        DO_SORTING = false;
    end
end

if ~DO_SORTING && DO_WAVECLUS
    error('DO_SORTING required to DO_WAVECLUS later.')
end

%%
if ~isempty(intersect(chs_th_pos,chs_th_abs))
    error('repeated channel in both deteccion signs')
end

poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool;
end


result_folder = [params.root_processing filesep experiment.folder_name filesep  'results'];
if exist(result_folder, 'dir')
    rmdir(result_folder,'s')
end
[~,~,ms_id] = mkdir(result_folder);
if par.preprocessing
    if exist([params.processing_rec_metadata filesep 'pre_processing_info.mat'],'file')
        load([params.processing_rec_metadata filesep 'pre_processing_info.mat'],'process_info')
    else
        error('pre_processing_info.mat not found')
    end
end

mat_comm = matconnect(params.beh_machine);
if ~params.disable_interactions
    questdlg('Ready to begin, you can pres ok and go to the other matlab','Ready?','Ok','Ok');
    msg_received = mat_comm.waitmessage();
    if strcmp(msg_received,experiment.msgs_Mat.error)
        error('error in task Matlab');
    end
    which_nsp = str2num(msg_received);
else
    which_nsp = 1;
end 

inst_num = which_nsp -1;
address = address{which_nsp};

if strcmp(params.system,'BRK')
    channel_id_offset = (inst_num>0)*(inst_num+1)*1000;
else
    channel_id_offset = 0; 
end


if strcmp(params.system, 'BRK')
    ev_channels = PHOTODIODE;
elseif strcmp(params.system, 'RIP')
    ev_channels = 1;
else
    error('Unsupported device')
end

device_com('open',params.system,'address',address,'instance',inst_num,'mapfile',MAPFILE,'nsp_type',NSP_TYPE);

if strcmp(params.system, 'BRK')
    cbmex('comment', 6724095, 0, 'online connected','instance',inst_num);
end

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
outputinfo = {};

channels = [];
ch_filters = {};
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
            [sos,g] = calc_sosg(preprocessing_info.notches, z_det,p_det,k_det,NOTCHS_USED);
            ch_filters{end+1}.det = {sos, g};
        else
            ch_filters{end+1}.det = {b_filter, a_filter};
            if DO_SORTING
                ch_filters{end}.sort = {b_sort_filter,a_sort_filter};
            end
            [sos_notch,g_notch] = calc_sosg(preprocessing_info.notches, [],[],1,NOTCHS_USED);
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
        disp('Inconsistency with messages sent')
    end
end
load('experiment_properties','experiment');
addpath([params.pics_root_processing filesep experiment.Path_pics])
Path_pics = experiment.Path_pics;
TRIAL_LEN = (experiment.ISI*experiment.seq_length + 3 + 5)*1.2;
[seq_length,NISI,~] = size(experiment.order_pic);

detecctions = repmat({NaN},experiment.Nseq,length(channels));
if DO_SORTING
    sorter = online_sorter(channels,priority_channels,SP2SORT,true, MAX_SORTING_CORES);
    if use_templates
        sorter.load_sorting_results(TEMPLATES_FILE)
    end
end

data = cell(length(channels),1);
for i = 1:length(data)
    data{i} = zeros(ceil(TRIAL_LEN*30000),1,'int16');
end
if params.disable_interactions
    TRIAL_LEN = 10 * TRIAL_LEN;
end
if ~params.disable_interactions
    message = mat_comm.waitmessage(); %wait for start recording
    fprintf('message read: %s\n',message)
end

Event_Time = cell(experiment.Nseq,1);
Event_Value = cell(experiment.Nseq,1);

if PHOTODIODE
    Event_Time_pdiode = cell(experiment.Nseq,1);
end

init_times = cell(experiment.Nseq,1);



device_com('enable_chs', channels, true, ev_channels);
mat_comm.send(experiment.msgs_Mat.ready_begin)

seq2sort = experiment.Nseq - (sort_at_least_seq);

try
for nseq = 1:experiment.Nseq
    datacounter = zeros(length(channels),1);
    lines_onoff = 0;    
    blank_on = 0;   
    vec_tini_buffer = {};    
    all_streams = {};
    if ~params.disable_interactions
        message = mat_comm.waitmessage(); %wait for trial to start
        if isempty(message) || strcmp(experiment.msgs_Mat.exper_aborted,message) || strcmp(experiment.msgs_Mat.error,message)
            break
        end    
        device_com('clear_buffer');
        pause(0.2);
    else
        if PHOTODIODE
            ph_done_counter = 0;
        end
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
            Event_Time_pdiode{nseq} = [Event_Time_pdiode{nseq}; double(streams.analog_ev_t{1}(:))/30]; %ms
            if params.disable_interactions && isempty(Event_Time_pdiode{nseq})
                pause(0.5)
                continue
            end
        end
        
        for j=1:size(streams.data,1)
            if streams.lost_prev(j)>0 %lost data before current data, adds zeros
                data{j}(datacounter(j)+(1:streams.lost_prev(j))) = 0;
                datacounter(j) = datacounter(j) + streams.lost_prev(j);
            end
            lseg = length(streams.data{j});
            channels_len{j} = [channels_len{j} lseg];
            data{j}(datacounter(j)+(1:lseg)) = streams.data{j};
            datacounter(j) = datacounter(j) + lseg;
            if streams.lost_post(j)>0 %lost data after current data, adds zeros
                data{j}(datacounter(j)+(1:streams.lost_post(j))) = 0;
                datacounter(j) = datacounter(j) + streams.lost_post(j);
            end
        end
        if (sum(streams.lost_prev)+sum(streams.lost_post))>0
            warning('DATA LOSS in loop')
            data_loss_flag = true;
        end
        vec_tini_buffer{end+1} = streams.timestamp;
        if params.disable_interactions && (blank_on == 2 || lines_onoff==2)
            if numel(Event_Time{nseq})<10
                blank_on = blank_on-(blank_on>1);
                lines_onoff = lines_onoff-(lines_onoff>1);
            else
                break
            end
        end
        if  ~isempty(streams.parallel.values)
            Event_Time{nseq} = [Event_Time{nseq}; double(streams.parallel.times)/30]; %ms
            Event_Value{nseq} = [Event_Value{nseq}; streams.parallel.values];            
            blank_on = blank_on+any(streams.parallel.values==experiment.blank_on);
            lines_onoff = lines_onoff+any(streams.parallel.values==experiment.lines_onoff);
        end
        
        
        if ~params.disable_interactions
            message = mat_comm.waitmessage_nofail(max(0.01,WAIT_LOOP - toc(start_loop))); %wait for trial to start
            if strcmp(experiment.msgs_Mat.trial_end,message) || strcmp(experiment.msgs_Mat.exper_aborted,message) || strcmp(experiment.msgs_Mat.error,message)
                break
            end  
        else
            if params.use_photodiodo
                 ph_done_counter = ph_done_counter + is_trial_done_pd(Event_Time_pdiode{nseq},experiment);
                 if ph_done_counter==2
                     break
                 end
            end
            pause(max(0.01,WAIT_LOOP - toc(start_loop)))
        end
    end
    % using vec_tini_buffer fix fisrt time in blk (one per channel?) needs
    % len of each channel
    init_times{nseq} = vec_tini_buffer;
    if ~data_loss_flag && strcmp(params.system,'BRK')
        cl_buffer_time = cellfun(@(x) x(1),vec_tini_buffer);
        lens = channels_len{j}; %uses only data of the first channel
        fixed_initial_t = median(cl_buffer_time  - [0 cumsum(lens(1:end-1))]);        
        times_diff = cl_buffer_time(1)-fixed_initial_t;
        if abs(times_diff)>300
            outputinfo{end+1} = sprintf('Trial %d: fixed error of %d samples on first trialdata',nseq,times_diff);
        end
        init_times{nseq}{1}(1:end) = fixed_initial_t;
    end
    
    if ~params.disable_interactions && (strcmp(experiment.msgs_Mat.exper_aborted,message) || strcmp(experiment.msgs_Mat.error,message))
        break
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
            [spikes{i}, detecctions{nseq,i}] = fetchOutputs(f(i));
        end        
        clear  f
        sorter.add_spikes(spikes);
    else
        parfor i = 1:length(channels)
            detecctions{nseq,i} = detect_mu_online(data{i}(1:datacounter(i)),ch_filters{i}.det, par,det_conf(i));
        end
    end
    
    
    fprintf('Detections for trial %d, done\n',nseq)
    if params.debug && ~params.disable_interactions
        last_trial = struct;
        trialname = ['trial_' num2str(nseq)];
        last_trial.(trialname).all_streams = all_streams;
        save('online_debug','-struct','last_trial','-append')
        clear last_trial
    end
    %after processing check if the trial was too long and breaks if it is
    if (~params.disable_interactions) && (elapsed_trial_time > TRIAL_LEN)
        break
    end
    
    if nseq >= ceil(EARLY_RESPONSE_AT*experiment.Nseq)
        if nseq > ceil(EARLY_RESPONSE_AT*experiment.Nseq)
            if ~isempty(early_fig_p) && strcmp(early_fig_p.State,'finished')
                fprintf('Early figures printed: time required: %0.f seconds.\n\n',toc(early_fig_tic))
                early_fig_p = [];
                if DO_SORTING
                    sorter.maxworkers = sorter.maxworkers + 1;
                end
            end
        else
            if DO_SORTING
                [classes_out, ch_ix_out] = sorter.get_done_sorts([]);
                sorter.maxworkers = sorter.maxworkers - 1;
            else
                classes_out = [];
                ch_ix_out = [];
            end
            
            early_fig_p = parfeval(@create_early_figures,0,params.early_figures_full_path,nseq,experiment,detecctions,...
                params.use_photodiodo,Event_Time_pdiode,Event_Time,Event_Value,...
                init_times, channels,priority_channels, nstd2plots, ...
                chan_label,TIME_PRE,TIME_POS, classes_out, ch_ix_out);
            early_fig_tic = tic();
            disp('Starting early figures.')
            
        end
    end
    
    if DO_SORTING  && nseq == seq2sort
        sorter.do_remaining_sorting();
    end
    
    mat_comm.send(experiment.msgs_Mat.process_ready)
end
%%
clear data
%%
if params.disable_interactions
    device_com('close')
end
catch ME
    mat_comm.send(experiment.msgs_Mat.error)
    if params.disable_interactions
        device_com('close')
    end
    custompath.rm()
    rethrow(ME)
end
%%
available_trials = nseq;

if available_trials ~= experiment.Nseq
	warning('Error in communication. Some trials lost')
    outputinfo{end+1} = sprintf('Only %d trials found',available_trials);
end

sp_index = cell(size(detecctions,2),1);

valid_trials = [];
pics_onset = [];
for nseq = 1:available_trials
    if isempty(init_times{nseq})
       break
    end
    for j=1:size(detecctions,2) %All trials has to be included because of the one to one relatinship between detecctions and spikes 
        sp_index{j} =  [sp_index{j}, (double(detecctions{nseq,j})+init_times{nseq}{1}(j))/30]; %to ms
    end
    if params.use_photodiodo
        [complete_times, text_out] = fix_photodiode_times(Event_Time_pdiode{nseq},nseq,experiment);
        if isempty(complete_times)
            text_out{end+1} = sprintf('Using DAC events in trial: %d',nseq);
            [complete_times, text_out{end+1}] = fix_onset_times(Event_Time{nseq},Event_Value{nseq} ,nseq,experiment);
        end    
    else
        [complete_times, text_out] = fix_onset_times(Event_Time{nseq},Event_Value{nseq} ,nseq,experiment);
    end

    if ~isempty(text_out)
        outputinfo = [outputinfo, text_out];
    end
        
    if ~isempty(complete_times)
        pics_onset = [pics_onset , complete_times];
    else
        continue
    end
    
    valid_trials(end+1) = nseq;

end
final_Nseq = length(valid_trials);

pics_onset = reshape(pics_onset,seq_length,NISI,final_Nseq);
%%
clear detecctions;
%%
stimulus = create_stimulus_online(experiment.order_pic(:,:,valid_trials),NISI,...
    experiment.ImageNames,experiment.ISI,experiment.order_ISI(:,valid_trials));

%creates grapes
grapes = struct;
grapes.exp_type = 'RSVPSCR'; 
grapes.time_pre = TIME_PRE;
grapes.time_pos = TIME_POS;

grapes.ImageNames = cell(length(stimulus),1);
for j=1:length(stimulus)                      %loop over stimuli
    grapes.ImageNames{j} = stimulus(j).name;
end


if nseq > ceil(EARLY_RESPONSE_AT*experiment.Nseq) && ~isempty(early_fig_p) && strcmp(early_fig_p.State,'finished')
    sprintf('Early figures printed: time required: %0.f seconds.\n',toc(early_fig_tic))
    if DO_SORTING
        sorter.maxworkers = sorter.maxworkers + 1;
    end
end


if DO_SORTING
    sorter.do_remaining_sorting();
    sorter.save_sorting_results([result_folder filesep 'templates_ms.mat'])
end

grapes = update_grapes(grapes,pics_onset,stimulus,sp_index,channels,chan_label,true,[],[]);
save([result_folder filesep 'online_finalevents'],'Path_pics','stimulus',...
        'pics_onset','TIME_PRE','TIME_POS','nstd2plots','chan_label','channels')

cd(result_folder)

if strcmp(ms_id,'MATLAB:MKDIR:DirectoryExists')
    delete('*.png')
end

sorted_channles = [priority_channels setdiff(channels,priority_channels,'stable')];
futures_mu = loop_plot_best_responses_BCM_rank_online(sorted_channles,grapes,'y',[],[],[],1,nstd2plots,[]);
if DO_SORTING
    done_chs_ix = [];
    disp('Finishing Spike Sorting:')
    while numel(done_chs_ix) < numel(channels)
        [classes_out, ch_ix_out] = sorter.get_done_sorts(done_chs_ix); % this is necesarry because check that all spikes are forced
        if ~isempty(ch_ix_out)
            fprintf('Channels: %d/%d done, start plotting.\n',numel(ch_ix_out),numel(channels));
            grapes = update_grapes(grapes,pics_onset,stimulus,sp_index(ch_ix_out),channels(ch_ix_out),chan_label(ch_ix_out),false,classes_out,[]);
            futures = loop_plot_best_responses_BCM_rank_online(channels(ch_ix_out),grapes,'n',[],[],[],1,nstd2plots,[]);
            done_chs_ix = [done_chs_ix ch_ix_out];
        end
    end
    wait(futures)
    disp('Plotting Rasters Done.')
end

if params.debug
    save('testing_data.mat')
end
save grapes_online.mat grapes
wait(futures_mu)
%%

if DO_SORTING
    create_sorting_figs(chan_label,sorter.spikes,sorter.classes,'scr_online',conversion)
end

if isempty(outputinfo)
    outputinfo = 'Everything: OK';
end

try 
    disp('Starting Dynamic ranking.')
    dinamic_ranking_folder = [result_folder filesep  'dynamic_ranking'];
    [~,~,ms_id] = mkdir(dinamic_ranking_folder);
    cd(dinamic_ranking_folder)
    if strcmp(ms_id,'MATLAB:MKDIR:DirectoryExists')
        delete('*.png')
    end
    ImageNames = struct();
    ImageNames.name = grapes.ImageNames;
    ImageNames.folder = arrayfun(@(x) [params.pics_root_processing filesep experiment.Path_pics],1:numel(ImageNames.name),'UniformOutput' , false);
    grapes.ImageNames = ImageNames;
    sigma_gauss = 10;
    alpha_gauss = 3.035;
    resp_conf = struct;
    resp_conf.from_onset = 1;
    resp_conf.smooth_bin=1500;
    resp_conf.min_spk_median=0.5;
    resp_conf.tmin_median=200;
    resp_conf.tmax_median=700;
    resp_conf.psign_thr = 0.05;
    resp_conf.t_down=20;
    resp_conf.over_threshold_time = 75;
    resp_conf.below_threshold_time = 100;
    resp_conf.nstd = nstd2plots;
    resp_conf.win_cent=1;
    
    ifr_resolution_ms = 1;

    ifr_calclator = IFRCalculator(alpha_gauss,sigma_gauss,ifr_resolution_ms,par.sr,TIME_PRE,TIME_POS);

    [data, rank_config] = create_responses_data_parallel(grapes,1:numel(grapes.ImageNames.name),{'mu','class'},ifr_calclator,resp_conf);
    %%
    sorted_datat = sort_responses_table(data);
    channel_grapes = struct;
    channel_grapes.time_pre = grapes.time_pre;
    channel_grapes.time_pos = grapes.time_pos;
    channel_grapes.exp_type =  grapes.exp_type; 
    channel_grapes.ImageNames =  grapes.ImageNames;
    channel_grapes.rasters = struct;
    [G, labels] = findgroups(sorted_datat.channel);

    %%
    ismu = cellfun(@(x) strcmp(x,'mu'),sorted_datat.class);
    futures = repmat(parallel.FevalFuture,numel(labels),1);
    for i = 1:numel(labels)    
        channel_grapes.rasters.(labels{i}).mu = grapes.rasters.(labels{i}).mu;
        futures(i) = parfeval(@loop_plot_responses_BCM_online,0,sorted_datat(G==i & ismu,:), channel_grapes,1,1,rank_config,ifr_calclator.ejex,true,sprintf('dinamic_rank_mu_%s',labels{i}),true);
        channel_grapes.rasters = struct;
    end
    %%
    for i=1:numel(labels)
        rasternames = fieldnames(grapes.rasters.(labels{i}));
        for rn = 1: numel(rasternames)
            rastername = rasternames{rn};
            if ~contains(rastername,'class')
                continue
            end
            channel_grapes.rasters.(labels{i}).(rastername) = grapes.rasters.(labels{i}).(rastername);
            thiscl = cellfun(@(x) strcmp(x,rastername),sorted_datat.class);
            futures(end+1) = parfeval(@loop_plot_responses_BCM_online,0,sorted_datat(G==i & thiscl,:), channel_grapes,1,1,rank_config,ifr_calclator.ejex,true,sprintf('dinamic_rank_%s_%s',rastername,labels{i}),true);
            channel_grapes.rasters = struct;    
        end
    end
    wait(futures)
    
    disp('Dynamic ranking ended.')
catch ME
    warning('Dinamic_ranking plotting, did not work. \nError in function %s() at line %d.\nError Message:\n%s', ...
     ME.stack(1).name, ME.stack(1).line, ME.message)
end

mat_comm.send(experiment.msgs_Mat.process_end)

cd(online_dir)
questdlg(outputinfo,'Online Report','Ok','Ok');
if par.preprocessing
    movefile([params.processing_rec_metadata filesep 'pre_processing_info.mat'],[result_folder filesep 'pre_processing_info.mat']  );
end
if use_templates
	movefile(TEMPLATES_FILE,[result_folder filesep 'used_templates.mat']);
end
if ~isempty(MAPFILE)
    copyfile(MAPFILE , result_folder);
end
if DO_WAVECLUS
    ws_folder = [result_folder filesep  'with_wc'];
    [~,~,ms_id] = mkdir(ws_folder);
    cd(ws_folder)
    if strcmp(ms_id,'MATLAB:MKDIR:DirectoryExists')
        delete('*.png');
        delete('times_*.mat');
    end
    disp('Starting waveclus processes...')
    
    
    futures = repmat(parallel.FevalFuture,numel(chan_label),1);
    for i = 1:numel(chan_label)    
        futures(i) = parfeval(@sort_and_plot_spikes,0,chan_label{i},sorter.spikes{i}, sp_index{i}, par,ws_folder);
    end
    wait(futures);
    disp('Waveclus processes ended.')

    wc_classes = cell(numel(chan_label),1);
    max_std = 3;
    cls_maxdist = cell(numel(chan_label),1);
    cls_centers = cell(numel(chan_label),1);
    
    for i = 1:numel(chan_label)
        timefile = [ws_folder filesep 'times_' chan_label{i} '.mat'];
        if ~exist(timefile,'file')
            cls_centers{i} = nan;
            cls_maxdist{i} = inf;
            wc_classes{i} = [];
            continue
        end
        load(timefile,'forced','cluster_class');
        wc_classes{i} = cluster_class(:,1)';
        sp4template = (forced == 0);
        [centers, sd, ~] = build_templates(wc_classes{i}(sp4template), sorter.spikes{i}(sp4template,:));
        cls_centers{i} = centers;
        cls_maxdist{i} = (max_std*sd).^2;
    end

    save([result_folder filesep 'templates_wc.mat'], 'cls_centers', 'cls_maxdist', 'channels', 'max_std')
    create_sorting_figs(chan_label,sorter.spikes,wc_classes,'scr_online',conversion)
    make_wc_rasters([], 0)
    make_wc_rasters([], 1)
    cd(online_dir)
end

custompath.rm()
end