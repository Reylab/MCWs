function get_spikes_and_online_sorting()

TOTAL_time = 1; %minutes to acquire data
outputfolder = 'online_test'; %
%results currently saved in C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\experimental_files\outputfolder\with_wc

TRIAL_LEN = 30; %seconds
which_nsp = 2; %which nsp has micros


addpath(fullfile(fileparts(mfilename('fullpath')),'..')) % always / for this trick
custompath = reylab_custompath({'useful_functions','wave_clus_reylab','wave_split','codes_for_analysis','mex','tasks/.','tasks/000rsvpscr','tasks/online_v3/online'});
params = location_setup('BCM-BRK');

MAX_STD = 3; %max standard deviations to force spikesinto class, use to define max distance in templates
TEMPLATE_FILE_WC = 'templates_wc.mat'; %name fo templates file



Ntrials = ceil(TOTAL_time*60 / TRIAL_LEN);



chs_th_abs = [];
 chs_th_pos = [];
%chs_th_pos = [9 57];
remove_channels = [];
%maybe move to locations:
DO_SORTING = true;
%TEMPLATES_FILE = [params.processing_rec_metadata filesep  'templates.mat'];
MAX_SORTING_CORES = 16; %[not used really]

WAIT_LOOP = 1; %seconds between buffer clearings, max 3seg

remove_channels_by_label = {'^[^[mc]](.*)$','^(micro \d*)$','(ref-\d*)$'};

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

%% sorting parameters
SP2SORT = 10000;% [not implemented] spikes used to detect classes

NSP_TYPE = 265;

if strcmp(params.system,'BRK')
    MAPFILE = [];
else
    mapfiles = dir([params.processing_rec_metadata filesep params.mapfile]);
    if numel(mapfiles)>1
        error('multiple mapfiles found')
    elseif numel(mapfiles)==1
        MAPFILE = [params.processing_rec_metadata filesep mapfiles(1).name];
    else
        error('mapfile not found')
    end
end

poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool;
end
NOTCHS_USED = 5;
%for blackrock use:
address = {'192.168.137.3','192.168.137.178'}; %index using which_nsp-1
result_folder = fullfile(params.root_processing, outputfolder);

[~,~,ms_id] = mkdir(result_folder);
if par.preprocessing
    if exist([params.processing_rec_metadata filesep 'pre_processing_info.mat'],'file')
        load([params.processing_rec_metadata filesep 'pre_processing_info.mat'],'process_info')
    else
        error('pre_processing_info.mat not found')
    end
end

inst_num = which_nsp -1;
address = address{which_nsp};

if strcmp(params.system,'BRK')
    channel_id_offset = (inst_num>0)*(inst_num+1)*1000;
else
    channel_id_offset = 0; 
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

det_conf = -1*ones(size(channels));
det_conf(ismember(channels,chs_th_pos)) = 1;
det_conf(ismember(channels,chs_th_abs)) = 0;

detecctions = repmat({NaN},Ntrials,length(channels));
if DO_SORTING
    sorter = online_sorter(channels,[],SP2SORT,true, MAX_SORTING_CORES);
end

data = cell(length(channels),1);
for i = 1:length(data)
    data{i} = zeros(ceil(TRIAL_LEN*30000),1,'int16');
end
device_com('enable_chs', channels, true, []);

init_times = cell(Ntrials,1);

for nseq = 1:Ntrials
    fprintf('Starting block %d of %d\n',nseq,Ntrials )
    datacounter = zeros(length(channels),1);
    vec_tini_buffer = {};    
    device_com('clear_buffer');
    pause(0.2);
    start_trial = tic();
    data_loss_flag = false;
    channels_len = cell(length(channels),1);
    while toc(start_trial) < TRIAL_LEN
        start_loop = tic();
        streams = device_com('get_stream');
        
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
        end
        vec_tini_buffer{end+1} = streams.timestamp;
        pause(max(0.01,WAIT_LOOP - toc(start_loop)))
    end
    
    
    init_times{nseq} = vec_tini_buffer;
    
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

end
%%
clear data

sp_index = cell(size(detecctions,2),1);
for nseq = 1:Ntrials
    for j=1:size(detecctions,2) %All trials has to be included because of the one to one relatinship between detecctions and spikes 
        sp_index{j} =  [sp_index{j}, (double(detecctions{nseq,j})+init_times{nseq}{1}(j))/30]; %to ms
    end
end
ws_folder = [result_folder filesep  'with_wc'];
[~,~,ms_id] = mkdir(ws_folder);
min_spikes4SPC = 60;
nspikes = cellfun(@(x) size(x,1), sorter.spikes);
summary = table(channels',nspikes);
channels2sort = find(cellfun(@(x) size(x,1)>min_spikes4SPC, sorter.spikes));

writetable(summary,fullfile(ws_folder,'summary.csv'))
futures = repmat(parallel.FevalFuture,numel(channels2sort) ,1);

for i = 1:numel(channels2sort) 
    chi = channels2sort(i);
    futures(i) = parfeval(@sort_and_plot_spikes,0,chan_label{chi},sorter.spikes{chi}*conversion(chi), sp_index{chi}, par,ws_folder);
end
wait(futures)

%creates templates from times
template_data = struct;
template_data.channels = channels;
template_data.cls_maxdist = cell(numel(channels),1);
template_data.cls_maxdist(:)  = {inf}; 
template_data.cls_centers = cell(numel(channels),1);
template_data.cls_centers(:) = {nan};

for i = 1:numel(channels2sort) 
    chi = channels2sort(i);
    times_file = fullfile(ws_folder,['times_' chan_label{chi}   '.mat']);
    if ~exist(times_file,'file')
        continue
    end
    load(times_file, 'cluster_class','forced','spikes')
    classes_out = cluster_class(:,1);
    sp4template = (forced == 0);
    [centers, sd, ~] = build_templates(classes_out(sp4template), spikes(sp4template,:));
    template_data.cls_centers{chi} = centers;
    template_data.cls_maxdist{chi} = (MAX_STD*sd).^2;
    
end
save(fullfile(ws_folder,TEMPLATE_FILE_WC), '-struct','template_data')


fprintf('Results in: %s\n',ws_folder)
custompath.rm()
end