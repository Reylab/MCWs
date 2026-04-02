function make_wc_rasters_online(channels_nums,nfigures,experiment,scr_config_cell,times_folder,online_data_folder,pics_folder)
% Default: all channels
% after changing the WC solution, run from folder results/with_wc to get
% the new raster (use the edited channel numbers only)

if ~exist('nfigures','var') || isempty(nfigures)
    nfigures=1;
end
if ~exist('times_folder','var') || isempty(times_folder)
    times_folder = '';
end
if ~exist('online_data_folder','var') || isempty(online_data_folder)
    online_data_folder = '';
end
if ~exist('pics_folder','var')
    pics_folder = '';
end

TEMPLATE_FILE_WC = 'templates_wc.mat';
MAX_STD = 3; %max standard deviations to force spikesinto class
online_data = fullfile(online_data_folder ,'online_data.mat');

load(online_data,'wc_sp_index_cell', 'stimulus_cell', 'pics_onset_cell','resp_conf', 'chan_label','channels');
if ~exist('experiment','var')
    load(['..' filesep '..' filesep 'experiment_properties_online3'],'experiment','scr_config_cell')
end
ifr_calculator = IFRCalculator(resp_conf.alpha_gauss,resp_conf.sigma_gauss,resp_conf.ifr_resolution_ms,resp_conf.sr,resp_conf.TIME_PRE,resp_conf.TIME_POS);
 
if exist('channels_nums','var') && ~isempty(channels_nums)
    ch_idx = [];
    for i =1:numel(chan_label)
        if any(channels_nums==channels(i))
            ch_idx(end+1) = i;
        end
    end
else
    ch_idx = 1:numel(chan_label);
end
if isempty(ch_idx)
    return
end
all_picsused = unique(cell2mat(cellfun(@(x)x.pics2use, scr_config_cell, 'UniformOutput', false)));

final_n_scr = numel(wc_sp_index_cell);

if exist(TEMPLATE_FILE_WC,'file')
    template_data = load(TEMPLATE_FILE_WC, 'cls_centers', 'cls_maxdist', 'channels', 'max_std');
    ch2tmp_ix = zeros(1,numel(ch_idx));
    for i=1:numel(ch_idx)
        tmp_ix= find(template_data.channels == channels(ch_idx(i)));
        if isempty(tmp_ix)
            ch2tmp_ix(i) = numel(template_data.cls_centers)+1;
            template_data.cls_centers{ch2tmp_ix(i)} = nan;
            template_data.cls_maxdist{ch2tmp_ix(i)} = inf;
            template_data.channels(ch2tmp_ix(i)) = channels(ch_idx(i));
        else
            ch2tmp_ix(i) = tmp_ix;
            if template_data.max_std ~= MAX_STD
                template_data.cls_maxdist{tmp_ix} = (sqrt(template_data.cls_maxdist{tmp_ix}).*(MAX_STD/template_data.max_std)).^2;
            end
        end

    end
else
    ch2tmp_ix = 1:numel(ch_idx);
    template_data = struct;
    template_data.channels = channels(ch_idx);
    template_data.cls_maxdist = cell(numel(ch_idx),1);
    template_data.cls_maxdist(:)  = {inf}; 
    template_data.cls_centers = cell(numel(ch_idx),1);
    template_data.cls_centers(:) = {nan};
end
template_data.max_std = MAX_STD;
channel_grapes = struct;
channel_grapes.exp_type = 'ONLINE_RSVPSCR'; 
channel_grapes.time_pre = resp_conf.TIME_PRE;
channel_grapes.time_pos = resp_conf.TIME_POS;
channel_grapes.ImageNames = experiment.ImageNames;
if isempty(pics_folder)
    channel_grapes.ImageNames.folder = fullfile( experiment.params.pics_root_processing,experiment.ImageNames.folder); %in grapes, the pics have the full path
else
    channel_grapes.ImageNames.folder(:) = {pics_folder};
end
futures = repmat(parallel.FevalFuture,1,0);
for chi = 1:numel(ch_idx)
    times_file = fullfile(times_folder,['times_' chan_label{ch_idx(chi)}   '.mat']);
    if ~exist(times_file,'file')
        continue
    end
    grapes_ch = struct;
    grapes_ch.exp_type = 'ONLINE_RSVPSCR'; 
    grapes_ch.time_pre = resp_conf.TIME_PRE;
    grapes_ch.time_pos = resp_conf.TIME_POS;
    grapes_ch.ImageNames = experiment.ImageNames;
    load(times_file, 'cluster_class','forced','spikes')
    classes_out = cluster_class(:,1);
    init_idx = 1;
    for nsc = 1:final_n_scr
        sp_times = wc_sp_index_cell{nsc}{ch_idx(chi)};
        nsps = numel(sp_times);
        grapes_ch = update_grapes(grapes_ch,pics_onset_cell{nsc},stimulus_cell{nsc},{sp_times},channels(ch_idx(chi)),chan_label(ch_idx(chi)),false,{classes_out(init_idx:init_idx+nsps-1)}, scr_config_cell{nsc}.pics2use,nsc);
        init_idx = init_idx + nsps;
    end
    [data, rank_config] = create_responses_data_parallel(grapes_ch,all_picsused,{'class'},ifr_calculator,resp_conf);
    data = sort_responses_table(data);
    chname = ['chan' num2str(channels(ch_idx(chi)))];
    rasternames = fieldnames(grapes_ch.rasters.(chname));
    if ~isempty(futures)
        wait(futures);
    end
    futures = repmat(parallel.FevalFuture,1,0);
    channel_grapes.rasters = struct;
    for rn = 1: numel(rasternames)
        rastername = rasternames{rn};
        if ~contains(rastername,'class')
            continue
        end
        isthisclass = cellfun(@(x) strcmp(x,rastername),data.class);
        channel_grapes.rasters.(chname).(rastername) = grapes_ch.rasters.(chname).(rastername);
        futures(end+1) = parfeval(@loop_plot_responses_BCM_online,0,data(isthisclass ,:), channel_grapes,final_n_scr,nfigures,rank_config,ifr_calculator.ejex,true,sprintf('finalwc_ch_%s_%s',chan_label{ch_idx(chi)},rastername),true);
        channel_grapes.rasters.(chname) = struct;
    end
    
    sp4template = (forced == 0);
    [centers, sd, ~] = build_templates(classes_out(sp4template), spikes(sp4template,:));
    template_data.cls_centers{ch2tmp_ix(chi)} = centers;
    template_data.cls_maxdist{ch2tmp_ix(chi)} = (MAX_STD*sd).^2;
    
end


save(TEMPLATE_FILE_WC, '-struct','template_data')

end
