function make_wc_rasters(use_channels, dynamic_rank )
%function to make new figures using 
%run from online folder
%it requires be in a folder with times and on level up the finalevents
% Default: all channels

%Some variables like nstd2plots, TIME_PRE, TIME_POS are taken from the
%caller enviroment

if ~exist('dynamic_rank','var') || isempty(dynamic_rank),  dynamic_rank=false; end
load([fileparts(pwd) filesep  'online_finalevents']);

if ~exist('use_channels','var') || isempty(use_channels)
    ch_ixs = true(1,numel(channels));
else
    ch_ixs = arrayfun(@(x) any(use_channels == x),channels);
end

classes_out = {};
sp_index = {};
for i=1:numel(ch_ixs)
    if ~ch_ixs(i)
        continue
    end
    times_file = ['times_' chan_label{i}   '.mat'];
    if ~exist(times_file,'file')
        ch_ixs(i) = false;
        continue
    end
    load(times_file, 'cluster_class','par')
    classes_out{end+1} = cluster_class(:,1);
    sp_index{end+1} = cluster_class(:,2)';
end
if sum(ch_ixs) == 0
    return
end

grapes = struct;
grapes.exp_type = 'RSVPSCR'; 
grapes.time_pre = TIME_PRE;
grapes.time_pos = TIME_POS;
grapes = update_grapes(grapes,pics_onset,stimulus,sp_index,channels(ch_ixs),chan_label(ch_ixs),false,classes_out,[]);
grapes.ImageNames = cell(length(stimulus),1);
for j=1:length(stimulus)                      %loop over stimuli
    grapes.ImageNames{j} = stimulus(j).name;
end
if ~dynamic_rank
    if sum(ch_ixs) == 1
        loop_plot_best_responses_BCM_rank_online(channels(ch_ixs),grapes,'n',[],[],[],1,nstd2plots,[],0);
    else
        futures = loop_plot_best_responses_BCM_rank_online(channels(ch_ixs),grapes,'n',[],[],[],1,nstd2plots,[],1);
        wait(futures)
    end

else
    ImageNames = struct();
    ImageNames.name = grapes.ImageNames;
    ImageNames.folder = arrayfun(@(x) Path_pics,1:numel(ImageNames.name),'UniformOutput' , false);
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
    
    
    ifr_calclator = IFRCalculator(alpha_gauss,sigma_gauss,1000/par.sr,par.sr,TIME_PRE,TIME_POS);

    [data, rank_config] = create_responses_data_parallel(grapes,1:numel(grapes.ImageNames.name),{'class'},ifr_calclator,resp_conf);
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
    futures = repmat(parallel.FevalFuture,0,1);
    
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
            futures(end+1) = parfeval(@loop_plot_responses_BCM_online,0,sorted_datat(G==i & thiscl,:), channel_grapes,1,1,rank_config,ifr_calclator.ejex,true,sprintf('dynamic_rank_%s_%s',rastername,labels{i}),true);
            channel_grapes.rasters = struct;    
        end
    end
    wait(futures)
    
end
end
