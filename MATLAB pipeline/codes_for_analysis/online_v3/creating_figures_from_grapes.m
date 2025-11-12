load('D:\fertest\EMU-005_task-RSVPscr_dynamic_run-03\results\grapes_online.mat')
load('D:\fertest\EMU-005_task-RSVPscr_dynamic_run-03\experiment_properties_online3.mat')



grapes = revert_grapes(grapes,2,scr_config_cell);





path_pics = 'D:\fertest\EMU-008_task-RSVPscr_dynamic_run-04\pics_used';
grapes.ImageNames.folder(:) = {path_pics};
MAX_NTRIALS = 15+5;
MAX_RASTERS_PER_STIM = 3;
RASTER_SIMILARITY_THR = 0.85;
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
resp_conf.sr = 30000;
resp_conf.TIME_PRE = grapes.time_pre;
resp_conf.TIME_POS = grapes.time_pos;

ifr_calculator= IFRCalculator(resp_conf.alpha_gauss,resp_conf.sigma_gauss,resp_conf.ifr_resolution_ms,resp_conf.sr,resp_conf.TIME_PRE,resp_conf.TIME_POS);
n_scr = 2;
scr_config = scr_config_cell{n_scr};

[datat, rank_config] = create_responses_data_parallel(grapes, scr_config.pics2use,{'mu','class'},ifr_calculator,resp_conf,[]);

datat = sort_responses_table(datat);
%save sorted_datat?
enough_trials = datat.ntrials >= MAX_NTRIALS;
stim_rm = unique(datat(enough_trials,:).stim_number);

datat = datat(~enough_trials, :);
[stim_best,istim_best,~] = unique(datat.stim_number,'stable');
%data2plot = datat(istim_best,:);

data2plot = [];
nwin_choose = 2;

datat.ismu = cellfun(@(x) strcmp(x,'mu'),datat.class);
data2plot = datat(1,:);

for itable = 2:size(datat,1)
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
            fprintf('Not showing: channel %s, class %s, stim number %d ()\n',datat.channel{itable},datat.class{itable},datat.stim_number(itable))
        end
    end
    data2plot = [data2plot; datat(itable,:)];
    if size(data2plot,1)==(nwin_choose*20) %all the needed
        break
    end
end
clear datat

fc=loop_plot_responses_BCM_online(data2plot, grapes,n_scr,nwin_choose,rank_config,ifr_calculator.ejex,false,'select_win');

sel_ix = select_stims(fc);

