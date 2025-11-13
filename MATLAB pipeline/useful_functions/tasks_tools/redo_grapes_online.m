clear all
load('experiment_properties_online3.mat')
load('results\grapes_online.mat')
priority_chs_ranking = [257:296 321:328 353:369];


n_scr = 2;

grapesNEW.time_pre = grapes.time_pre;
grapesNEW.time_pos = grapes.time_pos;
grapesNEW.ImageNames = grapes.ImageNames;

channels = fieldnames(grapes.rasters);

if n_scr == 1
    tr = experiment.NREP(1);
    stims{1} = scr_config_cell{1}.pics2use;
elseif n_scr == 2
   tr = [experiment.NREP(1) experiment.NREP(2) sum(experiment.NREP(1:2))];
% tr1 = experiment.NREP(1);
% tr2 = experiment.NREP(2);
% tr12 = sum(experiment.NREP(1:2));
    stims{1} = setdiff(scr_config_cell{1}.pics2use,scr_config_cell{2}.pics2use);
    stims{2} = setdiff(scr_config_cell{2}.pics2use,scr_config_cell{1}.pics2use);
    stims{3} = intersect(scr_config_cell{1}.pics2use,scr_config_cell{2}.pics2use);
end
% stims1 = setdiff(scr_config_cell{1}.pics2use,scr_config_cell{2}.pics2use);
% stims2 = setdiff(scr_config_cell{2}.pics2use,scr_config_cell{1}.pics2use);
% stims12 = intersect(scr_config_cell{1}.pics2use,scr_config_cell{2}.pics2use);


for ich = 1:length(channels)
    grapesNEW.rasters.(channels{ich}).details = grapes.rasters.(channels{ich}).details;
    r_names = fieldnames(grapes.rasters.(channels{ich}));
    r_names = r_names(cellfun(@(x)  startsWith( x , {'class'} ) ,r_names));
    max_cl_raster = max(cellfun(@(x)  str2double(x(6:end)), r_names));
    for ind=1:length(tr)
        for ist=stims{ind}
            grapesNEW.rasters.(channels{ich}).mu.stim{ist} = grapes.rasters.(channels{ich}).mu.stim{ist}(1:tr(ind));
            for icl=1:max_cl_raster
                clname = ['class' num2str(icl)];
                grapesNEW.rasters.(channels{ich}).(clname).stim{ist} = grapes.rasters.(channels{ich}).(clname).stim{ist}(1:tr(ind));
            end
        end
    end
%     for ist=stims1
%         grapesNEW.rasters.(channels{ich}).mu.stim{ist} = grapes.rasters.(channels{ich}).mu.stim{ist}(1:tr1);
%         for icl=1:max_cl_raster
%             clname = ['class' num2str(icl)];
%             grapesNEW.rasters.(channels{ich}).(clname).stim{ist} = grapes.rasters.(channels{ich}).(clname).stim{ist}(1:tr1);
%         end
%     end
%     for ist=stims2
%         grapesNEW.rasters.(channels{ich}).mu.stim{ist} = grapes.rasters.(channels{ich}).mu.stim{ist}(1:tr2);
%         for icl=1:max_cl_raster
%             clname = ['class' num2str(icl)];
%             grapesNEW.rasters.(channels{ich}).(clname).stim{ist} = grapes.rasters.(channels{ich}).(clname).stim{ist}(1:tr2);
%         end
%     end
%     for ist=stims12
%         grapesNEW.rasters.(channels{ich}).mu.stim{ist} = grapes.rasters.(channels{ich}).mu.stim{ist}(1:tr12);
%         for icl=1:max_cl_raster
%             clname = ['class' num2str(icl)];
%             grapesNEW.rasters.(channels{ich}).(clname).stim{ist} = grapes.rasters.(channels{ich}).(clname).stim{ist}(1:tr12);
%         end
%     end
end

%%

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

TIME_PRE=1e3;
TIME_POS=2e3;
resp_conf.sigma_gauss = 10;
resp_conf.alpha_gauss = 3.035;
resp_conf.ifr_resolution_ms = 1;
resp_conf.sr = 30000;
resp_conf.TIME_PRE = TIME_PRE;
resp_conf.TIME_POS = TIME_POS;
not_online_channels = [];
priority_channels = [];
MAX_NTRIALS = 15;
MAX_RASTERS_PER_STIM = 3;
RASTER_SIMILARITY_THR = 0.85;

new_pics2load = experiment.NPICS(2:end)-(experiment.NPICS(1:end-1)-experiment.P2REMOVE(1:end-1)); %pics to load after each sub_screening
totalp2load = experiment.NPICS(1) + sum(new_pics2load); % total amount of pics in the library
available_pics = 1:totalp2load;

ifr_calculator= IFRCalculator(resp_conf.alpha_gauss,resp_conf.sigma_gauss,resp_conf.ifr_resolution_ms,resp_conf.sr,resp_conf.TIME_PRE,resp_conf.TIME_POS);

scr_config = scr_config_cell{n_scr};

selected2notremove = [];
[datat, rank_config] = create_responses_data_parallel(grapesNEW, scr_config.pics2use,{'mu','class'},ifr_calculator,resp_conf,not_online_channels,priority_channels);

%%
datat = sort_responses_table(datat,priority_chs_ranking);
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
                        continue
                    end
                end
                data2plot = [data2plot; datat(itable,:)];
                if size(data2plot,1)==(nwin_choose*20) %all the needed
                    break
                end
            end
%             clear datat

%             if scr_config.manual_select
%                 fc=loop_plot_responses_BCM_online(data2plot, grapes,n_scr,nwin_choose,rank_config,ifr_calculator.ejex,true,'select_win');
%                 sel_ix = select_stims(fc);
%                 sel_ix = sel_ix(1:min( numel(sel_ix), height(data2plot)));%remove selected out of options
%                 selected2rm = unique(data2plot(sel_ix==-1,:).stim_number,'stable');
%                 selected2explore = unique(data2plot(sel_ix==1,:).stim_number,'stable');
%                 stim_rm = [stim_rm; selected2rm];
%             else
%                 selected2explore = [];
%                 selected2rm = [];
%             end
%             selected2explore_cell{end+1}=selected2explore;
            selected2explore = selected2explore_cell{n_scr};
            extra_stim_rm = experiment.P2REMOVE(n_scr) - length(stim_rm);
            selected2notremove = [selected2notremove;selected2explore]; %CHECK RO W OR COL
            
%             if extra_stim_rm>0
%                 extra_stims = setdiff(stim_best,[selected2notremove;stim_rm],'stable');
%                 stim_rm = [stim_rm; extra_stims(end-extra_stim_rm+1:end)];
%             else
%                 stim_rm = stim_rm(1:(end+extra_stim_rm));
%             end
% 
%             % save grapes_online.mat grapes
%             %%
%             stim_rm_cell{end+1} = stim_rm;
%             removed_stims = cell2mat(stim_rm_cell');
            pic2add = [];
            if n_scr < numel(experiment.P2REMOVE) && ~isempty(selected2explore)
                added_counter = 0 ;
                for c = selected2explore(:)'
                    if added_counter == new_pics2load(n_scr)
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
                    for rp = related_pics'
%                         if all(rp~=removed_stims) && all(rp~=stim_best) && all(rp~=pic2add) %not used
                        if all(rp~=stim_rm) && all(rp~=stim_best) && all(rp~=pic2add) %not used
                            added_counter = added_counter + 1;
                            this_concept_counter = this_concept_counter +1; %to report latter
                            pic2add(end+1) = rp;
                            if added_counter == new_pics2load(n_scr)
                                break
                            end
                        end
                    end
                    fprintf('%d pictures added related to %s\n', this_concept_counter, experiment.ImageNames.concept_name{c})

                end
                selected2notremove = [selected2notremove ; pic2add(:)];  %%%%%%

                available_pics = [pic2add setdiff(available_pics,pic2add,'stable')];
                selected2notremove = unique(selected2notremove);
            end
            if extra_stim_rm>0
                extra_stims = setdiff(stim_best,[selected2notremove;stim_rm],'stable');
                if numel(extra_stims)>extra_stim_rm
                    stim_rm = [stim_rm; extra_stims(end-extra_stim_rm+1:end)];
                else
                    extra_stims = setdiff(stim_best,stim_rm,'stable');
                    if numel(extra_stims)>extra_stim_rm
                        stim_rm = [stim_rm; extra_stims(end-extra_stim_rm+1:end)];                    
                    end
                end
            else
                stim_rm = stim_rm(1:(end+extra_stim_rm));
            end          
            stim_rm_cell{end+1} = stim_rm;
            removed_stims = cell2mat(stim_rm_cell');
       
        available_pics_cell{end+1} = setdiff(available_pics,removed_stims,'stable');