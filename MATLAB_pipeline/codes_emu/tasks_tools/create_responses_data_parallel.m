function [data, rank_config] = create_responses_data_parallel(grapes,selected_stim,unit_label, ...
                                                              ifr_calclator,resp_conf,remove_channels, ...
                                                              priority_channels)
%This funtion creates 
%output:
% -data: a table with the responses metrics and ifr of each stimulus
%presented
% -rank_config: struct with the parameters and internal macros of the
% constructed responses

    data=table;
    channels = fieldnames(grapes.rasters);
    channels_order = 1:numel(channels);
    min_ifr_thr_list = ones(numel(channels),1) * resp_conf.min_ifr_thr;
    if exist('remove_channels','var') && ~isempty(remove_channels)
        remove_channels = arrayfun(@(x)['chan' num2str(x)],remove_channels,'UniformOutput',false);
        channels = channels(~ismember(channels, remove_channels));
    end
    if exist('priority_channels','var') && ~isempty(priority_channels)
        priority_channels = arrayfun(@(x)['chan' num2str(x)],priority_channels,'UniformOutput',false);
        [~, inds_first] = ismember(priority_channels, channels);
        inds_others = setdiff(1:numel(channels), inds_first(inds_first > 0));
        channels_order = [inds_first(inds_first > 0), inds_others];
        priority_inds = ismember(channels, priority_channels);
        min_ifr_thr_list(priority_inds) = 3;  % Set to 3 Hz for priority channels
    end
    
	win_length = resp_conf.tmax_median - resp_conf.tmin_median;
    rank_config.tmin_base = resp_conf.tmin_base;
    rank_config.tmax_base = resp_conf.tmax_base;
    rank_config.FR_resol = resp_conf.FR_resol;
	rank_config.win_cent = resp_conf.win_cent;
	rank_config.nstd = resp_conf.nstd;
	rank_config.win_length = win_length;
    rank_config.min_ifr_thr = resp_conf.min_ifr_thr;
    
    futures(1:numel(channels)) = parallel.FevalFuture;
%     for chi = 1:numel(channels)
% %     for chi=channels_order
% %         data_aux_p=get_responses_channel(grapes.rasters.(channels{chi}),channels(chi),selected_stim,unit_label,ifr_calclator,resp_conf,rank_config);
% %         data = [data;data_aux_p];
% %     end
%     
    if numel(channels)>1
        for chi=channels_order
            futures(chi) = parfeval(@get_responses_channel,1,grapes.rasters.(channels{chi}), channels(chi), ...
                                                             selected_stim, unit_label, ifr_calclator, ...
                                                             resp_conf, rank_config, min_ifr_thr_list(chi));
        end
        for i=1:numel(channels)
            wait(futures(i));  
            data_aux_p = fetchOutputs(futures(i));
            data = [data;data_aux_p];
        end
    else
        data=get_responses_channel(grapes.rasters.(channels{1}), channels(1), ...
                                   selected_stim, unit_label, ifr_calclator, ...
                                   resp_conf,rank_config, min_ifr_thr_list(1));
    end

% if numel(channels)>1
%         for chi = 1:numel(channels)
%             futures(chi) = parfeval(@get_responses_channel,1,grapes.rasters.(channels{chi}),channels(chi),selected_stim,unit_label,ifr_calclator,resp_conf,rank_config);
%         end
%         for i=1:numel(channels)
%             wait(futures(i));  
%             data_aux_p = fetchOutputs(futures(i));
%             data = [data;data_aux_p];
%         end
%     else
%         data=get_responses_channel(grapes.rasters.(channels{1}),channels(1),selected_stim,unit_label,ifr_calclator,resp_conf,rank_config);
%     end
end

function data=get_responses_channel(chrasters, chlabel, selected_stim, ...
                                    unit_label, ifr_calclator, ...
                                    resp_conf, rank_config, min_ifr_thr)
    min_trial_for_median = 3;
    data=table;
    try
        classes = fieldnames(chrasters);
        classes = classes(cellfun(@(x)  startsWith( x , unit_label ) ,classes));
    
        for cli = 1:numel(classes)
            class = classes{cli};
            active_cluster_stim = chrasters.(class).stim(selected_stim);
    
    
            base_spikes_4mean = [];  % to compute baseline activity
            lstim = length(active_cluster_stim);
            IFR = NaN; onset = NaN; tons = NaN; dura = NaN; good_lat = false; ejex = NaN;
            if iscell(active_cluster_stim{1})
                nrows = max(cellfun(@(x) numel(x),active_cluster_stim));
            else
                [nrows, ~]  = cellfun(@size,active_cluster_stim);
            end
            sp_count_base = NaN*ones(lstim,max(nrows));
            sp_count_post = NaN*ones(lstim,max(nrows));
            p_value_sign = NaN*ones(lstim,1);
            ntrials = zeros(lstim,1);
            if ~resp_conf.from_onset
                for ist =1:lstim
                    spikes1 = active_cluster_stim{ist};
                    if iscell(spikes1)
                        for jj=1:numel(spikes1)
                            base_spikes_4mean = [base_spikes_4mean ; histcounts(spikes1{jj}((spikes1{jj}< rank_config.tmax_base) & (spikes1{jj}> rank_config.tmin_base)),(rank_config.tmin_base:rank_config.FR_resol:rank_config.tmax_base))];
                            sp_count_base(ist,jj)=sum((spikes1{jj}< -resp_conf.tmin_median) & (spikes1{jj}> -resp_conf.tmax_median));
                            %without onset
                            sp_count_post(ist,jj)=sum((spikes1{jj}< resp_conf.tmax_median) & (spikes1{jj}> resp_conf.tmin_median));
                        end  
                    else
                        for jj=1:size(spikes1,1)
                            base_spikes_4mean = [base_spikes_4mean ; histcounts(spikes1(jj,(spikes1(jj,:)< rank_config.tmax_base) & (spikes1(jj,:)> rank_config.tmin_base)),(rank_config.tmin_base:rank_config.FR_resol:rank_config.tmax_base))];
                            sp_count_base(ist,jj)=sum((spikes1(jj,:)< -resp_conf.tmin_median) & (spikes1(jj,:)> -resp_conf.tmax_median));
                            %without onset
                            sp_count_post(ist,jj)=sum((spikes1(jj,:)< resp_conf.tmax_median) & (spikes1(jj,:)> resp_conf.tmin_median));
                        end  
                    end
                    [p_value_sign(ist),~] = signtest(sp_count_base(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),sp_count_post(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),'Tail','left');
                end
                mu_baseFR = mean(base_spikes_4mean)*1000/rank_config.FR_resol;
                IFR_thr = mean(mu_baseFR) + resp_conf.nstd*std(mu_baseFR);
            else
                IFR = cell(lstim,1);
                onset = NaN*ones(lstim,1);
    
                tons = NaN*ones(lstim,1);
                dura = NaN*ones(lstim,1);
                good_lat = false(lstim,1);
    
                for ist =1:lstim
                    spikes1 = active_cluster_stim{ist};
                    ntrials(ist) = numel(spikes1);
                    IFR{ist} = ifr_calclator.get_ifr(spikes1);
                end
                IFR_mat = cell2mat(IFR);
    %             mu_IFR = mean(IFR_mat);
    %             std_IFR = std(IFR_mat);
                mu_IFR = mean(IFR_mat(ntrials>min_trial_for_median,:));
                std_IFR = std(IFR_mat(ntrials>min_trial_for_median,:));
                
                times_base = ifr_calclator.ejex<rank_config.tmax_base & ifr_calclator.ejex>rank_config.tmin_base;
                IFR_thr = mean(mu_IFR(times_base) + resp_conf.nstd*std_IFR(times_base));
                if IFR_thr < min_ifr_thr || isnan(IFR_thr)
                    IFR_thr = min_ifr_thr;
                end
                
                for ist =1:lstim               
                    [good_lat(ist),onset(ist),dura(ist)]= get_latency_BCM(IFR{ist},ifr_calclator.ejex,ifr_calclator.sample_period,IFR_thr,resp_conf.t_down,resp_conf.over_threshold_time,resp_conf.below_threshold_time);
                end
    
                for ist =1:lstim                
                    spikes1 = active_cluster_stim{ist};
                    if good_lat(ist)
            %             tons(ist) = onset(ist);
                        if resp_conf.win_cent
                            tons(ist) = max(onset(ist)-(rank_config.win_length-min(dura(ist),rank_config.win_length))/2,50);
                        else
                            tons(ist) = onset(ist)-100;
                        end
                    else
                        tons(ist) = resp_conf.tmin_median;
                    end
                    if iscell(spikes1)
                        for jj=1:numel(spikes1)
                            sp_count_base(ist,jj)=sum((spikes1{jj}< -resp_conf.tmin_median) & (spikes1{jj}> -resp_conf.tmax_median));
                            sp_count_post(ist,jj)=sum((spikes1{jj}< tons(ist)+rank_config.win_length) & (spikes1{jj}> tons(ist)));
                        end
                    else
                        for jj=1:size(spikes1,1)
                            sp_count_base(ist,jj)=sum((spikes1(jj,:)< -resp_conf.tmin_median) & (spikes1(jj,:)> -resp_conf.tmax_median));
                            sp_count_post(ist,jj)=sum((spikes1(jj,:)< tons(ist)+rank_config.win_length) & (spikes1(jj,:)> tons(ist)));
                        end
                    end
                    [p_value_sign(ist),~] = signtest(sp_count_base(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),sp_count_post(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),'Tail','left');
                end    
            end
            medians_base = median(sp_count_base,2,'omitnan');
    %         mu_base_med = mean(medians_base);
    %         std_base = std(medians_base);   
            mu_base_med = mean(medians_base(ntrials>min_trial_for_median),'omitnan');
            std_base = std(medians_base(ntrials>min_trial_for_median),'omitnan');   
            fprintf('%s (%s) mu_base_med:%.2f std_base_med:%.2f (ifr_t:%.2fHz(mu:%.2f,std:%.2f))\n', ...
                                                chlabel{1}, class, mu_base_med, std_base, ...
                                                IFR_thr, mean(mu_IFR), mean(std_IFR));
    
            median_post = median(sp_count_post,2,'omitnan');
            zscore = (median_post - mu_base_med) / (std_base + 1e-10);
            min_spk_test = median_post >= resp_conf.min_spk_median;
            p_test = p_value_sign <= resp_conf.psign_thr;
            mu_b = repmat(mu_base_med,lstim,1);
            s_b= repmat(std_base,lstim,1);
            data_u = table(ntrials,IFR,onset,tons,dura,good_lat,zscore,median_post,mu_b,s_b,p_value_sign,min_spk_test,p_test);
            data_u.IFR_thr = ones(size(data_u,1),1)*IFR_thr;
            data_u.stim_number = selected_stim(:);
            data_u.class = repmat({class}, size(data_u,1), 1);
            data_u.channel = repmat(chlabel, size(data_u,1), 1);
            data = [data; data_u];
        end
    catch ME
        fprintf('Could not compute metrics for %s (%s)\n', ...
                                                chlabel{1}, class);
        errMsg = getReport(ME);
        disp(errMsg);
    end
end
