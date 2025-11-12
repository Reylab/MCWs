function [stims_sorted,IFR_thr,ejex,IFR,onset,tons,dura,good_lat,zscore,median_post,p_value_sign] = create_ranking(active_cluster_stim,from_onset,min_spk_median,psign_thr,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms,tmin_median,tmax_median,t_down,over_threshold_time,below_threshold_time,nstd,win_cent)

tmin_base = -900;
tmax_base = -100;
FR_resol = 10; %bin size in ms
% nstd = 4;
win_length = tmax_median - tmin_median;

base_spikes_4mean = [];  % to compute baseline activity
lstim = length(active_cluster_stim);
IFR = NaN; onset = NaN; tons = NaN; dura = NaN; good_lat = false; ejex = NaN;

if iscell(active_cluster_stim{1})
    ntrials  = cellfun(@numel,active_cluster_stim);
else
    [ntrials, ~]  = cellfun(@size,active_cluster_stim);
end
sp_count_base = NaN*ones(lstim,max(ntrials));
sp_count_post = NaN*ones(lstim,max(ntrials));

p_value_sign = NaN*ones(lstim,1);

if ~from_onset
    for ist =1:lstim
        spikes1 = active_cluster_stim{ist};
        if iscell(spikes1) %current grapes with cells
            for jj=1:numel(spikes1)
                base_spikes_4mean = [base_spikes_4mean ; histcounts(spikes1{jj}((spikes1{jj}< tmax_base) & (spikes1{jj}> tmin_base)),(tmin_base:FR_resol:tmax_base))];
                sp_count_base(ist,jj)=sum((spikes1{jj}< -tmin_median) & (spikes1{jj}> -tmax_median));
                %without onset
                sp_count_post(ist,jj)=sum((spikes1{jj}< tmax_median) & (spikes1{jj}> tmin_median));
            end
        else
            for jj=1:size(spikes1,1)
                base_spikes_4mean = [base_spikes_4mean ; histcounts(spikes1(jj,(spikes1(jj,:)< tmax_base) & (spikes1(jj,:)> tmin_base)),(tmin_base:FR_resol:tmax_base))];
                sp_count_base(ist,jj)=sum((spikes1(jj,:)< -tmin_median) & (spikes1(jj,:)> -tmax_median));
                %without onset
                sp_count_post(ist,jj)=sum((spikes1(jj,:)< tmax_median) & (spikes1(jj,:)> tmin_median));
            end
        end
        [p_value_sign(ist),~] = signtest(sp_count_base(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),sp_count_post(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),'Tail','left');
    end
    mu_baseFR = mean(base_spikes_4mean)*1000/FR_resol;
    IFR_thr = mean(mu_baseFR) + nstd*std(mu_baseFR);
else
    IFR = cell(lstim,1);
    onset = NaN*ones(lstim,1);
    tons = NaN*ones(lstim,1);
    dura = NaN*ones(lstim,1);
    good_lat = false(lstim,1);
    
    for ist =1:lstim
        spikes1 = active_cluster_stim{ist};
        [ejex,IFR{ist}]= get_IFR_BCM(spikes1,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms);
    end
    ifr_mat = cell2mat(IFR);
    mu_IFR = mean(ifr_mat);
    std_IFR = std(ifr_mat);
    times_base = ejex<tmax_base & ejex>tmin_base;
    IFR_thr = mean(mu_IFR(times_base) + nstd*std_IFR(times_base));
%     IFR_thr = mean(mu_IFR(times_base)) + nstd*std(mu_IFR(times_base));

    for ist =1:lstim
        [good_lat(ist),onset(ist),dura(ist)]= get_latency_BCM(IFR{ist},ejex,sample_period,IFR_thr,t_down,over_threshold_time,below_threshold_time);
    end
        
    for ist =1:lstim
        spikes1 = active_cluster_stim{ist};
        if good_lat(ist)
%             tons(ist) = onset(ist);
            if win_cent
                tons(ist) = max(onset(ist)-(win_length-min(dura(ist),win_length))/2,50);
            else
                tons(ist) = onset(ist)-100;
            end
        else
            tons(ist) = tmin_median;
        end
        if iscell(spikes1) %current grapes with cells
            for jj=1:numel(spikes1)
                sp_count_base(ist,jj)=sum((spikes1{jj}< -tmin_median) & (spikes1{jj}> -tmax_median));
                sp_count_post(ist,jj)=sum((spikes1{jj}< tons(ist)+win_length) & (spikes1{jj}> tons(ist)));
            end  
        else
            for jj=1:size(spikes1,1)
                sp_count_base(ist,jj)=sum((spikes1(jj,:)< -tmin_median) & (spikes1(jj,:)> -tmax_median));
                sp_count_post(ist,jj)=sum((spikes1(jj,:)< tons(ist)+win_length) & (spikes1(jj,:)> tons(ist)));
            end  
        end
        [p_value_sign(ist),~] = signtest(sp_count_base(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),sp_count_post(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),'Tail','left');
    end    
end
medians_base = nanmedian(sp_count_base,2);
mu_base_med = mean(medians_base);
std_base = std(medians_base);        
median_post = nanmedian(sp_count_post,2);
zscore = (median_post - mu_base_med)/std_base;

if std_base==0
    [~, indz] = sort(median_post,'descend');        
else
    [~, indz] = sort(zscore,'descend');
end
rem_stim = setdiff(indz,find(median_post<=min_spk_median),'stable');        
if isempty(rem_stim)
    stims_sorted = indz;
else
    stims_sorted = setdiff(indz,rem_stim,'stable'); 
    stims_sign = find(p_value_sign<=psign_thr);
    stims_sorted = [setdiff(rem_stim,stims_sign,'stable') ; stims_sorted];
    stims_sorted = [setdiff(indz,stims_sorted,'stable') ; stims_sorted];
end 