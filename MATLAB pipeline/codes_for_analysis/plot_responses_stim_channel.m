function plot_responses_stim_channel(chaux,stims,nstd,is_gray,fixed_limy,phase)
% function plot_responses_stim_channel(chaux,stims,nstd,phase, concat_folder)
% Plots the best max_stims responses of a channel ch and cluster i and IFR of all responses.

% nstd = 3;
addpath('pics_used')
do_plot = 1;
from_onset = 1;
win_cent = 1;
min_trial_for_median = 5;
smooth_bin=1500/30;
sigma_gauss = 10;
alpha_gauss = 3.035;
% min_spk_median=1;
min_spk_median=0.5;
tmin_median=200;
tmax_median=700;
win_length = tmax_median - tmin_median;
strength_thr = 5;
psign_thr = 0.05;
t_down=20;
over_threshold_time = 75;
below_threshold_time = 100;
tmin_base = -900;
tmax_base = -100;
FR_resol = 10; %bin size in ms
load('experiment_properties_online3.mat','scr_config_cell')
all_picsused = unique(cell2mat(cellfun(@(x)x.pics2use, scr_config_cell, 'UniformOutput', false)));

% nstd = 4;

warning off
if ~exist('is_gray','var') || isempty(is_gray), is_gray=false; end
% if ~exist('concat_folder','var') || isempty(concat_folder), concat_folder='.'; end
if ~exist('fixed_limy','var') || isempty(fixed_limy), fixed_limy=0; end
if ~exist('effect_rows_2','var') || isempty(effect_rows_2), effect_rows_2=0; end
if ~exist('phase','var') || isempty(phase)
    grapes_name = 'grapes.mat';
    %     stimulus = load('stimulus','stimulus');
    %     stimulus = stimulus.stimulus;
    phase = [];
else
    grapes_name = ['grapes_' phase '.mat'];
    %     if strcmp(phase,'prescr') || strcmp(phase,'posscr')
    %         stimulus = load('stimulus','stimulus');
    %         stimulus = stimulus.stimulus;
    %     else
    %         stimulus = load(['stimulus_' phase '.mat'],'stimulus');
    %         stimulus = stimulus.stimulus;
    %     end
end

grapes = load(grapes_name);

time_pre_ms = grapes.time_pre;
time_pos_ms = grapes.time_pos;

pic_num=  [ 1:5  16:20 31:35 46:50];
rast_num= [ 6:10 21:25 36:40 51:55]';
hist_num= [11:15 26:30 41:45 56:60];
max_stims = 15;

rows = 12;
cols = 5;

col='brgcmybrgcmy';

fontsize_mult = 1.4;
LineFormat.Color = 'blue';
LineFormat.LineWidth = 0.35;

% for ic = 1:length(chaux)
set(groot,'defaultaxesfontsmoothing','off')
set(groot,'defaultfiguregraphicssmoothing','off')
set(groot,'defaultaxestitlefontsizemultiplier',1.1)
set(groot,'defaultaxestitlefontweight','normal')
set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})

% fig = figure('Visible','on','Position',[300   120   798   598.5],'Units','pixels','GraphicsSmoothing','off');

%     ch = chaux(ic);
ch = chaux;
chfield = ['chan' num2str(ch)];
%     texto = char(fieldnames(grapes.(['chan' num2str(ch)]).details));
% ncl = sum(contains(fieldnames(grapes.rasters.(chfield)),'class'));
ncl = sum(contains(fieldnames(grapes.rasters.(chfield)),'class'))+1;
%     [row,~]=find(texto(:,1:3)=='cla');
%     ncl = length(row);
clusters_to_plot = arrayfun(@(x)['class' num2str(x)],1:ncl-1,'UniformOutput',false)';
clusters_to_plot{end+1}='mu';
muonly_flag=0;

inds_cl = 1:ncl;

sample_period = 1000/grapes.rasters.(chfield).details.sr; % sample period for the spike list - window convolution in ms/sample
%     sample_period = 1000/NSx(find([NSx(:).chan_ID]==ch,1,'first')).sr; % sample period for the spike list - window convolution in ms/sample
%     sample_period = 300000/NSx(find([NSx(:).chan_ID]==ch,1,'first')).sr; % sample period for the spike list - window convolution in ms/sample
% %             sample_period = 1000/30000; % sample period for the spike list - window convolution in ms/sample
ch_label = grapes.rasters.(chfield).details.ch_label;
%     ch_label = NSx([NSx(:).chan_ID]==ch).output_name;
% % ch_label = sprintf('probando_ch%.2d',ch);
exp_type = grapes.exp_type;
% % exp_type = 'RSVPSCR';
zscore = NaN*ones(ncl,1);
p_value_sign = NaN*ones(ncl,1);
IFR = cell(ncl,1);
onset = NaN*ones(ncl,1);
tons = NaN*ones(ncl,1);
dura = NaN*ones(ncl,1);
good_lat = false(ncl,1);
IFR_thr = NaN*ones(ncl,1);
ifr_calclator = IFRCalculator(alpha_gauss,sigma_gauss,1,grapes.rasters.(chfield).details.sr,time_pre_ms,time_pos_ms);

for istnum=1:numel(stims)
    fig=figure(randi(100)+100);
    set(fig,'Visible','on','Position',[300   120   798   598.5],'Units','pixels','GraphicsSmoothing','off');
    stim_num=stims(istnum);

    for icl=inds_cl
        %         if muonly_flag
        % %             icl=size(clusters_to_plot,1);
        %             if icl~=size(clusters_to_plot,1)
        %                 continue
        %             end
        %         end
        %         clf;
        active_cluster = grapes.rasters.(chfield).(clusters_to_plot{icl}).stim;
%         whichstims = all_picsused;
%         lstim = length(active_cluster);
        lstim = length(all_picsused);
%         IFR_4thr = cell(lstim,1);
        IFR_4thr_2 = cell(lstim,1);


        %         [stims_sorted,IFR_thr,ejex,IFR,onset,tons,dura,good_lat,zscore,median_post,p_value_sign] = create_ranking(active_cluster.stim,from_onset,min_spk_median,psign_thr,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms,tmin_median,tmax_median,t_down,over_threshold_time,below_threshold_time,nstd,win_cent);

        %%% RANK with paired test and zscore median AND THEN PLOT??
        %         num_base = 0;
        %         base_spikes = 0;  % to compute baseline activity
        base_spikes_4mean = [];  % to compute baseline activity

        if iscell(active_cluster{1})
            ntrials  = cellfun(@numel,active_cluster);
        else
            [ntrials, ~]  = cellfun(@size,active_cluster);
        end
        %         [nrows, ~]  = cellfun(@size,active_cluster.stim);
        sp_count_base = NaN*ones(lstim,max(ntrials));
        sp_count_post = NaN*ones(1,max(ntrials));
        %         p_value_sign = NaN*ones(lstim,1);

        %         all_spks = [];
%         for ist =1:lstim
        for ist =all_picsused
            spikes1 = active_cluster{ist};
            %             for jj=1:size(spikes1,1)
            for jj=1:numel(spikes1)
                %                 num_base = num_base + 1;
                %                 base_spikes(num_base) = sum((spikes1(jj,:)< tmax_base) & (spikes1(jj,:)> tmin_base))/(tbase/1000); %in Hz
                base_spikes_4mean = [base_spikes_4mean ; histcounts(spikes1{jj}((spikes1{jj}< tmax_base) & (spikes1{jj}> tmin_base)),(tmin_base:FR_resol:tmax_base))];
                %                 base_spikes_4mean = [base_spikes_4mean ; histcounts(spikes1(jj,(spikes1(jj,:)< tmax_base) & (spikes1(jj,:)> tmin_base)),(tmin_base:FR_resol:tmax_base))];
                %                     all_spks=[all_spks cell2mat(spikes1')];
%                 [~,IFR_4thr{ist}]= get_IFR_BCM(spikes1,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms);
                IFR_4thr_2{ist} = ifr_calclator.get_ifr(spikes1);

                sp_count_base(ist,jj)=sum((spikes1{jj}< -tmin_median) & (spikes1{jj}> -tmax_median));
                %                 sp_count_base(ist,jj)=sum((spikes1(jj,:)< -tmin_median) & (spikes1(jj,:)> -tmax_median));
                %without onset
                %                 sp_count_post(ist,jj)=sum((spikes1(jj,:)< tmax_median) & (spikes1(jj,:)> tmin_median));
                %with onset
                %onset hasta onset+  (tmax_median-tmin_median)
            end
            %             [p_value_sign(ist),~] = signtest(sp_count_base(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),sp_count_post(ist,~isnan(sp_count_base(ist,:))&~isnan(sp_count_post(ist,:))),'Tail','left');
        end

        %         base_spikes = all_spks((all_spks< tmax_base+half_ancho_gauss) & (all_spks> tmin_baseFR-half_ancho_gauss));
        %         IFR_thr = mean(base_spikes) + nstd*std(base_spikes);
        %         mu_baseFR = mean(base_spikes_4mean)*1000/FR_resol;
        %         IFR_thr = mean(mu_baseFR) + nstd*std(mu_baseFR);
        IFR_4thr_2 = IFR_4thr_2(all_picsused);
        sp_count_base=sp_count_base(all_picsused,:);
                
        ifr_mat = cell2mat(IFR_4thr_2);
        mu_IFR = mean(ifr_mat(ntrials(all_picsused)>min_trial_for_median,:));
        std_IFR = std(ifr_mat(ntrials(all_picsused)>min_trial_for_median,:));

        spikes1 = active_cluster{stim_num};
%         [ejex,IFR{icl}]= get_IFR_BCM(spikes1,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms);
        IFR{icl}=IFR_4thr_2{all_picsused==stim_num};

%         if icl==ncl
%             disp('mu')
%         end
        times_base = ifr_calclator.ejex<tmax_base & ifr_calclator.ejex>tmin_base;
        IFR_thr(icl) = mean(mu_IFR(times_base) + nstd*std_IFR(times_base));
        [good_lat(icl),onset(icl),dura(icl)]= get_latency_BCM(IFR{icl},ifr_calclator.ejex,ifr_calclator.sample_period,IFR_thr(icl),t_down,over_threshold_time,below_threshold_time);

        if good_lat(icl)
            if win_cent
                tons(icl) = max(onset(icl)-(win_length-min(dura(icl),win_length))/2,50);
            else
                tons(icl) = onset(icl)-100;
            end
        else
            tons(icl) = tmin_median;
        end

        %         for jj=1:size(spikes1,1)
        for jj=1:numel(spikes1)
            %             sp_count_post(jj)=sum((spikes1(jj,:)< tons(icl)+win_length) & (spikes1(jj,:)> tons(icl)));
            sp_count_post(jj)=sum((spikes1{jj}< tons(icl)+win_length) & (spikes1{jj}> tons(icl)));
        end
        [p_value_sign(icl),~] = signtest(sp_count_base(all_picsused==stim_num,~isnan(sp_count_base(all_picsused==stim_num,:))&~isnan(sp_count_post)),sp_count_post(~isnan(sp_count_base(all_picsused==stim_num,:))&~isnan(sp_count_post)),'Tail','left');

        medians_base = median(sp_count_base,2,'omitnan');
        mu_base_med = mean(medians_base(ntrials(all_picsused)>min_trial_for_median));
        std_base = std(medians_base(ntrials(all_picsused)>min_trial_for_median));
        %         median_post = nanmedian(sp_count_post,2);
        sp_count_post = median(sp_count_post,'omitnan');
        zscore(icl) = (sp_count_post - mu_base_med)/std_base;

    end
    limits = 0;
    stim2plot = min(ncl,max_stims);

    for i = 1:stim2plot
        %pictures
        subplot_ax = subplot(rows,cols,pic_num(i),'align');
        %             imagename=stimulus(st).name;
        imagename=grapes.ImageNames{stim_num};
        %             flag=1;
        if contains(lower(imagename),'.jp')
            try
                A=imread(imagename,'jpg');
                image(A); set(subplot_ax,'PlotBoxAspectRatio',[1 1 1]);
                if is_gray
                    colormap(gray(256))
                end
            catch
                [~,fname,~] = fileparts(imagename);
                xlim([-time_pre_ms time_pos_ms]);
                text(500,0.5,fname,'fontsize',6*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
            end
            if zscore(i)>=strength_thr && p_value_sign(i)<=psign_thr && sp_count_post>min_spk_median && good_lat(i)
                text(210,40,'Z_m*','fontsize',8*fontsize_mult,'FontWeight','bold');
                text(190,110,sprintf('%2.2f',zscore(i)),'fontsize',8*fontsize_mult,'FontWeight','bold');
            elseif zscore(i)>=strength_thr && p_value_sign(i)<=psign_thr && sp_count_post>min_spk_median
                text(210,40,'Z_m','fontsize',8*fontsize_mult,'FontWeight','bold');
                text(190,110,sprintf('%2.2f',zscore(i)),'fontsize',8*fontsize_mult,'FontWeight','bold');
            else
                text(210,40,'Z_m','fontsize',6*fontsize_mult);
                text(190,110,sprintf('%2.2f',zscore(i)),'fontsize',6*fontsize_mult);
            end
        else
            if exist(imagename,'file')
                [y,Fs] = audioread(imagename);
                line(subplot_ax,(0:1/Fs:(length(y)-1)/Fs)*1000,y(:,1));
                xlim([-time_pre_ms time_pos_ms]);
                [~,fname,~] = fileparts(imagename);
                text(500,0.5,fname,'fontsize',6*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
            else
                xlim([-time_pre_ms time_pos_ms]);
                text(500,0.5,lower(imagename),'fontsize',6*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
            end
        end

        set(subplot_ax,'xtick',[],'ytick',[]);box off;
        if i==3
            dirs = split(pwd,filesep);
            if ~from_onset
                title(sprintf('%s, %s %s responses. Channel %s, %s. Stim %d (%s). resp_win =[%d, %d] ms; IFR_thr = %2.2f Hz (ntsd=%d)',dirs{end},exp_type,phase,ch_label,clusters_to_plot{icl},tmin_median,tmax_median,IFR_thr,nstd),'interpreter','none','units','normalized','Position',[0 1.5 0])
            elseif ~win_cent
                title(sprintf('%s, %s %s responses. Channel %s, %s. Stim %d (%s). resp_win: %dms ons-100; IFR_thr = %2.2f Hz (ntsd=%d)',dirs{end},exp_type,phase,ch_label,clusters_to_plot{icl},win_length,IFR_thr,nstd),'interpreter','none','units','normalized','Position',[0 1.5 0])
            else
                title(sprintf('%s responses. Channel %s. Stim %d (%s). resp_win: %dms cent dura; ntsd=%d',exp_type,ch_label,stim_num,imagename,win_length,nstd),'interpreter','none','units','normalized','Position',[0 1.5 0])
            end
        end
        %             ylabel(num2str(stim_num),'rotation',0,'fontsize',8*fontsize_mult,'FontWeight','bold','units','normalized','position',[-0.5 0.3000 0]);

        %              image_ylabel = {['#' num2str(data_table.ntrials(st))],[chaux ': ' erase(clusters_to_plot{i}, 'class' )],[grapes.rasters.(chfield).details.(clusters_to_plot{i}) ' spks']};
        image_ylabel = {[num2str(chaux) ': ' erase(clusters_to_plot{i}, 'class' )],sprintf('IFRth %2.1f',IFR_thr(i)),[num2str(grapes.rasters.(chfield).details.(clusters_to_plot{i})) ' sp']};

        ylabel(image_ylabel,'rotation',0,'fontsize',6.5*fontsize_mult,...
            'HorizontalAlignment','right','FontWeight','bold','units','normalized','position',[-0.1 -0.2 0]);

        %rasters
        subplot_ax = subplot(rows,cols,rast_num(i,:),'align');
        hold on;

        spikes1 = grapes.rasters.(chfield).(clusters_to_plot{i}).stim{stim_num};

        if iscell(spikes1)
            lst=numel(spikes1);
            if ~all(cellfun(@isempty,spikes1))
                plotSpikeRaster(spikes1,'PlotType','vertline','LineFormat',LineFormat);
            end
        else
            lst=size(spikes1,1);
            plotSpikeRaster( num2cell(spikes1,2),'PlotType','vertline','LineFormat',LineFormat);
        end
        if lst>0
            set(subplot_ax,'YLim',[0.5 lst+0.5])
        end
        set(subplot_ax,'XLim',[-time_pre_ms time_pos_ms]);
        limy=ylim;
        %             if ~from_onset
        %                 xpoints = [tmin_median tmax_median tmax_median tmin_median];
        %             else
        xpoints = [tons(i) tons(i)+win_length tons(i)+win_length tons(i)];
        %             end
        ypoints = [limy(1) limy(1) limy(2) limy(2)];
        p=patch(xpoints,ypoints,'y');
        set(p,'FaceAlpha',0.2,'linestyle','none');
        set(subplot_ax,'FontSize',7);
        axis off

        %IFR
        subplot_ax = subplot(rows,cols,hist_num(i),'align');
        hold on;
        %             if ~from_onset
        %                 [ejex,IFR_plot]= get_IFR_BCM(spikes1,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms);
        %                 [good_lat_plot(i),onset_plot(i),dura_plot(i)]= get_latency_BCM(IFR_plot,ejex,sample_period,IFR_thr,t_down,over_threshold_time,below_threshold_time);
        %             else
        %                 IFR_plot = IFR{st};
        %                 onset_plot(i) = onset(st);
        %                 dura_plot(i) = dura(st);
        %                 good_lat_plot(i) = good_lat(st);
        %             end


        IFR_sm = smooth(IFR{i},smooth_bin);
        line(subplot_ax,ifr_calclator.ejex,IFR_sm,'Color','r','Linewidth',1)
        aux = max(IFR_sm*1.15);
        %             limits = max(aux,limits);
        limits(i) = max(aux,IFR_thr(i)*1.15);
        %             box on
        xlim([-time_pre_ms time_pos_ms])

        if (i>(stim2plot-cols))&&(i<=max_stims)
            %                 set(subplot_ax,'fontsize',7*fontsize_mult);
            set(subplot_ax,'xtick',[-time_pre_ms 0 time_pos_ms/2 time_pos_ms]);
        else
            axis off
        end

        if ~good_lat(i)
            title(sprintf('lat = %3.0f, dura = %3.0f',onset(i),dura(i)),'fontsize',5*fontsize_mult,'interpreter','none','VerticalAlignment','middle');
        else
            title(sprintf('lat = %3.0f, dura = %3.0f',onset(i),dura(i)),'fontsize',6*fontsize_mult,'FontWeight','bold','interpreter','none','VerticalAlignment','middle');
        end
        box off
    end

    for i =1:stim2plot
        subplot_ax = subplot(rows,cols,hist_num(i),'align');
        line(xlim,[IFR_thr(i) IFR_thr(i)],'linestyle','--','color','k')
        if fixed_limy==0
            use_limy = limits(i);
        else
            use_limy = fixed_limy;
        end

%         if limits(i)~=0
        if use_limy~=0
            set(subplot_ax,'YLim',[0 use_limy]);
            line([0 0],[0 use_limy],'linestyle',':')
            line([1000 1000],[0 use_limy],'linestyle',':')
            line([onset(i) onset(i)],[0 use_limy],'linestyle',':','color','k','linewidth',0.5)
        end
        limy=ylim;
        xpoints = [onset(i) onset(i)+dura(i) onset(i)+dura(i) onset(i)];
        %             xpoints = [tmin_median tmax_median tmax_median tmin_median];
        ypoints = [limy(1) limy(1) limy(2) limy(2)];
        p=patch(xpoints,ypoints,'c');
        set(p,'FaceAlpha',0.2,'linestyle','none');

        subplot_ax = subplot(rows,cols,rast_num(i,:),'align');
        line([onset(i) onset(i)],ylim,'linestyle',':','color','r','linewidth',0.5)
    end

    %         % Ranking plot accessory calculations
    %         subplot_ax = subplot(4,6,19:23,'align');
    %         bar(zscore(stims_sorted));
    %         hold on
    %         limy = ylim;
    %         xpoints = [rankfirst-0.5 rankfirst+stim2plot-1+0.5 rankfirst+stim2plot-1+0.5 rankfirst-0.5];
    %         ypoints = [limy(1) limy(1) limy(2) limy(2)];
    %         p=patch(xpoints,ypoints,'g');
    %         set(p,'FaceAlpha',0.2,'linestyle','none');
    %         n_min_spk = lstim-sum(median_post<=min_spk_median);
    %         n_psign = sum(p_value_sign<=psign_thr & median_post>min_spk_median);
    %         line([n_min_spk+0.5 n_min_spk+0.5],ylim,'color','r','linestyle','--')
    %         line([n_psign+0.5 n_psign+0.5],ylim,'color','r','linestyle','--')
    %         xlim([0.5 lstim+0.5]);set(subplot_ax,'ygrid','on');
    %         if n_min_spk == 0
    %             title(sprintf('stims signif_psign & more_min_spk  = %d.  stims less_min_spk = %d',n_psign,lstim-n_min_spk),'fontsize',7*fontsize_mult,'color','r','interpreter','none','VerticalAlignment','baseline');
    %         else
    %             title(sprintf('stims signif_psign & more_min_spk  = %d.  stims less_min_spk = %d',n_psign,lstim-n_min_spk),'fontsize',7*fontsize_mult,'interpreter','none','VerticalAlignment','baseline');
    %         end

    % %% COMMENT THIS CELL WHEN ONLY GRAPES IS AVAILABLE (NO TIMES OR SPIKES)
    %         if exist(sprintf('%s%s%s_spikes.mat',concat_folder,filesep,ch_label),'file') || exist(sprintf('%s%stimes_%s.mat',concat_folder,filesep,ch_label),'file')
    %         subplot_ax = subplot(4,6,24,'align');
    %         if icl==size(clusters_to_plot,1)
    % %             q=load(sprintf('%s_spikes.mat',NSx([NSx(:).chan_ID]==ch).output_name));
    %             q=load(sprintf('%s%s%s_spikes.mat',concat_folder,filesep,ch_label));
    %             q.sp=-q.spikes;
    %             hold on; line(subplot_ax,1:size(q.sp,2),mean(q.sp),'color','k','linestyle','-','linewidth',1);
    %             line(subplot_ax,1:size(q.sp,2),mean(q.sp)+std(q.sp),'color','k','linestyle','--');
    %             line(subplot_ax,1:size(q.sp,2),mean(q.sp)-std(q.sp),'color','k','linestyle','--');
    %             ns=size(q.sp,1);
    %         else
    % %             q=load(sprintf('times_%s.mat',NSx([NSx(:).chan_ID]==ch).output_name));
    %             q=load(sprintf('%s%stimes_%s.mat',concat_folder,filesep,ch_label));
    %             q.sp=-q.spikes(q.cluster_class(:,1)==icl,:);
    %             hold on;
    %             line(subplot_ax,1:size(q.sp,2),mean(q.sp),'color',col(icl),'linestyle','-','linewidth',1);
    %             line(subplot_ax,1:size(q.sp,2),mean(q.sp)+std(q.sp),'color',col(icl),'linestyle','--');
    %             line(subplot_ax,1:size(q.sp,2),mean(q.sp)-std(q.sp),'color',col(icl),'linestyle','--');
    %             ns=sum(q.cluster_class(:,1)==icl);
    %         end
    %         xlim([1 64]);
    %         title(sprintf('#spks= %d',ns),'fontsize',6*fontsize_mult,'FontWeight','bold','VerticalAlignment','baseline');
    %         set(subplot_ax,'xtick',[2 20 63],'xticklabel',{'-0.67','0','1.5'},'fontsize',5*fontsize_mult);
    %         ylim_max=max(max(mean(q.sp)+std(q.sp)),max(mean(q.sp)-std(q.sp)))+25;
    %         ylim_min=min(min(mean(q.sp)+std(q.sp)),min(mean(q.sp)-std(q.sp)))-25;
    %         try
    %             ylim([ylim_min ylim_max])
    %         catch
    %             disp('problem with ylim')
    %         end
    %         line([20 20],ylim,'linestyle',':')
    %         line(xlim,[0 0],'linestyle',':')
    %%
    %         end
    if do_plot
        print(fig,'-dpng',sprintf('resp_%s_stim%d_ylim%d_nstd%d_wincent%d_%s.png',ch_label,stim_num,fixed_limy,nstd,win_cent,exp_type));
    end
    %     end
    %     close(fig);
    % end
end
warning on
set(groot,'defaultfiguregraphicssmoothing','remove')
set(groot,'defaultaxesfontsmoothing','remove')
set(groot,'defaultaxestitlefontsizemultiplier','remove')
set(groot,'defaultaxestitlefontweight','remove')
set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'remove','remove','remove'})
