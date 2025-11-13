function futures=loop_plot_best_responses_BCM_rank_online(chaux,grapes,muonly,rankfirst,ranklast_def,effect_rows_2,win_cent,nstd,phase,run_parallel)
% Plots the best max_stims responses of a channel ch and cluster i and IFR of all responses.

from_onset = 1;
smooth_bin=1500;
sigma_gauss = 10;
alpha_gauss = 3.035;
min_spk_median=0.5;
tmin_median=200;
tmax_median=700;
win_length = tmax_median - tmin_median;
strength_thr = 5;
psign_thr = 0.05;
t_down=20;
over_threshold_time = 75;
below_threshold_time = 100;

if ~exist('muonly','var') || isempty(muonly), muonly='n'; end
if ~exist('rankfirst','var') || isempty(rankfirst), rankfirst=1; end
if ~exist('ranklast_def','var') || isempty(ranklast_def), ranklast_def=15; end
if ~exist('effect_rows_2','var') || isempty(effect_rows_2), effect_rows_2=0; end
if ~exist('phase','var')|| isempty(phase),phase = []; end
if ~exist('run_parallel','var')|| isempty(run_parallel),run_parallel = true; end
time_pre_ms = grapes.time_pre;
time_pos_ms = grapes.time_pos;
    
if effect_rows_2==30
    pic_num=  [ 1:5  21:25];
    rast_num= [[6 11];[7 12];[8 13];[9 14];[10 15];20+[6 11];20+[7 12];20+[8 13];20+[9 14];20+[10 15]];
    hist_num= [16:20 36:40];
    max_stims = 10;
elseif effect_rows_2==10
    pic_num=  [ 1:5  21:25];
    rast_num= [[6 11];[7 12];[8 13];[9 14];[10 15];20+[6 11];20+[7 12];20+[8 13];20+[9 14];20+[10 15]];
    hist_num= [16:20 36:40];
    max_stims = 10;
elseif effect_rows_2==0
    pic_num=  [ 1:5  16:20 31:35 46:50];
    rast_num= [ 6:10 21:25 36:40 51:55]';
    hist_num= [11:15 26:30 41:45 56:60];
    max_stims = 15;
end    
rows = 12;    
cols = 5;

fontsize_mult = 1.4;
clusters_to_plot = {'class1'; 'class2';'class3';'class4';'class5';'class6';'class7';'class8';'class9';'class10';'class11';'mu'};
LineFormat.Color = 'blue';
LineFormat.LineWidth = 0.35;

if run_parallel
    futures(1:length(chaux)) = parallel.FevalFuture;
    for ic = 1:length(chaux)
        futures(ic) = parfeval(@loop_best_channel,0,grapes.rasters.(['chan' num2str(chaux(ic))]),...
            win_cent,nstd,phase,from_onset,smooth_bin,sigma_gauss,alpha_gauss,min_spk_median,...
            tmin_median,tmax_median,win_length,strength_thr,psign_thr,t_down,...
            over_threshold_time,below_threshold_time,muonly, rankfirst,...
            ranklast_def, time_pre_ms, time_pos_ms,...
            pic_num, rast_num,hist_num,max_stims,rows,cols,grapes.exp_type,...
            fontsize_mult,clusters_to_plot,LineFormat,grapes.ImageNames);

    end
else
    futures = [];
    for ic = 1:length(chaux)
        loop_best_channel(grapes.rasters.(['chan' num2str(chaux(ic))]),...
            win_cent,nstd,phase,from_onset,smooth_bin,sigma_gauss,alpha_gauss,min_spk_median,...
            tmin_median,tmax_median,win_length,strength_thr,psign_thr,t_down,...
            over_threshold_time,below_threshold_time,muonly, rankfirst,...
            ranklast_def, time_pre_ms, time_pos_ms,...
            pic_num, rast_num,hist_num,max_stims,rows,cols,grapes.exp_type,...
            fontsize_mult,clusters_to_plot,LineFormat,grapes.ImageNames);

    end
end

end




function loop_best_channel(ch_grapes,win_cent,nstd,phase,from_onset,smooth_bin,sigma_gauss,alpha_gauss,min_spk_median,tmin_median,tmax_median,win_length,strength_thr,psign_thr,t_down,over_threshold_time,below_threshold_time,muonly, rankfirst, ranklast_def, time_pre_ms, time_pos_ms, pic_num, rast_num,hist_num,max_stims,rows,cols,exp_type,fontsize_mult,clusters_to_plot,LineFormat,...
    ImageNames)
    set(groot,'defaultaxesfontsmoothing','off')
    set(groot,'defaultfiguregraphicssmoothing','off')
    set(groot,'defaultaxestitlefontsizemultiplier',1.1)
    set(groot,'defaultaxestitlefontweight','normal')
    set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})

    fig = figure('Visible','off','Position',[300   120   798   598.5],'Units','pixels','GraphicsSmoothing','off');
    
    ncl = sum(cellfun(@(x) contains(x,'class'),fieldnames(ch_grapes)));
    muonly_flag=0;
    if strcmp(muonly,'y')
        ncl=0;muonly_flag=1;%plot multi unit
    end
    if ~muonly_flag && ncl==0
        return
    end
    if muonly_flag || ncl==0
        inds_cl = size(clusters_to_plot,1);
    else
        inds_cl = 1:ncl;
    end
    sample_period = 1000/ch_grapes.details.sr; % sample period for the spike list - window convolution in ms/sample
    ch_label = ch_grapes.details.output_name;
    
    for icl=inds_cl
     
        clf;
        active_cluster = ch_grapes.(clusters_to_plot{icl});
        lstim = length(active_cluster.stim);
        ranklast = min(ranklast_def,lstim);
        
        
        [stims_sorted,IFR_thr,ejex,IFR,onset,tons,dura,good_lat,zscore,median_post,p_value_sign] = create_ranking(active_cluster.stim,from_onset,min_spk_median,psign_thr,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms,tmin_median,tmax_median,t_down,over_threshold_time,below_threshold_time,nstd,win_cent);
        
        limits = 0;
        stim2plot = min(ranklast-rankfirst+1,max_stims);
        
        if ~from_onset
            onset_plot = NaN*ones(1,stim2plot);
            dura_plot = NaN*ones(1,stim2plot);
%             tons_plot = NaN*ones(1,stim2plot);            
            good_lat_plot = false(1,stim2plot);
        end
        for i = 1:stim2plot
            st = stims_sorted(rankfirst+i-1);
            %pictures
            subplot_ax = subplot(rows,cols,pic_num(i),'align');
%             imagename=stimulus(st).name;
            imagename=ImageNames{st};
%             flag=1;
            if contains(lower(imagename),'.jp')
                try
                	A=imread(imagename,'jpg');
                    image(A); set(subplot_ax,'PlotBoxAspectRatio',[1 1 1]);                                            
                catch
                    [~,fname,~] = fileparts(imagename);
                    xlim([-time_pre_ms time_pos_ms]);
                    text(500,0.5,fname,'fontsize',6*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
                end
                if zscore(st)>=strength_thr && p_value_sign(st)<=psign_thr && median_post(st)>min_spk_median && good_lat(st)
                    text(210,40,'Z_m*','fontsize',8*fontsize_mult,'FontWeight','bold');
                    text(190,110,sprintf('%2.2f',zscore(st)),'fontsize',8*fontsize_mult,'FontWeight','bold');
                elseif zscore(st)>=strength_thr && p_value_sign(st)<=psign_thr && median_post(st)>min_spk_median
                    text(210,40,'Z_m','fontsize',8*fontsize_mult,'FontWeight','bold');
                    text(190,110,sprintf('%2.2f',zscore(st)),'fontsize',8*fontsize_mult,'FontWeight','bold');
                else
                    text(210,40,'Z_m','fontsize',6*fontsize_mult);
                    text(190,110,sprintf('%2.2f',zscore(st)),'fontsize',6*fontsize_mult);
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
                    title(sprintf('%s, %s %s responses. Channel %s, %s. resp_win =[%d, %d] ms; IFR_thr = %2.2f Hz (ntsd=%d)',dirs{end},exp_type,phase,ch_label,clusters_to_plot{icl},tmin_median,tmax_median,IFR_thr,nstd),'interpreter','none','units','normalized','Position',[0 1.5 0])
                elseif ~win_cent
                    title(sprintf('%s, %s %s responses. Channel %s, %s. resp_win: %dms ons-100; IFR_thr = %2.2f Hz (ntsd=%d)',dirs{end},exp_type,phase,ch_label,clusters_to_plot{icl},win_length,IFR_thr,nstd),'interpreter','none','units','normalized','Position',[0 1.5 0])
                else
                    title(sprintf('%s, %s %s responses. Channel %s, %s. resp_win: %dms cent dura; IFR_thr = %2.2f Hz (ntsd=%d)',dirs{end},exp_type,phase,ch_label,clusters_to_plot{icl},win_length,IFR_thr,nstd),'interpreter','none','units','normalized','Position',[0 1.5 0])
                end
            end
            ylabel(num2str(stims_sorted(rankfirst+i-1)),'rotation',0,'fontsize',8*fontsize_mult,'FontWeight','bold','units','normalized','position',[-0.5 0.3000 0]);
                        
            %rasters
            subplot_ax = subplot(rows,cols,rast_num(i,:),'align');
            hold on;
           
            spikes1 = active_cluster.stim{st};
            
            if numel(spikes1)>0 
                if iscell(spikes1)
                    lst = numel(spikes1);
                    if ~all(cellfun(@isempty,spikes1))
                        plotSpikeRaster(spikes1,'PlotType','vertline','LineFormat',LineFormat);
                    end
                else
                   lst = size(spikes1,1);
                   plotSpikeRaster(num2cell(spikes1,2),'PlotType','vertline','LineFormat',LineFormat);
                end
                set(subplot_ax,'YLim',[0.5 lst+0.5])
            end         
            set(subplot_ax,'XLim',[-time_pre_ms time_pos_ms]);  
            limy=ylim;
%             if ~from_onset
%                 xpoints = [tmin_median tmax_median tmax_median tmin_median];
%             else
                xpoints = [tons(st) tons(st)+win_length tons(st)+win_length tons(st)];
%             end
            ypoints = [limy(1) limy(1) limy(2) limy(2)]; 
            p=patch(xpoints,ypoints,'y');
            set(p,'FaceAlpha',0.2,'linestyle','none'); 
            set(subplot_ax,'FontSize',7);
            axis off
            
            %IFR
            subplot_ax = subplot(rows,cols,hist_num(i),'align');
            hold on;
            if ~from_onset
                [ejex,IFR_plot]= get_IFR_BCM(spikes1,alpha_gauss,sigma_gauss,sample_period,time_pre_ms,time_pos_ms);
                [good_lat_plot(i),onset_plot(i),dura_plot(i)]= get_latency_BCM(IFR_plot,ejex,sample_period,IFR_thr,t_down,over_threshold_time,below_threshold_time);
            else
                IFR_plot = IFR{st};
                onset_plot(i) = onset(st);
                dura_plot(i) = dura(st);
                good_lat_plot(i) = good_lat(st);
            end
            if  ~all(isnan(IFR_plot))
                IFR_sm = smooth(IFR_plot,smooth_bin);
            else
                IFR_sm = nan(size(IFR_plot,2),1);
            end
            line(subplot_ax,ejex,IFR_sm,'Color','r','Linewidth',1)
            aux = max(IFR_sm*1.15);
            limits = max(aux,limits);
%             box on
            xlim([-time_pre_ms time_pos_ms])                
            
            if (i>(stim2plot-cols))&&(i<=max_stims)
%                 set(subplot_ax,'fontsize',7*fontsize_mult);
                set(subplot_ax,'xtick',[-time_pre_ms 0 time_pos_ms/2 time_pos_ms]);
            else
                axis off
            end
                       
            if ~good_lat_plot(i)
                title(sprintf('lat = %3.0f, dura = %3.0f',onset_plot(i),dura_plot(i)),'fontsize',5*fontsize_mult,'interpreter','none','VerticalAlignment','middle');
            else
                title(sprintf('lat = %3.0f, dura = %3.0f',onset_plot(i),dura_plot(i)),'fontsize',6*fontsize_mult,'FontWeight','bold','interpreter','none','VerticalAlignment','middle');
            end
            box off
        end
        
        for i =1:stim2plot            
            subplot_ax = subplot(rows,cols,hist_num(i),'align');
            line(xlim,[IFR_thr IFR_thr],'linestyle','--','color','k')
            if limits~=0
                set(subplot_ax,'YLim',[0 limits]);
                line([0 0],[0 limits],'linestyle',':')
                line([1000 1000],[0 limits],'linestyle',':')
                line([onset_plot(i) onset_plot(i)],[0 limits],'linestyle',':','color','k','linewidth',0.5)
            end
            limy=ylim;
            xpoints = [onset_plot(i) onset_plot(i)+dura_plot(i) onset_plot(i)+dura_plot(i) onset_plot(i)];
%             xpoints = [tmin_median tmax_median tmax_median tmin_median];
            ypoints = [limy(1) limy(1) limy(2) limy(2)]; 
            p=patch(xpoints,ypoints,'c');
            set(p,'FaceAlpha',0.2,'linestyle','none');
            
            subplot_ax = subplot(rows,cols,rast_num(i,:),'align');
            line([onset_plot(i) onset_plot(i)],ylim,'linestyle',':','color','r','linewidth',0.5)
        end
        
        % Ranking plot accessory calculations
        subplot_ax = subplot(4,6,19:23,'align');
        bar(zscore(stims_sorted));
        hold on
        limy = ylim;
        xpoints = [rankfirst-0.5 rankfirst+stim2plot-1+0.5 rankfirst+stim2plot-1+0.5 rankfirst-0.5];
        ypoints = [limy(1) limy(1) limy(2) limy(2)]; 
        p=patch(xpoints,ypoints,'g');
        set(p,'FaceAlpha',0.2,'linestyle','none');                       
        n_min_spk = lstim-sum(median_post<=min_spk_median);
        n_psign = sum(p_value_sign<=psign_thr & median_post>min_spk_median);
        line([n_min_spk+0.5 n_min_spk+0.5],ylim,'color','r','linestyle','--')
        line([n_psign+0.5 n_psign+0.5],ylim,'color','r','linestyle','--')
        xlim([0.5 lstim+0.5]);set(subplot_ax,'ygrid','on');
        if n_min_spk == 0
            title(sprintf('stims signif_psign & more_min_spk  = %d.  stims less_min_spk = %d',n_psign,lstim-n_min_spk),'fontsize',7*fontsize_mult,'color','r','interpreter','none','VerticalAlignment','baseline');
        else
            title(sprintf('stims signif_psign & more_min_spk  = %d.  stims less_min_spk = %d',n_psign,lstim-n_min_spk),'fontsize',7*fontsize_mult,'interpreter','none','VerticalAlignment','baseline');
        end

%%
        print(fig,'-dpng',sprintf('best_resp_%s_%s_rank%d_to_%d_nstd%d_wincent%d_%s_%s_online.png',ch_label,clusters_to_plot{icl},rankfirst,ranklast,nstd,win_cent,exp_type,phase));        
    end
    close(fig); 
end