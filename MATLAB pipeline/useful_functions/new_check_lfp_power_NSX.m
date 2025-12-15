function new_check_lfp_power_NSX(channels, varargin)
% FJC oct 2020
% channels is a list of channel IDs as they were saved in the NSx.mat
% This function requires the parse_data_NSx to be applied before
    p = inputParser;
    p.addParameter('save_fig',false);
    p.addParameter('show_img',true);
    p.addParameter('direc_resus_base',pwd);
    p.addParameter('resus_folder_name','spectra');
    p.addParameter('direc_raw',pwd);
    p.addParameter('with_NoNotch',false);
    p.addParameter('time_plot_duration',1); %minutes
    p.addParameter('freq_line',60); %Hz
    p.addParameter('parallel',false);
%     p.addParameter('n_notchs',50);
    p.addParameter('k_periodograms',200);
    p.addParameter('notch_filter',true);
    p.addParameter('spectum_resolution',0.5); %periodogram resolution Hz 
%     p.addParameter('spectum_resolution',2); %periodogram resolution Hz 
    p.parse(varargin{:});
    par = p.Results;
    [~,par.session] = fileparts(par.direc_raw);
%     par.fsplit_thr = 2 * par.freq_line - 10;
    %par.min_freq_notch = 10;
    par.min_freq_notch = 40;
    par.Vlim = 50;
    par.span_smooth = 21;
    par.db_thr = 5;
    par.freqs_comp = [300,1000,2000,3000,6000];
    
    load(fullfile(par.direc_raw,'NSx'),'NSx','freq_priority');
    % NSx = NSx;
    % freq_priority = freq_priority;
    
    p = gcp('nocreate'); % If no pool, do not create new one.
    
%     if length(par.freq_line) ~= length(par.n_notchs)
%         error('freq_line and n_notchs should be the same length')
%     end
    
    if isempty(p)
        poolsize = 0;
    else
        poolsize = 1;
    end
    if par.parallel
        % warning('Parallel selected: show_img=false and save_fig=true')
        disp('Parallel selected: show_img=false and save_fig=true')
        par.show_img = false;
        par.save_fig = true;
    end
    
    conf_table = par_check_lfp_power_NSX();
    nchannels = length(channels);
    process_info_out = cell(nchannels,1);
    if par.parallel
        parfor i = 1:nchannels
            ch = channels(i);
            process_info_out{i} = new_check_lfp_power(ch,par,conf_table,NSx,freq_priority);
        end
    else
        for i = 1:nchannels
            ch = channels(i);
            process_info_out{i} = new_check_lfp_power(ch,par,conf_table,NSx,freq_priority);
        end
    end
    
    if exist(fullfile(par.direc_raw,'pre_processing_info.mat'),'file')
        load(fullfile(par.direc_raw,'pre_processing_info.mat'),'process_info')
        process_temp = cell2mat({process_info_out{~cellfun('isempty',process_info_out)}});
        for j=1:length(process_temp)
            if isempty(process_info)
                ind_preprocess = [];
            else
                ind_preprocess = find([process_info(:).chID]==process_temp(j).chID);
            end
            if isempty(ind_preprocess)
                ind_preprocess = length(process_info)+1;
            end
            process_info(ind_preprocess).chID = process_temp(j).chID;
            process_info(ind_preprocess).SOS =process_temp(j).SOS;
            process_info(ind_preprocess).G = process_temp(j).G;
            process_info(ind_preprocess).freqs_notch = process_temp(j).freqs_notch;
            process_info(ind_preprocess).BwHz_notch = process_temp(j).BwHz_notch;
        end
        save(fullfile(par.direc_raw,'pre_processing_info.mat'),'process_info','-append')
    else
        process_info = cell2mat({process_info_out{~cellfun('isempty',process_info_out)}});
%         for j=1:nchannels
%             if ~isempty(process_info_out{j}{1})
%                 process_info(1).chID = process_info_out{j}{1};
%                 process_info(1).SOS = process_info_out{j}{2};
%                 process_info(1).G = process_info_out{j}{3};
%                 process_info(1).freqs_notch = process_info_out{j}{4};
%                 process_info(1).BwHz_notch = process_info_out{j}{5};
%             end
%         end        
        save(fullfile(par.direc_raw,'pre_processing_info.mat'),'process_info')
    end
    
    if par.parallel && poolsize == 0
        delete(gcp('nocreate'))
    end
end

function info=new_check_lfp_power(channel,par,conf_table,NSx,freq_priority)
    ch_type = [];
    selected = arrayfun(@(x) (x.chan_ID==channel)*(find(freq_priority(end:-1:1)==x.sr)),NSx);

    if sum(selected)==0
        error('channel not found in NSx.mat')
    elseif length(nonzeros(selected))>1
        [tt,~] = max(selected);
        posch = find(selected==tt);
    else
        posch = find(selected);
    end
        
    ch_type = NSx(posch).ext;
    sr = NSx(posch).sr;
    lts = NSx(posch).lts;
    unit = NSx(posch).unit;
    label = NSx(posch).label;
    conversion = NSx(posch).conversion;
    output_name = NSx(posch).output_name;
    if isfield(NSx,'dc')
        dc = NSx(posch).dc;
    else
        dc=0;
    end
    if isempty(ch_type)
        warning('channel %d not parsed',channel)
        return
    end  
    if sr==30000
        par.freqs_comp = [300,1000,2000,3000,6000];
%     elseif sr==10000
    elseif sr==7500
        par.freqs_comp = [300,1000,2000,3000];
    else
        par.freqs_comp = [300,990];
    end
    conf = table2struct(conf_table([num2str(sr) '_' unit ],:));
    if ~par.notch_filter
        conf.notch_q = [];
        conf.notch_width = [];
    end
    nofilter = isempty(conf.notch_q) && isempty(conf.notch_width)&& isempty(conf.filter_stop) && isempty(conf.filter_order);
    b_pass =[]; a_pass=[];
    max_freq_line = sr/2;
    min_freq_line = 0;
    if ~isempty(conf.filter_stop)
        % max_freq_line = conf.filter_stop(2);
        max_freq_line = max([conf.filter_stop(2) 550]);
        % min_freq_line = conf.filter_stop(1);
        min_freq_line = 40;
        [orden_pass, Wnorm_pass] = ellipord(conf.filter_pass*2/sr,...
            conf.filter_stop*2/sr,conf.filter_Rp,conf.filter_Rs);
        [b_pass,a_pass] = ellip(orden_pass,conf.filter_Rp,conf.filter_Rs,Wnorm_pass);
    elseif ~isempty(conf.filter_order)
        % min_freq_line = conf.filter_pass(1);
        min_freq_line = 40;
        max_freq_line = max([conf.filter_pass(2) 550]);
         [b_pass,a_pass] = ellip(conf.filter_order,conf.filter_Rp,conf.filter_Rs,conf.filter_pass*2/sr);
%         [b_pass,a_pass] = butter(conf.filter_order,conf.filter_pass*2/sr);
    end
    

    N = 2^ceil(log2(sr/par.spectum_resolution));
    samples_spectrum = min(par.k_periodograms *N,lts);
    samples_timeplot = min(par.time_plot_duration * 60* sr,lts);
    
    f1 = fopen(sprintf('%s%s',output_name,ch_type),'r','l');
    initial_index = lts- max(samples_spectrum,samples_timeplot);
%     initial_index = 1;
    fseek(f1,initial_index*2,'bof'); %this moves to the end-lts of the file
    
    x_raw = fread(f1,samples_spectrum,'int16=>double')*conversion+dc;
    fclose(f1);
    
    [pxx,fs] = pwelch(x_raw(end+1-samples_spectrum:end),barthannwin(N),0,[],sr,'onesided');
    if nofilter
        extra_title = 'No filter';
    else
       extra_title = 'Filtered: '; 
    end
    
    if isempty(conf.filter_stop) && isempty(conf.filter_order)
        x = x_raw;
    else
        if exist('fast_filtfilt','file') 
            x = fast_filtfilt(b_pass,a_pass,x_raw);
        else
            x = filtfilt(b_pass,a_pass,x_raw);
        end
        extra_title = [extra_title sprintf('Band-pass: %.0f-%.0f Hz. ',conf.filter_pass(1),conf.filter_pass(2))];
    end
    [pxx_filtered,fs_filtered] = pwelch(x(end+1-samples_spectrum:end),barthannwin(N),0,[],sr,'onesided');
    
    if par.with_NoNotch
        x_NoNotch = x;
    end
    pxx_db = 10*log10(pxx_filtered);
    pxx_slideavg = movmedian(pxx_db,par.span_smooth);
    pxx_thr_db = pxx_slideavg+par.db_thr;   
    
    low_freqs = find(fs_filtered<=min_freq_line,1,'last');                     
    high_freqs = find(fs_filtered>=max_freq_line,1,'first');
    pxx_thr_db(1:low_freqs) = pxx_thr_db(low_freqs);
    pxx_thr_db(high_freqs:end) = pxx_thr_db(high_freqs);
        
    pow_comp = zeros(size(par.freqs_comp));
    for iii=1:numel(par.freqs_comp)
        indF = find(fs>par.freqs_comp(iii),1,'first');
        pow_comp(iii) = pxx_slideavg(indF);
    end
    used_notchs = [];
    bw_notchs = [];
%     inds_freqs_search_notch = 2:(numel(fs)-1); %usefull before
    min_ind = find(fs<par.min_freq_notch,1,'last');
    inds_freqs_search_notch = min_ind:(numel(fs)-1); %usefull before
    supra_thr = find(pxx_db(inds_freqs_search_notch) > pxx_thr_db(inds_freqs_search_notch))+inds_freqs_search_notch(1)-1;    
    if ~isempty(supra_thr)
        max_amp4notch = max(pxx_db(inds_freqs_search_notch)- pxx_thr_db(inds_freqs_search_notch));
        temp_supra=find(diff(supra_thr)>1);
        inds_to_explore=[supra_thr(1); supra_thr(temp_supra+1)];
        if isempty(temp_supra)
            sample_above = numel(supra_thr);
        else
            sample_above = [temp_supra(1); diff(temp_supra); numel(supra_thr)-find(supra_thr==inds_to_explore(end))+1];
        end
        for jj=1:length(inds_to_explore)
            if sample_above(jj)>1
                [~, iaux] = max(pxx_db(inds_to_explore(jj):inds_to_explore(jj)+sample_above(jj)-1));    %introduces alignment
                centre_sample = mean(inds_to_explore(jj):inds_to_explore(jj)+sample_above(jj)-1);
                ind_max = iaux + inds_to_explore(jj) - 1;
                %             amp4notch = mean(pxx_db(inds_to_explore(jj):inds_to_explore(jj)+sample_above(jj)-1)-pxx_thr_db(inds_to_explore(jj):inds_to_explore(jj)+sample_above(jj)-1));
                if mod(centre_sample,1)==0.5 && (pxx_db(floor(centre_sample))> pxx_db(ceil(centre_sample)))
                    centre_sample = floor(centre_sample);
                else
                    centre_sample = ceil(centre_sample);
                end
    %              extra=2;
    %              int_factor=5;             
    %              pico=pxx_db(inds_to_explore(jj)-extra:inds_to_explore(jj)+sample_above(jj)-1+extra);
    %              s = 1:numel(pico);
    %              ints = 1/int_factor:1/int_factor:numel(pico);
    %              intpico=spline(s,pico,ints);
    %                 [maxi iaux] = max(intspikes(:,(w_pre+extra-1)*int_factor:(w_pre+extra+1)*int_factor),[],2);
    %     iaux = iaux + (w_pre+extra-1)*int_factor -1;
    %         spikes1(i,:)= intspikes(i,iaux(i)-w_pre*int_factor+int_factor:int_factor:iaux(i)+w_post*int_factor);
                            amp4notch = pxx_db(ind_max)-pxx_thr_db(ind_max);

    %             if (ind_max<centre_sample) && (pxx_db(inds_to_explore(jj))>pxx_db(inds_to_explore(jj)+sample_above(jj)-1))
    %                   centre_sample = centre_sample - 1;
    %             end  
                used_notchs(end+1) = fs(centre_sample);
                bw_notchs(end+1) = (fs(2)-fs(1))*sample_above(jj)*2*amp4notch/max_amp4notch;
            end
        end   
    end
    
    info = [];
    if (~isempty(conf.notch_q) || ~isempty(conf.notch_width)) && ~isempty(supra_thr)
        Z = [];
        P = [];
        K = 1;        
%         for notch_i = 1:length(par.freq_line) 
%             notch_freq = par.freq_line(notch_i);
%             for i= 1:par.n_notchs(notch_i)
            for i= 1:numel(used_notchs)
%                 nfi = notch_freq*i;
                nfi = used_notchs(i);
%                 if nfi >= sr/2
%                     break
%                 end
%                 used_notchs(end+1) = nfi;
                w = nfi/(sr/2);
%                 if ~isempty(conf.notch_width)
%                     bw = conf.notch_width/(sr/2);
%                 else
%                     bw = w/conf.notch_q;
%                 end
%                 bw_notchs(end+1) = bw*(sr/2);
                if nfi<295
                    max_bw = 3;
                else
                    max_bw = 5;
                end
                bw_notchs(i) = min(max(bw_notchs(i),1),max_bw);
                bw = bw_notchs(i)/(sr/2);
%                 bw = 3/(sr/2);
                if isMATLABReleaseOlderThan("R2023b")
                    [b_notch,a_notch] = iirnotch(w,bw);
                else
                    [b_notch,a_notch] = designNotchPeakIIR(CenterFrequency=w,...
                                                       Bandwidth=bw, ...
                                                       Response="notch");
                end
                
                
                [zi,pi,ki] = tf2zpk(b_notch,a_notch);
                K = K * ki;
                Z(end+1:end+2) = zi;
                P(end+1:end+2) = pi;
            end
%         end
        if ~isempty(Z)
%             extra_title = [extra_title ' notchs: ' num2str(par.freq_line) '.  '];
            extra_title = [extra_title ' #notchs: ' num2str(numel(used_notchs)) '.  '];
            [S,G] = zp2sos(Z,P,K);       % Convert to SOS
            if exist('fast_filtfilt','file') 
                x = fast_filtfilt(S,G,x);  
            else
                x = filtfilt(S,G,x); 
            end
            info=struct();
            info.chID = channel;
            info.SOS = S;
            info.G = G;
            info.freqs_notch = used_notchs;
            info.BwHz_notch = bw_notchs;
        end
        %extra_title = [extra_title sprintf('Band-pass: %.0f-%.0f Hz. ',conf.filter_pass(1),conf.filter_pass(2))];
        [pxx_filtered,fs_filtered] = pwelch(x(end+1-samples_spectrum:end),barthannwin(N),0,[],sr,'onesided');

    end
    %we want at a resolution of 0.5Hz
    naverages = floor(samples_spectrum/N);
    if ~isempty(conf.freqs_fit)
        fit1 = logfit(fs,pxx,conf.freqs_fit(1),conf.freqs_fit(2));
        fit2 = logfit(fs,pxx,conf.freqs_fit(2),conf.freqs_fit(3));
        if size(conf.freqs_fit,2)==4
            fit3 = logfit(fs,pxx,conf.freqs_fit(3),conf.freqs_fit(4));
        end
    end
    
    xini_margin = 0.06;
    xend_margin = 0.01;
    yup_margin = 0.08;
    ydown_margin = 0.07;
    zoom_prop = 0.5;
    ytime_plots = 0.60;
    yw_time = (ytime_plots - yup_margin - ydown_margin*2)/2;
    pos.spectrum_ax = [xini_margin,ytime_plots,zoom_prop-(xini_margin+xend_margin),1-(ytime_plots+yup_margin) ]; %xini,yini,xw,yw
    pos.zoom_ax = [0.5+xini_margin,ytime_plots,zoom_prop-(xini_margin+xend_margin),1-(ytime_plots+yup_margin)];
    pos.rawtime_ax = [xini_margin,yw_time+ydown_margin*2 ,1-(xini_margin+xend_margin),yw_time];
    pos.filttime_ax = [xini_margin,ydown_margin,1-(xini_margin+xend_margin),yw_time];
    fig = figure('units','pixels','position',[100 100 1400 700]);
    fig.GraphicsSmoothing = 'off';
    %fig = figure();
    if par.show_img==0
        fig.Visible = 'off';
    end
    if par.parallel == false
        fig.WindowState = 'maximized';
    end
    figs = struct;
    for fn = fieldnames(pos)'
       figs.(fn{1}) = subplot('Position', pos.(fn{1}),'Units','normalized');
    end
    
    
    for fn={'zoom_ax','spectrum_ax'}
        set(fig,'CurrentAxes',figs.(fn{1}))
        hold on
        plot(fs,10*log10(pxx),'LineWidth',1.5);
        plot(fs,pxx_thr_db,'Color','m','LineWidth',0.7);        
%         plot(fs,pxx_thr_db2,'Color','g','LineStyle','--','LineWidth',0.5);        
        if ~isempty(used_notchs)
            for f = used_notchs
                xline(f,'Color','k','LineStyle',':','LineWidth',1.0);
            end
        end
        if strcmp(fn,'zoom_ax')
             xlim(conf.freqs_lim_zoom)
%            xlim([0 100])
            yl = [-30 50];
        else
            xlim(conf.freqs_lim)
            yl = [-50 50];
        end
%         ylim('auto');
%         yl = ylim();
%         yl(1) = yl(1) - 0.05*(yl(2)-yl(1)); 
        
        if ~nofilter
            plot(fs_filtered,10*log10(pxx_filtered),'r','LineWidth',1.0);   
        end
            grid minor
        if ~isempty(conf.freqs_fit)
            xline(conf.freqs_fit(2),'Color','g','LineStyle','--','LineWidth',1.5);
            xline(conf.freqs_fit(3),'Color','g','LineStyle','--','LineWidth',1.5);
        end
        xlabel('Frequency (Hz)','fontsize',14)
        ylabel('Power Spectrum (dB/Hz)','fontsize',14)
        set(gca,'fontsize',12)
        ylim(yl);
        box on
    end
    set(fig,'CurrentAxes',figs.spectrum_ax)

    if ~isempty(conf.freqs_fit)
        annotation('textbox',[0.25 0.81 0.8 0.1],'String',{'Power fit (1st order loglog)', ...
            sprintf('%d-%d Hz: \\alpha = %1.2f (r^2 = %1.2f)',conf.freqs_fit(1),conf.freqs_fit(2),fit1.k1,fit1.r2),...
            sprintf('%d-%d Hz: \\alpha = %1.2f (r^2 = %1.2f)',conf.freqs_fit(2),conf.freqs_fit(3),fit2.k1,fit2.r2)},...
            'FontSize',9,'BackgroundColor',[1 1 1],'FitBoxToText','on');
        if size(conf.freqs_fit,2)==4
            annotation('textbox',[0.25 0.81 0.8 0.1],'String',{'Power fit (1st order loglog)', ...
            sprintf('%d-%d Hz: \\alpha = %1.2f (r^2 = %1.2f)',conf.freqs_fit(1),conf.freqs_fit(2),fit1.k1,fit1.r2),...
            sprintf('%d-%d Hz: \\alpha = %1.2f (r^2 = %1.2f)',conf.freqs_fit(2),conf.freqs_fit(3),fit2.k1,fit2.r2),...
            sprintf('%d-%d Hz: \\alpha = %1.2f (r^2 = %1.2f)',conf.freqs_fit(3),conf.freqs_fit(4),fit3.k1,fit3.r2)},...
            'FontSize',9,'BackgroundColor',[1 1 1],'FitBoxToText','on');
        end
    end
    
    set(fig,'CurrentAxes',figs.rawtime_ax)
    plot(linspace(0,samples_timeplot/sr,samples_timeplot),x_raw(end+1-samples_timeplot:end))
    grid minor
    xlabel('Time (sec)','fontsize',12)
    ylabel({'Raw', conf.unit},'fontsize',11)
    set(gca,'fontsize',12)
    if ~nofilter

        set(fig,'CurrentAxes',figs.filttime_ax)
        if par.with_NoNotch
            plot(linspace(0,samples_timeplot/sr,samples_timeplot),x_NoNotch(end+1-samples_timeplot:end))
            hold on
        end
        plot(linspace(0,samples_timeplot/sr,samples_timeplot),x(end+1-samples_timeplot:end),'r')

        if ~isempty(conf.threshold)
            sigma = median(abs(x))/0.6745;
            thr = conf.threshold *sigma;
            yline(thr,'b','LineWidth',1.2);
            yline(-thr,'b','LineWidth',1.2);
            if par.with_NoNotch
                sigma_NoNotch = median(abs(x_NoNotch))/0.6745;
                thr_NoNotch = conf.threshold *sigma_NoNotch;
                yline(thr_NoNotch,'k','LineWidth',1);
                yline(-thr_NoNotch,'k','LineWidth',1);
            end
            extra_title = [extra_title sprintf('Thr = sigma*%.1f = %.2f %s. Power at ',conf.threshold,thr,conf.unit)];
            for iii=1:numel(par.freqs_comp)
                extra_title = [extra_title sprintf('%.1f, ',par.freqs_comp(iii)/1000)];
            end
            extra_title = [extra_title 'KHz = '];
            for iii=1:numel(par.freqs_comp)
                extra_title = [extra_title sprintf('%.2f, ',pow_comp(iii))];
            end    
           extra_title = [extra_title 'dB'];
        end
        
        ylim([-3*par.Vlim 2*par.Vlim])
        grid minor
        box on
        xlabel('Time (sec)','fontsize',11)
        ylabel({'Filtered',conf.unit},'fontsize',11)
        set(gca,'fontsize',12)
    end
    linkaxes([figs.rawtime_ax,figs.filttime_ax],'x')
    
    
    titletext = sprintf('Spectrum of Ch %d (%s) session %s. sr %1.0f Hz (%d averaged periodograms of length %d with barthannwin)',...
    channel,output_name,par.session,sr,naverages,N);
%     channel,label,par.session,sr,naverages,N);
    

    sgtitle({titletext,extra_title},'fontsize',13,'interpreter','none');  
    
    if par.save_fig == 1
        if ~exist(fullfile(par.direc_resus_base,par.resus_folder_name),'dir')
            mkdir(par.direc_resus_base,par.resus_folder_name);
        end
        outfile = fullfile(par.direc_resus_base,par.resus_folder_name,['spectrum_' par.session '_ch' num2str(channel) '_' label]);
%         savefig(fig,[outfile '.fig'],'compact');    
        print(fig,'-dpng',[outfile '.png']);           
    end
    
end

function fit = logfit(freqs,s,f1,f2)
    power_fit = log10(s(find(freqs>f1,1):find(freqs>=f2,1)));
    freqs_fit = log10(freqs(find(freqs>f1,1):find(freqs>=f2,1)));
    p_all1 = polyfit(freqs_fit,power_fit,1);
    fit = struct;
    fit.k1=p_all1(1);
    fit.loga=p_all1(2);    
    model = fit.k1*freqs_fit+fit.loga;
    fit.r2 = max(0,1 - sum((power_fit-model).^2) / sum((power_fit-mean(power_fit)).^2));
end