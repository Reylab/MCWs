function fig_cells = loop_plot_responses_BCM_online(data_table, grapes, ss_num, max_nwins, ...
                                                    rank_config, ejex, save_fig, extra_label, ...
                                                    close_fig, order_offset, ...
                                                    priority_chs_ranking, is_ch_grapes, ...
                                                    b_yellow_patch, b_short_win, ...
                                                    b_img_lbl_legend)
    
    if ~exist('b_img_lbl_legend', 'var')
        b_img_lbl_legend = false; % only for dynamic_scr, categ_localiz
    end    
    if ~exist('b_short_win', 'var')
        b_short_win = false; % default is to show the full response window
    end

    if ~exist('b_yellow_patch', 'var')
        b_yellow_patch = true; % default is to show the response window
    end 
    %this function uses data_table to know where go to look on the grapes
    if ~exist('order_offset','var') || isempty(order_offset)
        order_offset=0;
    end

    if ~exist('is_ch_grapes', 'var')         is_ch_grapes         = false; end
    if ~exist('priority_chs_ranking', 'var') priority_chs_ranking = []; end
    
    rows = 12;    
    cols = 5;
    
    nwins= min(max_nwins,ceil(height(data_table)/(rows*cols/3))); %3 rows per stimulus
    
    warning off
    
    if ~exist('save_fig','var'), save_fig = true; end
    if ~exist('extra_label','var'), extra_label = ''; end
    if ~exist('close_fig','var'), close_fig = false; end
    
    pic_num=  [ 1:5  16:20 31:35 46:50];
    rast_num= [ 6:10 21:25 36:40 51:55]';
    hist_num= [11:15 26:30 41:45 56:60];

    set(groot,'defaultaxesfontsmoothing','off')
    set(groot,'defaultfiguregraphicssmoothing','off')
    set(groot,'defaultaxestitlefontsizemultiplier',1.1)
    set(groot,'defaultaxestitlefontweight','normal')
    set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})

    fig_w = 798;
    fig_h = 598.5;

    screen_size = get(0, 'ScreenSize');
    screen_width = screen_size(3);
    screen_height = screen_size(4);
    
    max_stims = length(unique(pic_num));
    
    fontsize_mult = 1.4;
    
    fig_cells = cell(nwins,1);
    for nwin = 1:nwins        
        if nwin <=3
            pos_x = 10 + (nwin-1) * fig_w;
            pos_y = screen_height / 2;
        else
            pos_x = 10 + (nwin-4) * fig_w;
            pos_y = 0;
        end
        pos = [pos_x, pos_y, fig_w, fig_h];
    
        fig = figure('Visible','off','Position', pos,...
            'Units','pixels','GraphicsSmoothing','off','Color',[1,1,1],...
            'Name',sprintf('Best responses window: %d',nwin));
        
        labels = {'same_feats', 'diff_feats', 'all_but_x'};
        colors = {'magenta', 'blue', 'cyan'};
        if ~contains(grapes.ImageNames.folder{1},'categ_localiz')
            labels = {'same_unit', 'same_cat'};
            colors = {'magenta', 'blue'};
        end
        if b_img_lbl_legend
            x0 = 0.02; y0 = 0.98; w = 0.01; h = 0.01; dx = 0.12;
            for k = 1:numel(labels)
                annotation(fig, 'rectangle', [x0+(k-1)*dx, y0, w, h], 'FaceColor', colors{k}, 'EdgeColor', colors{k});
                annotation(fig, 'textbox', [x0+(k-1)*dx+w+0.005, y0, 0.07, h], 'String', labels{k}, ...
                            'Interpreter','none', 'LineStyle', 'none', 'FontSize', 8, ...
                            'VerticalAlignment', 'middle', 'Color', colors{k});
            end
        end
    
        y_max = 0;
        stims2plot = (1:max_stims)+(nwin-1)*max_stims+order_offset;
        real_max_stims = min(max_stims, size(data_table,1) - order_offset - (nwin-1)*max_stims);
        if is_ch_grapes
            ifr_y_max = max(cellfun(@(x) max(x(:),[],"omitnan"), data_table(stims2plot(1:real_max_stims),:).IFR),[],'omitnan');
        else
            ifr_y_max = -1;
        end
        for i = 1:real_max_stims %index in data_table, from data_table get grapes to plot
            stim_idx = stims2plot(i);

            subplot_ax = subplot(rows,cols,pic_num(i),'align');
            scale_factor = plot_img_zscore_class(stim_idx, data_table, grapes, ...
                                                    subplot_ax, fontsize_mult);
            
            if ~is_ch_grapes
                add_bundle_label(stim_idx, data_table, grapes, ...
                                 priority_chs_ranking, fontsize_mult, ...
                                 scale_factor);
            end

            set(subplot_ax,'xtick',[],'ytick',[]);box off;
            
            subplot_ax = subplot(rows,cols,rast_num(i),'align');
            plot_rasters(stim_idx, data_table, grapes, rank_config, ...
                            subplot_ax, b_yellow_patch, b_short_win);
            if b_short_win
                time_pre_ms = grapes.time_pre / 2;
                time_pos_ms = grapes.time_pos / 2;
                scale_ax(subplot_ax, 0.5);
            else
                time_pre_ms = grapes.time_pre;
                time_pos_ms = grapes.time_pos;
            end
            
            subplot_ax = subplot(rows,cols,hist_num(i),'align');
            plot_ifr(stim_idx, data_table, grapes, ejex, ifr_y_max, ...
                                subplot_ax, fontsize_mult, b_short_win)
            if (i>(real_max_stims-cols))&&(i<=real_max_stims)
                set(subplot_ax,'xtick',[-time_pre_ms 0 time_pos_ms/2 time_pos_ms]);
            else
                axis off
            end
            if b_short_win
                scale_ax(subplot_ax, 0.5);
            end
        end

        add_figure_title(grapes, rank_config, ss_num, nwin, extra_label)
        
        if save_fig
            extra_label_name = extra_label;
            aaa=strfind(extra_label,' (');
            if ~isempty(aaa)
                extra_label_name = extra_label(1:aaa-1);
            end
            print(fig,'-dpng',sprintf('%s_best_resp_subscr %d_fig %d.png',extra_label_name,ss_num,nwin));  
%             F = getframe(fig);
%             imwrite(F.cdata, [sprintf('%s_best_resp_subscr %d_fig %d.png',extra_label_name,ss_num,nwin)])
        end
        
        if close_fig
            close(fig)
        else
            fig_cells{nwin} = fig;
        end
    end
    
    warning on
    set(groot,'defaultfiguregraphicssmoothing','remove')
    set(groot,'defaultaxesfontsmoothing','remove')
    set(groot,'defaultaxestitlefontsizemultiplier','remove')
    set(groot,'defaultaxestitlefontweight','remove')
    set(groot,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'remove','remove','remove'})

end
function scale_ax(ax, perc)
    pos = get(ax, 'Position');
    new_width = pos(3) * perc; 
    pos(1) = pos(1) + (pos(3) - new_width)/2; % Shift to center
    pos(3) = new_width;
    set(ax, 'Position', pos);
end

function scale_factor = plot_img_zscore_class(stim_idx, data_table, grapes, subplot_ax, fontsize_mult)
    strength_thr = 5;
    img_idx = data_table.stim_number(stim_idx);
            
    imagename = fullfile(grapes.ImageNames.folder{img_idx} , grapes.ImageNames.name{img_idx});
    if ~exist(imagename, "file")
        fprintf("%s not found", imagename);
    end
    if contains(lower(imagename),'.jp')
        img_name_color = 'black';
        % Plot image name under the image
        [~,fname,~] = fileparts(imagename);
        if length(fname) > 25 % If image name is too long..
            fname = [fname(1:10) '..' fname(end-15:end)];
        end
        if any("selectable" == string(grapes.ImageNames.Properties.VariableNames))
            if grapes.ImageNames.selectable(img_idx) == 1 % same_unit, same_feats
                img_name_color = 'magenta';
            elseif grapes.ImageNames.selectable(img_idx) == 2 % same_cat, diff_feats
                img_name_color = 'blue';
            elseif grapes.ImageNames.selectable(img_idx) == 3 % all_but_x
                img_name_color = 'cyan';
            else
                img_name_color = 'black';
            end
        else
            if ismember('concept_number', data_table.Properties.VariableNames)
                if str2num(data_table.concept_number(stim_idx)) > 1
                    img_name_color = 'magenta';
                else
                    img_name_color = 'black';
                end
            end
        end
        
        img_name_pos_x = 50;
        img_name_pos_y = 195;
        try
            A = imread(imagename);
            [~, width, ~] = size(A);
            scale_factor = width/160; % Use 160px as base reference
            
            % Scale positions based on image size
            img_name_pos_x = 50 * scale_factor;
            img_name_pos_y = 195 * scale_factor;
            zm_pos_x = 210 * scale_factor;
            zm_pos_y = 40 * scale_factor;
            zscore_pos_x = 190 * scale_factor;
            zscore_pos_y = 110 * scale_factor;

            image(A); set(subplot_ax,'PlotBoxAspectRatio',[1 1 1]);  
            if contains(grapes.ImageNames.folder{img_idx},'RSVPCategLocaliz') || ...
                contains(grapes.ImageNames.folder{img_idx},'categ_pics') || ...
                contains(grapes.ImageNames.folder{img_idx},'categ_localiz')
                colormap(gray(256))
            end
        catch
            img_name_pos_x = 500;
            img_name_pos_y = 0.5;
            %xlim([-time_pre_ms time_pos_ms]);
            set(subplot_ax,'PlotBoxAspectRatio',[2 1 1]); %just a test
            xlim([0, 1000]);
        end
        text(img_name_pos_x,img_name_pos_y,fname,'fontsize',4*fontsize_mult, ...
            'HorizontalAlignment','center', 'interpreter','none', 'Color', img_name_color);
        if data_table.zscore(stim_idx) >= 1e10
            zscore_val = Inf;
        else
            zscore_val = data_table.zscore(stim_idx);
        end
        if data_table.zscore(stim_idx)>=strength_thr && data_table.p_test(stim_idx) && data_table.min_spk_test(stim_idx) && data_table.good_lat(stim_idx)
            text(zm_pos_x,zm_pos_y,'Z_m*','fontsize',5*fontsize_mult,'FontWeight','bold');
            
            text(zscore_pos_x,zscore_pos_y,sprintf('%2.2f',zscore_val),'fontsize',5*fontsize_mult,'FontWeight','bold');
        elseif data_table.zscore(stim_idx)>=strength_thr && data_table.p_test(stim_idx) && data_table.min_spk_test(stim_idx)
            text(zm_pos_x,zm_pos_y,'Z_m','fontsize',5*fontsize_mult,'FontWeight','bold');
            text(zscore_pos_x,zscore_pos_y,sprintf('%2.2f',zscore_val),'fontsize',5*fontsize_mult,'FontWeight','bold');
        else
            text(zm_pos_x,zm_pos_y,'Z_m','fontsize',5*fontsize_mult);
            text(zscore_pos_x,zscore_pos_y-10*scale_factor,sprintf('%2.2f',zscore_val),'fontsize',5*fontsize_mult);
        end            
    else
        if exist(imagename,'file')
            [y,Fs] = audioread(imagename);
            line(subplot_ax,(0:1/Fs:(length(y)-1)/Fs)*1000,y(:,1));
            xlim([-time_pre_ms time_pos_ms]);
            [~,fname,~] = fileparts(imagename);
            text(500,0.5,fname,'fontsize',5*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
        else
            xlim([-time_pre_ms time_pos_ms]);
            text(500,0.5,lower(imagename),'fontsize',5*fontsize_mult,'HorizontalAlignment','center','interpreter','none');
        end
    end

    if data_table.p_test(stim_idx)
        sign_test = '*';
    else
        sign_test = '';
    end
    image_ylabel = {[num2str(img_idx) sign_test],['#' num2str(data_table.ntrials(stim_idx))], ...
                    [erase( data_table.channel{stim_idx} , 'chan' ) ':' ...
                     erase( data_table.class{stim_idx} , 'class' )]};
    
    ylabel(image_ylabel,'rotation',0,'fontsize',5*fontsize_mult,...
        'HorizontalAlignment','right','FontWeight','bold','units','normalized','position',[-0.1 -0.2 0]);
end

function add_bundle_label(stim_idx, data_table, grapes, ...
                          priority_chs_ranking, fontsize_mult, ...
                          scale_factor)
    % Bundle label
    electrode_id = erase(data_table.channel{stim_idx},'chan');
    color = 'black';
    if exist('priority_chs_ranking', 'var') && numel(priority_chs_ranking)
        if sum(priority_chs_ranking == str2num(electrode_id)) == 1
            % Color label red to denote priority channel
            color = 'red';
        else
            color = 'black';
        end
    end
    try
        chan_rasters = grapes.rasters.(data_table.channel{stim_idx});
        if isfield(chan_rasters, 'details')
            bundle = chan_rasters.details.ch_label;
            % regex to get the bundle name
            bundle = regexprep(bundle, '[^A-Z]', '');
            text(190*scale_factor,160*scale_factor,bundle,'fontsize', 4*fontsize_mult, 'Color', color)
        end
    catch ME
        msg = getReport(ME)
    end

end

function plot_rasters(stim_idx, data_table, grapes, rank_config, subplot_ax, b_yellow_patch, b_short_win)
    LineFormat.Color = 'blue';
    LineFormat.LineWidth = 0.35;

    if b_short_win
        time_pre_ms = grapes.time_pre / 2; 
        time_pos_ms = grapes.time_pos / 2; 
    else
        time_pre_ms = grapes.time_pre;
        time_pos_ms = grapes.time_pos;
    end
    

    hold on;        
    img_idx = data_table.stim_number(stim_idx);
    spikes1 = grapes.rasters.(data_table.channel{stim_idx}).(data_table.class{stim_idx}).stim{img_idx};

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
    if b_yellow_patch
        % Highlight the response window with a yellow patch
        xpoints = [data_table.onset(stim_idx) data_table.onset(stim_idx) + rank_config.win_length ...
                   data_table.onset(stim_idx) + rank_config.win_length data_table.onset(stim_idx)];
        ypoints = [limy(1) limy(1) limy(2) limy(2)]; 
        p=patch(xpoints,ypoints,'y');
        set(p,'FaceAlpha',0.2,'linestyle','none'); 
        uistack(p, 'bottom'); % <-- Send patch to back
    end
    set(subplot_ax,'FontSize',7);
    axis off
    %subplot_ax = subplot(rows,cols,rast_idx,'align');
    line([data_table.onset(stim_idx) data_table.onset(stim_idx)],ylim,'linestyle',':','color','r','linewidth',0.5)
end

function plot_ifr(stim_idx, data_table, grapes, ifr_x, ...
                  ifr_y_max, subplot_ax, fontsize_mult, b_short_win)
    % Plots IFR curve, prints onset(lat) and duration
    if b_short_win
        time_pre_ms = grapes.time_pre / 2; 
        time_pos_ms = grapes.time_pos / 2; 
    else
        time_pre_ms = grapes.time_pre;
        time_pos_ms = grapes.time_pos;
    end
    hold on;
    if isnan(data_table.IFR{stim_idx}(1)) && all(isnan(data_table.IFR{stim_idx}))
        IFR_sm = data_table.IFR{stim_idx}';
    else
        smooth_bin = ceil(50 * (ifr_x(2)-ifr_x(1)));
        IFR_sm = smooth(data_table.IFR{stim_idx},smooth_bin);
    end
    
    lo=line(subplot_ax,ifr_x,IFR_sm,'Color','r','Linewidth',1);
    
    if ifr_y_max == -1 
        %ifr_y_max = max(IFR_sm) * 1.05; 
        limy=ylim; % (FC)
        ifr_y_max = (limy(2));% (FC)
    end
    xlim([-time_pre_ms time_pos_ms])     

    if ~data_table.good_lat(stim_idx)
        title(sprintf('lat=%3.0f dur=%3.0f',data_table.onset(stim_idx),data_table.dura(stim_idx)),'fontsize',5*fontsize_mult,'interpreter','none','VerticalAlignment','middle');
    else
        title(sprintf('lat=%3.0f dur=%3.0f',data_table.onset(stim_idx),data_table.dura(stim_idx)),'fontsize',5*fontsize_mult,'FontWeight','bold','interpreter','none','VerticalAlignment','middle');
    end
    box off
    %subplot_ax = subplot(rows,cols,ifr_idx,'align');
    line(xlim,[data_table.IFR_thr(stim_idx) data_table.IFR_thr(stim_idx)],'linestyle','--','color','k')
      
    set(subplot_ax,'YLim',[0 ifr_y_max]);
    line([0 0],[0 ifr_y_max],'linestyle',':')
    line([500 500],[0 ifr_y_max],'linestyle',':')
    line([1000 1000],[0 ifr_y_max],'linestyle',':')
    line([data_table.onset(stim_idx) data_table.onset(stim_idx)],[0 ifr_y_max],'linestyle',':','color','k','linewidth',0.5)
    
    %
    limy=ylim;
    xpoints = [data_table.onset(stim_idx) data_table.onset(stim_idx)+data_table.dura(stim_idx) data_table.onset(stim_idx)+data_table.dura(stim_idx) data_table.onset(stim_idx)];
    ypoints = [limy(1) limy(1) limy(2) limy(2)]; 
    p=patch(xpoints,ypoints,'c');
    set(p,'FaceAlpha',0.2,'linestyle','none');
    uistack(p, 'bottom'); % <-- Send patch to back
    uistack(lo, 'top'); % <-- Send ifr line to top
end

function add_figure_title(grapes, rank_config, ss_num, fig_num, extra_label)
    from_onset = 1;
    dirs = split(pwd,filesep);
    if ~from_onset
        title(sprintf('%s, %s responses. subscr:%d fig:%d. resp_win =[%d, %d] ms (ntsd=%d)(thr=%dHz)', ...
            extra_label,grapes.exp_type,ch_label,ss_num,fig_num,tmin_median,tmax_median,rank_config.nstd), ...
            'interpreter','none','units','normalized','Position',[0 1.5 0])
    elseif ~rank_config.win_cent
        title(sprintf('%s, %s responses. subscr:%d fig:%d. resp_win: %dms ons-100 (ntsd=%d)', ...
            extra_label,grapes.exp_type,ss_num,nwin,rank_config.win_length,rank_config.nstd), ...
            'interpreter','none','units','normalized','Position',[0 1.5 0])
    else
        sgtitle(sprintf('%s, %s responses. subscr:%d fig:%d. resp_win: %dms cent dura (ntsd=%d)', ...
            extra_label,grapes.exp_type,ss_num,fig_num,rank_config.win_length,rank_config.nstd), ...
            'interpreter','none', 'fontsize', 8);
    end
end