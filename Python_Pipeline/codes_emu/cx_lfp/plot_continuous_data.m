% function plot_continuous_data(data, par, chID, labels, conversion, folderpath)
function plot_continuous_data(data, par, chID, labels, conversion, folderpath,notches_folder,remove_artifacts)
% Plots the continuous data and does a first detection of spikes for each
% notches_folder = folderpath;
detect = par.detect_sign;
factor_thr = par.thr;
MAX_NOTCHES = [0;10;25;Inf];
detect_fmin = 300;
detect_fmax = 3000;
filt_order = 4;
sr = 30000;
min_ref_per = 1.5;     %detector dead time (in ms)
Vmin = 40;
% for_text = 10;
w_pre=20;                       %number of pre-event data points stored
w_post=44;                      %number of post-event data points stored
ref = floor(min_ref_per*sr/1000);    %number of counts corresponding the dead time
par.ref = ref;

if ~exist('remove_artifacts','var') || isempty(remove_artifacts)
    remove_artifacts = false;
end

[b_orig,a_orig]=ellip(filt_order,0.1,40,[detect_fmin detect_fmax]*2/(sr));
[z_det,p_det,k_det] = tf2zpk(b_orig,a_orig);
if isempty(chID)
    return
end

colours_cell = {[0 0 0],[0.4660, 0.6740, 0.1880],[0.8500, 0.3250, 0.0980],[0, 0.4470, 0.7410]};
line_labels = ['KGRB'];
max_record = size(data, 2);
tmax = max_record/sr;

% max_subplots = 8;

fig_num = ceil(10000*rand(1));
figure(fig_num)
set(fig_num, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1], 'Visible', 'off')

if exist(fullfile(notches_folder,'pre_processing_info.mat'),'file')
    load(fullfile(notches_folder,'pre_processing_info.mat'),'process_info')
elseif exist([pwd filesep 'pre_processing_info.mat'],'file')
    load([pwd filesep 'pre_processing_info.mat'],'process_info')
elseif exist(fullfile(folderpath,'pre_processing_info.mat'),'file')
    load(fullfile(folderpath,'pre_processing_info.mat'),'process_info')
else
    process_info = [];
end

% max_ch_per_macro = 8;

inds = cellfun(@(x)regexp(x,'\d+','once'),labels,'UniformOutput',false);
bundles = cellfun(@(x,y) x(1:y-1),labels,inds,'UniformOutput',false);

bundles_to_plot = unique(bundles);

% splitlabels = cellfun(@(x)regexp(x,'(?=\w*)\d*-(?=\w*$)','split'),labels,'UniformOutput',false);
% %if it looks fine: chanelname-chid
% labels_wm =  cellfun(@(x) numel(x)==2 && ~isempty(x{1}) &&  ~isempty(x{2}), splitlabels);
% [umacros,~,umacrosix]  = unique(cellfun(@(x)x{1},splitlabels(labels_wm),'UniformOutput',false));
% %check that all macros has max_ch_per_macro or less channels
% if max(arrayfun(@(x)sum(x==umacrosix),1:numel(umacros))) <= max_ch_per_macro
%     macronums =  cellfun(@(x) str2double(regexp(x,'(?=\w*)\d*(?=-\w*$)','match','once')) ,labels(labels_wm));
%     % starts with channels without macro name
%     plot_order = find(~labels_wm);
%     last_ch_of_macro = chID(plot_order);
%     % for each macro
%     macroix = find(labels_wm);
%     for i=1:max(umacrosix)
%         kidx = find(umacrosix==i);
%         lnums = macronums(kidx);
%         [~, sidx] = sort(lnums);
%         plot_order = [plot_order; macroix(kidx(sidx))]; %adds in order channes for this macro
%         last_ch_of_macro = [last_ch_of_macro;chID(plot_order(end))];
%     end
% else
%     plot_order = 1:length(chID);
%     last_ch_of_macro = chID(1);
%     for k= 2:length(chID)
%         if ceil(last_ch_of_macro(end)/8)==ceil(chID(k)/8)
%             last_ch_of_macro(end) = chID(k);
%         else
%             last_ch_of_macro(end+1) = chID(k);
%         end
%     end
% end

eje = linspace(0,tmax,max_record);
%ch = [];
% cont=1;

artifact_half_width = 500; % 500 samples

for ibun = 1:length(bundles_to_plot)
    pos_chans_to_plot = find(arrayfun(@(x) (strcmp(x,bundles_to_plot{ibun})),bundles));

    max_subplots = length(pos_chans_to_plot);
    cont=1;
    clf(fig_num)
    for k= 1:max_subplots
        channel1=chID(pos_chans_to_plot(k));
        posch = pos_chans_to_plot(k);

        notches = [];
        if ~isempty(process_info)
            pinfo_ix = find([process_info(:).chID]==channel1);
            if ~isempty(pinfo_ix)
                notches =  process_info(pinfo_ix).notches;
            end
        end
        subplot(max_subplots,1,cont)
        box off; hold on
        Vlim = Vmin;

        text2show = [];
        for n_notc=1:length(MAX_NOTCHES)
            if n_notc>1 && isempty(notches)
                break
            end

            K = k_det;
            P = p_det;
            Z = z_det;

            if ~isempty(notches)
                [~, ix_notch] = sort(notches.abs_db,'descend');
                n_notches = min(MAX_NOTCHES(n_notc), length(notches.abs_db));

                for ni = 1:n_notches
                    zpix = ix_notch(ni)*2+(-1:0);
                    Z(end+1:end+2) = notches.Z(zpix);
                    P(end+1:end+2) = notches.P(zpix);
                    K = K *notches.K(ix_notch(ni));
                end
            end
            [S,G] = zp2sos(Z,P,K);

            % HIGH-PASS FILTER OF THE DATA
            xd = fast_filtfilt(S,G, single(data(posch,:))* conversion(posch));

            line(eje,xd,'Color',colours_cell{n_notc})

            % GET THRESHOLD AND NUMBER OF SPIKES BETWEEN 0 AND TMAX
            thr = factor_thr * median(abs(xd))/0.6745;

            thrmax = 10 * thr;     %thrmax for artifact removal is based on sorted settings.

            if remove_artifacts
                artifact_idx = find(abs(xd(w_pre+2:end-w_post-2)) > abs(thrmax)) +w_pre+1;
                if numel(artifact_idx) > 0
                    clean_samples = xd;
                    % counter = 1;
                    for idx=artifact_idx
                        idx_val = str2num(sprintf('%.0f',idx));
                        clean_samples(idx_val-artifact_half_width:idx_val+artifact_half_width) = 0;
                        % artifact_idx_int(counter) = idx_val;
                        % counter = counter + 1;
                    end
    
                    % Plot artifactless signal
                    line(eje,clean_samples,'Color','r')
                end
            end

            switch detect
                case 'pos'
                    xaux = find((xd(w_pre+2:end-w_post-2) > thr) & (abs(xd(w_pre+2:end-w_post-2)) < thrmax)) +w_pre+1;
                    line([0 tmax],[thr thr],'color',colours_cell{n_notc},'LineWidth',1)
                    ylim([-Vlim 3*Vlim])
                case 'neg'
                    xaux = find((xd(w_pre+2:end-w_post-2) < -thr) & (abs(xd(w_pre+2:end-w_post-2)) < thrmax)) +w_pre+1;
                    line([0 tmax],[-thr -thr],'color',colours_cell{n_notc},'LineWidth',1)
                    ylim([-3*Vlim Vlim])
                case 'abs'
                    xaux = find((abs(xd(w_pre+2:end-w_post-2)) > thr) & (abs(xd(w_pre+2:end-w_post-2)) < abs(thrmax))) +w_pre+1;
                    line([0 tmax],[thr thr],'color',colours_cell{n_notc},'LineWidth',1)
                    line([0 tmax],[-thr -thr],'color',colours_cell{n_notc},'LineWidth',1)
                    ylim([-3*Vlim 3*Vlim])
            end
            clear xd;
            if ~isempty(xaux)
                nspk=nnz(diff(xaux)>ref)+1;
            else
                nspk = 0;
            end

            if isempty(notches)
                text2show = sprintf('%c: %.2f %d  spikes.',line_labels(n_notc),thr,nspk);
            else
                text2show =[text2show sprintf('[%c: %.2f | %d notches, %d spikes]',line_labels(n_notc),thr,n_notches, nspk) ];
            end
        end
        ylabel(['Ch.' num2str(channel1)])

        cont=cont+1;
        if (nspk > ceil(tmax/20) && nspk < tmax * 60)
            ylabel(['Ch.' num2str(channel1)],'fontsize',10,'fontweight','bold')
        end

%         text((tmax)/5,for_text+max(ylim),text2show,'fontsize',10)
        title(text2show,'fontsize',10)

        set(gca,'Xlim',[0 tmax],'Xtick',linspace(0,tmax,7))
        if any(k==max_subplots)
            fprintf('\n')
            %         cont=1;
            xlabel('Time (sec)')
            %if sum(diff(ceil(ch/8)))==0
            %             macro_i = regexp(labels{k},'\d+-\d+$','start','once');
            %             if isempty(macro_i) || macro_i<2
            %                 macro = num2str(ceil(chID(k)/8));
            %             else
            %                 macro = labels{k}(1:macro_i-1);
            %             end
            title_out = sprintf('%s.   bundle  %s.   fmin %d Hz. fmax %d Hz',pwd,bundles_to_plot{ibun},detect_fmin,detect_fmax);
            %end

            sgtitle(title_out,'fontsize',12,'interpreter','none','fontWeight','bold','HorizontalAlignment','left')

            outfile=sprintf('%s_%s_filtorder%d_withnotches%d_det%s','fig2print_bundle',bundles_to_plot{ibun},filt_order,~isempty(notches),detect);

            print(fig_num,fullfile(folderpath,outfile),'-dpng')

        end
    end
end
    % for k= plot_order(:)'
    %     % LOAD NSX DATA
    %     channel1 = chID(k);
    %     %ch = [ch channel1];
    %     notches = [];
    %     if ~isempty(process_info)
    %         pinfo_ix = find([process_info(:).chID]==chID(k));
    %         if ~isempty(pinfo_ix)
    %             notches =  process_info(pinfo_ix).notches;
    %         end
    %     end
    %
    %
    %     % MAKES PLOT
    %     subplot(max_subplots,1,cont)
    %     box off; hold on
    %     Vlim = Vmin;
    %
    %
    %
    %     text2show = [];
    %     for n_notc=1:length(MAX_NOTCHES)
    %         if n_notc>1 && isempty(notches)
    %             break
    %         end
    %
    %         K = k_det;
    %         P = p_det;
    %         Z = z_det;
    %
    %         if ~isempty(notches)
    %             [~, ix_notch] = sort(notches.abs_db,'descend');
    %             n_notches = min(MAX_NOTCHES(n_notc), length(notches.abs_db));
    %
    %             for ni = 1:n_notches
    %                 zpix = ix_notch(ni)*2+(-1:0);
    %                 Z(end+1:end+2) = notches.Z(zpix);
    %                 P(end+1:end+2) = notches.P(zpix);
    %                 K = K *notches.K(ix_notch(ni));
    %             end
    %         end
    %         [S,G] = zp2sos(Z,P,K);
    %
    %         % HIGH-PASS FILTER OF THE DATA
    %         xd = fast_filtfilt(S,G, single(data(k,:))* conversion(k));
    %
    %         % GET THRESHOLD AND NUMBER OF SPIKES BETWEEN 0 AND TMAX
    %         thr = factor_thr * median(abs(xd))/0.6745;
    %         thrmax = 10 * thr;     %thrmax for artifact removal is based on sorted settings.
    %         line(eje,xd,'Color',colours_cell{n_notc})
    %         switch detect
    %             case 'pos'
    %                 xaux = find((xd(w_pre+2:end-w_post-2) > thr) & (abs(xd(w_pre+2:end-w_post-2)) < thrmax)) +w_pre+1;
    %                 line([0 tmax],[thr thr],'color',colours_cell{n_notc},'LineWidth',1)
    %                 ylim([-Vlim 3*Vlim])
    %             case 'neg'
    %                 xaux = find((xd(w_pre+2:end-w_post-2) < -thr) & (abs(xd(w_pre+2:end-w_post-2)) < thrmax)) +w_pre+1;
    %                 line([0 tmax],[-thr -thr],'color',colours_cell{n_notc},'LineWidth',1)
    %                 ylim([-3*Vlim Vlim])
    %             case 'abs'
    %                 xaux = find((abs(xd(w_pre+2:end-w_post-2)) > thr) & (abs(xd(w_pre+2:end-w_post-2)) < abs(thrmax))) +w_pre+1;
    %                 line([0 tmax],[thr thr],'color',colours_cell{n_notc},'LineWidth',1)
    %                 line([0 tmax],[-thr -thr],'color',colours_cell{n_notc},'LineWidth',1)
    %                 ylim([-3*Vlim 3*Vlim])
    %         end
    %         clear xd;
    %         if ~isempty(xaux)
    %             nspk=nnz(diff(xaux)>ref)+1;
    %         else
    %             nspk = 0;
    %         end
    %
    %         if isempty(notches)
    %             text2show = sprintf('%c: %.2f %d  spikes.',line_labels(n_notc),thr,nspk);
    %         else
    %             text2show =[text2show sprintf('[%c: %.2f | %d notches, %d spikes]',line_labels(n_notc),thr,n_notches, nspk) ];
    %         end
    %     end
    %     ylabel(['Ch.' num2str(chID(k))])
    %
    %     cont=cont+1;
    %     ylabel(['Ch.' num2str(chID(k))],'fontsize',10,'fontweight','bold')
    %
    %     text((tmax)/5,for_text+max(ylim),text2show,'fontsize',10)
    %
    %     set(gca,'Xlim',[0 tmax],'Xtick',linspace(0,tmax,7))
    %     if any(channel1==last_ch_of_macro)
    %         fprintf('\n')
    %         cont=1;
    %         xlabel('Time (sec)')
    %         %if sum(diff(ceil(ch/8)))==0
    %             macro_i = regexp(labels{k},'\d+-\d+$','start','once');
    %             if isempty(macro_i) || macro_i<2
    %                 macro = num2str(ceil(chID(k)/8));
    %             else
    %                 macro = labels{k}(1:macro_i-1);
    %             end
    %             title_out = sprintf('%s.   bundle  %s.   fmin %d Hz. fmax %d Hz',pwd,macro,detect_fmin,detect_fmax);
    %         %end
    %
    %         sgtitle(title_out,'fontsize',12,'interpreter','none','fontWeight','bold','HorizontalAlignment','left')
    %
    %         outfile=sprintf('%s_%s_filtorder%d_withnotches%d_det%s','fig2print_macro',macro,filt_order,~isempty(notches),detect);
    %
    %         print(fig_num,fullfile(folderpath,outfile),'-dpng')
    %
    %
    %         %ch = [];
    %         clf(fig_num)
    %     end
    % end
close(fig_num)
