function sort_and_plot_spikes(channame,spikes, index,par_input,ws_folder)
min_spikes4SPC = 60;
resolution = '-r150';
cd(ws_folder);
[times_filename, status] = do_clustering_from_spikes(channame,spikes, index,min_spikes4SPC, par_input,true);
if status==0
    return
end
curr_fig = figure('Visible','Off');
set(curr_fig, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'RendererMode','manual','Renderer','painters')
if isfield(get(curr_fig),'GraphicsSmoothing')
    set(curr_fig,'GraphicsSmoothing','off');
end

par = set_parameters();
par.filename = times_filename;
data_handler = readInData(par);
par = data_handler.update_par(par);
par = update_parameters(par,par_input,'batch_plot');
par.min_clus = 15;
par.cont_segment = false;  %maybe true and save the sample in spikes
%% plotring data segment
subplot(3,1,1)
%     if par.cont_segment && data_handler.with_psegment
%         box off; hold on
%         %these lines are for plotting continuous data
%         [xd_sub, sr_sub] = data_handler.get_signal_sample();
%         lx = length(xd_sub);
%         plot((1:lx)/sr_sub,xd_sub)
%         noise_std_detect = median(abs(xd_sub))/0.6745;
%         xlim([0 lx/sr_sub])
%         thr = par.stdmin * noise_std_detect;
%         plotmax = 15 * noise_std_detect; %thr*par.stdmax/par.stdmin;
% 
%         if strcmp(par.detection,'pos')
%             line([0 length(xd_sub)/sr_sub],[thr thr],'color','r')
%             ylim([-plotmax/2 plotmax])
%         elseif strcmp(par.detection,'neg')
%             line([0 length(xd_sub)/sr_sub],[-thr -thr],'color','r')
%             ylim([-plotmax plotmax/2])
%         else
%             line([0 length(xd_sub)/sr_sub],[thr thr],'color','r')
%             line([0 length(xd_sub)/sr_sub],[-thr -thr],'color','r')
%             ylim([-plotmax plotmax])
%         end
%     end
%%
title([pwd '/' channame],'Interpreter','none','Fontsize',14)

if ~status
    print(curr_fig,'-dpng',['fig2print_' channame '.png'],resolution);
    return
end

% LOAD SPIKES
[clu, tree, spikes, index, inspk, ipermut, classes, forced,temp] = data_handler.load_results();
nspk = size(spikes,1);
Mclasses = max(classes);
auto_sort_info = [];
if data_handler.with_gui_status
    [original_classes, current_temp, auto_sort_info] = data_handler.get_gui_status();
    temperature = par.mintemp+current_temp*par.tempstep;
    org_class = zeros(1,Mclasses);
    for i = 1:Mclasses
        org_class(i) = original_classes(find(classes==i,1,'first'));
    end
end

%PLOTS
ylimit = [];
subplot(3,5,11)

color = [[0.0 0.0 1.0];[1.0 0.0 0.0];[0.0 0.5 0.0];[0.620690 0.0 0.0];[0.413793 0.0 0.758621];[0.965517 0.517241 0.034483];
[0.448276 0.379310 0.241379];[1.0 0.103448 0.724138];[0.545 0.545 0.545];[0.586207 0.827586 0.310345];
[0.965517 0.620690 0.862069];[0.620690 0.758621 1.]];
maxc = size(color,1);

hold on
num_temp = floor((par.maxtemp -par.mintemp)/par.tempstep);     % total number of temperatures
switch par.temp_plot
    case 'lin'
       if ~isempty(auto_sort_info)
            [xp, yp] = find(auto_sort_info.peaks);
            ptemps = par.mintemp+(xp)*par.tempstep;
            psize = tree(sub2ind(size(tree), xp,yp+4));
            plot(ptemps,psize,'xk','MarkerSize',7,'LineWidth',0.9);
            area(par.mintemp+par.tempstep.*[auto_sort_info.elbow,num_temp],max(ylim).*[1 1],'LineStyle','none','FaceColor',[0.9 0.9 0.9]);
        end
        plot([par.mintemp par.maxtemp-par.tempstep], ...
        [par.min_clus par.min_clus],'k:',...
        par.mintemp+(1:num_temp)*par.tempstep, ...
        tree(1:num_temp,5:size(tree,2)),[temperature temperature],[1 tree(1,5)],'k:')
        for i=1:Mclasses
            tree_clus = tree(temp(i),4+org_class(i));
            tree_temp = tree(temp(i)+1,2);
            plot(tree_temp,tree_clus,'.','color',color(mod(i-1,maxc)+1,:),'MarkerSize',20);
        end
    case 'log'
        set(gca,'yscale','log');
        if ~isempty(auto_sort_info)
            [xp, yp] = find(auto_sort_info.peaks);
            ptemps = par.mintemp+(xp)*par.tempstep;
            psize = tree(sub2ind(size(tree), xp,yp+4));
            semilogy(ptemps,psize,'xk','MarkerSize',7,'LineWidth',0.9);
            area(par.mintemp+par.tempstep.*[auto_sort_info.elbow,num_temp],max(ylim).*[1 1],'LineStyle','none','FaceColor',[0.9 0.9 0.9],'basevalue',1);
        end
        semilogy([par.mintemp par.maxtemp-par.tempstep], ...
        [par.min_clus par.min_clus],'k:',...
        par.mintemp+(1:num_temp)*par.tempstep, ...
        tree(1:num_temp,5:size(tree,2)),[temperature temperature],[1 tree(1,5)],'k:')

        for i=1:Mclasses
            tree_clus = tree(temp(i),4+org_class(i));
            tree_temp = tree(temp(i)+1,2);
            semilogy(tree_temp,tree_clus,'.','color',color(mod(i-1,maxc)+1,:),'MarkerSize',20);
        end
end
xlim([par.mintemp, par.maxtemp])
subplot(3,5,6)
hold on

class0 = find(classes==0);
    max_spikes=min(length(class0),par.max_spikes_plot);
    plot(spikes(class0(1:max_spikes),:)','k');
    xlim([1 size(spikes,2)]);
subplot(3,5,10);
    hold on
    plot(spikes(class0(1:max_spikes),:)','k');
    plot(mean(spikes(class0,:),1),'c','linewidth',2)
    xlim([1 size(spikes,2)]);
    title(['Cluster 0: # ' num2str(length(class0))],'Fontweight','bold')
subplot(3,5,15)
    xa=diff(index(class0));
    [n,c]=hist(xa,0:1:100);
    bar(c(1:end-1),n(1:end-1))
    xlim([0 100])
    xlabel('ISI (ms)');
    title([num2str(nnz(xa<3)) ' in < 3ms']);

    
numclus = max(classes);
subplot(3,5,6);   
for i = 1:numclus
    class = find(classes==i);
    max_spikes=min(length(class),par.max_spikes_plot);
    plot(spikes(class(1:max_spikes),:)','color',color(mod(i-1,maxc)+1,:));
    xlim([1 size(spikes,2)]);
end
ylim('auto')
ylimit = ylim;

for i = 1:min(numclus,3)
    class = find(classes==i);
    max_spikes=min(length(class),par.max_spikes_plot);
    subplot(3,5,6+i);
    hold on
    plot(spikes(class(1:max_spikes),:)','color',color(mod(i-1,maxc)+1,:));
    plot(mean(spikes(class,:),1),'k','linewidth',2)
    xlim([1 size(spikes,2)]);
    ylim(ylimit);
    title(['Cluster ' num2str(i) ': # ' num2str(length(class)) ' (' num2str(nnz(classes(:)==i & ~forced(:))) ')'],'Fontweight','bold')
    subplot(3,5,11+i)
    xa=diff(index(class));
    [n,c]=hist(xa,0:1:100);
    bar(c(1:end-1),n(1:end-1))
    xlim([0 100])
    xlabel('ISI (ms)');
    title([num2str(nnz(xa<3)) ' in < 3ms']);
end
print(curr_fig,'-dpng',['fig2print_' channame '.png'],resolution);

if numclus>3
    clf(curr_fig)
    title([pwd '/' channame 'a'],'Interpreter','none','Fontsize',14)
    for i = 4:min(8,max(classes))
        class = find(classes==i);
        max_spikes=min(length(class),par.max_spikes_plot);
        if i<=8
            subplot(3,5,2+i);
            hold on
            plot(spikes(class(1:max_spikes),:)','color',color(mod(i-1,maxc)+1,:));
            plot(mean(spikes(class,:),1),'k','linewidth',2)
            xlim([1 size(spikes,2)]);
            title(['Cluster ' num2str(i) ': # ' num2str(length(class)) ' (' num2str(nnz(classes(:)==i & ~forced(:))) ')'],'Fontweight','bold')
            ylim(ylimit);
            subplot(3,5,7+i)
            xa=diff(index(class));
            [n,c]=hist(xa,0:1:100);
            bar(c(1:end-1),n(1:end-1))
            xlim([0 100])
            xlabel('ISI (ms)');
            title([num2str(nnz(xa<3)) ' in < 3ms']);
        end
    end
    print(curr_fig,'-dpng',['fig2print_' channame 'a.png'],resolution);
end

close(curr_fig)
end