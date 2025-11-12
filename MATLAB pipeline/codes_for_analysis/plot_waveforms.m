% chan=259;
chan=334;
% chan=259;
% name = 'mLPBTG03';
name = 'mLFUS05';
% name = 'mLFUS08';
% classes_ids = [1 3];
classes_ids = [3 4];
% classes_ids = [2 8];
load(sprintf('times_%s raw_%d.mat',name,chan), 'spikes','cluster_class')
clus_colors = [[0.0 0.0 1.0];[1.0 0.0 0.0];[0.0 0.5 0.0];[0.620690 0.0 0.0];[0.413793 0.0 0.758621];[0.965517 0.517241 0.034483];
    [0.448276 0.379310 0.241379];[1.0 0.103448 0.724138];[0.545 0.545 0.545];[0.586207 0.827586 0.310345];
    [0.965517 0.620690 0.862069];[0.620690 0.758621 1.]];

fig = figure('Visible','on','Position',[300   120   798   598.5],'Units','pixels','GraphicsSmoothing','off');
k=1;
hold on;
for i=1:numel(classes_ids)
    spks = spikes(cluster_class(:,1)==classes_ids(i),:);
    nspks = size(spks,1);
    plot(mean(spks),'color',clus_colors(i,:),'linewidth',1);
    plot(mean(spks)-k*std(spks),'color',clus_colors(i,:),'linestyle','--');
    plot(mean(spks)+k*std(spks),'color',clus_colors(i,:),'linestyle','--');
end
xlim([1 64]);
set(gca,'xtick',[5 20 62],'xticklabel',{'-0.5','0','1.4'},'fontsize',20);
xlabel('Time from spike peak (ms)')
grid on
ylabel('Amplitude (uV)')
box on
print(fig,'-dpng',sprintf('spikes_ch%d_classes_%s.png',chan,num2str(classes_ids)));
