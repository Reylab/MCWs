function analyze_possible_artifacts(mat_file1,mat_file2)
%ANALYZE_POSSIBLE_ARTIFACTS Summary of this function goes here
%   Detailed explanation goes here
spikes_file1 = load(mat_file1);
spikes_file2 = load(mat_file2);

if isfield(spikes_file1,'spikes_all')
    spikes1 = spikes_file1.spikes_all;
else
    spikes1 = spikes_file1.spikes;
end
if isfield(spikes_file2,'spikes_all')
    spikes2 = spikes_file2.spikes_all;
else
    spikes2 = spikes_file2.spikes;
end

artifact_spikes1 = spikes1(find(spikes_file1.possible_artifact==1),:);
artifact_spikes2 = spikes2(find(spikes_file2.possible_artifact==1),:);



rr_ind = 207;
Nspks_plot = 1500;
rr_wav = artifact_spikes1
Nspks = size(rr_wav,1);
figure(35)
clf(35)
lsp = size(rr_wav,2);
x = repmat((1:lsp)',Nspks,1);
y = reshape(rr_wav',[Nspks*lsp,1]);
[aux,aux_c] = hist3([x,y],[lsp lsp]);
colormap('hot')
pcolor(aux_c{1},aux_c{2},aux')
shading('Flat');%interp
xlabel('samples')
ylabel('Amplitude (uV)')
% yline(0,'w:')
% xline(20,'w:')
% xline(40,'w:')
% colorbar
title(sprintf('Python pipeline %d spks',Nspks))
%title(sprintf('density %s Ch%d class%d. %d spks',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks))
%print(35,'-dpng', fullfile(resus_folder,sprintf('density_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(35,'-depsc','-vector', fullfile(resus_folder,sprintf('density_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
limy = ylim;
limx = xlim;
figure(34)
clf(34)
mu =mean(rr_wav);
sd = std(rr_wav);
% ord = randperm(Nspks,Nspks_plot);
% yline(0,':')
% xline(20,':')
% xline(40,':')
% plot(rr_wav(ord,:)','b');
hold on
plot(mu,'k')
plot(mu+sd,'k--')
plot(mu-sd,'k--')
xlim(limx)
ylim(limy)
xlabel('samples')
ylabel('Amplitude (uV)')
box on
title(sprintf('Python pipeline %d spks',Nspks))
%title(sprintf('muwaveform %s Ch%d class%d. %d spks',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks))
%print(34,'-dpng', fullfile(resus_folder,sprintf('spikesavg_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(34,'-depsc','-vector', fullfile(resus_folder,sprintf('spikesavg_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
figure(33)
clf(33)
mu =mean(rr_wav);
sd = std(rr_wav);
if Nspks < Nspks_plot
    ord = 1:Nspks;
else
    ord = randperm(Nspks,Nspks_plot);
end
% yline(0,':')
% xline(20,':')
% xline(40,':')
plot(rr_wav(ord,:)','b');
hold on
plot(mu,'k')
plot(mu+sd,'k--')
plot(mu-sd,'k--')
xlim(limx)
ylim(limy)
xlabel('samples')
ylabel('Amplitude (uv)')
box on
title(sprintf('Python pipeline %d spks',Nspks))
%title(sprintf('waveforms %s Ch%d class%d. %d spks. SNR=%2.2f. ISI=%1.1f%%',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks,spk_waveforms(rr_ind).SNR,spk_waveforms(rr_ind).perc_refrac3))
%print(33,'-dpng', fullfile(resus_folder,sprintf('spikes_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(33,'-depsc','-vector', fullfile(resus_folder,sprintf('spikes_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
figure(36)
clf(36)
lsp = size(rr_wav,2);
x = repmat((1:lsp)',Nspks,1);
y = reshape(rr_wav',[Nspks*lsp,1]);
[aux,aux_c] = hist3([x,y],[lsp lsp]);
aux=log10(aux+1);
colormap('hot')
pcolor(aux_c{1},aux_c{2},aux')
shading('Flat');%interp
xlabel('samples')
ylabel('Amplitude (uv)')
% yline(0,'w:')
% xline(20,'w:')
% xline(40,'w:')
% colorbar
title(sprintf('Python pipeline %d spks',Nspks))
%title(sprintf('log density %s Ch%d class%d. %d spks',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks))
%print(36,'-dpng', fullfile(resus_folder,sprintf('logdensity_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(36,'-depsc','-vector', fullfile(resus_folder,sprintf('logdensity_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))


rr_ind = 207;
Nspks_plot = 1500;
rr_wav = artifact_spikes2
Nspks = size(rr_wav,1);
figure(45)
clf(45)
lsp = size(rr_wav,2);
x = repmat((1:lsp)',Nspks,1);
y = reshape(rr_wav',[Nspks*lsp,1]);
[aux,aux_c] = hist3([x,y],[lsp lsp]);
colormap('hot')
pcolor(aux_c{1},aux_c{2},aux')
shading('Flat');%interp
xlabel('samples')
ylabel('Amplitude (uV)')
% yline(0,'w:')
% xline(20,'w:')
% xline(40,'w:')
% colorbar
title(sprintf('MATLAB pipeline %d spks',Nspks))
%title(sprintf('density %s Ch%d class%d. %d spks',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks))
%print(35,'-dpng', fullfile(resus_folder,sprintf('density_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(35,'-depsc','-vector', fullfile(resus_folder,sprintf('density_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
limy = ylim;
limx = xlim;
figure(44)
clf(44)
mu =mean(rr_wav);
sd = std(rr_wav);
% ord = randperm(Nspks,Nspks_plot);
% yline(0,':')
% xline(20,':')
% xline(40,':')
% plot(rr_wav(ord,:)','b');
hold on
plot(mu,'k')
plot(mu+sd,'k--')
plot(mu-sd,'k--')
xlim(limx)
ylim(limy)
xlabel('samples')
ylabel('Amplitude (uV)')
box on
title(sprintf('MATLAB pipeline %d spks',Nspks))
%title(sprintf('muwaveform %s Ch%d class%d. %d spks',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks))
%print(34,'-dpng', fullfile(resus_folder,sprintf('spikesavg_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(34,'-depsc','-vector', fullfile(resus_folder,sprintf('spikesavg_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
figure(43)
clf(43)
mu =mean(rr_wav);
sd = std(rr_wav);
if Nspks < Nspks_plot
    ord = 1:Nspks;
else
    ord = randperm(Nspks,Nspks_plot);
end
% yline(0,':')
% xline(20,':')
% xline(40,':')
plot(rr_wav(ord,:)','b');
hold on
plot(mu,'k')
plot(mu+sd,'k--')
plot(mu-sd,'k--')
xlim(limx)
ylim(limy)
xlabel('samples')
ylabel('Amplitude (uv)')
box on
title(sprintf('MATLAB pipeline %d spks',Nspks))
%title(sprintf('waveforms %s Ch%d class%d. %d spks. SNR=%2.2f. ISI=%1.1f%%',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks,spk_waveforms(rr_ind).SNR,spk_waveforms(rr_ind).perc_refrac3))
%print(33,'-dpng', fullfile(resus_folder,sprintf('spikes_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(33,'-depsc','-vector', fullfile(resus_folder,sprintf('spikes_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
figure(46)
clf(46)
lsp = size(rr_wav,2);
x = repmat((1:lsp)',Nspks,1);
y = reshape(rr_wav',[Nspks*lsp,1]);
[aux,aux_c] = hist3([x,y],[lsp lsp]);
aux=log10(aux+1);
colormap('hot')
pcolor(aux_c{1},aux_c{2},aux')
shading('Flat');%interp
xlabel('samples')
ylabel('Amplitude (uv)')
% yline(0,'w:')
% xline(20,'w:')
% xline(40,'w:')
% colorbar
title(sprintf('MATLAB pipeline %d spks',Nspks))
%title(sprintf('log density %s Ch%d class%d. %d spks',spk_waveforms(rr_ind).session,spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class,Nspks))
%print(36,'-dpng', fullfile(resus_folder,sprintf('logdensity_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))
%print(36,'-depsc','-vector', fullfile(resus_folder,sprintf('logdensity_ch%d_cl%d',spk_waveforms(rr_ind).channel,spk_waveforms(rr_ind).class)))

end

