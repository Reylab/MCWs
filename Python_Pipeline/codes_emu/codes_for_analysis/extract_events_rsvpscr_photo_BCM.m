% function extract_events_rsvpscr_photo_BCM

channel_photo = 257;
photo_thr = 2000;
samples_pre = 0.1*30000;
load 'experiment_properties.mat'
[seq_length,NISI,~] = size(experiment.order_pic);
Nseq = length(experiment.inds_pics)/seq_length;
create_stimulus_struct_rsvpscr(experiment,Nseq);
window_search = round(0.75*experiment.ISI*30000);
load('NSx.mat','NSx')    

posch = find(arrayfun(@(x) (x.chan_ID==channel_photo),NSx));
f1 = fopen(sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext),'r','l');
chphoto = fread(f1,'int16=>double')*NSx(posch).conversion;
% % plot((1:length(chphoto))/30,chphoto)
% % hold on
% % plot((1:length(chphoto))/30,smooth(chphoto,25),'r')

TS_from = find(diff(smooth(chphoto,25)>photo_thr)==1)+1-samples_pre;
% TS_from = find(diff(chphoto>photo_thr)==1)+1;
% % line(xlim,[photo_thr photo_thr],'color','m')

num_pics_onset=numel(TS_from);
should_be_pics = seq_length*NISI*Nseq;
fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,num_pics_onset)
photo_pic_timestamp = zeros(size(TS_from));

for j=1:numel(TS_from)
% %     line([TS_from(j)/30 TS_from(j)/30],ylim,'color','k')
    [~,indM] = max(chphoto(TS_from(j):TS_from(j)+window_search));
    photo_new=findchangepts(chphoto(TS_from(j):TS_from(j)+indM),'statistic','std');
    photo_pic_timestamp(j) = TS_from(j)+photo_new;
% %     line([photo_pic_timestamp(j)/30 photo_pic_timestamp(j)/30],ylim,'color','c')
end      

pics_onset_ph = reshape(photo_pic_timestamp'/30000*1000,seq_length,NISI,Nseq);

A=diff(squeeze(pics_onset_ph));
figure
set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
plot(A(:)-500,'.')
hold on
for ii=1:Nseq
    line([(ii-1)*(seq_length-1)+1 (ii-1)*(seq_length-1)+1],ylim,'color','c','linestyle','--')
end
xlim([0 seq_length*Nseq+1])
h_legend=legend('stimulus "error" with photodiode (ms)','location','best');legend('boxoff')
    
set(gcf,'PaperPositionMode','auto')
    
keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');

if strcmp(keyval,'y')
    %% save figure and events with Blackrock times
    print -dpng ttl_rsvp.png
    save finalevents pics_onset
end

close all