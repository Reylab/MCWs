function check_sync(filename,ftype,chsync,chsync1,chsync2,cell2use1,cell2use2)

% filename1='20160520-161205-001';
% filename2='20160520-161232-001';
elec_info128_ns3=[]; elec_info256_ns3=[]; 
eval(['elec_info128_' ftype '=[];'])
eval(['elec_info256_' ftype '=[];'])
   
samp_ini = 0*30000+1;
samp_end = 15*30000;

NSx1 = openNSx('read',['./' filename '_128.' ftype],sprintf('c:%d:%d',chsync1,chsync1));
ch1=[zeros(1,NSx1.MetaTags.Timestamp(cell2use1)) double(NSx1.Data{cell2use1}(1,samp_ini:samp_end))/4];
NSx2 = openNSx('read',['./' filename '_256.' ftype],sprintf('c:%d:%d',chsync2,chsync2));
ch2=[zeros(1,NSx2.MetaTags.Timestamp(cell2use2)) double(NSx2.Data{cell2use2}(1,samp_ini:samp_end))/4];

fini = min([numel(ch1) numel(ch2)]);
ch1=ch1(1:fini);
ch2=ch2(1:fini);

fig_num=2550; figure(fig_num); clf(fig_num);
set(fig_num, 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperPositionMode', 'auto','units','normalized','outerposition',[0 0 1 1],'Visible', 'off') 
subplot(2,1,1)
plot((1:length(ch1)),ch1,'b')
hold on
plot((1:length(ch2)),ch2,'r')
ylabel('Amplitude (uV)','fontsize',12)
legend({['NSP1 ch ' num2str(chsync)],['NSP2 ch ' num2str(chsync)]},'fontsize',12)    
title(sprintf('%s.   NSP1 channel %d, NSP2 channel %d. Cell %d. Timestamp1 %d, Timestamp2 %d. Offset = %d samples',pwd,chsync,chsync,cell2use1,NSx1.MetaTags.Timestamp(cell2use1),NSx2.MetaTags.Timestamp(cell2use2),(find(ch1>0.99*max(ch1),1)-find(ch2>0.99*max(ch1),1))),'fontsize',12,'interpreter','none')
set(gca,'fontsize',12)

subplot(2,1,2)
plot((1:length(ch1)),ch1-ch2,'k')    
set(gca,'fontsize',12)
xlabel('Time (sec)','fontsize',12) 
title('Difference between channels','fontsize',12)
saveas(fig_num,fullfile(pwd,['synctest_cell128-' num2str(cell2use1) '_cell256' num2str(cell2use2) '_ch' num2str(chsync)]),'png');

NSx = openNSx(['./' filename '_128.' ftype], 'noread');
eval(['elec_info128_' ftype '=NSx.ElectrodesInfo;'])
NSx = openNSx(['./' filename '_256.' ftype], 'noread');
eval(['elec_info256_' ftype '=NSx.ElectrodesInfo;'])
if exist(['./' filename '_128.ns3'],'file')
    NSx1ns3 = openNSx(['./' filename '_128.ns3'], 'noread');
    elec_info128_ns3=NSx1ns3.ElectrodesInfo;
else
    warning(['File ' filename '_128.ns3 does not exist'])
end
if exist(['./' filename '_256.ns3'],'file')
    NSx2ns3 = openNSx(['./' filename '_256.ns3'], 'noread');
    elec_info256_ns3=NSx2ns3.ElectrodesInfo;
else
    warning(['File ' filename '_256.ns3 does not exist'])
end
save('ElectrodesInfo',['elec_info128_' ftype ],['elec_info256_' ftype ],'elec_info128_ns3','elec_info256_ns3')