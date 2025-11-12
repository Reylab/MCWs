clearvars
close all
% dir_base = 'D:\OneDrive - Baylor College of Medicine\BCM\Behavioral Tasks\EMU-001_task-RSVPscr_run-01';
dir_base = 'D:\OneDrive - Baylor College of Medicine\BCM\Behavioral Tasks\daqtest_mac';
channel = 257;

figure(1)
clf
ISI = 0.5;
pic_onoff = [[1 4 16];[2 8 32]];  % first pic with row 2
seq_length = 60;
load(fullfile(dir_base,'NSx.mat'),'NSx')
posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));

f1 = fopen(fullfile(dir_base,sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext)),'r','l');
ch257 = fread(f1,'int16=>double')*NSx(posch).conversion;

window_search = 0.75*ISI;
nevs=dir('daq*.nev');
% openNEV(fullfile(dir_base,sprintf('%s.nev',dir_base(find(dir_base == '\', 1, 'last')+1:end))))
openNEV(fullfile(dir_base,nevs.name),'report','noread','8bits')
events = double(NEV.Data.SerialDigitalIO.TimeStamp);
pics_on = ismember(NEV.Data.SerialDigitalIO.UnparsedData,pic_onoff);
plot((1:length(ch257))/30,ch257)
hold on
inds= find(pics_on);
    
t_DAQ = zeros(1,numel(inds));
t_photo = zeros(1,numel(inds));
for j=1:numel(inds)
    t_DAQ(j) = events(inds(j));
    line([t_DAQ(j)/30 t_DAQ(j)/30],ylim,'color','k')
    [~,indM] = max(ch257(t_DAQ(j):t_DAQ(j)+round(window_search*30000)));
%     line([t_DAQ+indM t_DAQ+indM],ylim,'color','k')
    photo_new=findchangepts(ch257(t_DAQ(j):t_DAQ(j)+indM),'statistic','std');
    t_photo(j) = t_DAQ(j)+photo_new;
    line([t_photo(j)/30 t_photo(j)/30],ylim,'color','c')
end
figure(2)
clf
subplot(2,1,1)
x=1:numel(inds); y = (t_photo-t_DAQ)/30;
plot(x,y)
hold on
X = [ones(length(x),1) x'];
b = X\y';
y_lin = X*b;
plot(x,y_lin','k--')

nlines = floor(numel(inds)/seq_length);
for k=1:nlines
    line([seq_length*k seq_length*k],ylim,'color','r')
end
xlim([1 numel(inds)])
ylabel('Difference between photodiode and DAQ (ms)')
xlabel('Picture number')

%%
load(fullfile(dir_base,'experiment_properties.mat'))

inds_signature = find(NEV.Data.SerialDigitalIO.UnparsedData == experiment.data_signature(1)); %it should be (1 3 5)
fprintf('inds_signature = %s\n',num2str(inds_signature))
t = diff(events)/30; %time interval in samples
fprintf('t(1:7) = %s\n',num2str(t(1:7)))
if isequal(inds_signature(:),[1 3 5]') ||isequal(inds_signature(:),[2 4 6]')
%     Event_Time=events(inds_signature(3)+2:end);
    t_last_signat = events(inds_signature(3)+1)/30000;
end    
% pulses_skipped = 2;
t_Mat2Brk_sec = experiment.times(2)-t_last_signat;
t_DAQ_Mat_ms = (experiment.t_DAQpic-t_Mat2Brk_sec)*1000;
t_pic_Mat_ms = (experiment.times(experiment.inds_pics)-t_Mat2Brk_sec)*1000; 
t_stimon_Mat_ms = (experiment.t_stimon-t_Mat2Brk_sec)*1000; 
t_fliptime_Mat_ms = (experiment.t_fliptime-t_Mat2Brk_sec)*1000; 

% t_BR_sec = NEV.Data.SerialDigitalIO.TimeStampSec(find(double(NEV.Data.SerialDigitalIO.UnparsedData)==11,1,'first')-1);
% t_MAT_sec = experiment.times(experiment.inds_start_seq(1)-1);
% 
% t_MAT_pic_ms = (experiment.times(experiment.inds_pics) - t_MAT_sec + t_BR_sec)*1000;

subplot(2,1,2)
x=1:numel(inds); y = t_photo/30-t_pic_Mat_ms;
plot(x,y)
hold on
X = [ones(length(x),1) x'];
b = X\y';
y_lin = X*b;
plot(x,y_lin','k--')

nlines = floor(numel(inds)/seq_length);
for k=1:nlines
%     line([seq_length*k seq_length*k],ylim,'color','r')
    xline(seq_length*k,'color','r')
end
xlim([1 numel(inds)])
ylabel('Difference between photodiode and Matlab flip (ms)')
xlabel('Picture number')

figure(3)
subplot(2,1,1)
plot(t_pic_Mat_ms-t_DAQ/30)
hold on
plot(t_pic_Mat_ms-t_DAQ_Mat_ms,'k')
plot(t_stimon_Mat_ms-t_DAQ_Mat_ms,'c--')
plot(t_fliptime_Mat_ms-t_DAQ_Mat_ms,'r-.')
nlines = floor(numel(inds)/seq_length);
for k=1:nlines
    line([seq_length*k seq_length*k],ylim,'color','r')
end
xlim([1 numel(inds)])
ylabel('Difference between Matlab flip times and DAQ (ms)')
xlabel('Picture number')
legend('DAQ Blackrock','DAQ_Matlab','interpreter','none')

subplot(2,1,2)
plot(t_DAQ_Mat_ms-t_DAQ/30)
nlines = floor(numel(inds)/seq_length);
for k=1:nlines
    line([seq_length*k seq_length*k],ylim,'color','r')
end
xlim([1 numel(inds)])
ylabel('Difference between DAQ in BRK and MAT (ms)')
xlabel('Picture number')