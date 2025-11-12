clearvars
close all
% dir_base = 'D:\OneDrive - Baylor College of Medicine\BCM\Behavioral Tasks\EMU-001_task-RSVPscr_run-01';
dir_base = pwd;

% channel = 269;
% channel_bit = 270;

channel = 268;
channel_bit = 269;

figure(1)
clf
ISI = 0.5;
% pic_onoff = [[1 4 16];[2 8 32]];  % first pic with row 2
seq_length = 60;
load(fullfile(dir_base,'NSx.mat'),'NSx')
posch = find(arrayfun(@(x) (x.chan_ID==channel),NSx));

f1 = fopen(fullfile(dir_base,sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext)),'r','l');
ch_photo = fread(f1,'int16=>double')*NSx(posch).conversion;

posch = find(arrayfun(@(x) (x.chan_ID==channel_bit),NSx));

f1 = fopen(fullfile(dir_base,sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext)),'r','l');
ch_bit = fread(f1,'int16=>double')*NSx(posch).conversion;

window_search = 0.75*ISI;
nevs=dir('daaqq.nev');
% openNEV(fullfile(dir_base,sprintf('%s.nev',dir_base(find(dir_base == '\', 1, 'last')+1:end))))
openNEV(fullfile(dir_base,nevs.name),'report','noread','8bits')
events = double(NEV.Data.SerialDigitalIO.TimeStamp);
t_DAQ = events(3:2:end);

% pics_on = ismember(NEV.Data.SerialDigitalIO.UnparsedData,pic_onoff);
plot((1:length(ch_photo))/30,ch_photo)
hold on
% inds= find(pics_on);
    
% t_DAQ = zeros(1,numel(inds));
t_photo = zeros(1,numel(t_DAQ));
for j=1:numel(t_DAQ)
%     t_DAQ(j) = events(inds(j));
    line([t_DAQ(j)/30 t_DAQ(j)/30],ylim,'color','k')
    [~,indM] = max(ch_photo(t_DAQ(j):t_DAQ(j)+round(window_search*30000)));
    photo_new=findchangepts(ch_photo(t_DAQ(j):t_DAQ(j)+indM),'statistic','std');
    t_photo(j) = t_DAQ(j)+photo_new;
    line([t_photo(j)/30 t_photo(j)/30],ylim,'color','c')
end
plot((1:length(ch_bit))/30,ch_bit,'r')



figure(2)
clf
subplot(2,1,1)
x=1:numel(t_DAQ); y = (t_photo-t_DAQ)/30;
plot(x,y)
hold on
X = [ones(length(x),1) x'];
b = X\y';
y_lin = X*b;
plot(x,y_lin','k--')

nlines = floor(numel(t_DAQ)/seq_length);
for k=1:nlines
    line([seq_length*k seq_length*k],ylim,'color','m')
end
xlim([1 numel(t_DAQ)])
ylabel('Difference between photodiode and DAQ (ms)')
xlabel('Picture number')

%%
DAQ_WS=load(fullfile(dir_base,'daqqq_WS.mat'));

t_last_signat = events(1)/30000;
    
t_Mat2Brk_sec = DAQ_WS.times(2)-t_last_signat;
t_DAQ_Mat_ms = (DAQ_WS.t_DAQpic(2:2:end)-t_Mat2Brk_sec)*1000;
% t_pic_Mat_ms = (DAQ_WS.times(2:end-1)-t_Mat2Brk_sec)*1000; 
t_pic_Mat_ms = (DAQ_WS.times(3:end)-t_Mat2Brk_sec)*1000; 
t_stimon_Mat_ms = (DAQ_WS.t_stimon-t_Mat2Brk_sec)*1000; 
t_fliptime_Mat_ms = (DAQ_WS.t_fliptime-t_Mat2Brk_sec)*1000; 

% t_BR_sec = NEV.Data.SerialDigitalIO.TimeStampSec(find(double(NEV.Data.SerialDigitalIO.UnparsedData)==11,1,'first')-1);
% t_MAT_sec = experiment.times(experiment.inds_start_seq(1)-1);
% 
% t_MAT_pic_ms = (experiment.times(experiment.inds_pics) - t_MAT_sec + t_BR_sec)*1000;

subplot(2,1,2)
x=1:numel(t_photo); y = t_photo/30-t_pic_Mat_ms;
plot(x,y)
hold on
X = [ones(length(x),1) x'];
b = X\y';
y_lin = X*b;
plot(x,y_lin','k--')

nlines = floor(numel(t_photo)/seq_length);
for k=1:nlines
%     line([seq_length*k seq_length*k],ylim,'color','r')
    xline(seq_length*k,'color','r')
end
xlim([1 numel(t_photo)])
ylabel('Difference between photodiode and Matlab flip (ms)')
xlabel('Picture number')

figure(3)
subplot(2,1,1)
plot(t_pic_Mat_ms-t_DAQ/30)
hold on
% plot(t_pic_Mat_ms-t_DAQ_Mat_ms,'k')
% plot(t_stimon_Mat_ms-t_DAQ_Mat_ms,'c--')
% plot(t_fliptime_Mat_ms-t_DAQ_Mat_ms,'r-.')
plot(t_pic_Mat_ms-t_stimon_Mat_ms,'k')
plot(t_pic_Mat_ms-t_fliptime_Mat_ms,'c--')
nlines = floor(numel(t_photo)/seq_length);
for k=1:nlines
    line([seq_length*k seq_length*k],ylim,'color','r')
end
xlim([1 numel(t_photo)])
ylabel('Difference between Matlab flip times and DAQ (ms)')
xlabel('Picture number')
legend('DAQ Blackrock','DAQ_Matlab','interpreter','none')

subplot(2,1,2)
plot(t_DAQ_Mat_ms-t_DAQ/30)
nlines = floor(numel(t_photo)/seq_length);
for k=1:nlines
    line([seq_length*k seq_length*k],ylim,'color','r')
end
xlim([1 numel(t_photo)])
ylabel('Difference between DAQ in BRK and MAT (ms)')
xlabel('Picture number')