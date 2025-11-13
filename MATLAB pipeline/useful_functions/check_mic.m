function check_mic(t_start,t_play,dirmain)

if ~exist('t_start','var')|| isempty(t_start),  t_start = 0.5; end %start at tmin secs
if ~exist('t_play','var')|| isempty(t_play), t_play = 30; end %seconds to reproduce
if ~exist('dirmain','var')|| isempty(dirmain), dirmain = pwd; end %data folder


%% check sound recorded in analogue input
load(fullfile(dirmain,'NSx.mat'),'NSx');
% poschmic = find(arrayfun(@(x) (contains(x.label,'Mic2')),NSx));
poschmic = find(arrayfun(@(x) (contains(x.label,'MicL')),NSx));
% poschmic = find(arrayfun(@(x) (contains(x.label,'Aud')),NSx));
 
if NSx(poschmic).lts<NSx(poschmic).sr * t_start
    disp('t_start is smaller than the recording length')
else
    min_record = NSx(poschmic).sr * t_start;
end
max_record = floor(min(NSx(poschmic).lts,min_record + NSx(poschmic).sr * t_play));
f1 = fopen(fullfile(dirmain,sprintf('%s%s',NSx(poschmic).output_name,NSx(poschmic).ext)),'r','l');
fseek(f1,(min_record-1)*2,'bof');
y=fread(f1,(max_record-min_record+1),'int16=>double')*NSx(poschmic).conversion + NSx(poschmic).dc;
fclose(f1);
soundsc(y,NSx(poschmic).sr,16); 


%%
% % close all
% which_wav = 'ECOGS001R09_in.wav';
% [y_wav,fs_wav] = audioread(fullfile(dirmain,'\wav_files\1 - AuditoryTask',which_wav));
% y_resamp = resample(y_wav,NSx(poschmic).sr,fs_wav);
% % soundsc(y_resamp(1:NSx(poschmic).sr*10),NSx(poschmic).sr,16);
% 
% figure
% secs = 20;
% offset = 32400000-45000;
% sr = 30000;
% c(1) = subplot(2,1,1);
% plot([1:secs*sr]/sr,y(offset:offset+secs*sr-1),'color','b')
% grid minor
% ylabel('RIP mic')
% 
% c(2) = subplot(2,1,2);
% plot([1:secs*fs_wav]/fs_wav,y_wav(1:secs*fs_wav),'color','r')
% hold on
% plot([1:secs*sr]/sr,y_resamp(1:secs*30000),'g--')
% grid minor
% linkaxes(c,'x');
% ylabel(which_wav)
% xlabel ('Time (sec)')
% sgtitle(sprintf('RIP offset = %d samples (@ %dHz)',offset,sr))
% 
% % audiowrite('description_first20_BCI2000.wav',y_wav(1:44100*20),44100);
% % audiowrite('description_first30_RIP.wav',y(offset:offset+30*sr-1)/max(abs(y(offset:offset+30*sr-1))),sr);
% %%
% 
% [C21,lag21] = xcorr(mean(y_resamp,2),y);
% C21 = C21/max(C21);
% figure
% plot(lag21,C21)
% 
% [~,I21] = max(C21);
% t21 = lag21(I21);
% 
% figure
% c(1) = subplot(2,1,1);
% plot(y,'color','b')
% xline(t21,'color','k')
% xline(32400000,'color','g')
% ylabel('RIP mic')
% grid minor
% 
% c(2) = subplot(2,1,2);
% plot(y_resamp,'color','r')
% ylabel(which_wav)
% grid minor
% linkaxes(c,'x');
% 
% if t21 < 0
%     y = y(-t21:end);
% else 
%     y = y(t21:end);
% end 
% 
% figure
% c(1) = subplot(2,1,1);
% plot(y,'color','b')
% ylabel('RIP mic')
% grid minor
% 
% c(2) = subplot(2,1,2);
% plot(y_resamp,'color','r')
% ylabel(which_wav)
% grid minor
% linkaxes(c,'x');

