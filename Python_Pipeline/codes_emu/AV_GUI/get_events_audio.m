function get_events_audio(folder_base,tbeg,rec_length,folder_resus)
clc
close all
if ~exist('tbeg','var') || isempty(tbeg), tbeg=0; end % in seconds. to decide from which point I playback the sound and show the FR

if ~exist('folder_base','var') || isempty(folder_base) 
    folder_base = pwd;
end
if ~exist('folder_resus','var') || isempty(folder_resus) 
    folder_resus = pwd;
end

load(fullfile(folder_base,'NSX_TimeStamps'));

if ~exist('rec_length','var') || isempty(rec_length)   % rec_length in seconds. to decide from which point I playback the sound and show the FR
    tend=lts/sr; 
    rec_length = tend - tbeg;
else
    tend = rec_length + tbeg;
end

inds_sep = strfind(folder_base,filesep);
session = folder_base(inds_sep(end)+1:end);
events = [];
ev=1;

figu = 55;
figure(figu)
set(gcf,'Color',[1 1 1],'PaperUnits', 'inches','PaperType', 'A4','PaperPositionMode', 'auto')

if exist(fullfile(folder_resus,'finalevents_audio.mat'),'file')==2
    L=load(fullfile(folder_resus,'finalevents_audio.mat'));
    if ~strcmp(session,L.session)
        error('check the session as there seems to be a discrepancy between the one loaded and the one you are trying to process')
    end
    events = L.events;
    ev=size(events,1)+1;
    
    if ~isempty(events)
        plot(events(:,2)/1e6,events(:,1),'linestyle','none','linewidth',2,'marker','x','color','b','markersize',16)
    end
    hold on
end  

pauseKey = 28; % left key
resumeKey = 29; % right key
stopKey = 27; % Esc key
zeroKey = 48;

min_record = sr * tbeg;
if tbeg==0
    min_record=1;
end
max_record = floor(min(lts,min_record + sr * rec_length));
f1=fopen(fullfile(folder_base,'NSX129.NC5'),'r','l');
fseek(f1,(min_record-1)*2,'bof');
samples_2_play = max_record-min_record+1;
y=fread(f1,samples_2_play,'int16=>double');
fclose(f1);
% soundsc(y,sr,16);
ymin = min(y(:));
ymax = max(y(:));
ysc = (y-ymin)/(ymax-ymin)*2-1;
player = audioplayer(ysc,sr,16);

% samp_2_draw = (player.CurrentSample + min_record - 1);

play(player);
keyval = [];
% pause(1);
hlin = plot([(player.CurrentSample + min_record - 1)/sr (player.CurrentSample + min_record - 1)/sr],ylim,'linewidth',1,'color','k');
valid = 0;

while ~isequal(keyval,stopKey)
    keyval=double(get(figu,'CurrentCharacter'));
    figure(figu)
    set(hlin, 'Xdata', [(player.CurrentSample + min_record - 1)/sr (player.CurrentSample + min_record - 1)/sr])
    title(num2str((player.CurrentSample + min_record - 1)/sr,'%3.3f'),'fontsize',24)
    if isequal(keyval,pauseKey),  
        pause(player);
        keyval = [];
        while ~sum(ismember([stopKey resumeKey],keyval)), 
            keyval=double(get(figu,'CurrentCharacter'));
            figure(figu)
            if sum(ismember([zeroKey:zeroKey+9],keyval))
                events(ev,2) = (player.CurrentSample + min_record - 1)/sr*1e6; % in microsec from the beginning of the recording
                events(ev,1) = keyval - zeroKey + 1;
                hold on
                plot(events(ev,2)/1e6,events(ev,1),'linestyle','none','linewidth',2,'marker','o','color','r','markersize',16)
                set(hlin, 'Xdata', [(player.CurrentSample + min_record - 1)/sr (player.CurrentSample + min_record - 1)/sr],'Ydata',ylim)
                valid=1;                
            end
        end
        if isequal(keyval,resumeKey)            
            if valid==0
                events(ev,2) = NaN; 
                events(ev,1) = NaN;
            else
                valid=0;
            end
            ev=ev+1;
            resume(player);
        end
    end
    if isequal(keyval,stopKey)  
        stop(player); 
        break; 
    end
end
disp('finito')
events(isnan(events(:,1)),:)=[];
[~,Ieve]=sort(events,1);
events=events(Ieve(:,2),:);
% close all
save(fullfile(folder_resus,'finalevents_audio.mat'),'events','session')
