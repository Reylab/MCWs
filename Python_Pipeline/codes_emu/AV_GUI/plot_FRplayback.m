function plot_FRplayback(channel,class,folder_base,sigma_gauss,tbeg,rec_length)

close all
if ~exist('sigma_gauss','var') || isempty(sigma_gauss), sigma_gauss=49.42; end  % sigma_gauss=49.42;  sigma_gauss=82.4  sigma_gauss=115.32
if ~exist('tbeg','var') || isempty(tbeg), tbeg=0; end % in seconds. to decide from which point I playback the sound and show the FR

load(fullfile(folder_base,'NSX_TimeStamps'));
if ~exist('rec_length','var') || isempty(rec_length)   % rec_length in seconds. to decide from which point I playback the sound and show the FR
    tend=lts/sr; 
    rec_length = tend - tbeg;
else
    tend = rec_length + tbeg;
end

tbeg_ms = tbeg*1000;
tend_ms = tend*1000;
FRlength = 10; % secs

figu = 55;
figure(figu)
set(gcf,'Color',[1 1 1],'PaperUnits', 'inches','PaperType', 'A4','PaperPositionMode', 'auto')
% Maximize_pure(figu)

alpha_gauss = 3.035;
half_ancho_gauss = alpha_gauss * sigma_gauss;
sample_period = 1000/sr; % sample period for the spike list - window convolution in ms/sample
N_gauss = 2*round(half_ancho_gauss/sample_period)+1; % Number of points of the gaussian window
int_window = gausswin(N_gauss, alpha_gauss);
int_window = 1000*int_window/sum(int_window)/sample_period;

load(fullfile(folder_base,['times_NSX' num2str(channel)]));
sorted_times = cluster_class(cluster_class(:,1)==class,2 ); 

tic
spike_timeline1 = zeros(ceil((tend-tbeg)*sr),1);
spike_timeline1(ceil(sorted_times*sr/1e3)) = 1;
toc
tic
spike_timeline = hist(sorted_times(sorted_times>=tbeg_ms & sorted_times<=tend_ms),(tbeg_ms:sample_period:tend_ms));
toc
sum(~spike_timeline1(:)*spike_timeline(:))
n_spike_timeline = length(spike_timeline);
ejex_temp = (tbeg_ms:sample_period:tend_ms*1.1); % in ms
ejex_temp = ejex_temp(1:n_spike_timeline);
integ_timeline_stim = conv(spike_timeline, int_window);
integ_timeline_stim_cut = integ_timeline_stim(round(half_ancho_gauss/sample_period)+1:n_spike_timeline+round(half_ancho_gauss/sample_period));
FRmax = max(integ_timeline_stim_cut);
ejex_temp_sec =  ejex_temp/1000;          

% KbName('UnifyKeyNames');
% pauseKey= KbName('space');  %to pause
% stopKey= KbName('s');  %to stop

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
% play(player);   % start the playback
% pause(player);  % pause the playback
% resume(player)  % resume the playback
% stop(player)    % stop the playback

%         % cant_plots = ceil(rec_length/FRlength);
%         current_frame = 0;
%         next_time = 0;
%         t_elapsed = 0;
%         tcum = 0;
%         t_step = 0.1; % in sec


current_frame = 0;
% keyCode=zeros(1,256);
hlin = 0;

% % myStruct.playHeadLoc = playHeadLoc;
% % myStruct.frameT = frameT;
% % myStruct.ax = ax;
% % 
% % set(player, 'UserData', myStruct);
% % set(player, 'TimerFcn', @apCallback);
% % set(player, 'TimerPeriod', frameT);

% player.TimerPeriod = 0.1;
player.TimerPeriod = 1;
% player.TimerFcn = {@calltest, player};
% player.TimerFcn = {@update_line, player, figu, FRmax,ejex_temp_sec};
play(player);
tic

while player.CurrentSample<samples_2_play-sr/10
%     player.CurrentSample
    if player.CurrentSample>current_frame*FRlength*sr
        clf(figu)
        current_frame = current_frame + 1;
        ind_beg = (current_frame-1)*FRlength*sr+1;
        ind_end = current_frame*FRlength*sr;
        plot(ejex_temp_sec(ind_beg:ind_end),integ_timeline_stim_cut(ind_beg:ind_end),'color','b','linewidth',2);
    %     axis tight
        xlim([ejex_temp_sec(ind_beg) ejex_temp_sec(ind_end)]);
        ylim([0 1.05*FRmax]);
        hold on
        samp_2_draw = player.CurrentSample;
        hlin = line([ejex_temp_sec(samp_2_draw) ejex_temp_sec(samp_2_draw)],[0 FRmax],'linewidth',1,'color','r');
        xlabel('Time [s]')    
        drawnow
        toc
%     while player.CurrentSample<current_frame*FRlength*sr
%         [~, ~, keyCode] = KbCheck;
% %         if keyCode(pauseKey),  
% %             pause(player);        
% %             WaitSecs(2)
% %             FlushEvents('keydown');
% %             while ~sum(keyCode([stopKey pauseKey])), [~, ~, keyCode] = KbCheck; end
% %             if keyCode(pauseKey)
% %                 resume(player);
% %             end
% %         end
% %         if keyCode(stopKey),  stop(player); break; end   
    else
        newcurrent_time = ejex_temp_sec(player.CurrentSample);
        set(hlin, 'Xdata', [newcurrent_time newcurrent_time])
        drawnow
    end
end


stop(player)





% % figure;
% % plot(tAxis(1:end/2), x(1:end/2));
% % ylim([-mag mag])
% % xlim([0 durT/2])
% % xlabel('Time [s]')
% % 
% % KbName('UnifyKeyNames');
% % pauseKey= KbName('space');  %to pause
% % stopKey= KbName('s');  %to stop
% % 
% % playHeadLoc = 0;
% % hold on; ax = plot([playHeadLoc playHeadLoc], [-mag mag], 'r', 'LineWidth', 2);
% % 
% % player = audioplayer(x, fs);
% % myStruct.playHeadLoc = playHeadLoc;
% % myStruct.frameT = frameT;
% % myStruct.ax = ax;
% % myStruct.durT = durT;
% % myStruct.next_time = durT/2;
% % myStruct.tAxis = tAxis;
% % myStruct.x = x;
% % myStruct.player = player;
% % myStruct.pauseKey = pauseKey;
% % myStruct.stopKey = stopKey;
% % 
% % set(player, 'UserData', myStruct);
% % set(player, 'TimerFcn', @apCallback);
% % set(player, 'TimerPeriod', frameT);
% % play(player);
