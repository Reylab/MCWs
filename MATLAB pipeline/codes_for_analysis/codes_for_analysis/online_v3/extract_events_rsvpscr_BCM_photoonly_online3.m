function extract_events_rsvpscr_BCM_photoonly_online3(which_system_micro,nowait)
%% set paradigm pulses characteristics and create stimulus structure
% matlab_times=0;
if strcmp(which_system_micro,'BRK')
    channel_photo = 257;
elseif strcmp(which_system_micro,'RIP')
    channel_photo = 10241;
end
if ~exist('nowait','var')
    nowait=false;
end

thr = 2000;
square_duration = 50*30; % time in msec (3 ifi) but result in samples
% photo_corrected = 1;
load('experiment_properties_online3.mat','experiment','scr_config_cell','scr_end_cell')
ISI = unique(cellfun(@(x) x.ISI,scr_config_cell));
window_search = round(0.75*ISI*30000);
Nscr = numel(scr_config_cell);
n_scr_ended = min(Nscr, numel(scr_end_cell));

if Nscr>n_scr_ended
    warning('some subscreening not ended, using just the completed screenings');
    Nscr = n_scr_ended;
end

seq_length = zeros(Nscr,1);
NISI = zeros(Nscr,1);
Nseq= zeros(Nscr,1);
stimulus = cell(Nscr,1);
npics = zeros(Nscr,1);
for i = 1:Nscr
    seq_length(i) = scr_config_cell{i}.seq_length;
    NISI(i) = numel(scr_config_cell{i}.ISI);
    Nseq(i) = numel(scr_end_cell{i}.inds_start_seq) - scr_end_cell{i}.abort;
    stimulus{i} = create_stimulus_online(scr_config_cell{i}.order_pic(:,:,1:Nseq(i)),NISI(i),...
        scr_config_cell{i}.pics2use, experiment.ImageNames.name(scr_config_cell{i}.pics2use), ...
        scr_config_cell{i}.ISI,scr_config_cell{i}.order_ISI);
    npics(i) =  numel(scr_config_cell{i}.order_pic(:,:,1:Nseq(i)));
end
save('stimulus.mat', 'stimulus')
should_be_pics = sum(npics);

%% 
load('NSx.mat','NSx')
posch = find(arrayfun(@(x) (x.electrode_ID==channel_photo),NSx));
if isempty(posch)
    error('Photodiode channel not parsed.')
end
f1 = fopen(sprintf('%s%s',NSx(posch).output_name,NSx(posch).ext),'r','l');
ch_photo = fread(f1,'int16=>double')*NSx(posch).conversion;
        
smooth_ch_photo = smooth(ch_photo,150);
ff=figure();
ff.GraphicsSmoothing = 'off';
ff.Visible = 'off';
    
plot((1:length(ch_photo))/30,ch_photo), xlabel('Time (msec)')
hold on
% plot((1:length(ch_photo))/30,smooth_ch_photo,'r')
start_points=find(diff(smooth_ch_photo>thr)==1)-square_duration;
if should_be_pics~=numel(start_points)
    %try to fix removing extra blocks of pulses at the end
    start_block = [0; find(diff(start_points)>(60)*NSx(posch).sr)];
    if numel(start_block) > Nscr
        fist2rm = start_block(end-(numel(start_block)-Nscr)+1);
        start_points = start_points(1:(fist2rm));
    end
    if should_be_pics~=numel(start_points)
        warning('separate in segments and call code to fix photodiode times... for now:')
%         error('check manually')
    end
end
t_photo = zeros(size(start_points));
for j=1:numel(start_points)
    [~,indM] = max(ch_photo(start_points(j):start_points(j)+window_search));
    photo_new=findchangepts(ch_photo(start_points(j):start_points(j)+indM),'statistic','std');
    t_photo(j) = start_points(j)+photo_new; % in samples
    line([t_photo(j)/30 t_photo(j)/30],ylim,'color','c')
end
fprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,numel(t_photo))
title(sprintf('There should be %d stimulus onsets. There are %d.\n',should_be_pics,numel(t_photo)))
print -dpng photo_cont.png

%%

pics_onset =  cell(Nscr,1);
% pics_onset_test =  cell(Nscr,1);
% onset_counter = 1;
% seq_start = 1;
tphoto_sec = t_photo/30000;
t_photo1 = tphoto_sec(1);
t_matlab1 = scr_end_cell{1}.t_stimon(1);
% for i = 1:numel(scr_config_cell)
for i = 1:Nscr
%     ss_onsets = seq_length(i) * NISI(i) *Nseq(i);
%     seqs_end = find(diff(tphoto_sec)>ISI*1.1);
%     ttt = diff(seqs_end);
%     if all(ttt(seq_start:seq_start+Nseq(i)-1) ==seq_length(i))
%         pics_onset{i} = reshape(tphoto_sec(onset_counter:onset_counter+ss_onsets-1)*1000,seq_length(i),Nseq(i));
%         onset_counter = onset_counter + ss_onsets;
%     else
        % USE DAQ TIMES (OR MATLAB TIMES) TO DEFINE START AND END OF EACH
        % SEQUENCE AND IDENTIFY WHICH ARE WRONG AND NEED TO BE FIXED
        
        t_photo_M_start =scr_end_cell{i}.times(scr_end_cell{i}.inds_start_seq)-t_matlab1+t_photo1;        
        t_photo_M_end = [scr_end_cell{i}.times(scr_end_cell{i}.inds_start_seq(2:end)) scr_end_cell{i}.times(end)]-t_matlab1+t_photo1;        

        for iseq=1:Nseq(i)
            seq_times = find((tphoto_sec>t_photo_M_start(iseq)) & (tphoto_sec<t_photo_M_end(iseq)));            
            [complete_times, text_out] = fix_photodiode_times(tphoto_sec(seq_times)*1000,iseq,scr_config_cell{i});
%             pics_onset{i}(1:seq_length(i),iseq) =complete_times;
            if ~isempty(complete_times)
                pics_onset{i}(1:seq_length(i),1,iseq) =complete_times;                               
%                 pics_onset{i}(1:seq_length(i),iseq) =complete_times;
%                 pics_onset_test{i}(1:seq_length(i),iseq) =complete_times;
                if ~strcmp(text_out,'no changes needed')
                    warning(text_out)
                end 
            else
                % NEED TO USE THE DAQ TIMES FOR THAT SCREENING
                error('run again with digital events')
            end
        end     
%     end
end
% if (onset_counter-1) < size(t_photo,1)
%     warning('%n additional pulses at the end.', size(t_photo,1)-onset_counter+1)
% end

% produce figure to check pulses
figure
for i = 1:n_scr_ended
    subplot(5,4,i)
    set(gcf,'Units','normalized', 'OuterPosition',[0 0 1 1]);
    tdiff = diff(squeeze(pics_onset{i}));
    plot(tdiff(:)-500,'b'); %should be 0 ms
    xlim([0 seq_length(i)*Nseq(i)+1])
    h_legend = legend(sprintf('stimulus "error" (ms). %2.0f%% lower than 2ms',100*sum(abs(tdiff(:)-500)<2)/numel(tdiff)),'location','best');legend('boxoff')
end
set(gcf,'PaperPositionMode','auto')

if ~nowait
    keyval = input('Is the figure reasonable? If so, press ''y'' followed by ENTER to create the finalevents and save the figure  ','s');

    if strcmp(keyval,'y')
        % save figure and events with Blackrock times
        print -dpng ttl_rsvp.png
        save finalevents pics_onset
    end
else
    print -dpng ttl_rsvp.png
    save finalevents pics_onset
end