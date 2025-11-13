% COPY SESSION FILES FROM THE BEHAVIORAL COMPUTER IF NEEDED, INCLUDING THE PICS
% COPY FILES FROM BLACKROCK COMPUTER TO THE BEHAVIORAL FOLDER
% DEFINE dirmain AS THE EXPERIMENT FOLDER AND THEN RUN THIS SCRIPT or RUN FIRST CELL, AND THEN THE WHOLE SCRIPT FROM THE EXPERIMENT FOLDER

clearvars
clc
% dirmain = 'D:\march7AM\EMU-027_task-RSVPscr_run-03'; 
% dirmain = 'F:\OneDrive - Baylor College of Medicine\BCM\Birds\Birds Data for Hernan\Bad Session_ YDJ Oct 8'; 
% filenames={'sHiCikEPkEMkGm_20161018-160552-001'};
% channels = [65:80 97:104 113:120];

% dualNSP =0;
NSPs={'_NSP-1';'_NSP-2'};

do_power_plot = 1;
notchfilter=1;

do_sorting = 1;

do_loop_plot = 1;
extract_events = 1;
do_extra_stims = 1;

%%
[~,name] = system('hostname');
name=[ name repmat(' ',1,30)];
if strcmp(name(1:9),'REY-LT-01'), dir_base = 'E:\Google Drive FIUBA\Fer Chaure';  % HERNAN'S DELL LAPTOP
elseif strcmp(name(1:15),'DESKTOP-DQR054O'), dir_base = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR';  % EMU BEH PC  
elseif strcmp(name(1:14),'NPB-CSNzbookG5'), dir_base = 'D:\Google Drive FIUBA\Fer Chaure';  % HERNAN'S HP laptop ZBOOK15   
end

addpath(genpath(fullfile(dir_base,'NPMK')));
addpath(genpath(fullfile(dir_base,'BCM-EMU','wave_clus_BCM')));
addpath(genpath(fullfile(dir_base,'BCM-EMU','codes_for_analysis_BCM')));
addpath(genpath(fullfile(dir_base,'BCM-EMU','mex_HGR')));
addpath(genpath(fullfile(dir_base,'BCM-EMU','useful_functions')));
addpath(genpath(fullfile(dir_base,'pics_USA')));

%%
exp_type = 'RSVPSCR';  phase=[];

if ~exist('dirmain','var')
    dirmain = pwd;
end

addpath(genpath([dirmain '_pic']))
set(groot,'defaultaxesfontsmoothing','off')
set(groot,'defaultfiguregraphicssmoothing','off')
set(groot,'defaultaxestitlefontsizemultiplier',1.1)
set(groot,'defaultaxestitlefontweight','normal')
%%
cd(dirmain)
ftype = 'ns5';
%     ftype = 'ns3';

if ~exist('filenames','var')
    A=dir(['*.' ftype]);
    if isempty(A)
        error('There are no %s files in this folder',ftype);
    else
        filenames = {A.name};
    end
else
    fprintf('variable filenames already exists and is equal to %s\n',filenames{:})
end
%%
% max_memo_GB = 16;
max_memo_GB = [];

if all(cellfun(@(x) ~contains(x,NSPs{1}) && ~contains(x,NSPs{2}),filenames))
    parse_NSx(filenames,[],max_memo_GB);
    filename_NEV = [filenames{1}(1:end-3) 'nev'];
else
    parse_NSx(filenames(cellfun(@(x) contains(x,NSPs{1}),filenames)),1,max_memo_GB);
    parse_NSx(filenames(cellfun(@(x) contains(x,NSPs{2}),filenames)),2,max_memo_GB);
    ind_file=cellfun(@(x) contains(x,NSPs{1}),filenames);
    filename_NEV = [filenames{ind_file(1)}(1:end-3) 'nev'];
    % INCLUDE SYNC_CHECK BETWEEN NSPS   
end
if ~exist('channels','var')
    load('NSx','NSx');
    channels = cell2mat({NSx(arrayfun(@(x) (strcmp(x.unit,'uV') && x.sr==30000),NSx)).chan_ID});
end

disp('channel parsing DONE')
%% check sound recorded in analogue input
load('NSx','NSx');
poschmic = find(arrayfun(@(x) (contains(x.label,'Mic2')),NSx));
% poschmic = find(arrayfun(@(x) (contains(x.label,'MicL')),NSx));
% poschmic = find(arrayfun(@(x) (contains(x.label,'Aud')),NSx));

t_start = 0.5; %start at tmin secs
t_play = 200; %seconds to reproduce
if NSx(poschmic).lts<NSx(poschmic).sr * t_start
    disp('t_start is smaller than the recording length')
else
    min_record = NSx(poschmic).sr * t_start;
end
max_record = floor(min(NSx(poschmic).lts,min_record + NSx(poschmic).sr * t_play));
f1 = fopen(fullfile(dirmain,sprintf('%s%s',NSx(poschmic).output_name,NSx(poschmic).ext)),'r','l');
fseek(f1,(min_record-1)*2,'bof');
y=fread(f1,(max_record-min_record+1),'int16=>double')*NSx(poschmic).conversion;
fclose(f1);
soundsc(y,NSx(poschmic).sr,16); 

%% opens parallel pool
poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool
end

%% the events from the NEV file should be processed here if there was a break during the experiment

%% power spectrum and potential notches
if do_power_plot
    new_check_lfp_power_NSX(channels,'parallel',true)
    disp('power spectra DONE')
end
%% plot continuous data (spike filtered) for each macro (we dont really gain time by running it in parallel, at least when plotting just 2 mins per channel)

% sequential
filt_order=4;
tic
Plot_continuous_channels_BCM(channels,notchfilter,filt_order,'neg'); % CHECK MACRO NAMES
toc
disp('plot continuous DONE')
%% read pulses STEP 1: open NEV file (to read pulses sent with the DAQ)
if extract_events
    %ONLY FOR SINGLE FILE AT THE MOMENT
    openNEV(['./' filename_NEV],'report','noread','8bits')
    
    if strcmp(exp_type,'RSVPSCR')
        extract_events_rsvpscr_BCM(NEV)
%         extract_events_rsvpscr_BCM_photo(NEV)
    elseif strcmp(exp_type,'MOVIE_SCR')
        %     edit extract_events_movies_express_nevfiles.m % (manual supervision required)
        edit extract_events_movies_split_express_nevfiles.m % (manual supervision required)
    elseif strcmp(exp_type,'MOVIE')
        edit extract_events_movies_split_nevfiles.m % (manual supervision required)
        % CREATE IMAGENAMES.TXT
        % RUN AV_GUI TO CREATE FINALEVENT_AUDIO or FINALEVENT_VIDEO
        create_stimulus_struct_AV(phase)
    elseif strcmp(exp_type,'STORY')
        % CREATE IMAGENAMES.TXT
        % RUN AV_GUI TO CREATE FINALEVENT_AUDIO
        create_stimulus_struct_AV(phase)
    end
end
%% spike detection

ch_temp = input(sprintf('Currently, channels = %s.\nIf you want to keep it like that, press enter.\nOtherwise, enter the new vector and press enter  ',num2str(channels)));
if ~isempty(ch_temp)
    channels = ch_temp;
end
    
parallel = true;
% CHECK VALUES IN COMPARISON WITH SET_PARAMETERS.TXT
param.detect_order = 4;
param.sort_order = 2;
param.detect_fmin = 300;
param.sort_fmin = 300;
param.stdmin = 5;
param.stdmax = 50;                     % maximum threshold for detection
param.ref_ms = 1.5;
param.preprocessing = true;

neg_thr_channels = input('Enter the vector with neg_thr_channels and press enter. Press enter to use all channels ');
if isempty(ch_temp)
    neg_thr_channels = channels;
    pos_thr_channels = [];
else
    pos_thr_channels = input('Enter the vector with pos_thr_channels and press enter. Press enter for empty array ');
end

tic
param.detection = 'neg';
% neg_thr_channels=[2097:2104];
if ~isempty(neg_thr_channels)
    Get_spikes(neg_thr_channels,'parallel',parallel,'par',param);
end

% pos_thr_channels=[];
param.detection = 'pos';
if ~isempty(pos_thr_channels)
    Get_spikes(pos_thr_channels,'parallel',parallel,'par',param);
end

param.detection = 'both';
% both_thr_channels=[65 67:74 76:80 82:86 88:96 113:122 126 128];
both_thr_channels=setdiff(setdiff(channels,neg_thr_channels),pos_thr_channels);
if ~isempty(both_thr_channels)
    Get_spikes(both_thr_channels,'parallel',parallel,'par',param);
end
toc
disp('spike detection DONE')

%% build data structure (grapes) and compute best responses for multiunits
if do_loop_plot
    % channels = [2097:2104];    
    muonly = 'y';

    if strcmp(exp_type,'STORY') || strcmp(exp_type,'MOVIE')
        skip = 1; ons_ind =1; effect_rows_2=30;
        do_structure_mu_NSXfiles(channels,skip,ons_ind,exp_type,phase)
        loop_plot_best_responses_s_NSXfiles_profile(channels,muonly,1,10,effect_rows_2,phase)
        loop_plot_best_responses_s_NSXfiles_profile(channels,muonly,11,20,effect_rows_2,phase)
    elseif strcmp(exp_type,'RSVPSCR')
        skip = 0; ons_ind =0;effect_rows_2=0;
        do_structure_mu_BCM(channels,skip,ons_ind,exp_type,phase)
        rankfirst=1; ranklast=15;    
        loop_plot_best_responses_BCM_rank(channels,muonly,rankfirst,ranklast,effect_rows_2,1,3)
    elseif strcmp(exp_type,'MOVIE_SCR')
        skip = 5; ons_ind =2; effect_rows_2=10; rankfirst=1; ranklast=10;
        do_structure_mu_NSXfiles(channels,skip,ons_ind,exp_type,'prescr')
        loop_plot_best_responses_s_NSXfiles_profile(channels,muonly,rankfirst,ranklast,effect_rows_2,'prescr')
        do_structure_mu_NSXfiles(channels,skip,ons_ind,exp_type,'posscr')
        loop_plot_best_responses_s_NSXfiles_profile(channels,muonly,rankfirst,ranklast,effect_rows_2,'posscr')
    end

    disp('plot best responses DONE')

    %% same as before in case there are channels where more than the "best" 15 responses are needed
    if do_extra_stims
        channels_more = channels;
        step_pic = 15;
        if strcmp(exp_type,'SCR') || strcmp(exp_type,'RSVPSCR')
            rankfirst=16; ranklast=rankfirst+step_pic-1;
            do_structure_mu_BCM(channels_more,skip,ons_ind)
            loop_plot_best_responses_BCM_rank(channels_more,muonly,rankfirst,ranklast,effect_rows_2)      
        end
        
        disp('plot best responses DONE')
    end
end
%% sorting
if do_sorting
    % channels_active = [17 18 21 22 24 33:35 38:40];
    param.min_clus = 15;
    param.max_spk = 30000;
    param.mintemp = 0.00;                  % minimum temperature for SPC
    param.maxtemp = 0.251;                 % maximum temperature for SPC
    param.tempstep = 0.01;
    
    Do_clustering(channels,'parallel',true,'par',param)

    disp('spike sorting DONE')

    %%  build data structure (grapes) and compute best responses for single units
    if do_loop_plot
        clustered_channels = channels;
        muonly = 'n';

        if strcmp(exp_type,'SCR') || strcmp(exp_type,'RSVPSCR')
            rankfirst=1; ranklast=15;
        else
            rankfirst=1; ranklast=10;
        end

        if strcmp(exp_type,'MOVIE_SCR')
            do_structure_sorted_NSXfiles(clustered_channels,'prescr')
            loop_plot_best_responses_s_NSXfiles_profile(clustered_channels,muonly,rankfirst,ranklast,effect_rows_2,'prescr')
            do_structure_sorted_NSXfiles(clustered_channels,'posscr')
            loop_plot_best_responses_s_NSXfiles_profile(clustered_channels,muonly,rankfirst,ranklast,effect_rows_2,'posscr')
        elseif strcmp(exp_type,'STORY') || strcmp(exp_type,'MOVIE')
            do_structure_sorted_NSXfiles(clustered_channels,phase)
            loop_plot_best_responses_s_NSXfiles_profile(clustered_channels,muonly,1,10,effect_rows_2,phase)
            loop_plot_best_responses_s_NSXfiles_profile(clustered_channels,muonly,11,20,effect_rows_2,phase)
        elseif strcmp(exp_type,'RSVPSCR')
            do_structure_sorted_BCM(clustered_channels)
            loop_plot_best_responses_BCM_rank(clustered_channels,muonly,rankfirst,ranklast,effect_rows_2,1,3)
        else
            do_structure_sorted_NSXfiles(clustered_channels,phase)
            loop_plot_best_responses_s_NSXfiles_profile(clustered_channels,muonly,rankfirst,ranklast,effect_rows_2,phase)
        end

        disp('plot best responses DONE')
        %% same as before in case there are channels where more than the "best" 15 responses are needed
        if do_extra_stims
            channels_more_clus = clustered_channels;
            step_pic = 15;
            if strcmp(exp_type,'SCR') || strcmp(exp_type,'RSVPSCR')
                rankfirst=16; ranklast=rankfirst+step_pic-1;
                do_structure_sorted_BCM(channels_more_clus)
                loop_plot_best_responses_BCM_rank(channels_more_clus,muonly,rankfirst,ranklast,effect_rows_2)                
            end

            disp('plot best responses DONE')
        end
    end
end

set(groot,'defaultaxesfontsmoothing','remove')
set(groot,'defaultfiguregraphicssmoothing','remove')
set(groot,'defaultaxestitlefontsizemultiplier','remove')
set(groot,'defaultaxestitlefontweight','remove')

rmpath(genpath(fullfile(dir_base,'NPMK')));
rmpath(genpath(fullfile(dir_base,'wave_clus_BCM')));
rmpath(genpath(fullfile(dir_base,'codes_for_analysis_BCM')));
% rmpath(genpath(fullfile(dirmain,'000rsvpscr_pic')))
rmpath(genpath([dirmain '_pic']))
