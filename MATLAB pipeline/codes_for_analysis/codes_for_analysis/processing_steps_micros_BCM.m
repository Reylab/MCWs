% FIRST: Move to the folder where the data is stored.

clearvars
clc
% dirmain = 'D:\march7AM\EMU-027_task-RSVPscr_run-03';
% filenames={'sHiCikEPkEMkGm_20161018-160552-001'};
% which_system_micro = 'BRK'; % 'BRK' or 'RIP'
which_system_micro = 'RIP'; % 'BRK' or 'RIP'
remove_ref_chs = [265,274,297,306]; 
bundle_labels = [];
% bundle_labels = {'RA';'RMH';'LA';'LMH'};
% bundle_labels = {'LPH';'LInsula';'bla1';'bla2'};
% bundle_labels = {'LIFGCingGyrus';'LAH';'LSTG';'LParietalOperc';'LPH';'LInsula';...
%     'RIFGCingGyrus';'RAH';'RSTG';'RParietalOperc';'RPH';'RInsula'};
% bundle_labels = {'mROF';'mRF1aCa';'mLOF';'mLT2bHb'};
% bundle_labels = {'mLF1aCa';'mLT2aHa';'mLOF';'mLT2bHb'};
% bundle_labels = {'mLF1bCa';'mRF1aCa';'mLT2aHa';'mLT2bHb'};
% bundle_labels = {'mRT2aHa';'mRT2bHb';'mLT2aHa';'mLT2bHb'};

% bundle_labels = {'mRT2aHa';'mRT2bE';'mLT2aHa';'mLT2bE'};


NSPs={'_NSP-1';'_NSP-2'};
% fast_analysis = 0;
fast_analysis = 1;
do_power_plot = 1;
notchfilter=1;
mex_folder = 'mex_HGR';
ch_photo_BRK = 1257;
ch_photo_RIP = 10241;
do_sorting = 1;
do_loop_plot = 1;
extract_events = 1;
% do_extra_stims = 1;

%%
[~,name] = system('hostname');
% if contains(name,'REY-LT-01'), dir_base = 'E:\Google Drive FIUBA\Fer Chaure';  % HERNAN'S DELL LAPTOP
if contains(name,'REY-LT-01'), dir_base = 'I:\My Drive\Fer Chaure';  % HERNAN'S DELL LAPTOP
elseif contains(name,'DESKTOP-DQR054O'), dir_base = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR';  % EMU BEH PC  
% elseif contains(name,'NPB-CSNzbookG5'), dir_base = 'D:\Google Drive FIUBA\Fer Chaure';  % HERNAN'S HP laptop ZBOOK15   
elseif contains(name,'NPB-CSNzbookG5'), dir_base = 'F:';  % HERNAN'S HP laptop ZBOOK15   
end

paths2add = {
    {dir_base,'BCM-EMU','NPMK'};
    {dir_base,'BCM-EMU','wave_clus_BCM'};
    {dir_base,'BCM-EMU', 'codes_for_analysis_BCM'};
    {dir_base,'BCM-EMU',mex_folder};
    {dir_base,'BCM-EMU','useful_functions'};
    {dir_base,'BCM-EMU','neuroshare'};
    {dir_base,'pics_USA'}
};

cellfun(@(x) addpath(genpath(strjoin(x, filesep))),paths2add);

if ~exist('dirmain','var')
    dirmain = pwd;
end

% addpath(genpath([dirmain '_pic'])) % srtimuli folder
set(groot,'defaultaxesfontsmoothing','off')
set(groot,'defaultfiguregraphicssmoothing','off')
set(groot,'defaultaxestitlefontsizemultiplier',1.1)
set(groot,'defaultaxestitlefontweight','normal')
%%
exp_type = 'RSVPSCR';  phase=[];
addpath([dirmain filesep '000rsvpscr_pic'])
cd(dirmain)
ftype = 'ns5';
% ftype = 'ns6';
%     ftype = 'ns3';

if strcmp(which_system_micro,'BRK')
    fname_prefix = '*.';
elseif strcmp(which_system_micro,'RIP')
    fname_prefix = '*_RIP.';
end

if ~exist('filenames','var')
    A=dir([fname_prefix ftype]);
    if isempty(A)
        error('There are no %s files in this folder',ftype);
    else
        filenames = {A.name};
    end
else
    fprintf('variable filenames already exists and is equal to %s\n',filenames{:})
end
%%
max_memo_GB = [];

if ~IsWin
%     [s,m]=unix('vm_stat | grep free');
%     spaces=strfind(m,' ');
%     max_memo_GB = str2double(m(spaces(end):end))*4096*0.8/((1024)^3);
end

if strcmp(which_system_micro,'BRK')
    if fast_analysis
        if ~exist('bundle_labels','var') 
            parse_NSx(filenames,2,max_memo_GB);
        else
            parse_NSx(filenames,2,max_memo_GB,bundle_labels);
        end
%          if length(filenames)>1                                                   
%         parse_NSx(filenames(cellfun(@(x) contains(x,NSPs{2}),filenames)),2,max_memo_GB)  
%        else                                                                     
%         parse_NSx(filenames,2,max_memo_GB);                                  
%  end 
    else        
        if all(cellfun(@(x) ~contains(x,NSPs{1}) && ~contains(x,NSPs{2}),filenames))
            parse_NSx(filenames,[],max_memo_GB);
            filename_NEV = [filenames{1}(1:end-3) 'nev'];
        else
            parse_NSx(filenames(cellfun(@(x) contains(x,NSPs{2}),filenames)),2,max_memo_GB);
            ind_file=cellfun(@(x) contains(x,NSPs{2}),filenames);
            filename_NEV = [filenames{ind_file(2)}(1:end-3) 'nev'];
        end
    end
elseif strcmp(which_system_micro,'RIP')
        parse_ripple(filenames,remove_ref_chs,bundle_labels)

%     [~, hFile] = ns_OpenFile(filenames{1},'single');    
%     channels_RIP = double(setdiff(cell2mat({hFile.Entity.ElectrodeID}),remove_ref_chs));
%     read_ripple(filenames{1},hFile,bundle_labels,channels_RIP);
end

% %%
% A=dir('*_NSP-2.ns5');
% if isempty(A)
%     disp('There is no ns5 file for NSP2');
% else
% %     parse_NSx({A.name},1,max_memo_GB);
%     parse_NSx({A.name},2,max_memo_GB);
% end

%%
% channels = [65:80 97:104 113:120];

if ~exist('channels','var')
    channels=[];
    load('NSx','NSx');
%     channels = double(cell2mat({NSx(arrayfun(@(x) (strcmp(x.unit,'uV') &&
%     x.sr==30000),NSx)).chan_ID})); % DON'T KNOW WHY IT DOES NOT WORK
    AA = {NSx(arrayfun(@(x) (strcmp(x.unit,'uV') && x.sr==30000),NSx)).chan_ID};
    for i=1:length(AA)
        channels(i)=double(AA{i});
    end
end

%% opens parallel pool
poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool
end

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
if extract_events && ~fast_analysis
    %ONLY FOR SINGLE FILE AT THE MOMENT
    if strcmp(which_system_micro,'BRK')
        openNEV(['./' filename_NEV],'report','noread','8bits')
    end
    if strcmp(exp_type,'RSVPSCR')
%         extract_events_rsvpscr_BCM(NEV)
        extract_events_rsvpscr_BCM_photoonly(which_system_micro)  %ver correccion FER
%         extract_events_rsvpscr_BCM_photo(NEV)
    end
end
%% spike detection
if fast_analysis
    neg_thr_channels = channels;
    pos_thr_channels = [];
else
    ch_temp = input(sprintf('Currently, channels = %s.\nIf you want to keep it like that, press enter.\nOtherwise, enter the new vector and press enter  ',num2str(channels)));
    if ~isempty(ch_temp)
        channels = ch_temp;
    end

    neg_thr_channels = input('Enter the vector with neg_thr_channels and press enter. Press enter to use all channels ');
    if isempty(ch_temp)
        neg_thr_channels = channels;
        pos_thr_channels = [];
    else
        pos_thr_channels = input('Enter the vector with pos_thr_channels and press enter. Press enter for empty array ');
    end
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
if do_loop_plot  && ~fast_analysis
    muonly = 'y';

    if strcmp(exp_type,'RSVPSCR')
        skip = 0; ons_ind =0;effect_rows_2=0;
        do_structure_mu_BCM(channels,skip,ons_ind,exp_type,phase)
        rankfirst=1; ranklast=15;    
        loop_plot_best_responses_BCM_rank(channels,muonly,rankfirst,ranklast,effect_rows_2,1,3)
    end

    disp('plot best responses DONE')

    % same as before in case there are channels where more than the "best" 15 responses are needed
    if do_extra_stims
        channels_more = channels;
        step_pic = 15;
        if strcmp(exp_type,'SCR') || strcmp(exp_type,'RSVPSCR')
            rankfirst=16; ranklast=rankfirst+step_pic-1;
            do_structure_mu_BCM(channels_more,skip,ons_ind)
            loop_plot_best_responses_BCM_rank(channels_more,muonly,rankfirst,ranklast,effect_rows_2,1,3)      
        end
        
        disp('plot best responses DONE')
    end
end
%% sorting
if do_sorting
    param.min_clus = 15;
    param.max_spk = 20000;
    param.mintemp = 0.00;                  % minimum temperature for SPC
    param.maxtemp = 0.251;                 % maximum temperature for SPC
    param.tempstep = 0.01;
    
    Do_clustering(channels,'parallel',true,'par',param)

    disp('spike sorting DONE')
%%
    % build data structure (grapes) and compute best responses for single units
    if do_loop_plot  && ~fast_analysis
        clustered_channels = channels;
        muonly = 'n';

        if strcmp(exp_type,'SCR') || strcmp(exp_type,'RSVPSCR')
            rankfirst=1; ranklast=15;
        else
            rankfirst=1; ranklast=10;
        end

        if strcmp(exp_type,'RSVPSCR')
            do_structure_sorted_BCM(clustered_channels)
            loop_plot_best_responses_BCM_rank(clustered_channels,muonly,rankfirst,ranklast,effect_rows_2,1,3)
        end

        disp('plot best responses DONE')
        % same as before in case there are channels where more than the "best" 15 responses are needed
        if do_extra_stims
            channels_more_clus = clustered_channels;
            step_pic = 15;
            if strcmp(exp_type,'SCR') || strcmp(exp_type,'RSVPSCR')
                rankfirst=16; ranklast=rankfirst+step_pic-1;
                do_structure_sorted_BCM(channels_more_clus)
                loop_plot_best_responses_BCM_rank(channels_more_clus,muonly,rankfirst,ranklast,effect_rows_2,1,3)                
            end

            disp('plot best responses DONE')
        end
    end
end

%%
set(groot,'defaultaxesfontsmoothing','remove')
set(groot,'defaultfiguregraphicssmoothing','remove')
set(groot,'defaultaxestitlefontsizemultiplier','remove')
set(groot,'defaultaxestitlefontweight','remove')

rmpath([dirmain filesep '000rsvpscr_pic'])
cellfun(@(x) rmpath(genpath(strjoin(x, filesep))),paths2add);
