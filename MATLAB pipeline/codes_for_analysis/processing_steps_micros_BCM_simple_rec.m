% FIRST: Move to the folder where the data is stored.

clearvars
clc
% dirmain = 'D:\march7AM\EMU-027_task-RSVPscr_run-03';
% filenames={'sHiCikEPkEMkGm_20161018-160552-001'};
which_system_micro = 'BRK'; % 'BRK' or 'RIP'
%which_system_micro = 'RIP'; % 'BRK' or 'RIP'
remove_ref_chs = [265,274,297,306]; 
bundle_labels = {'mRT2H';'mRF2aCa';'mLOF';'mLF2aCa'};
    
NSPs={'_NSP-1';'_NSP-2'};
do_power_plot = 1;
notchfilter=1;
mex_folder = 'mex_HGR';
% ch_photo_BRK = 1257;
% ch_photo_RIP = 10241;
% do_sorting = 1;
% do_loop_plot = 1;
% extract_events = 1;
% do_extra_stims = 1;

%%
[~,name] = system('hostname');
if contains(name,'REY-LT-01'), dir_base = 'E:\Google Drive FIUBA\Fer Chaure';  % HERNAN'S DELL LAPTOP
elseif contains(name,'DESKTOP-DQR054O'), dir_base = 'C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR';  % EMU BEH PC  
elseif contains(name,'NPB-CSNzbookG5'), dir_base = 'D:\Google Drive FIUBA\Fer Chaure';  % HERNAN'S HP laptop ZBOOK15   
end

paths2add = {
    {dir_base,'NPMK'};
    {dir_base,'BCM-EMU','wave_clus_BCM'};
    {dir_base,'BCM-EMU', 'codes_for_analysis_BCM'};
    {dir_base,'BCM-EMU',mex_folder};
    {dir_base,'BCM-EMU','useful_functions'};
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
cd(dirmain)
ftype = 'ns5';
%     ftype = 'ns3';

if strcmp(which_system_micro,'BRK')
    fname_prefix = '*.';
elseif strcmp(which_system_micro,'RIP')
    addpath(genpath(fullfile(dir_base,'neuroshare')));
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
    if all(cellfun(@(x) ~contains(x,NSPs{1}) && ~contains(x,NSPs{2}),filenames))
%         parse_NSx(filenames,2,max_memo_GB);
        parse_NSx(filenames,[],max_memo_GB);
        filename_NEV = [filenames{1}(1:end-3) 'nev'];
    else
        parse_NSx(filenames(cellfun(@(x) contains(x,NSPs{2}),filenames)),2,max_memo_GB);
        ind_file=cellfun(@(x) contains(x,NSPs{1}),filenames);
        filename_NEV = [filenames{ind_file(1)}(1:end-3) 'nev'];
    end
elseif strcmp(which_system_micro,'RIP')
    [~, hFile] = ns_OpenFile(filenames{1});    
%    CHECK CHANNEL NUMBERS BEFORE CONTINUING
    channels_RIP = double(setdiff(cell2mat({hFile.Entity.ElectrodeID}),remove_ref_chs));
    
%     CONFIRM BUDLE NAME AND CH NUM
%     keyval = input('If this is fine by you, press y to continue; any other key will terminate this program  ','s');
% 
%     if strcmpi(keyval,'y')
%         asdad
%     end

    read_ripple(filenames{1},hFile,bundle_labels,channels_RIP);
%     CHECK FILES VARIABLE IN PARSENSX FOR RIPPLE
end
A=dir('*_NSP-2.ns5');
if isempty(A)
    disp('There is no ns5 file for NSP2');
else
%     parse_NSx({A.name},1,max_memo_GB);
    parse_NSx({A.name},2,max_memo_GB);
end

%%
% channels = [65:80 97:104 113:120];

if ~exist('channels','var')
    channels=[];
    load('NSx','NSx');
%     channels = double(cell2mat({NSx(arrayfun(@(x) (strcmp(x.unit,'uV') && x.sr==30000),NSx)).chan_ID}));
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

%%
set(groot,'defaultaxesfontsmoothing','remove')
set(groot,'defaultfiguregraphicssmoothing','remove')
set(groot,'defaultaxestitlefontsizemultiplier','remove')
set(groot,'defaultaxestitlefontweight','remove')

cellfun(@(x) rmpath(genpath(strjoin(x, filesep))),paths2add);

