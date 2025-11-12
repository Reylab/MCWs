function processing_steps_MCW_reduced(varargin)

% clearvars
% clc
% filenames={'sHiCikEPkEMkGm_20161018-160552-001'};
%  par.which_system_micro = 'BRK'; % 'BRK' or 'RIP'
processing_steps_start = tic;
ipr = inputParser;
addParameter(ipr,'which_system_micro','RIP'); % 'BRK' or 'RIP'
addParameter(ipr,'fast_analysis',0); % fast_analysis = 1;
addParameter(ipr,'nowait',true); % nowait = false;
addParameter(ipr,'micros',true); % micros = false;
addParameter(ipr,'remove_ref_chs',[]); % remove_ref_chs = [265,274,297,306,329,338];
addParameter(ipr,'do_power_plot',1); % do_power_plot = 1;
addParameter(ipr,'notchfilter',1); % notchfilter = 1;
addParameter(ipr,'do_sorting',1); % do_sorting = 1;
addParameter(ipr,'do_loop_plot',1); % do_loop_plot = 1;
addParameter(ipr,'extract_events',1); % extract_events = 1;
addParameter(ipr,'extra_stims_win',0); % extra_stims_win = 0;
addParameter(ipr,'is_online',false); % is_online = false;
addParameter(ipr,'plot_best_stims_only',false); % plot_best_stims_only = false;
addParameter(ipr,'copy2miniscrfolder', false); % copy2miniscrfolder = false;
%addParameter(ipr,'copy2dailyminiscrfolder', false); % copy2dailyminiscrfolder = false;
addParameter(ipr,'show_sel_count', false); % show_sel_count = false;
addParameter(ipr,'show_best_stims_wins', false); % show_best_stims_wins = false;
addParameter(ipr,'best_stims_nwins', 8); % best_stims_nwins = 8;
addParameter(ipr,'ch_grapes_nwins', 2); % ch_grapes_nwins = 2;
addParameter(ipr,'max_spikes_plot', 5000); % max_spikes_plot = 5000;
addParameter(ipr,'use_blanks', true); % use_blanks = false;
addParameter(ipr,'circshiftblanks', true); % circshiftblanks = true;
addParameter(ipr,'make_templates', false); % make_templates = false;


% check if varargin is empty
if ~exist('varargin', 'var') || isempty(varargin)
    parse(ipr);
else
    parse(ipr,varargin{:});
end

par = ipr.Results;
par.qc_params = struct();
par.qc_params.min_amplitude_percentile = 5; % Spikes below this P2P amplitude percentile are quarantined
par.qc_params.min_width_idx = 3;            % Min width of main feature (in samples)
par.qc_params.max_width_idx = 15;           % Max width of main feature (in samples)
par.qc_params.prominence_ratio_threshold = 0.01; % Secondary feature prominence must be > 1% of main peak amp
par.qc_params.final_prominence_ratio_pass = 0.8;

% RIP_hours_offset = 5;
NSPs={'_NSP-1';'_NSP-2'};
USE_PHOTODIODE = 1;
ch_photo_BRK = 1257;
ch_photo_RIP = 10241;
step_pic = 15;


%%
% Define the path where the codes emu repository is located
[~,name] = system('hostname');
if contains(name,'BEH-REYLAB'), dir_base = '/home/user/share/codes_emu'; 
elseif contains(name,'TOWER-REYLAB') || contains(name,'RACK-REYLAB') || contains(name, 'ABTL')
%     current_user = 'sofiad';  % replace with appropriate user name  
    current_user = getenv('USER');    dir_base = sprintf('/home/%s/Documents/GitHub/codes_emu',current_user); 
elseif contains(name,'NSRG-HUB-15446'), dir_base = 'D:\codes_emu'; % Hernan's desktop
elseif contains(name,'NSRG-HUB-16167'), dir_base = 'D:\bcm_emu'; % Hernan's laptop
elseif contains(name,'NSRG-HUB-17988'), dir_base = 'C:\Users\al58796\Documents\GitHub\codes_emu'; % aj desktop
elseif contains(name,'AJ-PC'), dir_base = 'C:\Users\betan\Documents\Research\codes_emu'; % aj desktop

elseif contains(name,'DESKTOP-OO8FBF4'), dir_base = 'D:\BCM-EMU'; % Hernan's laptop home
%elseif contains(name,'ABT-REYLAB'), dir_base = 'C:\Users\user\Documents\GitHub\codes_emu'; % ABT-REYLAB
elseif contains(name,'ABT-REYLAB') || contains(name,'NRSG-HUB-18687'), dir_base = 'C:\Users\smathew\OneDrive - mcw.edu\Rey lab\codes_emu'; % ABT-REYLAB
elseif contains(name,'MCW-20880'), dir_base = 'C:\Users\de31182\Documents\GitHub\codes_emu'; %Dewan's laptop
end

addpath(dir_base);
% custompath = reylab_custompath({'wave_clus_reylab','NPMK','codes_for_analysis','mex','useful_functions','neuroshare','tasks/.','tasks/locations/'});
custompath = reylab_custompath({'wave_clus_reylab','NPMK-master_Gemini','codes_for_analysis','mex','useful_functions','neuroshare','tasks/.','tasks/locations/'});

if contains(name,'REYLAB')
    params = MCW_location(['MCW-' par.which_system_micro]);
    param.processing_rec_metadata = params.processing_rec_metadata;
end

if ~exist('dirmain','var')
    dirmain = pwd;
end

% addpath(genpath([dirmain '_pic'])) % srtimuli folder
set(groot,'defaultaxesfontsmoothing','off')
set(groot,'defaultfiguregraphicssmoothing','off')
set(groot,'defaultaxestitlefontsizemultiplier',1.1)
set(groot,'defaultaxestitlefontweight','normal')
%%
% exp_type = 'RSVPSCR';  phase=[];
exp_type = 'RSVP_online';  phase=[];
% exp_type = 'RSVP_online_mini_scr';  phase=[];

addpath(genpath([dirmain filesep 'pics_used']))
cd(dirmain)
if par.micros
    ftype = 'ns5';
else
    if strcmp(par.which_system_micro,'RIP')
        ftype = 'nf3';
    elseif strcmp(par.which_system_micro,'BRK')
        ftype = 'ns3';
    end
end

if strcmp(par.which_system_micro,'BRK')
    fname_prefix = '*.';
elseif strcmp(par.which_system_micro,'RIP')
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

if strcmp(par.which_system_micro,'BRK')
    if par.fast_analysis
        if ~exist('bundle_labels','var')
%             parse_NSx(filenames,[1 2],max_memo_GB);
            parse_NSx_Gemi_brief(filenames)
        else
            parse_NSx(filenames,[1 2],max_memo_GB,bundle_labels);
        end
    else
        if all(cellfun(@(x) ~contains(x,NSPs{1}) && ~contains(x,NSPs{2}),filenames))
            parse_NSx(filenames,[],max_memo_GB);
            filename_NEV = [filenames{1}(1:end-3) 'nev'];
        else
            parse_NSx(filenames(cellfun(@(x) contains(x,NSPs{2}),filenames)),2,max_memo_GB);
            ind_file=cellfun(@(x) contains(x,NSPs{2}),filenames);
            filename_NEV = [filenames{ind_file(1)}(1:end-3) 'nev'];
        end
    end
elseif strcmp(par.which_system_micro,'RIP')
    %         parse_ripple(filenames,par.remove_ref_chs,bundle_labels)
    parse_ripple(filenames,par.remove_ref_chs)
    filename_NEV = [filenames{1}(1:end-3) 'nev'];
end
clear filenames

%%
if ~exist('channels','var')
    channels=[];
    load('NSx','NSx');
    if par.micros
        AA = {NSx(arrayfun(@(x) (startsWith(x.label,'m') && strcmp(x.unit,'uV') && x.sr==30000),NSx)).chan_ID};
    else
        AA = {NSx(arrayfun(@(x) (x.sr==2000),NSx)).chan_ID};
    end

    for i=1:length(AA)
        channels(i)=double(AA{i});
    end
end
% channels = [1:246];
%% opens parallel pool
poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool
end

%% power spectrum and potential notches
if par.do_power_plot
%     new_check_lfp_power_NSX(channels,'parallel',true)
    do_power_plot_start = tic;
    fprintf('power spectrum BEGIN..\n')
    new_check_lfp_power_NSX(channels,'parallel',true,'with_NoNotch',true)
    time_taken = toc(do_power_plot_start);
    time_taken_mins = floor(time_taken / 60);
    time_taken_secs = mod(time_taken, 60);
    fprintf('Power spectrum DONE in %d mins %0.0f seconds.\n', time_taken_mins, time_taken_secs)
end

%% spike detection
if par.micros
    if par.fast_analysis || par.nowait
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
    % param.preprocessing = false;
    param.preprocessing = true;

    disp('spike detection BEGIN..')
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
    disp('spike detection DONE.')

    
    %% collision and quarantine within probes
    disp('separate_collisions BEGIN..')
    separate_collisions(channels)

    % quarantine squikes 
    artifact_removal(channels)

       %% sorting
    if par.do_sorting
        disp('spike sorting BEGIN..')
        param.min_clus = 15;
        param.max_spk = 30000;
        param.mintemp = 0.00;                  % minimum temperature for SPC
        param.maxtemp = 0.251;                 % maximum temperature for SPC
        param.tempstep = 0.01;
        param.max_std_templates = 3;
        param.max_spikes_plot = par.max_spikes_plot; % Default: 5000
        
        Do_clustering(channels, 'parallel', true, 'make_times', true, ...
                      'make_templates', par.make_templates, 'make_plots', false, 'par', param);
        disp('spike sorting DONE')

        Do_clustering(channels,'parallel',true,'make_times',false,'make_templates',false,'make_plots',true,'par',param)    

        compute_metrics_batch('all','parallel',false,'quar',false);
    end
        % can be used on single channels
        % [metrics_table, SS] = compute_cluster_metrics(data, ...
        %     'exclude_cluster_0', params.exclude_cluster_0, ...
        %     'n_neighbors', params.n_neighbors, ...
        %     'bin_duration', params.bin_duration, ...
        %     'make_plots',false, ...
        %     'save_plots',true);        

        % % and if you want to merge clusters use this function
        % [new_data, ~, metrics, SS] = merge_and_report('times_CH123.mat', [1, 2, 3], ...
        %         'calc_metrics', true, ...
        %         'make_plots', false);

    %% reintroduce quarantined spikes
    % see if they match any templates
    rescue_spikes(channels,'parallel',true);

    %fix need something to separate this ones quar
    compute_metrics_batch('all','quar',true);
            
    

    end
end