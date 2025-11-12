function plot_grapes_as_online(varargin)

    resp_conf = struct;
    ipr = inputParser;
    addParameter(ipr, 'grapes_offline', true, @islogical);
    addParameter(ipr, 'channels2plot','all');
    addParameter(ipr, 'is_online', false, @islogical);
    addParameter(ipr, 'plot_best_stims_only', false, @islogical);
    addParameter(ipr, 'copy2miniscrfolder', true, @islogical);
    %addParameter(ipr, 'copy2dailyminiscrfolder', true, @islogical);
    addParameter(ipr, 'show_sel_count', true, @islogical);
    addParameter(ipr, 'show_best_stims_wins', true, @islogical);
    addParameter(ipr, 'best_stims_nwins', 8);
    addParameter(ipr, 'ch_grapes_nwins', 2);
    addParameter(ipr, 'stim_list', 'all');
    addParameter(ipr, 'order_by_rank', true, @islogical);
    addParameter(ipr, 'extra_lbl', '');
    addParameter(ipr, 'use_blanks', false, @islogical);    
    addParameter(ipr, 'circshiftblanks', false, @islogical);    
    addParameter(ipr, 'short_win', false, @islogical);    

    % Check if varargin exists or is empty
    if ~isempty(varargin)
        parse(ipr,varargin{:});
    else
        parse(ipr);
    end
    grapes_offline = ipr.Results.grapes_offline;
    channels2plot = ipr.Results.channels2plot;
    is_online = ipr.Results.is_online;
    plot_best_stims_only = ipr.Results.plot_best_stims_only;
    copy2miniscrfolder = ipr.Results.copy2miniscrfolder;
    %copy2dailyminiscrfolder = ipr.Results.copy2dailyminiscrfolder;
    show_sel_count = ipr.Results.show_sel_count;
    show_best_stims_wins = ipr.Results.show_best_stims_wins;
    nwins = ipr.Results.best_stims_nwins;
    ch_grapes_nwins = ipr.Results.ch_grapes_nwins;
    stim_list = ipr.Results.stim_list;
    order_by_rank = ipr.Results.order_by_rank;
    extra_lbl = ipr.Results.extra_lbl;   
    use_blanks = ipr.Results.use_blanks;
    circshiftblanks = ipr.Results.circshiftblanks;
    short_win = ipr.Results.short_win;

    if use_blanks
        if circshiftblanks
            grapes_name = 'grapes_blanks_circ';
            if isempty(extra_lbl)
                extra_lbl = 'offline_circblanks';
            end
        else
            grapes_name = 'grapes_blanks';
            if isempty(extra_lbl)
                extra_lbl = 'offline_blanks';
            end
        end
        
        resp_conf.tmin_median = 50;
        resp_conf.tmax_median = 550;
        resp_conf.tmin_base = -550;
	    resp_conf.tmax_base = -50;
    else
        grapes_name = 'grapes';
        resp_conf.tmin_median = 200;
        resp_conf.tmax_median = 700;
        resp_conf.tmin_base = -900;
	    resp_conf.tmax_base = -100;
    end
    
    %this should be imported from config files
    TIME_PRE=1e3;
    TIME_POS=2e3;
    SR = 30000;
    
    resp_conf.from_onset = 1;
    resp_conf.smooth_bin=1500;
    resp_conf.min_spk_median=1;
    
    resp_conf.psign_thr = 0.05;
    resp_conf.t_down=20;
    resp_conf.over_threshold_time = 75;
    resp_conf.below_threshold_time = 100;
    resp_conf.nstd = 3;
    resp_conf.win_cent=1;
    resp_conf.FR_resol = 10; %bin size in ms
    resp_conf.min_ifr_thr = 4; % 4 Hz
    
    N_RASTER_PER_WIN = 20;
    % win2plot = 'all';%'all'; %all for all
    sigma_gauss = 10;
    alpha_gauss = 3.035;
    ifr_resolution = 1;
    
    
    if ~grapes_offline
        folder = 'online_grapes_new_order_results';
        if is_online
            folder = [folder filesep 'plots_without_online_best'];
        end
        
    else
        folder = 'offline_grapes_new_order_results';
        if is_online
            folder = [folder filesep 'plots_without_online_best'];
        end
        if ~strcmp(channels2plot,'all')
            if min(channels2plot)>2000
                channels2plot = channels2plot - 2000;
            elseif min(channels2plot)>1000
                channels2plot = channels2plot - 1000;
            end
        end
    end
    
    %%
    
    addpath(fullfile(fileparts(mfilename('fullpath')),'../..')) % always / for this trick
    
    custompath = reylab_custompath({'wave_clus_reylab','tasks/online_v3/online','codes_for_analysis','mex','useful_functions'});
    
    load('experiment_properties_online3.mat')

    if ~exist('priority_chs_ranking', 'var')
        priority_chs_ranking = []; % It should be in patient 12 & onwards in experiment prop.
    end
    
    if grapes_offline
        grapes = load(grapes_name);
        grapes.ImageNames=table(repmat({[pwd filesep 'pics_used']}, ...
                                numel(grapes.ImageNames),1),grapes.ImageNames, ...
                                'VariableNames',{'folder', 'name'});
    else
        load(['results' filesep 'grapes_online.mat']);
    %     picusa = cellfun(@(x) contains(x,'pics_USA'),grapes.ImageNames.folder);
         grapes.ImageNames.folder(:)={[pwd filesep 'pics_used']};
    end

    final_n_scr = numel(scr_end_cell);
    
    if ~strcmp(channels2plot,'all')
        new_rasters = struct;
        for i =1:numel(channels2plot)
            ch_field = ['chan' num2str(channels2plot(i))];
            new_rasters.(ch_field) = grapes.rasters.(ch_field);
        end
        grapes.rasters = new_rasters;
    end
    
    if ~exist(['.' filesep folder],'dir') 
        mkdir(folder)
    end
    cd(folder)
    all_picsused = unique(cell2mat(cellfun(@(x)x.pics2use, scr_config_cell, 'UniformOutput', false)));
    
    ifr_calculator = IFRCalculator(alpha_gauss,sigma_gauss,ifr_resolution,SR,TIME_PRE,TIME_POS);
    
    %is this doing things for every response?
    [data, rank_config] = create_responses_data_parallel(grapes,all_picsused, ...
                                                         {'mu','class'},ifr_calculator,resp_conf, ...
                                                         [], priority_chs_ranking);
    if strcmp(ch_grapes_nwins,'all')
        ch_grapes_nwins = ceil(numel(unique(data.stim_number))/ N_RASTER_PER_WIN);
    end
    
    if is_online && contains(experiment.subtask, 'DynamicScr')
        online_miniscr_selections_file = ['..' filesep '..' filesep 'results' filesep 'selected4miniscr.csv'];
        if ~exist(online_miniscr_selections_file, "file")
            fprintf("%s not found", online_miniscr_selections_file)
        else
            online_miniscr_selections = readtable(online_miniscr_selections_file,'Delimiter',',');
            if height(online_miniscr_selections) > 0
                data_wo_online_best = data(~ismember(string(experiment.ImageNames.concept_name(data.stim_number)), ...
                                      online_miniscr_selections.concept_name),:);
                data_wo_online_best = sort_responses_table(data_wo_online_best);
            end
        end
    end

    data = sort_responses_table(data);
    
    if strcmp(channels2plot,'all') || plot_best_stims_only        
        if is_online && contains(experiment.subtask, 'DynamicScr')
            % plot best stims without online
            % miniscr selections
            data_to_plot = create_best_stims_table(experiment, grapes, data_wo_online_best, nwins, true, priority_chs_ranking, [], [], 0, is_online);
            
            disp('Plotting best stims without online miniscr selections')
            lbl = sprintf('EMU-%.3d_best_stim_online',experiment.params.EMU_num);
            [~, s4miniscr_tbl, ~] = stimulus_selection_windows( ...
                                                              data_to_plot, grapes, rank_config, ...
                                                              final_n_scr, ifr_calculator, nwins, ...
                                                              lbl, priority_chs_ranking, ...
                                                              experiment, copy2miniscrfolder, ...
                                                              show_sel_count, show_best_stims_wins, short_win);
            writetable(s4miniscr_tbl, 'offline_miniscr_stims.csv','Delimiter',',');
            disp('Finished plotting best stims offline without online miniscr selections & creating offline_miniscr_stims.csv')
            cd('..')
        end

        % plot best stims without removing any miniscr selections
        data_to_plot = create_best_stims_table(experiment, grapes, data, nwins, true, priority_chs_ranking, [], [], 0, is_online);
        lbl = sprintf('EMU-%.3d_best_stim',experiment.params.EMU_num);
        
        [~, s4miniscr_tbl, ~] = stimulus_selection_windows( ...
                                                          data_to_plot, grapes, rank_config, ...
                                                          final_n_scr, ifr_calculator, nwins, ...
                                                          lbl, priority_chs_ranking, ...
                                                          experiment, copy2miniscrfolder, ...
                                                          show_sel_count, false, short_win);
        writetable(s4miniscr_tbl, 'offline_best_stims.csv','Delimiter',',');
        disp('Finished plotting best stims offline & creating offline_best_stims.csv')      
        
        
    end
    
    %tic
    if ~plot_best_stims_only
        plot_channel_grapes('channels2plot', channels2plot, 'stim_list', stim_list, 'order_by_rank', order_by_rank, ...
                                'data', data, 'grapes', grapes, 'n_scr', final_n_scr, 'nwins2plot', ch_grapes_nwins, ...
                                'rank_config', rank_config, 'ifr_x', ifr_calculator.ejex, ...
                                'save_fig', true, 'emu_num', experiment.params.EMU_num, ...
                                'close_fig', true, 'order_offset', 0, ...
                                'priority_chs_ranking', priority_chs_ranking, ...
                                'parallel_plots', true, 'extra_lbl', extra_lbl);
    end
    cd('..')
    % 
    % isthisclass = cellfun(@(x) strcmp(x,'class1'),data.class);
    % isthischannel = cellfun(@(x) strcmp(x,'chan43'),data.channel);
    % isthisstim = data.stim_number==113;
    % 
    % f=loop_plot_responses_BCM_online(data(isthischannel & isthisclass & isthisstim,:), ...
    %                                  grapes,final_n_scr,1,rank_config,ifr_calclator.ejex, ...
    %                                  false,sprintf('test_%s_ch_%s',rastername,labels{i}),false)