% Run from a run folder like EMU-007_subj-MCW-FH_010_task-RSVPDynamicScr_run-02
% Add grapes_debug to path, don't change folder

clear all;
grapes_offline = false;
block_by_block = true;
block_stim_popup = true;
save_dl_data = false;
plot_best_stims = false;
plot_ch_grapes = true;
parallel_plots = true;
use_blanks = true;
circshiftblanks = true;

WAIT_LOOP = 1; %seconds between buffer clearings, max 3seg

TIME_PRE=1e3;
TIME_POS=2e3;
RASTER_SIMILARITY_THR = 0.85;
MAX_NTRIALS = 15;

remove_channels = [];
priority_channels = [];
priority_chs_ranking = [];
% priority_chs_ranking = [257:274 298:306 321:338];
%priority_chs_ranking = [257:274 284:319]; % EMU-007_subj-MCW-FH_010_task-RSVPDynamicScr_run-02
% priority_chs_ranking = [298:306]; % EMU-007_subj-MCW-FH_010_task-RSVPDynamicScr_run-02
not_online_channels = [];



%grapes and plot parameters
MAX_RASTERS_PER_STIM = 3;
resp_conf = struct;
resp_conf.from_onset = 1;
resp_conf.smooth_bin=1500;
resp_conf.min_spk_median=1;
resp_conf.tmin_median=200;
resp_conf.tmax_median=700;
resp_conf.psign_thr = 0.05;
resp_conf.nstd = 3;
resp_conf.win_cent=1;

resp_conf.sigma_gauss = 10;
resp_conf.alpha_gauss = 3.035;
resp_conf.ifr_resolution_ms = 1;
resp_conf.sr = 30000;
resp_conf.TIME_PRE = TIME_PRE;
resp_conf.TIME_POS = TIME_POS;
resp_conf.tmin_base = -900; % ms
resp_conf.tmax_base = -100; % ms
resp_conf.FR_resol = 10; % ms
resp_conf.min_ifr_thr = 4; % Hz
nwin = 4;

if use_blanks
    if circshiftblanks
        grapes_name = 'grapes_blanks_circ';
        % if isempty(extra_lbl)
        %     extra_lbl = 'circblanks';
        % end
    else
        grapes_name = 'grapes_blanks';
        % if isempty(extra_lbl)
        %     extra_lbl = 'blanks';
        % end
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

% add_folder_path('codes_emu');

ifr_calc= IFRCalculator(resp_conf.alpha_gauss,resp_conf.sigma_gauss, ...
                              resp_conf.ifr_resolution_ms,resp_conf.sr,TIME_PRE,TIME_POS);

ifr_calc_blanks= IFRCalculator(resp_conf.alpha_gauss,resp_conf.sigma_gauss, ...
                               resp_conf.ifr_resolution_ms,resp_conf.sr,0,500);

exp_props_file = 'experiment_properties_online3.mat';
load(exp_props_file);

if grapes_offline
    grapes = load(grapes_name);
    grapes.ImageNames=table(repmat({[pwd filesep 'pics_used']}, ...
                            numel(grapes.ImageNames),1),grapes.ImageNames, ...
                            'VariableNames',{'folder', 'name'});
    % grapes.ImageNames.name = cellfun(@(x) regexprep(x, '(\D)(\d)', '$1_$2'), grapes.ImageNames.name, 'UniformOutput', false);
    % grapes.ImageNames.name = cellfun(@(x) replace(x,'-','_'), grapes.ImageNames.name, 'UniformOutput', false);
else
    load(['results' filesep 'grapes_online.mat']);
    grapes.ImageNames.folder(:)={[pwd filesep 'pics_used']};
end

result_folder = [pwd filesep 'results'];
cd(result_folder);

if ~block_by_block
    % Call at the end of online.
    all_picsused = unique(cell2mat(cellfun(@(x)x.pics2use, scr_config_cell, 'UniformOutput', false)));
    b_parallel = true; 
    % [data, rank_config] = create_responses_data(grapes,all_picsused, ...
    %                       {'mu','class'},ifr_calc, ifr_calc_blanks, ...
    %                       resp_conf,[],priority_channels, b_parallel, use_blanks);
    [data, rank_config] = create_responses_data_parallel(grapes,all_picsused, ...
                          {'mu','class'},ifr_calc, ...
                          resp_conf,[],priority_channels);
    data = sort_responses_table(data);
    
    data_class = data(~strcmp(data.class, 'mu'), :);
    class_unique = unique(data_class(:, {'class', 'channel'}));
    fprintf("Single unit count: %d \n", height(class_unique));

    if save_dl_data
        save_dl_raster_data(data, grapes);
    end

    final_n_scr = numel(scr_config_cell);
    
    if plot_best_stims
        show_best_stims_wins = true;
        copy2miniscrfolder = false; show_sel_count = true;
        % best stims plot
        data_to_plot = create_best_stims_table(experiment, grapes, data, nwin, true, priority_chs_ranking);
        lbl = sprintf('EMU-%.3d_best_stim',experiment.params.EMU_num);
        [~, s4miniscr_tbl, ~] = stimulus_selection_windows( ...
                                                          data_to_plot, grapes, rank_config, ...
                                                          final_n_scr, ifr_calc, nwin, ...
                                                          lbl, priority_chs_ranking, ...
                                                          experiment, copy2miniscrfolder, ...
                                                          show_sel_count, show_best_stims_wins);
        
        % writetable(s4miniscr_tbl, 'selected4miniscr.csv','Delimiter',',');
    end
        
    if plot_ch_grapes
        channels2plot = 'all';
        % stim_list = 'all';
        % stim_list = [175,1528,880,376,898,1050,971,1113,1299]; % word-42
        stim_list = fetched_pics_cell{1, 1}{1, 1}.categ_info{10, 1}.pic_ids; % same_feats
        % stim_list = fetched_pics_cell{1, 1}{1, 2}.categ_info{10, 1}.pic_ids; % diff_feats
        % stim_list = fetched_pics_cell{1, 1}{1, 3}.categ_info{10, 1}.pic_ids; % all_but_1
        % stim_list = fetched_pics_cell{1, 1}{1, 4}.categ_info{10, 1}.pic_ids; % all_but_2
        % stim_list = fetched_pics_cell{1, 1}{1, 5}.categ_info{10, 1}.pic_ids; % all_but_3
        % stim_list = fetched_pics_cell{1, 1}{1, 6}.categ_info{10, 1}.pic_ids; % all_but_4
        channels2plot = [270,273];
        % stim_list = [20,88,161,13,14,154,288,144,155,238,267,47,273,278,253,298,177,28,106,72];
        % stim_list = [130,17,161,62,148,125,97,33,46,160,86,73,61,94,140,11,2,134,37,30];
        
        plot_channel_grapes('channels2plot', channels2plot, 'stim_list', stim_list, 'order_by_rank', false, ...
                            'data', data, 'grapes', grapes, 'n_scr', final_n_scr, 'nwins2plot', 2, ...
                            'rank_config', rank_config, 'ifr_x', ifr_calc.ejex, ...
                            'save_fig', true, 'emu_num', experiment.params.EMU_num, ...
                            'close_fig', true, 'order_offset', 0, ...
                            'priority_chs_ranking', priority_chs_ranking, ...
                            'parallel_plots', parallel_plots);
    end
else
    grapes_block = struct;
    grapes_block.time_pre   = grapes.time_pre;
    grapes_block.time_pos   = grapes.time_pos;
    grapes_block.exp_type   = grapes.exp_type;
    grapes_block.ImageNames = grapes.ImageNames;

    for scr_id=1:length(scr_config_cell)
        scr_config = scr_config_cell{scr_id};
        fns = fieldnames(grapes.rasters);
        for j=1:length(fns)
            chan = grapes.rasters.(fns{j});
            chan_fns = fieldnames(chan);
            grapes_block.rasters.(fns{j}).details = chan.details;
            for k=2:length(chan_fns)
                cluster = chan.(chan_fns{k});
                for l=1:length(scr_config.pics2use)
                    stim_idx = scr_config.pics2use(l);
                    if length(cluster.stim) < scr_config.pics2use(l)
                        fprintf('Picture %d not in %s %s \n', scr_config.pics2use(l), fns{j}, chan_fns{k});
                        grapes_block.rasters.(fns{j}).(chan_fns{k}).stim{stim_idx} = [];
                        continue
                    end
                    stim = cluster.stim{stim_idx};
                    if length(stim) == 0
                        continue
                    end
                    num_stim = 0;
                    for m=1:scr_id
                        if length(find(scr_config_cell{m}.pics2use == stim_idx)) > 0
                            num_stim = num_stim + scr_config_cell{m}.Nrep;
                        end
                    end
                    grapes_block.rasters.(fns{j}).(chan_fns{k}).stim{stim_idx} = stim(1,1:num_stim);
                end
            end
        end
        [datat, rank_config] = create_responses_data_parallel(grapes_block, scr_config.pics2use, ...
                                                              {'mu','class'},ifr_calc,resp_conf, ...
                                                              not_online_channels,priority_channels);
        [used_stims, ia, ~] = unique(datat.stim_number);
        
        experiment.ImageNames(used_stims,:).stim_trial_count = datat(ia,:).ntrials;
        if scr_id == 1 && contains(experiment.subtask, 'CategLocaliz')
            % experiment.task_pics_folder = 'C:\ReyLab\experimental_files\pics\pics_space\categ_localiz'; % ABT
            experiment.task_pics_folder = '/home/user/ReyLab/experimental_files/pics/pics_space/categ_localiz'; % ABTL
            [categ_localiz_history, img_info_table] = init_categ_localiz_history( ...
                                                      experiment, ...
                                                      used_stims, 'all');
        end
    
        datat = sort_responses_table_online(datat, priority_chs_ranking);
        [stim_best,ia,~] = unique(datat.stim_number,'stable');
        datat_best = datat(ia,:);
        datat_best = datat_best(datat_best.min_spk_test==1 & ~(datat_best.zscore<4.5) & datat_best.good_lat==1,:);

        % Reorder stim_best to have datat_best stim_numbers first
        stim_best_reordered = [stim_best(ismember(stim_best, datat_best.stim_number)); stim_best(~ismember(stim_best, datat_best.stim_number))];
            
        data_class = datat(~strcmp(datat.class, 'mu'), :);
        class_unique = unique(data_class(:, {'class', 'channel'}));
        fprintf("Single unit count block %d: %d \n", scr_id, height(class_unique));
        
        if exist('selected2notremove_cell', 'var') 
            if scr_id <= length(selected2notremove_cell)
                selected2notremove = selected2notremove_cell{scr_id};
            end
        else
            selected2notremove = [];
        end
        if save_dl_data
            datat_modified = datat;
            datat_modified.IFR = [];
            rasters = grapes_block.rasters;
            writetable(datat_modified, 'block_1_sorted_table.csv','Delimiter',',')
            writetable(grapes_block.ImageNames, 'block_1_image_info_table.csv','Delimiter',',')
            save("block_1_rasters.mat", "rasters")
            % ifr = datat.IFR;
            % save("block_1_ifr.mat", "ifr")
        end
        % data_to_plot = create_data_to_plot(grapes_block, datat, selected2notremove, selected2explore_cell, scr_id);
        data2plot = create_best_stims_table(experiment, grapes_block, datat, ...
                                                    nwin, true, priority_chs_ranking, ...
                                                    selected2notremove, selected2explore_cell, scr_id);
        
        lbl = 'select_win_dbg';
        if block_stim_popup
            [selected2explore, ~, selected2rm] = stimulus_selection_windows( ...
                                                     data2plot, grapes_block, rank_config, ...
                                                     scr_id, ifr_calc, nwin, ...
                                                     lbl, priority_chs_ranking, ...
                                                     experiment, false, false, true); 
            % We only select same_units, same_category stimuli after the
            % first and second subscreening blocks of dynamicscr, third onwards we only
            % remove stimuli
            if experiment.MANUAL_SELECT(scr_id) && ~isempty(selected2explore)
                if contains(experiment.subtask, 'DynamicScr')
                    scr_fetched_cell = get_same_unit_categ_pics(scr_id, experiment, tbl_unused_pics, ...
                                                                selected2explore, selected2notremove, ...
                                                                new_pics2load);
                elseif contains(experiment.subtask, 'CategLocaliz')
                    [scr_fetched_cell, experiment, categ_localiz_history] = get_categ_localiz_pics(scr_id, experiment, img_info_table, ...
                                                                categ_localiz_history, selected2explore);
                    grapes_block.ImageNames = experiment.ImageNames;
                    curr_fetched_pics_list = [];
                    for i = 1:numel(scr_fetched_cell)
                        rule_info = scr_fetched_cell{i};
                        categ_info_list = rule_info.categ_info;
                        for k=1:numel(categ_info_list)
                            categ_info = categ_info_list{k};
                            curr_fetched_pics_list = [curr_fetched_pics_list; categ_info.pic_ids];
                        end
                    end
                    curr_fetched_pics_list = [];
                end
            end
        else
            fc=loop_plot_responses_BCM_online(data_to_plot, ...
                                          grapes_block,scr_id,nwin,rank_config, ...
                                          ifr_calc.ejex,true,lbl);
        end       
        
    end
end

function data2plot = create_data_to_plot(grapes, datat, selected2notremove, selected2explore_cell, n_scr)
    MAX_NTRIALS = 15;
    RASTER_SIMILARITY_THR = 0.85;
    MAX_RASTERS_PER_STIM = 3;
    
    enough_trials = datat.ntrials >= MAX_NTRIALS;
    stim_rm = unique(datat(enough_trials,:).stim_number);

    if exist('selected2notremove', 'var') 
        selected2notremove = setdiff(selected2notremove,stim_rm);
    end

    datat = datat(~enough_trials, :);
    [stim_best,istim_best,~] = unique(datat.stim_number,'stable');
    %data2plot = datat(istim_best,:);

    data2plot = [];
    nwin_choose = 12;

    datat.ismu = cellfun(@(x) strcmp(x,'mu'),datat.class);
    data2plot = datat(1,:);
    prim=1;

%             if ~isempty(selected2explore_cell)
%                 while ismember(datat.stim_number(prim),cat(1,selected2explore_cell{1:n_scr-1}))
    if exist('selected2notremove', 'var')  && ~isempty(selected2notremove)
        while ismember(datat.stim_number(prim),selected2notremove)
            prim = prim+1;
        end
        data2plot = datat(prim,:);
    end

    for itable = prim+1:size(datat,1)
        if ~isempty(selected2explore_cell)
            if ismember(datat.stim_number(itable),cat(1,selected2explore_cell{1:n_scr-1}))
                continue
            end
        end
        same_cltype = datat.ismu(itable) == data2plot.ismu;
        same_ch = cellfun(@(x) strcmp(x,datat.channel{itable}),data2plot.channel);
        same_stim = datat.stim_number(itable) == data2plot.stim_number;
        if any(~same_cltype & same_ch & same_stim)
            continue
        end
        if sum(same_stim)>=MAX_RASTERS_PER_STIM
            continue
        end
        ss_dc = find(same_stim & ~same_ch);
        for ss_dc_i = 1:numel(ss_dc)
            rasters_similarty = calculate_raster_similarty(...
                grapes.rasters.(data2plot.channel{ss_dc(ss_dc_i)}).(data2plot.class{ss_dc(ss_dc_i)}).stim{data2plot.stim_number(ss_dc(ss_dc_i))},...
                grapes.rasters.(datat.channel{itable}).(datat.class{itable}).stim{datat.stim_number(itable)});
            if rasters_similarty > RASTER_SIMILARITY_THR
                continue
            end
        end
        data2plot = [data2plot; datat(itable,:)];
        if size(data2plot,1)==(nwin_choose*20) %all the needed
            break
        end
    end
end

function save_dl_raster_data(data, grapes)
    % Save data for deep learning
    datat_modified = data;
    datat_modified.IFR = [];
    rasters = grapes.rasters;
    writetable(datat_modified, 'sorted_table.csv','Delimiter',',')
    writetable(grapes.ImageNames, 'image_info_table.csv','Delimiter',',')
    save("rasters.mat", "rasters")
    ifr = data.IFR;
    % Convert to single to save space
    ifrmat = cell2mat(ifr);
    % ifrmat = single(ifrmat);
    % save("ifr.mat", "ifrmat", "-v7.3")
    save("ifr.mat", "ifrmat")
end

function add_folder_path(folder_name)

    % Search for the folder recursively across all drives
    drives = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    folder_path = '';
    
    for drive = drives
        curr_path = [drive, ':\'];
        folder_path = search_for_folder(curr_path, folder_name);
        if ~isempty(folder_path)
            break;
        end
    end
    
    % Add the folder to the path if found
    if ~isempty(folder_path)
        addpath(genpath(folder_path));
        disp(['Folder ', folder_path, ' added to the MATLAB path.']);
    else
        disp(['Folder ', folder_name, ' not found on any drive.']);
    end
end

% Recursive function to search for the folder within subfolders
function folder_path = search_for_folder(curr_path, target_folder)
    folder_path = '';
    contents = dir(curr_path);
    for i = 1:length(contents)
        if contents(i).isdir && ~strcmp(contents(i).name, '.') && ~strcmp(contents(i).name, '..')
            sub_folder_path = fullfile(curr_path, contents(i).name);
            if strcmp(contents(i).name, target_folder)
                folder_path = sub_folder_path;
                break;
            else
                folder_path = search_for_folder(sub_folder_path, target_folder);
                if ~isempty(folder_path)
                    break;
                end
            end
        end
    end
end
    