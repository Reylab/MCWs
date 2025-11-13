function [selected2explore, s2exp_block_tbl, selected2rm] = stimulus_selection_windows( ...
                                                  data2plot, grapes, rank_config, ...
                                                  n_scr, ifr_calculator, win_choose, ...
                                                  lbl, priority_chs_ranking, ...
                                                  experiment, copy2miniscrfolder, ...
                                                  show_sel_count, showwins, short_win)
    
% function [selected2explore, s2exp_block_tbl, selected2rm] = stimulus_selection_windows( ...
%                                                   data2plot, grapes, rank_config, ...
%                                                   n_scr, ifr_calculator, win_choose, ...
%                                                   lbl, priority_chs_ranking, ...
%                                                   experiment, copy2miniscrfolder, ...
%                                                   copy2dailyminiscrfolder,show_sel_count, showwins)
%     
    b_img_lbl_legend = false;
    if contains(experiment.subtask, "DynamicScr") || ...
       contains(experiment.subtask, "CategLocaliz")
        b_img_lbl_legend = true;
    end
    save_fig = ~showwins;
    if showwins
        fc=loop_plot_responses_BCM_online(data2plot, grapes,n_scr,win_choose,rank_config, ...
                                          ifr_calculator.ejex,save_fig,lbl, ...
                                          save_fig, 0, priority_chs_ranking, ...
                                          false, true, short_win, b_img_lbl_legend);
    else
        futures = parfeval(@loop_plot_responses_BCM_online, 0, ...
                                          data2plot, grapes,n_scr,win_choose,rank_config, ...
                                          ifr_calculator.ejex,save_fig,lbl, ...
                                          save_fig, 0, priority_chs_ranking, ...
                                          false, true, short_win, b_img_lbl_legend);
        wait(futures);
    end
    if showwins
        miniscr_pics_count = get_minscr_sel_count(experiment);
        sel_ix = select_stims(fc, n_scr, true, lbl, show_sel_count, miniscr_pics_count, true);
        sel_ix = sel_ix(1:min( numel(sel_ix), height(data2plot)));%remove selected out of options
        selected2rm = unique(data2plot(sel_ix==-1,:).stim_number,'stable');
        selected2explore = unique(data2plot(sel_ix==1,:).stim_number,'stable');
            
        if copy2miniscrfolder
            copy_to_miniscr_folder(experiment, selected2explore);
        end

%         if copy2dailyminiscrfolder
%             copy_to_dailyminiscr_folder(experiment, selected2explore);
%         end
    
        s2exp_block_tbl = table;
        s2exp_block_tbl.ranking = data2plot(sel_ix==1,:).ranking;
        s2exp_block_tbl.stim_number = data2plot(sel_ix==1,:).stim_number;
        s2exp_block_tbl.name = string(experiment.ImageNames.name(data2plot(sel_ix==1,:).stim_number));
        s2exp_block_tbl.channel = cellfun(@(x) grapes.rasters.(x).details.ch_label, data2plot(sel_ix==1,:).channel, 'UniformOutput', false);
        s2exp_block_tbl.class = data2plot(sel_ix==1,:).class;
        s2exp_block_tbl.concept_name = data2plot(sel_ix==1,:).concept_name;
    else
        [~, iu, ~] = unique(data2plot.stim_number, 'stable');
        best_stims_tbl = data2plot(iu, :);

        s2exp_block_tbl = table;
        s2exp_block_tbl.ranking = best_stims_tbl.ranking;
        s2exp_block_tbl.stim_number = best_stims_tbl.stim_number;
        s2exp_block_tbl.name = string(experiment.ImageNames.name(best_stims_tbl.stim_number));
        s2exp_block_tbl.channel = cellfun(@(x) grapes.rasters.(x).details.ch_label, best_stims_tbl.channel, 'UniformOutput', false);
        s2exp_block_tbl.class = best_stims_tbl.class;
        s2exp_block_tbl.concept_name = best_stims_tbl.concept_name;

        selected2explore = [];
        selected2rm = [];
    end
end

function copy_to_miniscr_folder(experiment, selected2explore_all)
    miniscr_pics_count = get_minscr_sel_count(experiment); % current miniscr folder pics count 
    temp_folder = [experiment.params.backup_path filesep 'temp'];
    if ~exist(temp_folder, 'dir')
       mkdir(temp_folder)
    end
    for s2exp=1:length(selected2explore_all)
        if s2exp + miniscr_pics_count > 40
            disp('There are 40 pics in miniscr folder, skipping copy_to_miniscr_folder');
            break;
        end
        stim_num = selected2explore_all(s2exp);
        % Copy pic to mini screening folder
        img_path = fullfile(experiment.params.pics_root_processing,experiment.ImageNames.folder{stim_num}, ...
                            experiment.ImageNames.name{stim_num});
        miniscr_folder = fullfile(experiment.params.pics_root_processing, 'miniscr_pics');
        copy_file(img_path, miniscr_folder, temp_folder);
    end
end

% function copy_to_dailyminiscr_folder(experiment, selected2explore_all)
%     temp_folder = [experiment.params.backup_path filesep 'temp'];
%     if ~exist(temp_folder, 'dir')
%        mkdir(temp_folder)
%     end
%     for s2exp=1:length(selected2explore_all)
%         stim_num = selected2explore_all(s2exp);
%         % Copy pic to mini screening folder
%         img_path = fullfile(experiment.params.pics_root_processing,experiment.ImageNames.folder{stim_num}, ...
%                             experiment.ImageNames.name{stim_num});
%         dailyminiscr_folder = fullfile(experiment.params.pics_root_processing, 'dailyminiscr_pics');
%         copy_file(img_path, dailyminiscr_folder, temp_folder);
%     end
% end

function miniscr_pics_count = get_minscr_sel_count(experiment)
    miniscr_pics_count = 0;
    % Get how many pics are already there in the miniscr folder
    miniscr_folder = fullfile(experiment.params.pics_root_processing, 'miniscr_pics');
    if exist(miniscr_folder, 'dir')
        miniscr_pics = dir(miniscr_folder);
        % Get the number of pics in the miniscr folder
        miniscr_pics_count = length(miniscr_pics)-2;
    end
end
    