function create_early_figures(early_results_path, available_trials,experiment,detecctions,...
    USE_PHOTODIODE,Event_Time_pdiode,Event_Time,Event_Value, init_times,...
    channels,priority_channels, nstd2plots,chan_label,TIME_PRE,TIME_POS,...
    classes_out, ch_ix_out)

    outputinfo = {};

    sp_index = cell(size(detecctions,2),1);

    valid_trials = [];
    pics_onset = [];
    for ntrial = 1:available_trials

        if USE_PHOTODIODE
            [complete_times, text_out] = fix_photodiode_times(Event_Time_pdiode{ntrial},ntrial,experiment);
            if isempty(complete_times)
                text_out{end+1} = sprintf('Using DAC events in trial: %d',ntrial);
                [complete_times, text_out{end+1}] = fix_onset_times(Event_Time{ntrial},Event_Value{ntrial} ,ntrial,experiment);
            end    
        else
            [complete_times, text_out] = fix_onset_times(Event_Time{ntrial},Event_Value{ntrial} ,ntrial,experiment);
        end

        if ~isempty(text_out)
            outputinfo = [outputinfo, text_out];
        end

        if ~isempty(complete_times)
            pics_onset = [pics_onset , complete_times];
        else
            continue
        end

        valid_trials(end+1) = ntrial;
        for j=1:size(detecctions,2)
            sp_index{j} =  [sp_index{j}, (double(detecctions{ntrial,j})+init_times{ntrial}{1}(j))/30]; %to ms
        end
    end
    final_Nseq = length(valid_trials);
    [seq_length,NISI,~] = size(experiment.order_pic);
    pics_onset = reshape(pics_onset,seq_length,NISI,final_Nseq);
    %%
    clear detecctions;
    %%
    stimulus = create_stimulus_online(experiment.order_pic(:,:,valid_trials),NISI,...
        experiment.ImageNames,experiment.ISI,experiment.order_ISI(:,valid_trials));

    %creates grapes
    grapes = struct;
    grapes.exp_type = 'RSVPSCR'; 
    grapes.time_pre = TIME_PRE;
    grapes.time_pos = TIME_POS;

    grapes.ImageNames = cell(length(stimulus),1);
    for j=1:length(stimulus)                      %loop over stimuli
        grapes.ImageNames{j} = stimulus(j).name;
    end

    grapes = update_grapes(grapes,pics_onset,stimulus,sp_index,channels,chan_label,true,[],[]);


    [~,~,ms_id] = mkdir(early_results_path);
    cd(early_results_path)
    %charlar si borra todo o no aca

    sorted_channles = [priority_channels setdiff(channels,priority_channels,'stable')];

    if ~isempty(ch_ix_out)
        grapes = update_grapes(grapes,pics_onset,stimulus,sp_index(ch_ix_out),channels(ch_ix_out),chan_label(ch_ix_out),false,classes_out,[]);
        % loop_plot_best with muonly=='n' plot classes as well
        loop_plot_best_responses_BCM_rank_online(channels(ch_ix_out),grapes,'n',[],[],[],1,nstd2plots,[],false);
    end
    loop_plot_best_responses_BCM_rank_online(sorted_channles,grapes,'y',[],[],[],1,nstd2plots,[],false);


end