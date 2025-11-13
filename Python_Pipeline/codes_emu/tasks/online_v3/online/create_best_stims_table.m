function data2plot = create_best_stims_table(experiment, grapes, datat, ...
                                             nwin_choose, group_stims, priority_chs_ranking, ...
                                             selected2notremove, selected2explore_cell, n_scr, is_online)
    RASTER_SIMILARITY_THR = 0.85;
    MAX_RASTERS_PER_STIM = 3;

    data2plot = [];
    ranking_list = [];

    datat.ismu = cellfun(@(x) strcmp(x,'mu'),datat.class);
    data2plot = datat(1,:);
    ranking_list = [1];
    prim=1;

    if exist('selected2notremove', 'var') && ~isempty(selected2notremove)
        while prim < length(datat.stim_number) && ismember(datat.stim_number(prim),selected2notremove)
            prim = prim+1;
        end
        if prim ~= length(datat.stim_number)
            data2plot = datat(prim,:);
            ranking_list = [prim];
        end
    end

    for itable = prim+1:size(datat,1)
        if exist('selected2explore_cell', 'var') && ~isempty(selected2explore_cell)
            if ismember(datat.stim_number(itable),cat(1,selected2explore_cell{1:n_scr-1}))
                continue
            end
        end
        if any("selectable" == string(experiment.ImageNames.Properties.VariableNames))
            if experiment.ImageNames(datat.stim_number(itable), :).selectable == -1
                continue
            end
        end
        if ~ismember(str2num(datat.channel{itable}(end-2:end)), priority_chs_ranking) && datat.onset(itable) > 600
            continue
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
        ranking_list = [ranking_list;itable];
        if size(data2plot,1)==(nwin_choose*20) %all the needed
            break
        end
    end
    data2plot.ranking = ranking_list;
    data2plot.name = string(experiment.ImageNames.name(data2plot.stim_number));
    data2plot.concept_name = string(experiment.ImageNames.concept_name(data2plot.stim_number));
    % Set concept_number = 1 if experiment.subtask is 'DynamicSeman' or 'CategLocaliz'
    if isfield(experiment, 'subtask') && ...
       (contains(experiment.subtask, 'Seman') || ...
        contains(experiment.subtask, 'CategLocaliz'))
        data2plot.concept_number = repmat("1", height(data2plot), 1);
    else
        data2plot.concept_number = string(experiment.ImageNames.concept_number(data2plot.stim_number));
    end
    
    % Display how many unique best stims were found
    [data2plot_unique, ~, io] = unique(data2plot.concept_name, 'stable');       
    fprintf('Unique best stims found: %d\n', height(data2plot_unique));

    if group_stims && is_online     
        tbl_tmp = table;
        tbl_tmp.io = io;
        tbl_tmp.concept_number = data2plot.concept_number;
        [~, sorted_idxs] = sortrows(tbl_tmp, {'io','concept_number'});
        
        % Get same stimulus together
        % [~,sorted_idxs] = sort(io);
        % Reorder data2plot
        data2plot = data2plot(sorted_idxs, :);
    end
end