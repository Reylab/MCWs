function plot_channel_grapes(varargin)

    ipr = inputParser;
    addParameter(ipr, 'channels2plot','all');
    addParameter(ipr, 'stim_list', 'all');
    addParameter(ipr, 'order_by_rank', true, @islogical);
    addParameter(ipr, 'data',{});
    addParameter(ipr, 'grapes', struct);
    addParameter(ipr, 'n_scr', 2);
    addParameter(ipr, 'nwins2plot', 2);
    addParameter(ipr, 'rank_config', 2);
    addParameter(ipr, 'ifr_x', 2);
    addParameter(ipr, 'save_fig', true, @islogical);
    addParameter(ipr, 'emu_num', 2);
    addParameter(ipr, 'close_fig', true, @islogical);
    addParameter(ipr, 'order_offset', 2);
    addParameter(ipr, 'priority_chs_ranking', 2);
    addParameter(ipr, 'parallel_plots', true, @islogical);    
    addParameter(ipr, 'extra_lbl', '');    

    % Check if varargin exists or is empty
    if ~isempty(varargin)
        parse(ipr,varargin{:});
    else
        parse(ipr);
    end
    channels2plot = ipr.Results.channels2plot;
    stim_list = ipr.Results.stim_list;
    order_by_rank = ipr.Results.order_by_rank;
    data = ipr.Results.data;
    grapes = ipr.Results.grapes;
    n_scr = ipr.Results.n_scr;
    nwins2plot = ipr.Results.nwins2plot;
    rank_config = ipr.Results.rank_config;
    ifr_x = ipr.Results.ifr_x;
    save_fig = ipr.Results.save_fig;
    emu_num = ipr.Results.emu_num;
    close_fig = ipr.Results.close_fig;
    order_offset = ipr.Results.order_offset;
    priority_chs_ranking = ipr.Results.priority_chs_ranking;
    parallel_plots = ipr.Results.parallel_plots;
    extra_lbl = ipr.Results.extra_lbl;
    

    begin_time = tic;
    disp('Channel grapes: BEGIN')
    if ~isempty(extra_lbl)
        stim_list_all_str = ['_' extra_lbl];
    else
        stim_list_all_str = '';
    end
    if strcmp(stim_list,'all')
        % Remove rows where its non priority channels & latency(onset) is >600
        data = data(~(cellfun(@(x) ~ismember(str2num(x(end-2:end)),priority_chs_ranking),data.channel) & data.onset > 600),:);
        stim_list_all_str = [stim_list_all_str '_' stim_list];
    else
        if order_by_rank
            stim_list_all_str = [stim_list_all_str '_r'];
        else
            stim_list_all_str = [stim_list_all_str ''];
        end
    end
    [G, labels] = findgroups(data.channel);
    if ~strcmp(channels2plot,'all')
        sel_chs = cellfun(@(x) any(arrayfun(@(chnum) strcmp(x,['chan' num2str(chnum)]),channels2plot)),labels);
        labels = labels(sel_chs);
        fsch = find(sel_chs);
        newG = zeros(size(G))+numel(labels)+1;
        for gi = 1:numel(fsch)
            newG(G ==fsch(gi))=gi;
        end
        G = newG;
        remove_existing_ch_plots(emu_num, grapes, labels, stim_list_all_str);
    else
        remove_existing_ch_plots(emu_num, grapes, labels, stim_list_all_str);
    end
    % Iniitalize futures for plotting in parallel
    if parallel_plots
        futures(1:numel(labels))  = parallel.FevalFuture;
    end

    channel_grapes            = struct;
    channel_grapes.time_pre   = grapes.time_pre;
    channel_grapes.time_pos   = grapes.time_pos;
    channel_grapes.exp_type   = grapes.exp_type;
    channel_grapes.ImageNames = grapes.ImageNames;
    channel_grapes.rasters    = struct;

    for i=1:numel(labels)
        classes = fieldnames(grapes.rasters.(labels{i}));
        classes = classes(cellfun(@(x) startsWith(x, {'mu','class'}), classes));
        for rn = 1: numel(classes)
            class = classes{rn};
            isthisclass = cellfun(@(x) strcmp(x,class),data.class);
            
             % Skip this class if it has no matching entries
            if sum(isthisclass) == 0
                 continue;
            end
            channel_grapes.rasters.(labels{i}).(class) = grapes.rasters.(labels{i}).(class);
            data_to_plot = data(G==i & isthisclass, :);
            if ~strcmp(stim_list,'all')
                if order_by_rank
                    isstim_list = ismember(data_to_plot.stim_number, stim_list);
                    data_to_plot = data_to_plot(isstim_list, :);
                else
                    [~, idx] = ismember(stim_list, data_to_plot.stim_number);
                    data_to_plot = data_to_plot(idx, :);
                end
            end
            channel_grapes.rasters.(labels{i}).details = grapes.rasters.(labels{i}).details;
            numspks = grapes.rasters.(labels{i}).details.(class);
            lbl = sprintf('EMU-%.3d_final%s_ch_%s_%s (%d spks)(ifr_t=%.1fHz)',emu_num, stim_list_all_str, ...
                  grapes.rasters.(labels{i}).details.ch_label,class, numspks, data_to_plot.IFR_thr(1));
            if parallel_plots
                futures = parfeval(@loop_plot_responses_BCM_online, 0, ...
                                          data_to_plot, channel_grapes, ...
                                          n_scr, nwins2plot, rank_config, ...
                                          ifr_x, save_fig, lbl, close_fig,  ...
                                          order_offset, priority_chs_ranking, true);
            else
                loop_plot_responses_BCM_online(data_to_plot, channel_grapes, ...
                                          n_scr, nwins2plot, rank_config, ...
                                          ifr_x, save_fig, lbl, close_fig, ...
                                          order_offset, priority_chs_ranking, true);
            end
            channel_grapes.rasters = struct;
        end
    end
    if parallel_plots
        wait(futures);
    end
    %futures = futures([]);
    tot_time = toc(begin_time);
    fprintf("Channel grapes: END (%0.2f seconds)\n", tot_time)
end

function remove_existing_ch_plots(emu_num, grapes, labels, stim_list_all_str)
    % Delete existing channel plots
    for i=1:numel(labels)
        lbl = sprintf('EMU-%.3d_final%s_ch_%s',emu_num, stim_list_all_str, ...
                      grapes.rasters.(labels{i}).details.ch_label);
        delete([lbl '*.png']);
    end
end