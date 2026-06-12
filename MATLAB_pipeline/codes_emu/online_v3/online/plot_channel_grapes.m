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
    addParameter(ipr, 'output_dir', pwd);

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
    output_dir = ipr.Results.output_dir;
    

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
    
    %% OPTIMIZATION 1: Pre-cache all images to avoid repeated disk I/O
    fprintf('Pre-caching images...\n');
    cache_time = tic;
    unique_stims = unique(data.stim_number);
    image_cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for si = 1:numel(unique_stims)
        img_idx = unique_stims(si);
        imagename = fullfile(grapes.ImageNames.folder{img_idx}, grapes.ImageNames.name{img_idx});
        if exist(imagename, 'file') && contains(lower(imagename), '.jp')
            try
                image_cache(img_idx) = imread(imagename);
            catch
                image_cache(img_idx) = [];
            end
        else
            image_cache(img_idx) = [];
        end
    end
    fprintf('Image cache built: %d images in %.1fs\n', numel(unique_stims), toc(cache_time));
    
    %% OPTIMIZATION 2: Batch jobs by channel instead of per-channel/class
    % Collect all channel jobs first, then batch them
    job_list = {}; % Each entry: {data_to_plot, channel_grapes, lbl}

    channel_grapes_base            = struct;
    channel_grapes_base.time_pre   = grapes.time_pre;
    channel_grapes_base.time_pos   = grapes.time_pos;
    channel_grapes_base.exp_type   = grapes.exp_type;
    channel_grapes_base.ImageNames = grapes.ImageNames;
    channel_grapes_base.rasters    = struct;
    if isfield(grapes,'ISI_min')
        channel_grapes_base.ISI_min   = grapes.ISI_min;
    end

    % Collect all jobs first
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
            
            channel_grapes = channel_grapes_base;
            channel_grapes.rasters.(labels{i}).(class) = grapes.rasters.(labels{i}).(class);
            data_to_plot = data(G==i & isthisclass, :);
            if ~strcmp(stim_list,'all')
                if order_by_rank
                    isstim_list = ismember(data_to_plot.stim_number, stim_list);
                    data_to_plot = data_to_plot(isstim_list, :);
                else
                    [~, idx] = ismember(stim_list, data_to_plot.stim_number);
                    idx(idx == 0) = []; 
                    data_to_plot = data_to_plot(idx, :);
                end
                
                % Check for empty AFTER both filtering methods
                if isempty(data_to_plot)
                    continue;
                end
            end
            channel_grapes.rasters.(labels{i}).details = grapes.rasters.(labels{i}).details;
            numspks = grapes.rasters.(labels{i}).details.(class);
            lbl = sprintf('EMU-%.3d_final%s_ch_%s_%s (%d spks)(ifr_t=%.1fHz)',emu_num, stim_list_all_str, ...
                  grapes.rasters.(labels{i}).details.ch_label,class, numspks, data_to_plot.IFR_thr(1));
            
            % Add to job list instead of immediate execution
            job_list{end+1} = {data_to_plot, channel_grapes, lbl};
        end
    end
    
    fprintf('Processing %d channel/class combinations...\n', numel(job_list));
    
    %% OPTIMIZATION 2: Batch parallel jobs - group multiple channels per worker
    % When saving figures, force sequential execution so paths are predictable
    if (parallel_plots && ~save_fig) && numel(job_list) > 0
        % Determine batch size based on number of workers and jobs
        pool = gcp('nocreate');
        if ~isempty(pool)
            num_workers = pool.NumWorkers;
        else
            num_workers = 4; % fallback
        end
        % Aim for ~2-4 jobs per worker for good load balancing
        batch_size = max(1, ceil(numel(job_list) / (num_workers * 3)));
        num_batches = ceil(numel(job_list) / batch_size);
        
        fprintf('Batching into %d jobs (batch_size=%d)\n', num_batches, batch_size);
        
        futures = parallel.FevalFuture.empty(0, 1);
        for bi = 1:num_batches
            batch_start = (bi-1) * batch_size + 1;
            batch_end = min(bi * batch_size, numel(job_list));
            batch_jobs = job_list(batch_start:batch_end);
            
            futures(bi) = parfeval(@process_batch_jobs, 0, batch_jobs, ...
                n_scr, nwins2plot, rank_config, ifr_x, save_fig, close_fig, ...
                order_offset, priority_chs_ranking, image_cache, output_dir);
        end
        wait(futures);
    else
        % Sequential execution with image cache
        for ji = 1:numel(job_list)
            job = job_list{ji};
            loop_plot_responses_BCM_online(job{1}, job{2}, ...
                n_scr, nwins2plot, rank_config, ...
                ifr_x, save_fig, job{3}, close_fig, ...
                order_offset, priority_chs_ranking, true, ...  
                false, false, false, 'image_cache',image_cache,'output_dir', output_dir);
        end
    end
    tot_time = toc(begin_time);
    fprintf("Channel grapes: END (%0.2f seconds)\n", tot_time)
end

function process_batch_jobs(batch_jobs, n_scr, nwins2plot, rank_config, ifr_x, ...
                            save_fig, close_fig, order_offset, priority_chs_ranking, image_cache, output_dir)
    % Process a batch of channel/class jobs sequentially within this worker
    for ji = 1:numel(batch_jobs)
        job = batch_jobs{ji};
        data_to_plot = job{1};
        channel_grapes = job{2};
        lbl = job{3};
        
        loop_plot_responses_BCM_online(data_to_plot, channel_grapes, ...
            n_scr, nwins2plot, rank_config, ...
            ifr_x, save_fig, lbl, close_fig, ...
            order_offset, priority_chs_ranking, true, ...
            false, false, false, image_cache, output_dir);
    end
end

function remove_existing_ch_plots(emu_num, grapes, labels, stim_list_all_str)
    % Delete existing channel plots
    for i=1:numel(labels)
        lbl = sprintf('EMU-%.3d_final%s_ch_%s',emu_num, stim_list_all_str, ...
                      grapes.rasters.(labels{i}).details.ch_label);
        delete([lbl '*.png']);
    end
end