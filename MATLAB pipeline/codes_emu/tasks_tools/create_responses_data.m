function [data, rank_config] = create_responses_data(grapes,selected_stim,unit_label, ...
                                                    ifr_calc,ifr_calc_blanks,resp_conf,remove_channels, ...
                                                    priority_channels, b_parallel, b_use_blanks)
% This funtion creates
% output:
% -data: a table with the responses metrics and ifr of each stimulus
% presented
% -rank_config: struct with the parameters and internal macros of the
% constructed responses
if exist('b_parallel')
    b_parallel = b_parallel;
else
    b_parallel = true;
end

tmin_base = -900;
tmax_base = -100;
FR_resol = 10; %bin size in ms
min_ifr_thr = 5; % 5 Hz

win_length = resp_conf.tmax_median - resp_conf.tmin_median;
rank_config.tmin_base = tmin_base;
rank_config.tmax_base = tmax_base;
rank_config.FR_resol = FR_resol;
rank_config.win_cent = resp_conf.win_cent;
rank_config.nstd = resp_conf.nstd;
rank_config.win_length = win_length;
rank_config.min_ifr_thr = min_ifr_thr;

b_parallel = false;
data = table;
channels = fieldnames(grapes.rasters);
channels_order = 1:numel(channels);
if exist('remove_channels','var') && ~isempty(remove_channels)
    remove_channels = arrayfun(@(x)['chan' num2str(x)],remove_channels,'UniformOutput',false);
    for i = 1:numel(remove_channels)
        keep = cellfun(@(x) ~strcmp(x,remove_channels{i}),channels);
        channels = channels(keep);
    end
end
if exist('priority_channels','var') && ~isempty(priority_channels)
    priority_channels = arrayfun(@(x)['chan' num2str(x)],priority_channels,'UniformOutput',false);
    inds_first = zeros(size(priority_channels));
    for i = 1:numel(priority_channels)
        inds_first(i) = find(cellfun(@(x) strcmp(x,priority_channels{i}),channels));
    end
    channels_order = [inds_first setdiff(1:numel(channels),inds_first,'stable')];
end

if b_parallel
    futures(1:numel(channels)) = parallel.FevalFuture;
    if numel(channels)>1
        for chi=channels_order
            futures(chi) = parfeval(@get_responses_channel_blanks,1, ...
                grapes.rasters.(channels{chi}),channels(chi), ...
                selected_stim,unit_label,ifr_calc, ifr_calc_blanks, ...
                resp_conf,rank_config, b_use_blanks);
        end
        for i=1:numel(channels)
            wait(futures(i));
            data_aux_p = fetchOutputs(futures(i));
            data = [data;data_aux_p];
        end
    else
        data=get_responses_channel_blanks(grapes.rasters.(channels{1}),channels(1), ...
            selected_stim,unit_label,ifr_calc, ifr_calc_blanks, ...
            resp_conf,rank_config, b_use_blanks);
    end
else
    for chi=channels_order
        data_aux_p=get_responses_channel_blanks(grapes.rasters.(channels{chi}), ...
            channels(chi),selected_stim,unit_label, ...
            ifr_calc, ifr_calc_blanks, resp_conf,rank_config, b_use_blanks);
        data = [data;data_aux_p];
    end
end
end

function data=get_responses_channel_blanks(chrasters, chlabel, selected_stim, ...
                                            unit_label, ifr_calc, ifr_calc_blanks, resp_conf, ...
                                            rank_config, b_use_blanks)
b_use_blanks_for_baseline = b_use_blanks;
min_trial_for_median = 5;
b_plot_blanks = true;
data=table;
classes = fieldnames(chrasters);
classes = classes(cellfun(@(x) startsWith(x, unit_label), classes));

for cli = 1:numel(classes)
    class = classes{cli};
    active_cluster_stim = chrasters.(class).stim(selected_stim);

    lstim = length(active_cluster_stim);
    if iscell(active_cluster_stim{1})
        nrows = max(cellfun(@(x) numel(x),active_cluster_stim));
    else
        [nrows, ~]  = cellfun(@size,active_cluster_stim);
    end
    sp_count_base = NaN*ones(lstim,max(nrows));
    sp_count_post = NaN*ones(lstim,max(nrows));
    p_value_sign = NaN*ones(lstim,1);
    ntrials = zeros(lstim,1);
    

    if b_use_blanks_for_baseline
        blank_on_spikes_list = [];
        if isfield(chrasters.(class), 'blank_on_onset')
            blank_on_onset = chrasters.(class).blank_on_onset;
            blank_on_spikes = chrasters.(class).blank_on_spikes;
            num_blank_spikes = count_nested_cell_elements(blank_on_spikes);
            % fprintf('num_blank_spikes: %d\n', num_blank_spikes);
            % blank_on_spikes_list = unravel_cell_array(blank_on_spikes);
        else
            blank_on_onset = {};
            blank_on_spikes = {};
        end
        b_plot_blanks = b_plot_blanks && strcmp(chlabel, 'chan368');
        if b_plot_blanks
            plot_blank_rasters(blank_on_spikes, blank_on_onset, chlabel, class)
        end
        
        IFR_blanks = cell(0);
        num_blank_spikes = 0;
        last_blank_end = 0;
        for subscr_idx = 1:numel(blank_on_spikes)
            subscr_blanks = blank_on_onset{subscr_idx};
            subscr_blanks_spks = blank_on_spikes{subscr_idx};
            
            for seq_idx = 1:numel(subscr_blanks_spks)
                seq_blanks = subscr_blanks{seq_idx};
                if iscell(subscr_blanks_spks{seq_idx})
                    seq_spks = cell2mat(subscr_blanks_spks{seq_idx});
                else
                    seq_spks = subscr_blanks_spks{seq_idx};
                end
                
                % if numel(seq_spks) == 1
                %     continue
                % end
                if numel(seq_spks) >= 1 & seq_spks ~= 9999
                    num_blank_spikes = num_blank_spikes + numel(seq_spks);
                    zero_idx_seq_spks = seq_spks - seq_blanks(1);
                    if seq_idx > 1
                        zero_idx_seq_spks = zero_idx_seq_spks + last_blank_end;
                    end
                    blank_on_spikes_list= [blank_on_spikes_list; zero_idx_seq_spks'];
                end
                last_blank_end = last_blank_end + (seq_blanks(2) - seq_blanks(1));
                ifr_blank = ifr_calc_blanks.get_ifr_pre_pos(seq_spks, seq_blanks(1), seq_blanks(2));
                % ifr_blank = ifr_calc_blanks.kde_ifr(seq_spks, seq_blanks(1), seq_blanks(2));
                IFR_blanks{end+1} = ifr_blank;
            end
        end
        % fprintf('num_blank_spikes: %d\n', num_blank_spikes);
    end
    IFR      = cell(lstim,1);
    onset    = NaN*ones(lstim,1);
    tons     = NaN*ones(lstim,1);
    dura     = NaN*ones(lstim,1);
    good_lat = false(lstim,1);

    for ist =1:lstim
        spikes1 = active_cluster_stim{ist};
        ntrials(ist) = numel(spikes1);
        % IFR{ist} = ifr_calc.get_ifr_pre_pos(spikes1, -1000, 2000);
        IFR{ist} = ifr_calc.get_ifr(spikes1);
        % IFR{ist} = ifr_calc.kde_ifr(spikes1, -1000-1, 2000+1);
    end
    
    if b_use_blanks_for_baseline
        % Stitch together blanks
        IFR_blanks_1d = [];
        % figure;
        for blank_idx=1:numel(IFR_blanks)
            % plot(IFR_blanks{blank_idx})
            % hold on
            IFR_blanks_1d = [IFR_blanks_1d; IFR_blanks{blank_idx}'];
        end
        % hold off
        IFR_blanks_length = length(IFR_blanks_1d);
        % Divide into chunks
        chunk_size = 100; % ms 
        num_chunks = floor(IFR_blanks_length/chunk_size);
        IFR_mat = cell(num_chunks,1);
        blank_on_spikes_count = zeros(num_chunks,1);
        num_blank_spikes = 0;
        for i = 1:num_chunks
            IFR_mat{i} = IFR_blanks_1d((i-1)*chunk_size+1:min(i*chunk_size, IFR_blanks_length));
            num_spikes = sum(blank_on_spikes_list >= (i-1)*chunk_size & blank_on_spikes_list <= i*chunk_size);
            blank_on_spikes_count(i) = num_spikes;
            num_blank_spikes = num_blank_spikes + num_spikes;
        end
        if num_chunks * chunk_size < IFR_blanks_length
            blank_on_spikes_count(i) = blank_on_spikes_count(i) + sum(blank_on_spikes_list >= ...
                                        i*chunk_size & blank_on_spikes_list <= IFR_blanks_length);
        end
        % fprintf('num_blank_spikes: %d\n', num_blank_spikes);
        % Plot each chunk
        % figure;
        % for i = 1:num_chunks
        %     plot(IFR_mat{i})
        %     hold on
        % end
        % hold off
        
        IFR_mat = cell2mat(IFR_mat');
        [IFR_sorted, I] = sort(IFR_mat, 2);
        idx_95 = ceil(size(IFR_mat, 2) * 0.95);
        % Remove top 5% from sorted matrix
        IFR_sorted = IFR_sorted(:, 1:idx_95);

        IFR_mat_removed = zeros(size(IFR_mat, 1), idx_95);
        for i = 1:size(IFR_mat, 1)
            row = IFR_mat(i, :);
            row(I(i,idx_95+1:end)) = [];
            IFR_mat_removed(i, :) = row;
        end

        % Calculate mean and std
        mu_IFR = mean(IFR_sorted', 'omitnan');
        std_IFR = std(IFR_sorted', 'omitnan');
        % IFR_thr = mean(mu_IFR + resp_conf.nstd*std_IFR);
        IFR_thr = mean(mu_IFR + resp_conf.nstd*std_IFR, 'omitnan');
        fprintf('%s (%s) IFR thr:%.2f Hz (%dms (%d))\n', chlabel{1}, class, IFR_thr, chunk_size, num_chunks);
        if IFR_thr < rank_config.min_ifr_thr || isnan(IFR_thr)
            IFR_thr = rank_config.min_ifr_thr;
        end

        % figure;
        % plot(mu_IFR)
        % hold on
        % plot(mu_IFR+std_IFR)
        % hold off
        % title(sprintf('Blanks %s (%s) IFR thr:%.2f Hz \n', chlabel{1}, class, IFR_thr))
    else
        IFR_mat = cell2mat(IFR);
        % mu_IFR = mean(IFR_mat);
        % std_IFR = std(IFR_mat);
        mu_IFR = mean(IFR_mat(ntrials>min_trial_for_median,:));
        std_IFR = std(IFR_mat(ntrials>min_trial_for_median,:));
        times_base = ifr_calc.ejex<rank_config.tmax_base & ifr_calc.ejex>rank_config.tmin_base;
        IFR_thr = mean(mu_IFR(times_base) + resp_conf.nstd*std_IFR(times_base));

        % figure;
        % plot(mu_IFR)
        % hold on
        % plot(mu_IFR+std_IFR)
        % hold off
        % title(sprintf('Stims %s (%s) IFR thr:%.2f Hz \n', chlabel{1}, class, IFR_thr))
    end
    
    for ist =1:lstim
        [good_lat(ist),onset(ist),dura(ist)] = get_latency_BCM( ...
                                                IFR{ist},ifr_calc.ejex, ...
                                                ifr_calc.sample_period, ...
                                                IFR_thr,resp_conf.t_down, ...
                                                resp_conf.over_threshold_time, ...
                                                resp_conf.below_threshold_time);
    end

    for ist=1:lstim
        spikes1 = active_cluster_stim{ist};
        if good_lat(ist)
            % tons(ist) = onset(ist);
            if resp_conf.win_cent
                tons(ist) = max(onset(ist) - (rank_config.win_length - min(dura(ist), ...
                    rank_config.win_length))/2,50);
            else
                tons(ist) = onset(ist) - 100;
            end
        else
            tons(ist) = resp_conf.tmin_median;
        end
        win_length = dura(ist);
        if win_length > rank_config.win_length
            win_length = rank_config.win_length;
        elseif win_length < 300
            win_length = 300;
        end
        if iscell(spikes1)
            for jj=1:numel(spikes1)
                sp_count_base(ist,jj)=sum((spikes1{jj}  < -resp_conf.tmin_median) & ...
                    (spikes1{jj} > -(resp_conf.tmin_median+win_length)));
                sp_count_post(ist,jj)=sum((spikes1{jj}< tons(ist)+win_length) & ...
                    (spikes1{jj}> tons(ist)));
            end
        else
            for jj=1:size(spikes1,1)
                sp_count_base(ist,jj)=sum((spikes1(jj,:)  < -resp_conf.tmin_median) & ...
                    (spikes1(jj,:) > -(resp_conf.tmin_median+win_length)));
                sp_count_post(ist,jj)=sum((spikes1(jj,:)  < tons(ist)+win_length) & ...
                    (spikes1(jj,:) > tons(ist)));
            end
        end
        [p_value_sign(ist),~] = signtest(sp_count_base(ist,~isnan(sp_count_base(ist,:)) & ...
                                         ~isnan(sp_count_post(ist,:))), ...
                                         sp_count_post(ist,~isnan(sp_count_base(ist,:))& ...
                                         ~isnan(sp_count_post(ist,:))),'Tail','left');
    end
    medians_base = median(sp_count_base,2,'omitnan');
    % mu_base_med = mean(medians_base);
    % std_base = std(medians_base);
    mu_base_med = mean(medians_base(ntrials>min_trial_for_median),'omitnan');
    std_base = std(medians_base(ntrials>min_trial_for_median),'omitnan');

    % With blanks
    med_idx = 2:3:length(blank_on_spikes_count)-1; % Get middle value for every triplet
    medians_base = blank_on_spikes_count(med_idx);
    mu_base_med = mean(medians_base);
    std_base = std(medians_base);

    median_post = median(sp_count_post,2,'omitnan');
    zscore = (median_post - mu_base_med)/std_base;
    min_spk_test = median_post >= resp_conf.min_spk_median;
    p_test = p_value_sign <= resp_conf.psign_thr;
    data_u = table(ntrials,IFR,onset,tons,dura,good_lat,zscore,median_post,p_value_sign,min_spk_test,p_test);
    data_u.IFR_thr = ones(size(data_u,1),1)*IFR_thr;
    data_u.stim_number = selected_stim(:);
    data_u.class = repmat({class}, size(data_u,1), 1);
    data_u.channel = repmat(chlabel, size(data_u,1), 1);
    data = [data; data_u];
end
end

function plot_blank_rasters(blank_on_spikes, blank_on_onset, chlabel, class)
% Plot blank_on_spikes for subscreenings with subplots for each subscr
figure;
sgtitle(sprintf('Blanks %s (%s)\n', chlabel{1}, class))
for subscr_idx = 1:numel(blank_on_spikes)
    subscr_blank_length = 0;
    subplot(1,numel(blank_on_spikes),subscr_idx)
    subscr_blanks = blank_on_onset{subscr_idx};
    subscr_blanks_spks = blank_on_spikes{subscr_idx};
    for seq_idx = 1:numel(subscr_blanks_spks)
        seq_blanks = subscr_blanks{seq_idx};
        if iscell(subscr_blanks_spks{seq_idx})
            seq_spks = cell2mat(subscr_blanks_spks{seq_idx});
        else
            seq_spks = subscr_blanks_spks{seq_idx};
        end
        x = seq_spks - seq_blanks(1);
        y = ones(size(x))*seq_idx;
        % scatter plot
        sz = 25;
        scatter(x, y, sz, '|')
        hold on
        blank_length = seq_blanks(2) - seq_blanks(1);
        subscr_blank_length = subscr_blank_length + blank_length;
        scatter(blank_length, seq_idx, sz, "filled")
        hold on                
    end
    hold off
    title(sprintf('subscr:%d seq count:%d(%.2fms)\n', subscr_idx, numel(subscr_blanks_spks), subscr_blank_length))
    % x limit between 0 and 750 ms
    xlim([0 750])
    ylim([0 numel(subscr_blanks_spks)+1])            
end
end

function unraveled_array = unravel_cell_array(cell_array)
    unraveled_array = [];
    for i = 1:numel(cell_array)
        if iscell(cell_array{i})
            unraveled_array = [unraveled_array, unravel_cell_array(cell_array{i})];
        else
            unraveled_array = [unraveled_array, cell_array{i}];
        end
    end
end

function num_elements = count_nested_cell_elements(nested_cell_array)
    num_elements = 0;
    stack = {nested_cell_array};
    
    while ~isempty(stack)
        current_cell = stack{end};
        stack(end) = [];
        
        for i = 1:numel(current_cell)
            if iscell(current_cell{i})
                stack{end+1} = current_cell{i};
            elseif isa(current_cell{i}, 'double') || isa(current_cell{i}, 'single') || isa(current_cell{i}, 'uint8')
                num_elements = num_elements + numel(current_cell{i});
            else
                num_elements = num_elements + 1;
            end
        end
    end
end