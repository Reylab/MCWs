function [figs, df_metrics, SS] = make_cluster_report(data, varargin)
    % MAKE_CLUSTER_REPORT - MATLAB port to match Python plotting layout
    p = inputParser;
    addParameter(p, 'calc_metrics', false, @islogical);
    addParameter(p, 'metrics_df', [], @(x) isempty(x) || istable(x));
    addParameter(p, 'SS', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'exclude_cluster_0', true, @islogical);
    addParameter(p, 'samplerate_hz', [], @isscalar);
    addParameter(p, 'clusters_per_page', 6, @isscalar);
    addParameter(p, 'bin_duration_ms', 60000.0, @isscalar);
    addParameter(p, 'refractory_ms', 3.0, @isscalar);
    addParameter(p, 'n_neighbors', 5, @isscalar);
    addParameter(p, 'max_waveforms_per_cluster', 1000, @isscalar);
    addParameter(p, 'show_figures', false, @islogical);
    parse(p, varargin{:});

    % --- Setup ---
    visstr = 'off'; % Default to off (prevents pop-ups)
    
    if ~isstruct(data) || ~isfield(data,'cluster_class') || ~isfield(data,'spikes') || ~isfield(data,'inspk')
        error('data must contain cluster_class, spikes, and inspk');
    end
    has_spikes_all = isfield(data,'spikes_all') && isfield(data,'index_all');
    cluster_class = data.cluster_class;

    if isfield(data,'cluster_class_pre_rescue')
        cluster_class_pre = data.cluster_class_pre_rescue;
        cluster_ids_pre = int32(cluster_class_pre(:,1));
    end

    waveforms = double(data.spikes);
    features = double(data.inspk);
    cluster_ids = int32(cluster_class(:,1));


    spike_times_ms = double(cluster_class(:,2));
    unique_clusters = unique(cluster_ids);
    recording_duration_ms = max(spike_times_ms);
    if has_spikes_all
        spikes_all = data.spikes_all;
        index_all = data.index_all;
    else
        spikes_all = [];
        index_all = [];
    end

    if isfield(data,'mask_nonart')
        mask_bund = ~data.mask_nonart;
    else
        mask_bund = false(size(index_all));
    end

    if isfield(data,'mask_non_quarantine')
        mask_art_chan = ~data.mask_non_quarantine;
    else
        mask_art_chan = false(size(index_all));
    end    
    
    if isfield(data,'mask_taskspks')
        mask_tsk = ~data.mask_taskspks;
    else
        mask_tsk = false(size(index_all));
    end

    mask_quarantine = mask_art_chan(:) | mask_bund(:) | mask_tsk(:);
    tot_quar = sum(mask_quarantine);
    if has_spikes_all
        tot_spikes = numel(index_all);
    else
        % No pre-quarantine spike record available; use clustered spike count instead.
        tot_spikes = numel(cluster_ids);
    end

    rescued = 0;
    if isfield(data, 'rescue_mask') && ~isempty(data.rescue_mask)
        rescued = sum(data.rescue_mask);
    end
    final_count = tot_spikes - tot_quar + rescued;

    % Store as fields in data for passing to overview page
    data.spike_summary = struct('total', tot_spikes, 'quarantined', tot_quar, 'rescued', rescued, 'final', final_count);

    % Extract forced  if available (indicates which spikes were force-classified)
    % Match forced to pre-rescue cluster_ids if rescue happened
    if isfield(data,'cluster_class_pre_rescue') && isfield(data, 'forced') && ~isempty(data.forced)
        % Rescue happened: forced should match pre-rescue spike count
        forced = logical(data.forced);
        forced = forced(:);
        if numel(forced) ~= numel(cluster_ids_pre)
            % If sizes still don't match, fallback to false array
            forced = false(size(cluster_ids_pre));
        end
    elseif isfield(data, 'forced') && ~isempty(data.forced)
        % No rescue: forced should match post-rescue spike count
        forced = logical(data.forced);
        forced = forced(:);
        if numel(forced) ~= numel(cluster_ids)
            forced = false(size(cluster_ids));
        end
    else
        % No forced information: default based on what exists
        if isfield(data,'cluster_class_pre_rescue')
            forced = false(size(cluster_ids_pre));
        else
            forced = false(size(cluster_ids));
        end
    end

    if p.Results.calc_metrics
        [df_metrics, SS] = compute_cluster_metrics(data, 'exclude_cluster_0', p.Results.exclude_cluster_0, ...
            'n_neighbors', p.Results.n_neighbors, 'bin_duration', p.Results.bin_duration_ms, 'make_plots', false);
    else
        df_metrics = p.Results.metrics_df;
        SS = p.Results.SS;
        if isempty(df_metrics)
            error('If calc_metrics=false, you must provide metrics_df');
        end
        % SS may legitimately be empty (e.g. only 1-2 clusters after filtering,
        % silhouette undefined). The metrics-overview page is skipped further
        % down in that case; waveform/ISI plots still get generated.
    end
    figs = {};
    leicolors = [0 0 0; 0 0 1; 1 0 0; 0 0.5 0; 0.62 0 0; 0.42 0 0.76; 0.97 0.52 0.03; 0.52 0.25 0; 1 0.10 0.72; 0.55 0.55 0.55; 0.59 0.83 0.31; 0.97 0.62 0.86; 0.62 0.76 1.0];
    
    % --- Paging Logic (4 clusters per page) ---
    clusters_per_page = 4;
    pages = {};
    K = numel(unique_clusters);
    for i = 1:clusters_per_page:K
        pages{end+1} = unique_clusters(i:min(i+clusters_per_page-1,K));
    end

    % Compute shared y-limits from page 1 real clusters (exclude cluster 0).
    % This shared scale is used on all pages for nonzero clusters.
    ylimit_shared = [];
    if ~isempty(pages)
        page1_clusters_for_limits = pages{1};
        page1_clusters_for_limits = page1_clusters_for_limits(page1_clusters_for_limits ~= 0);
        for page1_c_idx = 1:numel(page1_clusters_for_limits)
            cid_temp = page1_clusters_for_limits(page1_c_idx);
            mask_temp = (cluster_ids == cid_temp);
            W_temp = waveforms(mask_temp, :);
            if ~isempty(W_temp)
                ylim_temp = [min(W_temp(:)), max(W_temp(:))];
                if isempty(ylimit_shared)
                    ylimit_shared = ylim_temp;
                else
                    ylimit_shared = [min(ylimit_shared(1), ylim_temp(1)), max(ylimit_shared(2), ylim_temp(2))];
                end
            end
        end
    end
    if ~isempty(ylimit_shared)
        y_padding = 0.1 * (ylimit_shared(2) - ylimit_shared(1));
        ylimit_shared = [ylimit_shared(1) - y_padding, ylimit_shared(2) + y_padding];
    end

    % --- Summary + per-page plotting ---
    for page_i = 1:length(pages)
        page = pages{page_i};

        % Reorder page 1: put cluster 0 at the END (rightmost position on page 1)
        if page_i == 1
            idx_0 = find(page == 0, 1);
            if ~isempty(idx_0)
                page(idx_0) = []; 
                page = [page(:); 0];  % Append cluster 0 while preserving vector shape
            end
        end

        ylimit = ylimit_shared;

        nclusters_on_page = numel(page);
        show_summary_col = (page_i == 1);
        ncols = clusters_per_page + double(show_summary_col);

        if p.Results.show_figures
            visstr = 'on';
        end
        fig = figure('Visible', visstr, 'Units','normalized','OuterPosition',[0 0 1 1], ...
                    'PaperUnits', 'inches', 'PaperType', '<custom>', 'PaperSize', [16 8], 'PaperPosition', [0 0 16 8], 'PaperPositionMode', 'manual', ...
                    'RendererMode', 'manual', 'Renderer', 'painters');

        % grid 3 rows x ncols (summary column only on page 1)
        for col = 1:ncols
            if show_summary_col && col == 1
                % --- SUMMARY COLUMN ---
                ax1 = report_subplot(3, ncols, 1);
                hold(ax1,'on');
                T = size(waveforms,2);
                
                % FIX 1: Summary Mean Plot -> SAMPLES (ignore samplerate)
                tvec = 1:T;
                xlabel_str = 'Samples';
                
                tot_clust_spikes = numel(cluster_ids);
                for k = 1:numel(unique_clusters)
                    c = unique_clusters(k);
                    Wc = waveforms(cluster_ids==c, :);
                    if isempty(Wc), continue; end
                    colc = leicolors(mod(c, size(leicolors,1))+1,:);
                    plot(ax1, tvec, mean(Wc,1), 'Color', colc, 'LineWidth', 1.8);
                end
                
                rescued_all = 0;
                if isfield(data, 'rescue_mask') && ~isempty(data.rescue_mask)
                    rescued_all = sum(data.rescue_mask);
                end
                
                if rescued_all > 0
                    title(ax1, sprintf('Means (clust n = %d) tot.=%d R=%d', tot_clust_spikes, tot_spikes, rescued_all));
                else
                    title(ax1, sprintf('Means (clustered n = %d) total=%d', tot_clust_spikes, tot_spikes));
                end
                xlabel(ax1, xlabel_str); ylabel(ax1,'Amplitude'); grid(ax1,'on');
                
                % Style Update (Match Python)
                box(ax1, 'off');
                set(ax1, 'GridAlpha', 0.25, 'LineWidth', 0.8);

                % presence KDE (row2)
                ax2 = report_subplot(3, ncols, ncols+1);
                cluster_activity_kde_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax2, 100, 'inferno');

                % SNR bar (row3)
                ax3 = report_subplot(3, ncols, 2*ncols+1);
                if istable(df_metrics) && all(ismember({'cluster_id','snr'}, df_metrics.Properties.VariableNames))
                    [lia, ~] = ismember(df_metrics.cluster_id, unique_clusters);
                    table_clusters = df_metrics.cluster_id(lia);
                    table_snrs = df_metrics.snr(lia);
                    
                    [~, color_indices] = ismember(table_clusters, unique_clusters);
                    bar_colors = leicolors(mod(table_clusters, size(leicolors,1))+1, :);

                    h = bar(ax3, 1:numel(table_clusters), table_snrs, 'FaceColor', 'flat', 'BarWidth', 0.9);
                    h.CData = bar_colors; 
                    
                    set(ax3, 'XTick', 1:numel(table_clusters), 'XTickLabel', arrayfun(@(c)sprintf('C%d',c), table_clusters, 'UniformOutput',false), 'XTickLabelRotation',90);ylabel(ax3,'SNR'); title(ax3,'SNR by cluster');
                    
                    box(ax3, 'off');
                    grid(ax3, 'on');
                    set(ax3, 'GridAlpha', 0.25, 'LineWidth', 0.8, 'XGrid', 'off');
                else
                    text(ax3,0.5,0.5,'metrics_df missing (cluster_id,snr)','HorizontalAlignment','center');
                    axis(ax3,'off');
                end
            else
                page_col_idx = col - double(show_summary_col);

                % Leave empty subplots for unused cluster positions on this page
                if page_col_idx > nclusters_on_page
                    axEmpty1 = report_subplot(3, ncols, col);
                    axis(axEmpty1, 'off');
                    axEmpty2 = report_subplot(3, ncols, col + ncols);
                    axis(axEmpty2, 'off');
                    axEmpty3 = report_subplot(3, ncols, col + 2*ncols);
                    axis(axEmpty3, 'off');
                    continue;
                end

                % --- CLUSTER COLUMNS ---
                cid = page(page_col_idx);
                
                % waveform panel (row1)
                axW = report_subplot(3, ncols, col);
                if isfield(data,'rescue_mask')
                    mask = (cluster_ids_pre==cid);
                else
                    mask = (cluster_ids==cid);
                end
                    W = waveforms(mask,:);
                if isempty(W)
                    axis(axW,'off'); continue;
                end
                n_here = size(W,1);
                take = min(p.Results.max_waveforms_per_cluster, n_here);
                rng = RandStream('mlfg6331_64'); % Use fixed-seed stream
                if n_here > take
                    idx = randperm(rng, n_here, take);
                else
                    idx = 1:n_here;
                end
                
                hold(axW,'on');
                colc = leicolors(mod(cid, size(leicolors,1))+1,:);
                
                % FIX 2: Individual Cluster Plots -> SAMPLES (ignore samplerate)
                tvec = 1:size(W,2);
                xlabel_str = 'Samples';
                
                plot(axW, tvec, W(idx,:)', 'Color', [colc, 0.15], 'LineWidth', 0.8); 
                plot(axW, tvec, mean(W,1), 'Color', 'k', 'LineWidth', 2.4);
                
                % Calculate unforced spike count (spikes that weren't force-classified)
                n_unforced = sum(mask(:) & ~forced(:));
                
                % Updated title format matching Do_clustering
                tot_cluster_spikes = sum(cluster_ids == cid);
                
                % Build merge annotation for column header
                merge_tag = get_merge_tag(double(cid), data);

                has_rescued = isfield(data, 'rescue_mask') && ~isempty(data.rescue_mask) && has_spikes_all && cid ~= 0;
                if has_rescued
                    cluster_spike_times = spike_times_ms(cluster_ids == cid);
                    rescued_here = 0;
                    for st = cluster_spike_times(:)'
                        idx_in_all = find(index_all == st, 1);
                        if ~isempty(idx_in_all) && idx_in_all <= numel(data.rescue_mask)
                            if data.rescue_mask(idx_in_all)
                                rescued_here = rescued_here + 1;
                            end
                        end
                    end
                    title(axW, sprintf('Cluster %d%s: # %d (%d) R: %d', cid, merge_tag, tot_cluster_spikes, n_unforced, rescued_here), 'FontSize', 10);
                else
                    title(axW, sprintf('Cluster %d%s: # %d (%d)', cid, merge_tag, tot_cluster_spikes, n_unforced), 'FontSize', 10);
                end
                
                xlabel(axW, xlabel_str);
                ylabel(axW, 'Amplitude');
                xlim(axW, [1, size(W,2)]);
                if size(W,2) >= 20
                    set(axW, 'XTick', 20, 'XTickLabel', {'20'});
                else
                    set(axW, 'XTick', size(W,2), 'XTickLabel', {sprintf('%d', size(W,2))});
                end
                grid(axW, 'on');
                box(axW, 'off');
                set(axW, 'GridAlpha', 0.25, 'LineWidth', 0.8);
                
                % Cluster 0 is noise, so scale it by itself instead of the page.
                if cid == 0
                    ylimit_noise = [min(W(:)), max(W(:))];
                    if ylimit_noise(1) == ylimit_noise(2)
                        ylimit_noise = ylimit_noise + [-1, 1];
                    else
                        y_padding_noise = 0.1 * (ylimit_noise(2) - ylimit_noise(1));
                        ylimit_noise = [ylimit_noise(1) - y_padding_noise, ylimit_noise(2) + y_padding_noise];
                    end
                    ylim(axW, ylimit_noise);
                elseif ~isempty(ylimit)
                    ylim(axW, ylimit);
                end

                % Density image (row2) - match Python layout
                axD = report_subplot(3, ncols, col + ncols);
                
                % FIX 3: Density Plot -> SAMPLES (pass [] for samplerate)
                density_image_matlab(W, axD, [], 'cmap', 'inferno');
                
                % ISI histogram (row3)
                axI = report_subplot(3, ncols, col + 2*ncols);
                plot_isi_histogram(axI, spike_times_ms(mask), p.Results.refractory_ms, 60, colc);
            end
        end

        % --- Build two-line sgtitle ---
        % Line 1: channel ID + page x/n
        chan_str = '';
        if isfield(data, 'filename') && ~isempty(data.filename)
            % Strip 'times_' prefix first
            name_clean = regexprep(data.filename, '^times[_\-]*', '', 'ignorecase');
            tok_ch = regexp(name_clean, '[cC][hH](\d+)', 'tokens', 'once');
            if isempty(tok_ch)
                % Look for trailing _NN or space NN (e.g., mRAMY04 raw_333)
                tok_ch = regexp(name_clean, '[_\s](\d+)(?:\.mat)?$', 'tokens', 'once');
            end
            if isempty(tok_ch)
                % Fallback: grab the first numbers we see
                tok_ch = regexp(name_clean, '(\d+)', 'tokens', 'once');
            end
            if ~isempty(tok_ch)
                chan_str = sprintf(' | Ch%s', tok_ch{1});
            end
        end
        title_line1 = sprintf('Cluster Report%s    Page %d/%d', chan_str, page_i, length(pages));

        % Line 2: directory up to and including pwd (trim prefix above pwd)
        if isfield(data, 'fullpath') && ~isempty(data.fullpath)
            file_dir = fileparts(data.fullpath);
        else
            file_dir = pwd;
        end
        % Show up to 3 trailing path components of the file's directory so that
        % for merged data the times folder is always visible:
        %   &/times_20260615_1623/ch333_merge[3_6]
        % and for normal data:
        %   &/times_20260615_1623
        parts = strsplit(file_dir, filesep);
        parts = parts(~cellfun(@isempty, parts));
        n_show = min(3, numel(parts));
        if n_show > 0
            dir_display = ['&/' strjoin(parts(end-n_show+1:end), '/')];
        else
            dir_display = file_dir;
        end
        title_line2 = dir_display;

        sgtitle(sprintf('%s\n%s', title_line1, title_line2), 'FontSize', 12, 'Interpreter', 'none');
        figs{end+1} = fig;
    end

    % Metrics overview page
    if ~isempty(df_metrics) && ~isempty(SS)
        % Compute mean waveforms per cluster present in df_metrics and pass
        cluster_list_for_metrics = double(df_metrics.cluster_id(:));
        mean_wfs = NaN(numel(cluster_list_for_metrics), size(waveforms,2));
        for kk = 1:numel(cluster_list_for_metrics)
            cid = cluster_list_for_metrics(kk);
            Wc = waveforms(cluster_ids == cid, :);
            if ~isempty(Wc)
                mean_wfs(kk, :) = mean(Wc, 1);
            end
        end

        fig_metrics = create_metrics_overview_page(df_metrics, SS, visstr, leicolors, mean_wfs, cluster_list_for_metrics, data);
        figs{end+1} = fig_metrics;
    end

    % --- Correlogram page(s) ---
    % Build correlograms between each pair of clusters and put them on one page
    try
        % choose clusters to include (respect exclude_cluster_0)
        if p.Results.exclude_cluster_0
            cluster_list = unique_clusters(unique_clusters ~= 0);
        else
            cluster_list = unique_clusters;
        end
        Kc = numel(cluster_list);
        if Kc >= 1
            % Create Kc x Kc grid: diagonal = auto-correlograms, off-diagonal = cross-correlograms
            ncols = Kc;
            nrows = Kc;
            fig_corr = figure('Visible', visstr, 'Units','normalized','OuterPosition',[0 0 1 1], ...
                             'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperOrientation', 'landscape', 'PaperPositionMode', 'auto', ...
                             'RendererMode', 'manual', 'Renderer', 'painters');
            for i = 1:Kc
                for j = 1:Kc
                    ax = subplot(nrows, ncols, (i-1)*ncols + j);
                    a = cluster_list(i);
                    b = cluster_list(j);
                    times_a = spike_times_ms(cluster_ids == a);
                    times_b = spike_times_ms(cluster_ids == b);
                    if isempty(times_a) || isempty(times_b)
                        axis(ax,'off'); continue;
                    end
                    [lags, counts] = compute_cross_correlogram(times_a, times_b, 1.0, 50.0); % bin=1ms, maxlag=50ms
                    if i == j
                        bar_color = [0.05 0.05 0.4]; % Darker blue for auto-correlograms
                    else
                        bar_color = [0.2 0.2 0.7]; % Lighter blue for cross-correlograms
                    end
                    bar(ax, lags, counts, 'FaceColor', bar_color, 'EdgeColor','none');
                    xlabel(ax,'Lag (ms)', 'FontSize', 8); 
                    ylabel(ax,'Count', 'FontSize', 8);
                    if i == j
                        title(ax, sprintf('C%d (auto)', a), 'FontSize', 9, 'FontWeight', 'bold');
                    else
                        title(ax, sprintf('C%d � C%d', a, b), 'FontSize', 9);
                    end
                    box(ax,'off'); grid(ax,'on');
                    set(ax, 'FontSize', 7);
                end
            end
            sgtitle(sprintf('Correlograms (diagonal: auto, off-diagonal: cross)', Kc), 'FontSize', 14);
            figs{end+1} = fig_corr;
        end
    catch ME_corr
        warning('Failed to generate correlograms: %s', ME_corr.message);
    end

    % --- Cluster Breakdown Page(s) ---
    % Appended after correlograms; page numbers continue from previous count.
    try
        figs_bd = make_cluster_breakdown_pages(data, unique_clusters, cluster_ids, spike_times_ms, ...
            index_all, mask_quarantine, forced, leicolors, visstr, clusters_per_page, length(figs));
        figs = [figs, figs_bd];
    catch ME_bd
        warning('Failed to generate cluster breakdown pages: %s', ME_bd.message);
    end

    fprintf('Generated %d figure pages for cluster report %s\n', length(figs),data.filename);
end

% -------------------------------------------------------------------------
% Cluster Breakdown Pages
% -------------------------------------------------------------------------
function figs_bd = make_cluster_breakdown_pages(data, unique_clusters, cluster_ids, spike_times_ms, ...
        index_all, mask_quarantine, forced, leicolors, visstr, clusters_per_page, total_pages_before)
% MAKE_CLUSTER_BREAKDOWN_PAGES  One column per cluster, rows = original / forced / rescued

    figs_bd = {};
    if isempty(unique_clusters), return; end

    % Only non-noise clusters on breakdown page
    bd_clusters = unique_clusters(unique_clusters ~= 0);
    if isempty(bd_clusters), return; end

    has_rescue  = isfield(data,'rescue_mask') && ~isempty(data.rescue_mask);
    has_forced  = isfield(data,'forced')      && ~isempty(data.forced);
    has_pre     = isfield(data,'cluster_class_pre_rescue');

    % ALIGNMENT FIX: We MUST use the post-rescue arrays (cluster_ids, spike_times_ms)
    % so they index perfectly into data.spikes.
    cids_bd = cluster_ids;
    sts_bd  = spike_times_ms;
    waveforms_all = double(data.spikes);    % full post-rescue spike matrix
    index_all_vec = index_all(:);
    
    % 1) Pre-calculate forced and rescued spike times for the whole dataset
    forced_sts = [];
    if has_forced
        f_vec = forced(:);
        if has_pre && numel(f_vec) == size(data.cluster_class_pre_rescue, 1)
            forced_sts = data.cluster_class_pre_rescue(f_vec, 2);
        elseif numel(f_vec) == numel(cids_bd)
            forced_sts = sts_bd(f_vec);
        end
    end
    
    rescued_sts = [];
    if has_rescue
        rescued_sts = index_all_vec(data.rescue_mask(:));
    end

    K = numel(bd_clusters);
    n_pages = ceil(K / clusters_per_page);
    page_offset = total_pages_before;  

    for pg = 1:n_pages
        idx_start = (pg-1)*clusters_per_page + 1;
        idx_end   = min(pg*clusters_per_page, K);
        page_clusters = bd_clusters(idx_start:idx_end);
        nc = numel(page_clusters);
        ncols = clusters_per_page;
        nrows = 3;

        fig = figure('Visible', visstr, 'Units','normalized','OuterPosition',[0 0 1 1], ...
                    'PaperUnits','inches','PaperType','<custom>','PaperSize',[16 8], ...
                    'PaperPosition',[0 0 16 8],'PaperPositionMode','manual', ...
                    'RendererMode','manual','Renderer','painters');

        for ci = 1:ncols
            if ci > nc
                for rr = 1:nrows
                    axE = subplot(nrows, ncols, (rr-1)*ncols + ci);
                    axis(axE,'off');
                end
                continue;
            end

            cid = page_clusters(ci);
            colc = leicolors(mod(cid, size(leicolors,1))+1, :);
            merge_tag = get_merge_tag(cid, data);

            % --- Masks for this cluster ---
            mask_cid = (cids_bd == cid);

            % Forced mask
            mask_forced_cid = false(size(mask_cid));
            if ~isempty(forced_sts)
                mask_forced_cid(ismember(sts_bd, forced_sts) & mask_cid) = true;
            end
            n_forced = sum(mask_forced_cid);

            % Rescued mask
            mask_resc_cid = false(size(mask_cid));
            if ~isempty(rescued_sts)
                mask_resc_cid(ismember(sts_bd, rescued_sts) & mask_cid) = true;
            end
            n_resc = sum(mask_resc_cid);

            % Pure original spikes
            mask_orig_cid = mask_cid(:) & ~mask_forced_cid(:) & ~mask_resc_cid(:);
            n_orig = sum(mask_orig_cid);
            % ---- ROW 1: Bar Summary  original, forced, rescued ----
            ax1 = subplot(nrows, ncols, ci);
            counts = [n_orig, n_forced, n_resc];
            labels = {'Orig','Forced','Rescued'};
            bar_cols = [colc; [0.97 0.52 0.03]; [0.1 0.75 0.1]];
            bh = bar(ax1, 1:3, counts, 'FaceColor','flat','EdgeColor','k','LineWidth',0.5);
            bh.CData = bar_cols;
            set(ax1,'XTick',1:3,'XTickLabel',labels,'XTickLabelRotation',30,'FontSize',7);
            ylabel(ax1,'Count','FontSize',7);
            grid(ax1,'on'); box(ax1,'off');
            set(ax1,'GridAlpha',0.2,'XGrid','off');
            
            % Count labels above each bar
            y_max_bar = max(counts);
            ylim(ax1, [0, max(y_max_bar * 1.18, 1)]);
            for bi = 1:numel(counts)
                text(ax1, bi, counts(bi) + y_max_bar * 0.04, num2str(counts(bi)), ...
                    'HorizontalAlignment','center','FontSize',7,'FontWeight','bold');
            end
            ttl = sprintf('C%d%s', cid, merge_tag);
            title(ax1, ttl, 'FontSize', 9, 'FontWeight', 'bold', 'Interpreter','none');

            % ---- ROW 2: Waveforms  pure original, forced overlay ----
            ax2 = subplot(nrows, ncols, ncols + ci);
            hold(ax2,'on');
            tvec = 1:size(waveforms_all,2);
            max_disp = 300;

            % Plot ONLY the pure original spikes
            W_orig = waveforms_all(mask_orig_cid, :);
            if ~isempty(W_orig)
                n_disp = min(max_disp, size(W_orig,1));
                idx_disp = randperm(size(W_orig,1), n_disp);
                plot(ax2, tvec, W_orig(idx_disp,:)', 'Color', [colc, 0.12], 'LineWidth', 0.6);
                plot(ax2, tvec, mean(W_orig,1), 'Color', 'k', 'LineWidth', 2.0);
            end
            
            % Forced overlay (orange)
            if any(mask_forced_cid)
                W_f = waveforms_all(mask_forced_cid,:);
                n_f = min(max_disp, size(W_f,1));
                plot(ax2, tvec, W_f(randperm(size(W_f,1),n_f),:)', 'Color', [0.97 0.52 0.03 0.25], 'LineWidth', 0.5);
            end
            
            hold(ax2,'off');
            title(ax2, sprintf('Orig (n=%d) / Forced', n_orig), 'FontSize', 7);
            xlabel(ax2,'Samples','FontSize',7); ylabel(ax2,'Amp','FontSize',7);
            grid(ax2,'on'); box(ax2,'off');
            set(ax2,'GridAlpha',0.2,'FontSize',7);

            % ---- ROW 3: Rescued spikes only (green) ----
            ax3 = subplot(nrows, ncols, 2*ncols + ci);
            if has_rescue && any(mask_resc_cid)
                hold(ax3,'on');
                W_r = waveforms_all(mask_resc_cid,:);
                n_r = min(max_disp, size(W_r,1));
                plot(ax3, tvec, W_r(randperm(size(W_r,1),n_r),:)', 'Color', [0.1 0.75 0.1 0.25], 'LineWidth', 0.5);
                plot(ax3, tvec, mean(W_r,1), 'Color', [0 0.5 0], 'LineWidth', 1.8);
                hold(ax3,'off');
                title(ax3, sprintf('Rescued  (n=%d)', n_resc), 'FontSize', 7);
                xlabel(ax3,'Samples','FontSize',7); ylabel(ax3,'Amp','FontSize',7);
                grid(ax3,'on'); box(ax3,'off');
                set(ax3,'GridAlpha',0.2,'FontSize',7);
            elseif ~has_rescue
                text(ax3, 0.5, 0.5, 'No rescue performed', ...
                    'HorizontalAlignment','center','FontSize',8,'Color',[0.5 0.5 0.5]);
                axis(ax3,'off');
            else
                text(ax3, 0.5, 0.5, 'No rescued spikes', ...
                    'HorizontalAlignment','center','FontSize',8,'Color',[0.5 0.5 0.5]);
                axis(ax3,'off');
            end
        end

        % --- Page title ---
        bd_total_pages = page_offset + n_pages;
        sgtitle(sprintf('Cluster Breakdown    Page %d/%d', page_offset + pg, bd_total_pages), 'FontSize', 13, 'Interpreter', 'none');
        figs_bd{end+1} = fig;
    end
end

% Helper: merge annotation tag for a cluster ID given the data struct
function tag = get_merge_tag(cid, data)
    tag = '';
    % Preferred: merge_groups struct array (new multi-group format)
    if isfield(data, 'merge_groups') && ~isempty(data.merge_groups)
        for gi = 1:numel(data.merge_groups)
            grp = data.merge_groups(gi);
            if grp.new_id == cid && numel(grp.former_ids) > 1
                tag = sprintf(' [merged: %s]', strjoin(arrayfun(@(x)sprintf('%d',x), grp.former_ids(:)', 'UniformOutput',false), '+'));
                return;
            end
        end
    end
    % Fallback: flat merge_map (legacy single-group saves)
    if isfield(data,'merge_map') && isstruct(data.merge_map) && ...
       isfield(data.merge_map,'former') && isfield(data.merge_map,'new')
        src = data.merge_map.former(data.merge_map.new == cid);
        if numel(src) > 1
            tag = sprintf(' [merged: %s]', strjoin(arrayfun(@(x)sprintf('%d',x),src(:)','UniformOutput',false),'+'));
        end
    end
end

% Helper: return only the last component of a path
function name = basename_only(p)
    parts = strsplit(p, filesep);
    parts = parts(~cellfun(@isempty, parts));
    if isempty(parts)
        name = p;
    else
        name = parts{end};
    end
end

function plot_isi_histogram(ax, spike_times, refractory_ms, line_freq, color)
    % PLOT_ISI_HISTOGRAM - Plot ISI distribution with violations (Python Style)
    if nargin < 1 || isempty(ax) || ~isgraphics(ax,'axes')
        ax = gca;
    end
    if nargin < 2, spike_times = []; end
    if nargin < 3 || isempty(refractory_ms), refractory_ms = 3.0; end
    if nargin < 4, line_freq = []; end
    if nargin < 5, color = [0.3 0.3 0.3]; end

    times_sorted = sort(spike_times);
    isi_ms = diff(times_sorted);
    isi_ms = isi_ms(isi_ms >= 0 & isfinite(isi_ms));

    if ~isempty(isi_ms)
        nbins = 50;
        bin_step = 1.5;
        ISI_max_ms = nbins * bin_step;
        bin_edges = 0:bin_step:ISI_max_ms;

        [n, c] = histcounts(isi_ms, bin_edges);

        bar(ax, c(1:end-1) + (bin_step/2), n, 'EdgeColor', 'none', 'FaceColor', color, 'BarWidth', 1);
        hold(ax, 'on');

        if ~isempty(line_freq) && line_freq > 0
            line_period_ms = 1000 / line_freq;
            for i_freq = 1:floor(ISI_max_ms / line_period_ms)
                interval = i_freq * line_period_ms;
                line([interval, interval], ylim(ax), 'LineWidth', 0.8, 'LineStyle', ':', 'Color', 'r', 'Parent', ax);
            end
        end
        hold(ax, 'off');

        xlim(ax, [0 ISI_max_ms]);
        n_viol = sum(isi_ms < refractory_ms);
        try
            title(ax, sprintf('%d in < %.1fms', n_viol, refractory_ms), 'FontSize', 10);
        catch
            t = get(ax, 'Title'); set(t, 'String', sprintf('%d in < %.1fms', n_viol, refractory_ms));
        end
        xlabel(ax, 'ISI (ms)');
        ylabel(ax, 'Count');
        box(ax, 'off'); grid(ax, 'on');
        set(ax, 'GridAlpha', 0.25, 'LineWidth', 0.8, 'TickDir', 'out');
    else
        try
            title(ax, 'No ISIs', 'FontSize', 10);
        catch
            t = get(ax, 'Title'); set(t, 'String', 'No ISIs');
        end
        axis(ax, 'off');
    end
end

% Replace plot_cluster_metrics_summary to be axes-aware
function plot_cluster_metrics_summary(ax, df_metrics, cluster_id, color)
% PLOT_CLUSTER_METRICS_SUMMARY - Display key metrics for cluster into provided axes
    
    % Find the row for this cluster
    row_idx = find(df_metrics.cluster_id == cluster_id, 1);
    
    if isempty(row_idx)
        % Cluster not in metrics table
        text(ax, 0.5, 0.5, sprintf('No metrics for C%d', cluster_id), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.5 0.5 0.5]);
        axis(ax, 'off');
        return;
    end
    
    m = df_metrics(row_idx, :);
    
    % Build metrics text, replacing NaN/Inf with '--'
    metrics_text = {
        sprintf('Cluster %d', cluster_id), ...
        sprintf('N spikes: %d', m.num_spikes), ...
        sprintf('FR: %.2f Hz', m.firing_rate), ...
        sprintf('SNR: %s', format_metric(m.snr)), ...
        sprintf('Presence: %s', format_metric(m.presence_ratio)), ...
        sprintf('Amp cutoff: %s', format_metric(m.amplitude_cutoff)), ...
        sprintf('ISI viol: %s', format_metric(m.isi_violation_rate)), ...
        sprintf('ISO dist: %s', format_metric(m.isolation_distance)), ...
        sprintf('L-ratio: %s', format_metric(m.l_ratio)), ...
        sprintf('d'': %s', format_metric(m.d_prime)), ...
        sprintf('NN hit: %s', format_metric(m.nn_hit_rate)), ...
        sprintf('Silhouette: %s', format_metric(m.silhouette_score))
    };
    
    % Display as text in axes
    text(ax, 0.05, 0.95, strjoin(metrics_text, '\n'), ...
        'VerticalAlignment', 'top', 'FontSize', 8, ...
        'FontName', 'FixedWidth', 'Color', color);
    
    axis(ax, 'off');
end

function str = format_metric(val)
    % Helper to format metric values (handle NaN/Inf gracefully)
    if isnan(val) || isinf(val)
        str = '--';
    elseif abs(val) < 0.01 && val ~= 0
        str = sprintf('%.2e', val);
    else
        str = sprintf('%.3f', val);
    end
end

function figM = create_metrics_overview_page(df_metrics, SS, visstr, leicolors, mean_waveforms, mean_waveform_cluster_ids, data)
    % CREATE_METRICS_OVERVIEW_PAGE - Create summary page with all metrics
    figM = figure('Visible', visstr, 'Units','normalized','OuterPosition',[0 0 1 1], ...
                 'PaperUnits', 'inches', 'PaperType', 'A4', 'PaperOrientation', 'landscape', 'PaperPositionMode', 'auto', ...
                 'RendererMode', 'manual', 'Renderer', 'painters');

    % Extract cluster info from data for spike counting
    if isfield(data, 'cluster_class')
        cluster_ids = int32(data.cluster_class(:,1));
        spike_times_ms = double(data.cluster_class(:,2));
    else
        cluster_ids = [];
        spike_times_ms = [];
    end

    exclude = {'cluster_id', 'snr', 'SNR', 'presence_ratio', 'presence ratio', 'PresenceRatio', 'num_spikes'};
    cols = df_metrics.Properties.VariableNames;
    metrics_cols = cols(~ismember(cols, exclude));

    all_unique_clusters = unique(df_metrics.cluster_id); 
    
    if ismember(0, all_unique_clusters)
        cluster_list_for_colors = all_unique_clusters;
    else
        cluster_list_for_colors = df_metrics.cluster_id; 
    end

    n_metrics = numel(metrics_cols);
    cols_per_row = 6;
    
    % Define supplementary visualizations (add new ones here, grid auto-adjusts)
    supplementary = {'silhouette', 'spike_counts', 'spikes_per_cluster_rescued'};
    n_supplementary = numel(supplementary);
    n_total_slots = n_metrics + n_supplementary;
    n_rows = max(1, ceil(n_total_slots / cols_per_row));
    
    % Create slot assignment map for clean access without magic offsets
    slot_map = containers.Map(supplementary, n_metrics + (1:n_supplementary));

    axes_handles = gobjects(n_rows, cols_per_row);
    idx = 1;
    for r = 1:n_rows
        for c = 1:cols_per_row
            if idx <= n_metrics
                ax = subplot(n_rows, cols_per_row, (r-1)*cols_per_row + c);
                y = df_metrics{:, metrics_cols{idx}};
                
                num_bars = height(df_metrics);
                bar_colors_rep = zeros(num_bars, 3);
                
                color_indices = df_metrics.cluster_id + 1;
                valid_indices = color_indices(color_indices <= size(leicolors, 1));
                bar_colors_rep(1:numel(valid_indices), :) = leicolors(valid_indices, :);

                h = bar(ax, (1:num_bars), y, 'FaceColor', 'flat', 'BarWidth', 0.9);
                h.CData = bar_colors_rep; 

                metric_name_lower = lower(metrics_cols{idx});
                if contains(metric_name_lower, 'isolation') || contains(metric_name_lower, 'l_ratio') || contains(metric_name_lower, 'lratio')
                    set(ax, 'YScale', 'log');
                    ylabel(ax, [metrics_cols{idx} ' (log)']);
                else
                    ylabel(ax, metrics_cols{idx});
                end

                if r == n_rows
                    xticks(ax, 1:num_bars);
                    xticklabels(ax, arrayfun(@(x)sprintf('c%d', df_metrics.cluster_id(x)), 1:num_bars, 'UniformOutput',false));
                    xtickangle(ax, 90);
                else
                    set(ax, 'XTick', []);
                end
                title(ax, metrics_cols{idx}, 'Interpreter', 'none');
                grid(ax, 'on');
                box(ax, 'off');
                set(ax, 'GridAlpha', 0.3, 'LineWidth', 0.8, 'XGrid', 'off');
                
                axes_handles(r,c) = ax;
                idx = idx + 1;
            else
                axes_handles(r,c) = subplot(n_rows, cols_per_row, (r-1)*cols_per_row + c);
                axis(axes_handles(r,c), 'off');
            end
        end
    end

    % Plot supplementary visualizations using systematic slot assignment
    if ismember('silhouette', supplementary)
        sil_slot = slot_map('silhouette');
        ax_sil = axes_handles(ceil(sil_slot/cols_per_row), mod(sil_slot-1, cols_per_row) + 1);
        if nargin >= 6 && ~isempty(mean_waveforms)
            ids = mean_waveform_cluster_ids(:)';
            M = mean_waveforms;
            nC = size(M,1);
            labels = {};
            dists = [];
            for i = 1:(nC-1)
                for j = (i+1):nC
                    mi = M(i,:);
                    mj = M(j,:);
                    if any(isnan(mi)) || any(isnan(mj))
                        continue;
                    end
                    labels{end+1} = sprintf('C%d-C%d', ids(i), ids(j));
                    dists(end+1) = sum(abs(mi - mj));
                end
            end

            if isempty(dists)
                text(0.5,0.5,'No L1 distances', 'HorizontalAlignment','center','Parent', ax_sil);
                axis(ax_sil,'off');
            else
                bar(ax_sil, dists, 'FaceColor', [0.2 0.2 0.7]);
                ax_sil.XTick = 1:numel(dists);
                ax_sil.XTickLabel = labels;
                xtickangle(ax_sil, 90);
                ylabel(ax_sil, 'L1 distance (sum |A-B|)');
                title(ax_sil, 'Pairwise L1 dist, between mean plots');
                grid(ax_sil, 'on');
                box(ax_sil, 'off');
            end
        else
            if isempty(SS)
                text(0.5,0.5,'No silhouette matrix', 'HorizontalAlignment','center','Parent', ax_sil);
                axis(ax_sil,'off');
            else
                S = SS;
                for i = 1:size(S, 1)
                    for j = 1:size(S, 2)
                        if isnan(S(i, j)) && ~isnan(S(j, i))
                            S(i, j) = S(j, i);
                        end
                    end
                end
                S(1:size(S,1)+1:end) = NaN; 
                mask_upper_tri = triu(true(size(S)), 1); 
                S(mask_upper_tri) = NaN; 
                
                cluster_labels = arrayfun(@(x)sprintf('c%d', x), df_metrics.cluster_id, 'UniformOutput', false);

                imagesc(ax_sil, S, [-1 1]);
                colormap(ax_sil, 'parula');
                cb = colorbar(ax_sil);
                cb.Label.String = 'Silhouette Score';

                ax_sil.XTick = 1:size(S,2);
                ax_sil.YTick = 1:size(S,1);
                ax_sil.XTickLabel = cluster_labels;
                ax_sil.YTickLabel = cluster_labels;
                xtickangle(ax_sil, 90);
                title(ax_sil, 'Silhouette Score Heatmap');
                axis(ax_sil, 'equal', 'tight'); 
                box(ax_sil, 'off');
            end
        end
    end
    
    % Plot spike counts summary if available
    if ismember('spike_counts', supplementary) && isfield(data, 'spike_summary')
        spike_slot = slot_map('spike_counts');
        ax_spike = axes_handles(ceil(spike_slot/cols_per_row), mod(spike_slot-1, cols_per_row) + 1);
        s = data.spike_summary;
        spike_counts = [s.total, s.quarantined, s.rescued, s.final];
        spike_labels = {'Total', 'Quarantined', 'Rescued', 'Final'};
        bar(ax_spike, 1:4, spike_counts, 'FaceColor', [0.2 0.6 0.8]);
        set(ax_spike, 'XTick', 1:4, 'XTickLabel', spike_labels, 'XTickLabelRotation', 45);
        ylabel(ax_spike, 'Spike Count');
        title(ax_spike, 'Spike Counts');
        grid(ax_spike, 'on');
        box(ax_spike, 'off');
        set(ax_spike, 'GridAlpha', 0.25, 'LineWidth', 0.8);
    end
    
    % Plot spikes per cluster with rescue overlay if requested
    if ismember('spikes_per_cluster_rescued', supplementary) && ~isempty(cluster_ids) && ~isempty(spike_times_ms)
        rescue_slot = slot_map('spikes_per_cluster_rescued');
        ax_rescue = axes_handles(ceil(rescue_slot/cols_per_row), mod(rescue_slot-1, cols_per_row) + 1);
        
        cluster_list = double(df_metrics.cluster_id(:));
        n_clusters = numel(cluster_list);
        
        % Count spikes per cluster (base) and rescued per cluster
        base_spikes = zeros(n_clusters, 1);
        rescued_spikes = zeros(n_clusters, 1);
        
        has_rescue = isfield(data, 'rescue_mask') && ~isempty(data.rescue_mask) && has_spikes_all;
        if has_rescue
            index_all_vec = data.index_all(:);
        else
            index_all_vec = [];
        end
        
        for ci = 1:n_clusters
            cid = cluster_list(ci);
            cluster_spike_times = spike_times_ms(cluster_ids == cid);
            base_spikes(ci) = numel(cluster_spike_times);
            
            if has_rescue
                rescued_count = 0;
                for st = cluster_spike_times(:)'
                    idx_in_all = find(index_all_vec == st, 1);
                    if ~isempty(idx_in_all) && idx_in_all <= numel(data.rescue_mask)
                        if data.rescue_mask(idx_in_all)
                            rescued_count = rescued_count + 1;
                        end
                    end
                end
                rescued_spikes(ci) = rescued_count;
            end
        end
        
        x_pos = 1:n_clusters;
        bar_colors = leicolors(mod(cluster_list, size(leicolors,1))+1, :);
        
        % Plot full bar (total spikes) in dark color (rescued portion)
        bar_colors_dark = bar_colors * 0.6;
        if has_rescue && any(rescued_spikes > 0)
            % Plot the full height in dark
            b_total = bar(ax_rescue, x_pos, base_spikes, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'k');
            b_total.CData = bar_colors_dark;
            hold(ax_rescue, 'on');
            
            % Plot the unrescued height over it in normal color
            unrescued_spikes = max(0, base_spikes - rescued_spikes);
            b_base = bar(ax_rescue, x_pos, unrescued_spikes, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'k');
            b_base.CData = bar_colors;
            hold(ax_rescue, 'off');
            
            % Create legend with custom patches for clarity
            % Create invisible objects for legend (since bars have multiple colors)
            hold(ax_rescue, 'on');
            h1 = bar(ax_rescue, NaN, NaN, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k', 'DisplayName', 'Rescued Spikes');
            h2 = bar(ax_rescue, NaN, NaN, 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'k', 'DisplayName', 'Original Clusters');
            legend(ax_rescue, [h2, h1], 'Location', 'northeast', 'Box', 'on', 'FontSize', 8);
            hold(ax_rescue, 'off');
            
        else
            % Just plot the base spikes in normal color
            b_base = bar(ax_rescue, x_pos, base_spikes, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'k');
            b_base.CData = bar_colors;
            
            % Simple legend for no rescue case
            %legend(ax_rescue, 'Original Clusters', 'Location', 'northeast', 'Box', 'on');
        end
        
        set(ax_rescue, 'XTick', x_pos);
        tick_labels = arrayfun(@(c) sprintf('c%d', c), cluster_list, 'UniformOutput', false);
        set(ax_rescue, 'XTickLabel', tick_labels, 'XTickLabelRotation', 45);
        title(ax_rescue,'Cluster Spikes')

        ylabel(ax_rescue, 'Spike Count');
        grid(ax_rescue, 'on');
        box(ax_rescue, 'off');
        set(ax_rescue, 'GridAlpha', 0.25, 'LineWidth', 0.8, 'XGrid', 'off');
    end

    try
        sgtitle(figM, 'Cluster Metrics Overview', 'FontSize', 14);
    catch
        prevFig = gcf;
        try set(0,'CurrentFigure',double(figM)); catch; end
        sgtitle('Cluster Metrics Overview', 'FontSize', 14);
        try set(0,'CurrentFigure',prevFig); catch; end
    end
end

function ax = report_subplot(nrows, ncols, idx)
    fig = gcf;
    if isappdata(fig, 'cluster_report_layout')
        tl = getappdata(fig, 'cluster_report_layout');
        if ~isvalid(tl) || tl.GridSize(1) ~= nrows || tl.GridSize(2) ~= ncols
            tl = tiledlayout(fig, nrows, ncols, 'TileSpacing', 'compact', 'Padding', 'compact');
            setappdata(fig, 'cluster_report_layout', tl);
        end
    else
        tl = tiledlayout(fig, nrows, ncols, 'TileSpacing', 'compact', 'Padding', 'compact');
        setappdata(fig, 'cluster_report_layout', tl);
    end
    ax = nexttile(tl, idx);
end

% -----------------------
% Helper: 2D waveform density image
function density_image_matlab(W, ax, samplerate_hz, varargin)
    p = inputParser;
    addParameter(p, 'w', 400, @isscalar);
    addParameter(p, 'h', 240, @isscalar);
    addParameter(p, 'cmap', 'inferno', @ischar); 
    parse(p, varargin{:});

    if isempty(W)
        axis(ax,'off'); return;
    end

    [n, T] = size(W);
    
    % CHANGE 1: Add max waveforms limit
    max_waveforms = 5000;
    if n > max_waveforms
        rng(42, 'twister');
        subsample_idx = randperm(n, max_waveforms);
        W = W(subsample_idx, :);
        n = max_waveforms;
    end
    
    x_min = 0; x_max = T-1;
    y_min = min(W(:)); y_max = max(W(:));
    
    pad = 1e-6 * (y_max - y_min);
    if pad == 0, pad = 1e-6; end
    y_min = y_min - pad; y_max = y_max + pad;

    x_edges = linspace(x_min, x_max, p.Results.w+1);
    y_edges = linspace(y_min, y_max, p.Results.h+1);

    % CHANGE 2: Reduce interpolation factor
    interpolation_factor = 10;  % was 200
    Ti = max(2, min(T * interpolation_factor, p.Results.w));
    
    x_orig = 0:(T-1);
    x_hi = linspace(x_min, x_max, Ti);
    hi = zeros(n, Ti);
    for i = 1:n
        hi(i,:) = interp1(x_orig, W(i,:), x_hi, 'linear');
    end

    t_flat = repmat(x_hi, 1, n);
    a_flat = hi';
    a_flat = a_flat(:);
    t_flat = t_flat(:);

    H = histcounts2(t_flat, a_flat, x_edges, y_edges);
    
    D = H' ./ (numel(t_flat) * mean(diff(x_edges)) * mean(diff(y_edges)));
    
    D_log = log10(D + eps); 

    D_pos = D(D>0);
    if isempty(D_pos), D_pos = 1; end
    vmin_log = log10(max(min(D_pos), 1e-12));
    vmax_log = log10(max(D(:)));
    if vmax_log <= vmin_log, vmax_log = vmin_log + 1; end 
    
    imagesc(ax, [x_min x_max], [y_min y_max], D_log);
    set(ax, 'YDir', 'normal');

    try colormap(ax, p.Results.cmap); catch, colormap(ax, 'hot'); end
    caxis(ax, [vmin_log, vmax_log]);

    set(ax, 'Color', 'white', 'XColor', [0 0 0], 'YColor', [0 0 0], 'Box', 'off');

    if ~isempty(samplerate_hz)
        nx = min(5, max(2, ceil((x_max - x_min)/max(1, round((T)/10)))));
        xt_pos = round(linspace(x_min, x_max, nx));
        xt_pos = unique(max(0, min(T-1, xt_pos)));
        if ~isempty(xt_pos)
            set(ax, 'XTick', xt_pos);
            xtlbls = arrayfun(@(x) sprintf('%.0f', x/samplerate_hz*1e3), xt_pos, 'UniformOutput', false);
            set(ax, 'XTickLabel', xtlbls, 'XColor', [0 0 0]);
            xlabel(ax, 'Time (ms)', 'Color', [0 0 0]);
        end
    else
        % For report heatmaps in sample mode, hide x ticks/labels to reduce clutter.
        set(ax, 'XTick', [], 'XTickLabel', {}, 'XColor', [0 0 0]);
        xlabel(ax, '');
    end

    axis(ax, 'tight');
    set(ax, 'YColor', [0 0 0], 'XColor', [0 0 0], 'Visible', 'on');
end

% Helper: cluster activity KDE matrix (per-cluster rows)
function cluster_activity_kde_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax, time_pixels, cmapname)
    if nargin < 4 || isempty(ax), ax = gca; end
    if nargin < 5 || isempty(time_pixels), time_pixels = 100; end
    if nargin < 6, cmapname = 'inferno'; end

    clusters = unique(cluster_ids);
    clusters = sort(clusters,'descend');
    K = numel(clusters);
    if K == 0
        axis(ax,'off'); return;
    end

    T = double(recording_duration_ms);
    valid_spike_mask = (spike_times_ms >= 0) & (spike_times_ms < T);
    t_grid_min = linspace(0, T/60000.0, time_pixels); 
    kde_matrix = zeros(K, time_pixels);

    for r = 1:K
        cid = clusters(r);
        t = spike_times_ms((cluster_ids == cid) & valid_spike_mask) / 60000.0;
        if numel(t) > 1
            try
                % Match scipy gaussian_kde default (Scott's rule) more closely.
                n_t = numel(t);
                sd_t = std(t);
                if isfinite(sd_t) && sd_t > 0
                    bw = sd_t * n_t^(-1/5);
                    f = ksdensity(t, t_grid_min, 'Bandwidth', bw);
                else
                    f = ksdensity(t, t_grid_min);
                end
                kde_matrix(r,:) = f;
            catch
                kde_matrix(r,:) = zeros(1, time_pixels);
                [~, j] = min(abs(t_grid_min - mean(t)));
                kde_matrix(r,j) = 1;
            end
        elseif numel(t) == 1
            kde_matrix(r,:) = zeros(1, time_pixels);
            [~, j] = min(abs(t_grid_min - t));
            kde_matrix(r,j) = 1;
        else
            kde_matrix(r,:) = zeros(1, time_pixels);
        end
    end

    imagesc(ax, 1:time_pixels, 1:K, kde_matrix);
    colormap(ax, get_named_colormap(cmapname, 256));

    % Match seaborn heatmap behavior more closely by scaling to data min/max.
    vmin = min(kde_matrix(:));
    vmax = max(kde_matrix(:));
    if vmax <= vmin
        vmax = vmin + eps;
    end
    caxis(ax, [vmin, vmax]);

    set(ax, 'XColor', [0 0 0], 'YColor', [0 0 0], 'Box', 'off');

    nx = min(6, time_pixels);
    xt_idx = unique(round(linspace(1, time_pixels, nx)));
    xt_idx = max(1, min(time_pixels, xt_idx));
    xt_vals_min = t_grid_min(xt_idx); 

    % Use whole-minute labels to avoid clutter on the presence plot.
    xt_lbls = arrayfun(@(v) sprintf('%.0f', v), xt_vals_min, 'UniformOutput', false);
    set(ax, 'XTick', xt_idx, 'XTickLabel', xt_lbls, 'XColor', [0 0 0]);

    set(ax, 'YTick', 1:K, 'YTickLabel', arrayfun(@(c)sprintf('%d', c), clusters, 'UniformOutput', false), 'YColor', [0 0 0]);

    xlabel(ax, 'Time (min)', 'Color', [0 0 0]);
    ylabel(ax, 'Cluster', 'Color', [0 0 0]);
    set(ax, 'YDir', 'reverse');
    set(ax, 'Visible', 'on');
end

function cmap = get_named_colormap(cmapname, n)
    if nargin < 2 || isempty(n)
        n = 256;
    end

    if strcmpi(cmapname, 'inferno')
        % Approximation of Matplotlib inferno anchors in RGB [0..255].
        anchors = [
              0,   0,   4;
             31,  12,  72;
             85,  15, 109;
            136,  34, 106;
            186,  54,  85;
            227,  89,  51;
            249, 140,  10;
            249, 201,  50;
            252, 255, 164
        ] / 255;
        x = linspace(0, 1, size(anchors, 1));
        xi = linspace(0, 1, n);
        cmap = interp1(x, anchors, xi, 'linear');
        return;
    end

    try
        cmap_fun = str2func(lower(cmapname));
        cmap = cmap_fun(n);
    catch
        cmap = parula(n);
    end
end

% Helper: compute cross-correlogram between two spike-time lists (ms)
function [lags_ms, counts] = compute_cross_correlogram(times1_ms, times2_ms, bin_ms, maxlag_ms)
    if nargin < 3 || isempty(bin_ms), bin_ms = 1.0; end
    if nargin < 4 || isempty(maxlag_ms), maxlag_ms = 50.0; end
    % ensure column vectors and sorted
    t1 = sort(times1_ms(:));
    t2 = sort(times2_ms(:));
    maxlag = double(maxlag_ms);
    binw = double(bin_ms);

    diffs = [];
    % iterate over shorter list to limit work
    if numel(t1) <= numel(t2)
        for i = 1:numel(t1)
            window_idx = find(t2 >= (t1(i)-maxlag) & t2 <= (t1(i)+maxlag));
            if ~isempty(window_idx)
                diffs = [diffs; (t2(window_idx) - t1(i))]; 
            end
        end
    else
        for i = 1:numel(t2)
            window_idx = find(t1 >= (t2(i)-maxlag) & t1 <= (t2(i)+maxlag));
            if ~isempty(window_idx)
                diffs = [diffs; (t1(window_idx) - t2(i))]; 
            end
        end
    end

    if isempty(diffs)
        lags_ms = (-maxlag+binw/2):binw:(maxlag-binw/2);
        counts = zeros(size(lags_ms));
        return;
    end

    edges = -maxlag:binw:maxlag;
    counts = histcounts(diffs, edges);
    % center lags
    lags_ms = edges(1:end-1) + binw/2;
end