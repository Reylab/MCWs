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
        error('data must contain cluster_class, spikes, inspk');
    end
    cluster_class = data.cluster_class;
    waveforms = double(data.spikes);
    features = double(data.inspk);
    cluster_ids = int32(cluster_class(:,1));
    spike_times_ms = double(cluster_class(:,2));
    unique_clusters = unique(cluster_ids);
    recording_duration_ms = max(spike_times_ms);

    if p.Results.calc_metrics
        [df_metrics, SS] = compute_cluster_metrics(data, 'exclude_cluster_0', p.Results.exclude_cluster_0, ...
            'n_neighbors', p.Results.n_neighbors, 'bin_duration', p.Results.bin_duration_ms, 'make_plots', false);
    else
        df_metrics = p.Results.metrics_df;
        SS = p.Results.SS;
        if isempty(df_metrics) || isempty(SS)
            error('If calc_metrics=false, you must provide metrics_df and SS');
        end
    end
    figs = {};
    leicolors = [0 0 0; 0 0 1; 1 0 0; 0 0.5 0; 0.62 0 0; 0.42 0 0.76; 0.97 0.52 0.03; 0.52 0.25 0; 1 0.10 0.72; 0.55 0.55 0.55; 0.59 0.83 0.31; 0.97 0.62 0.86; 0.62 0.76 1.0];
    
    % --- Paging Logic ---
    pages = {};
    K = numel(unique_clusters);
    for i = 1:p.Results.clusters_per_page:K
        pages{end+1} = unique_clusters(i:min(i+p.Results.clusters_per_page-1,K));
    end

    % --- Summary + per-page plotting ---
    for page_i = 1:length(pages)
        page = pages{page_i};

        % Reorder page 1 to put cluster 0 last
        if page_i == 1
            idx_0 = find(page == 0, 1);
            if ~isempty(idx_0)
                page(idx_0) = []; 
                page(end+1) = 0;
            end
        end

        nclusters_on_page = numel(page);
        ncols = 1 + nclusters_on_page; % 1 summary + clusters

        if p.Results.show_figures
            visstr = 'on';
        end
        fig = figure('Visible', visstr, 'Units','normalized','Position',[0.1 0.1 0.8 0.8]);

        % grid 3 rows x ncols
        for col = 1:ncols
            if col == 1
                % --- SUMMARY COLUMN ---
                ax1 = subplot(3, ncols, 1);
                hold(ax1,'on');
                T = size(waveforms,2);
                
                % FIX 1: Summary Mean Plot -> SAMPLES (ignore samplerate)
                tvec = 1:T;
                xlabel_str = 'Samples';
                
                total_spikes = numel(cluster_ids);
                for k = 1:numel(unique_clusters)
                    c = unique_clusters(k);
                    Wc = waveforms(cluster_ids==c, :);
                    if isempty(Wc), continue; end
                    colc = leicolors(mod(k-1,size(leicolors,1))+1,:);
                    plot(ax1, tvec, mean(Wc,1), 'Color', colc, 'LineWidth', 1.8);
                end
                title(ax1, sprintf('Means (total n = %d)', total_spikes));
                xlabel(ax1, xlabel_str); ylabel(ax1,'Amplitude'); grid(ax1,'on');
                
                % Style Update (Match Python)
                box(ax1, 'off');
                set(ax1, 'GridAlpha', 0.25, 'LineWidth', 0.8);

                % presence KDE (row2)
                ax2 = subplot(3, ncols, ncols+1);
                cluster_activity_kde_ax_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax2, 100, 'inferno');

                % SNR bar (row3)
                ax3 = subplot(3, ncols, 2*ncols+1);
                if istable(df_metrics) && all(ismember({'cluster_id','snr'}, df_metrics.Properties.VariableNames))
                    [lia, ~] = ismember(df_metrics.cluster_id, unique_clusters);
                    table_clusters = df_metrics.cluster_id(lia);
                    table_snrs = df_metrics.snr(lia);
                    
                    [~, color_indices] = ismember(table_clusters, unique_clusters);
                    bar_colors = leicolors(mod(color_indices-1, size(leicolors,1))+1, :);

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
                % --- CLUSTER COLUMNS ---
                cid = page(col-1);
                
                % waveform panel (row1)
                axW = subplot(3,ncols, col);
                mask = (cluster_ids==cid);
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
                colc = leicolors(mod(find(unique_clusters==cid, 1)-1,size(leicolors,1))+1,:);
                
                % FIX 2: Individual Cluster Plots -> SAMPLES (ignore samplerate)
                tvec = 1:size(W,2);
                xlabel_str = 'Samples';
                
                plot(axW, tvec, W(idx,:)', 'Color', [colc, 0.15], 'LineWidth', 0.8); 
                plot(axW, tvec, mean(W,1), 'Color', 'k', 'LineWidth', 2.4); 
                title(axW, sprintf('Cluster %d (n=%d)', cid, sum(mask)), 'FontSize', 10);
                xlabel(axW, xlabel_str);
                ylabel(axW, 'Amplitude');
                grid(axW, 'on');
                box(axW, 'off');
                set(axW, 'GridAlpha', 0.25, 'LineWidth', 0.8);

                % Density image (row2) - match Python layout
                axD = subplot(3,ncols, col + ncols);
                
                % FIX 3: Density Plot -> SAMPLES (pass [] for samplerate)
                density_image_matlab(W, axD, [], 'cmap', 'inferno');
                
                % ISI histogram (row3)
                axI = subplot(3,ncols, col + 2*ncols);
                plot_isi_histogram(axI, spike_times_ms(mask), p.Results.refractory_ms, 60, colc);
            end
        end

        sgtitle(sprintf('Cluster Report - Page %d/%d', page_i, length(pages)), 'FontSize', 14);
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

        fig_metrics = create_metrics_overview_page(df_metrics, SS, visstr, leicolors, mean_wfs, cluster_list_for_metrics);
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
        if Kc >= 2
            pairs = nchoosek(1:Kc,2);
            npairs = size(pairs,1);
            % layout: try square-ish grid
            ncols = ceil(sqrt(npairs));
            nrows = ceil(npairs / ncols);
            fig_corr = figure('Visible', visstr, 'Units','normalized','Position',[0.1 0.1 0.8 0.8]);
            for pi = 1:npairs
                ax = subplot(nrows, ncols, pi);
                a = cluster_list(pairs(pi,1));
                b = cluster_list(pairs(pi,2));
                times_a = spike_times_ms(cluster_ids == a);
                times_b = spike_times_ms(cluster_ids == b);
                if isempty(times_a) || isempty(times_b)
                    axis(ax,'off'); continue;
                end
                [lags, counts] = compute_cross_correlogram(times_a, times_b, 1.0, 50.0); % bin=1ms, maxlag=50ms
                bar(ax, lags, counts, 'FaceColor', [0.2 0.2 0.7], 'EdgeColor','none');
                xlabel(ax,'Lag (ms)'); ylabel(ax,'Count');
                title(ax, sprintf('C%d vs C%d', a, b), 'FontSize', 9);
                box(ax,'off'); grid(ax,'on');
            end
            sgtitle(sprintf('Cross-correlograms (%d pairs)', npairs), 'FontSize', 14);
            figs{end+1} = fig_corr;
        end
    catch ME_corr
        warning('Failed to generate correlograms: %s', ME_corr.message);
    end

    fprintf('Generated %d figure pages for cluster report %s\n', length(figs),data.filename);
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

function figM = create_metrics_overview_page(df_metrics, SS, visstr, leicolors, mean_waveforms, mean_waveform_cluster_ids)
    % CREATE_METRICS_OVERVIEW_PAGE - Create summary page with all metrics
    figM = figure('Visible', visstr, 'Units','normalized','Position',[0.1 0.1 0.8 0.8]);

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
    n_rows = max(1, ceil((n_metrics + 1) / cols_per_row)); 

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

    sil_slot = n_metrics + 1;
    if sil_slot <= n_rows * cols_per_row
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
                title(ax_sil, 'Pairwise L1 distances between mean waveforms');
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

    try
        sgtitle(figM, 'Cluster Metrics Overview', 'FontSize', 14);
    catch
        prevFig = gcf;
        try set(0,'CurrentFigure',double(figM)); catch; end
        sgtitle('Cluster Metrics Overview', 'FontSize', 14);
        try set(0,'CurrentFigure',prevFig); catch; end
    end
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
    x_min = 0; x_max = T-1;
    y_min = min(W(:)); y_max = max(W(:));
    
    pad = 1e-6 * (y_max - y_min);
    if pad == 0, pad = 1e-6; end
    y_min = y_min - pad; y_max = y_max + pad;

    x_edges = linspace(x_min, x_max, p.Results.w+1);
    y_edges = linspace(y_min, y_max, p.Results.h+1);

    interpolation_factor = 200; 
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

    nx = min(5, max(2, ceil((x_max - x_min)/max(1, round((T)/10)))));
    xt_pos = round(linspace(x_min, x_max, nx));
    xt_pos = unique(max(0, min(T-1, xt_pos)));
    if ~isempty(xt_pos)
        set(ax, 'XTick', xt_pos);
        if ~isempty(samplerate_hz)
            xtlbls = arrayfun(@(x) sprintf('%.0f', x/samplerate_hz*1e3), xt_pos, 'UniformOutput', false);
            set(ax, 'XTickLabel', xtlbls, 'XColor', [0 0 0]);
            xlabel(ax, 'Time (ms)', 'Color', [0 0 0]);
        else
            xtlbls = arrayfun(@(x) sprintf('%d', x), xt_pos, 'UniformOutput', false);
            set(ax, 'XTickLabel', xtlbls, 'XColor', [0 0 0]);
            xlabel(ax, 'Sample', 'Color', [0 0 0]);
        end
    end

    axis(ax, 'tight');
    set(ax, 'YColor', [0 0 0], 'XColor', [0 0 0], 'Visible', 'on');
end

% Helper: cluster activity KDE matrix (per-cluster rows)
function cluster_activity_kde_ax_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax, time_pixels, cmapname)
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
    t_grid_min = linspace(0, T/60000.0, time_pixels); 
    kde_matrix = zeros(K, time_pixels);

    for r = 1:K
        cid = clusters(r);
        t = spike_times_ms(cluster_ids == cid) / 60000.0; 
        if numel(t) > 1
            try
                f = ksdensity(t, t_grid_min);
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
    try colormap(ax, cmapname); catch, colormap(ax, 'hot'); end

    vmax = quantile(kde_matrix(:), 0.98);
    if vmax <= 0, vmax = max(kde_matrix(:)); end
    if vmax == 0, vmax = 1; end
    caxis(ax, [0, vmax]);

    set(ax, 'XColor', [0 0 0], 'YColor', [0 0 0], 'Box', 'off');

    nx = min(6, time_pixels);
    xt_idx = unique(round(linspace(1, time_pixels, nx)));
    xt_idx = max(1, min(time_pixels, xt_idx));
    xt_vals_min = t_grid_min(xt_idx); 

    scale_factor = 10;
    xt_lbls = arrayfun(@(v) sprintf('%.2f', v * scale_factor), xt_vals_min, 'UniformOutput', false);
    if numel(unique(xt_lbls)) < numel(xt_lbls)
        xt_lbls = arrayfun(@(v) sprintf('%.3f', v * scale_factor), xt_vals_min, 'UniformOutput', false);
    end
    set(ax, 'XTick', xt_idx, 'XTickLabel', xt_lbls, 'XColor', [0 0 0]);

    set(ax, 'YTick', 1:K, 'YTickLabel', arrayfun(@(c)sprintf('%d', c), clusters, 'UniformOutput', false), 'YColor', [0 0 0]);

    xlabel(ax, 'Time (min) \times10', 'Color', [0 0 0]);
    ylabel(ax, 'Cluster', 'Color', [0 0 0]);
    set(ax, 'YDir', 'reverse');
    set(ax, 'Visible', 'on');
end

% -----------------------
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
                diffs = [diffs; (t2(window_idx) - t1(i))]; %#ok<AGROW>
            end
        end
    else
        for i = 1:numel(t2)
            window_idx = find(t1 >= (t2(i)-maxlag) & t1 <= (t2(i)+maxlag));
            if ~isempty(window_idx)
                diffs = [diffs; (t1(window_idx) - t2(i))]; %#ok<AGROW>
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