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

    % Ensure all figures created inside use the requested visibility.
    % Use root DefaultFigureVisible so even figure() calls without explicit
    % 'Visible' follow the flag.
    if p.Results.show_figures
        visstr = 'on';
    else
        visstr = 'off';
    end
    origDefault = get(0, 'DefaultFigureVisible');
    set(0, 'DefaultFigureVisible', visstr);

    % make sure we restore the original DefaultFigureVisible on exit
    cleanupObj = onCleanup(@() set(0, 'DefaultFigureVisible', origDefault));

    % Validate data
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

    % Compute metrics if requested
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

    % plotting helpers
    figs = {};
    % color list
    leicolors = [0 0 0; 0 0 1; 1 0 0; 0 0.5 0; 0.62 0 0; 0.42 0 0.76; 0.97 0.52 0.03; 0.52 0.25 0; 1 0.10 0.72; 0.55 0.55 0.55; 0.59 0.83 0.31; 0.97 0.62 0.86; 0.62 0.76 1.0];

    % Build pages (first column summary + per-cluster columns)
    pages = {};
    K = numel(unique_clusters);
    for i = 1:p.Results.clusters_per_page:K
        pages{end+1} = unique_clusters(i:min(i+p.Results.clusters_per_page-1,K));
    end

    % Summary + per-page plotting
    for page_i = 1:length(pages)
        page = pages{page_i};
        nclusters_on_page = numel(page);
        ncols = 1 + nclusters_on_page; % 1 summary + clusters

        % determine visibility string
        if p.Results.show_figures
            visstr = 'on';
        else
            visstr = 'off';
        end
        fig = figure('Visible', visstr, 'Units','normalized','Position',[0.1 0.1 0.8 0.8]);

        % grid 3 rows x ncols
        for col = 1:ncols
            if col == 1
                % Summary column: row1 mean overlays; row2 presence KDE; row3 SNR bar
                ax1 = subplot(3, ncols, 1);
                hold(ax1,'on');
                T = size(waveforms,2);
                if ~isempty(p.Results.samplerate_hz)
                    tvec = (0:T-1)/p.Results.samplerate_hz*1000;
                    xlabel_str = 'Time (ms)';
                else
                    tvec = 1:T;
                    xlabel_str = 'Samples';
                end
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

                % presence KDE (row2)
                ax2 = subplot(3, ncols, ncols+1);
                cluster_activity_kde_ax_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax2, 100, 'inferno');

                % SNR bar (row3)
                ax3 = subplot(3, ncols, 2*ncols+1);
                if istable(df_metrics) && all(ismember({'cluster_id','snr'}, df_metrics.Properties.VariableNames))
                    [~,order] = ismember(unique_clusters, df_metrics.cluster_id);
                    snrs = nan(size(unique_clusters));
                    for ii=1:numel(unique_clusters)
                        idx = find(df_metrics.cluster_id==unique_clusters(ii),1);
                        if ~isempty(idx), snrs(ii) = df_metrics.snr(idx); end
                    end
                    bar(ax3, 1:numel(unique_clusters), snrs, 'FaceColor',[0.2 0.2 0.8]);
                    set(ax3, 'XTick', 1:numel(unique_clusters), 'XTickLabel', arrayfun(@(c)sprintf('C%d',c), unique_clusters, 'UniformOutput',false), 'XTickLabelRotation',90);
                    ylabel(ax3,'SNR'); title(ax3,'SNR by cluster');
                else
                    text(ax3,0.5,0.5,'metrics_df missing (cluster_id,snr)','HorizontalAlignment','center');
                    axis(ax3,'off');
                end
            else
                % cluster columns start at col index 2..ncols
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
                rng = RandStream('mlfg6331_64'); inds = 1:n_here;
                if n_here > take
                    idx = randperm(rng, n_here, take);
                else
                    idx = 1:n_here;
                end
                hold(axW,'on');
                colc = leicolors(mod(find(unique_clusters==cid)-1,size(leicolors,1))+1,:);
                if ~isempty(p.Results.samplerate_hz)
                    tvec = (0:size(W,2)-1)/p.Results.samplerate_hz*1000;
                    xlabel_str = 'Time (ms)';
                else
                    tvec = 1:size(W,2);
                    xlabel_str = 'Samples';
                end
                plot(axW, tvec, W(idx,:)', 'Color', [colc, 0.1], 'LineWidth', 0.5);
                plot(axW, tvec, mean(W,1), 'Color', colc, 'LineWidth', 2);
                title(axW, sprintf('Cluster %d (n=%d)', cid, sum(mask)), 'FontSize', 10);
                xlabel(axW, xlabel_str);
                ylabel(axW, 'Amplitude');
                grid(axW, 'on');

                % Density image (row2) - match Python layout
                axD = subplot(3,ncols, col + ncols);
                density_image_matlab(W, axD, p.Results.samplerate_hz, 'cmap', 'inferno');

                % ISI histogram (row3)
                axI = subplot(3,ncols, col + 2*ncols);
                plot_isi_histogram(spike_times_ms(mask), p.Results.refractory_ms, colc);
            end
        end

        sgtitle(sprintf('Cluster Report - Page %d/%d', page_i, length(pages)), 'FontSize', 14);
        figs{end+1} = fig;
    end

    % Metrics overview page
    if ~isempty(df_metrics) && ~isempty(SS)
        fig_metrics = create_metrics_overview_page(df_metrics, SS, p.Results.show_figures);
        figs{end+1} = fig_metrics;
    end

    fprintf('Generated %d figure pages for cluster report\n', length(figs));
end

function plot_isi_histogram(spike_times, refractory_ms, color)
% PLOT_ISI_HISTOGRAM - Plot ISI distribution with violations
    times_sorted = sort(spike_times);
    isi_ms = diff(times_sorted);
    isi_ms = isi_ms(isi_ms >= 0 & isfinite(isi_ms));
    
    if ~isempty(isi_ms)
        nbins = 50;
        histogram(isi_ms, nbins, 'FaceColor', color, 'EdgeColor', 'none');
        n_viol = sum(isi_ms < refractory_ms);
        title(sprintf('ISI: %d < %dms', n_viol, refractory_ms), 'FontSize', 10);
        xlabel('ISI (ms)');
        grid on;
    else
        title('No ISIs', 'FontSize', 10);
    end
end

function plot_cluster_metrics_summary(df_metrics, cluster_id, color)
% PLOT_CLUSTER_METRICS_SUMMARY - Display key metrics for cluster
    cluster_metrics = df_metrics(df_metrics.cluster_id == cluster_id, :);
    
    if height(cluster_metrics) == 1
        text(0.1, 0.8, sprintf('SNR: %.1f', cluster_metrics.snr), 'FontSize', 9);
        text(0.1, 0.6, sprintf('Isolation: %.1f', cluster_metrics.isolation_distance), 'FontSize', 9);
        text(0.1, 0.4, sprintf('L-ratio: %.3f', cluster_metrics.l_ratio), 'FontSize', 9);
        text(0.1, 0.2, sprintf('Silhouette: %.2f', cluster_metrics.silhouette_score), 'FontSize', 9);
    end
    axis off;
end

function fig = create_metrics_overview_page(df_metrics, SS, show_figures)
% CREATE_METRICS_OVERVIEW_PAGE - Create summary page with all metrics
    if show_figures
        fig = figure('Visible', 'on', 'Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    else
        fig = figure('Visible', 'off');
    end

    % Determine metric columns (exclude identifiers)
    exclude = {'cluster_id', 'snr', 'SNR', 'presence_ratio', 'presence ratio', 'PresenceRatio', 'num_spikes'};
    cols = df_metrics.Properties.VariableNames;
    metrics_cols = cols(~ismember(cols, exclude));

    n_metrics = numel(metrics_cols);
    cols_per_row = 6;
    n_rows = max(1, ceil((n_metrics + 1) / cols_per_row)); % +1 for silhouette heatmap

    axes_handles = gobjects(n_rows, cols_per_row);
    idx = 1;
    for r = 1:n_rows
        for c = 1:cols_per_row
            if idx <= n_metrics
                ax = subplot(n_rows, cols_per_row, (r-1)*cols_per_row + c);
                y = df_metrics{:, metrics_cols{idx}};
                bar(ax, (1:height(df_metrics)), y, 'FaceColor', [0.2 0.4 0.7]);
                if r == n_rows
                    xticks(ax, 1:height(df_metrics));
                    xticklabels(ax, arrayfun(@(x)sprintf('c%d', df_metrics.cluster_id(x)), 1:height(df_metrics), 'UniformOutput',false));
                    xtickangle(ax, 90);
                else
                    set(ax, 'XTick', []);
                end
                ylabel(ax, metrics_cols{idx});
                title(ax, metrics_cols{idx}, 'Interpreter', 'none');
                grid(ax, 'on');
                axes_handles(r,c) = ax;
                idx = idx + 1;
            else
                axes_handles(r,c) = subplot(n_rows, cols_per_row, (r-1)*cols_per_row + c);
                axis(axes_handles(r,c), 'off');
            end
        end
    end

    % Silhouette heatmap in the next available slot
    sil_slot = n_metrics + 1;
    if sil_slot <= n_rows * cols_per_row
        ax_sil = axes_handles(ceil(sil_slot/cols_per_row), mod(sil_slot-1, cols_per_row) + 1);
        if isempty(SS)
            text(0.5,0.5,'No silhouette matrix', 'HorizontalAlignment','center','Parent', ax_sil);
            axis(ax_sil,'off');
        else
            S = SS;
            STri = tril(true(size(S)), -1); % show lower triangle to match python masking of upper
            imagesc(ax_sil, S, [-1 1]);
            % Use a MATLAB builtin colormap (parula) instead of 'viridis'
            colormap(ax_sil, 'parula');
             colorbar(ax_sil);
             ax_sil.XTick = 1:size(S,1);
             ax_sil.YTick = 1:size(S,1);
             ax_sil.XTickLabel = arrayfun(@(x)sprintf('c%d', x), 1:size(S,1), 'UniformOutput', false);
             ax_sil.YTickLabel = ax_sil.XTickLabel;
             title(ax_sil, 'Silhouette Score Heatmap');
         end
    end

    sgtitle('Cluster Metrics Overview', 'FontSize', 14);
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
    if y_min == y_max
        y_min = y_min - 1; y_max = y_max + 1;
    end

    x_edges = linspace(x_min, x_max, p.Results.w+1);
    y_edges = linspace(y_min, y_max, p.Results.h+1);

    % interpolate each waveform to higher temporal resolution
    Ti = max(2, min(T * 20, p.Results.w));
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
    % plot (use log scaling for visual dynamic range)
    axes(ax);
    imagesc(ax, [x_min x_max], [y_min y_max], log10(D + eps));
    set(ax, 'YDir','normal');
    if ~isempty(samplerate_hz)
        xlabel(ax, 'Time (ms)');
        xticks = get(ax,'XTick');
        set(ax,'XTickLabel', arrayfun(@(x) sprintf('%.0f', x/samplerate_hz*1e3), xticks, 'UniformOutput', false));
    else
        xlabel(ax, 'Sample');
    end
    colormap(ax, parula);
    axis(ax, 'tight');
    grid(ax, 'on');
end

% -----------------------
% Helper: cluster activity KDE matrix (per-cluster rows)
function cluster_activity_kde_ax_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax, time_pixels, cmapname)
    if nargin < 4, ax = gca; end
    if nargin < 5 || isempty(time_pixels), time_pixels = 100; end
    if nargin < 6, cmapname = 'parula'; end

    clusters = unique(cluster_ids);
    clusters = sort(clusters,'descend');
    K = numel(clusters);

    T = double(recording_duration_ms);
    t_grid_min = linspace(0, T/60000.0, time_pixels); % minutes
    kde_matrix = zeros(K, time_pixels);

    for r = 1:K
        cid = clusters(r);
        t = spike_times_ms(cluster_ids == cid) / 60000.0;
        if numel(t) > 1
            try
                % ksdensity over t_grid_min
                [f,xi] = ksdensity(t, t_grid_min);
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

    axes(ax);
    imagesc(ax, 1:time_pixels, 1:K, kde_matrix);
    colormap(ax, parula);
    set(ax, 'YTick', 1:K, 'YTickLabel', arrayfun(@(c)sprintf('Cl: %d', c), clusters, 'UniformOutput', false));
    xlabel(ax, 'Time (min)');
    ylabel(ax, 'Presence Plot');
    set(ax, 'YDir', 'reverse');
end