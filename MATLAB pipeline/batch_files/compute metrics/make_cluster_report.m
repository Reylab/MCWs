function [figs, df_metrics, SS] = make_cluster_report(data, varargin)
    % MAKE_CLUSTER_REPORT - MATLAB port to match Python plotting layout
    p = inputParser;
    addParameter(p, 'calc_metrics', false, @islogical);
    addParameter(p, 'metrics_df', [], @(x) isempty(x) || istable(x));
    addParameter(p, 'SS', [], @(x) isempty(x) || isnumeric(x));
     addParameter(p, 'cross_correlograms', [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'l1_distances', [], @(x) isempty(x) || isnumeric(x));
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
    if p.Results.show_figures
        visstr = 'on';
    else
        visstr = 'off';
    end
    origDefault = get(0, 'DefaultFigureVisible');
    set(0, 'DefaultFigureVisible', visstr);
    cleanupObj = onCleanup(@() set(0, 'DefaultFigureVisible', origDefault));
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
    cross_correlograms = p.Results.cross_correlograms;
    l1_distances = p.Results.l1_distances;
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

                    % --- FIX START: Use h.CData instead of ax3.CData ---
                    h = bar(ax3, 1:numel(table_clusters), table_snrs, 'FaceColor', 'flat', 'BarWidth', 0.9);
                    h.CData = bar_colors; 
                    % --- FIX END ---
                    
                    set(ax3, 'XTick', 1:numel(table_clusters), 'XTickLabel', arrayfun(@(c)sprintf('C%d',c), table_clusters, 'UniformOutput',false), 'XTickLabelRotation',90);ylabel(ax3,'SNR'); title(ax3,'SNR by cluster');
                    
                    % Style Update (Match Python)
                    box(ax3, 'off');
                    grid(ax3, 'on');
                    set(ax3, 'GridAlpha', 0.25, 'LineWidth', 0.8, 'XGrid', 'off');
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
                rng = RandStream('mlfg6331_64'); % Use fixed-seed stream
                if n_here > take
                    idx = randperm(rng, n_here, take);
                else
                    idx = 1:n_here;
                end
                hold(axW,'on');
                colc = leicolors(mod(find(unique_clusters==cid, 1)-1,size(leicolors,1))+1,:);
                if ~isempty(p.Results.samplerate_hz)
                    tvec = (0:size(W,2)-1)/p.Results.samplerate_hz*1000;
                    xlabel_str = 'Time (ms)';
                else
                    tvec = 1:size(W,2);
                    xlabel_str = 'Samples';
                end
                plot(axW, tvec, W(idx,:)', 'Color', [colc, 0.15], 'LineWidth', 0.8); % Python alpha=0.15, lw=0.8
                plot(axW, tvec, mean(W,1), 'Color', 'k', 'LineWidth', 2.4); % Python color='k', lw=2.4
                title(axW, sprintf('Cluster %d (n=%d)', cid, sum(mask)), 'FontSize', 10);
                xlabel(axW, xlabel_str);
                ylabel(axW, 'Amplitude');
                grid(axW, 'on');

                % Style Update (Match Python)
                box(axW, 'off');
                set(axW, 'GridAlpha', 0.25, 'LineWidth', 0.8);

                % Density image (row2) - match Python layout
                axD = subplot(3,ncols, col + ncols);
                density_image_matlab(W, axD, p.Results.samplerate_hz, 'cmap', 'inferno');

                % ISI histogram (row3)
                axI = subplot(3,ncols, col + 2*ncols);
                % Note: Hardcoding line_freq=60Hz. Change if needed.
                plot_isi_histogram(spike_times_ms(mask), p.Results.refractory_ms, 60, colc);
            end
        end

        sgtitle(sprintf('Cluster Report - Page %d/%d', page_i, length(pages)), 'FontSize', 14);
        figs{end+1} = fig;
    end

    % Metrics overview page
    if ~isempty(df_metrics) && ~isempty(SS)
        fig_metrics = create_metrics_overview_page(df_metrics, SS, p.Results.show_figures, leicolors);
        figs{end+1} = fig_metrics;
    end
    
    % cross-correlogram page
    if ~isempty(cross_correlograms)
        fig_ccg = create_cross_correlogram_page(cross_correlograms, l1_distances, p.Results.show_figures, leicolors);
        figs{end+1} = fig_ccg;
    end

    fprintf('Generated %d figure pages for cluster report\n', length(figs));
end

function plot_isi_histogram(spike_times, refractory_ms, line_freq, color)
    % PLOT_ISI_HISTOGRAM - Plot ISI distribution with violations (Python Style)
    ax = gca; % Get the current axes handle
    
    times_sorted = sort(spike_times);
    isi_ms = diff(times_sorted);
    isi_ms = isi_ms(isi_ms >= 0 & isfinite(isi_ms));
    
    if ~isempty(isi_ms)
        % Binning (Match Python: 2ms bins up to 100ms)
        nbins = 50;
        bin_step = 2;
        ISI_max_ms = nbins * bin_step;
        bin_edges = 0:bin_step:ISI_max_ms;
        
        [n, c] = histcounts(isi_ms, bin_edges);
        
        % Plot using bar
        bar(ax, c(1:end-1) + (bin_step/2), n, 'EdgeColor', 'none','FaceColor',color, 'BarWidth', 1);
        
        hold(ax, 'on');
        % Add line_freq lines (if any)
        if ~isempty(line_freq) && line_freq > 0
            line_period_ms = 1000 / line_freq; 
            for i_freq = 1:floor(ISI_max_ms / line_period_ms)
                interval = i_freq * line_period_ms;
                line([interval, interval], ylim(ax), 'LineWidth', 0.8, 'LineStyle', ':', 'Color', 'r');
            end
        end
        hold(ax, 'off');
        
        % Formatting (Match Python)
        xlim(ax, [0 ISI_max_ms]);
        n_viol = sum(isi_ms < refractory_ms);
        title(ax, sprintf('%d in < %.1fms', n_viol, refractory_ms), 'FontSize', 10);
        xlabel(ax, 'ISI (ms)');
        ylabel(ax, 'Count');
        box(ax, 'off');
        grid(ax, 'on');
        set(ax, 'GridAlpha', 0.25, 'LineWidth', 0.8);
        set(ax, 'TickDir', 'out');

    else
        title(ax, 'No ISIs', 'FontSize', 10);
        axis(ax, 'off');
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

function fig = create_metrics_overview_page(df_metrics, SS, show_figures, leicolors)
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

    all_unique_clusters = unique(df_metrics.cluster_id); 
    
    % Check for the presence of Cluster 0. If it's excluded from df_metrics, 
    % we adjust the index lookup to skip the first color (black).
    if ismember(0, all_unique_clusters)
        % If C0 IS in df_metrics (unlikely, but safe), colors are 1-based index
        cluster_list_for_colors = all_unique_clusters;
    else
        % If C0 is NOT in df_metrics (typical for metrics), skip the index 0 -> color 1 (black)
        cluster_list_for_colors = df_metrics.cluster_id; % [1, 2, 3, 4, 5, ...]
    end

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
                
                % Use cluster colors
                num_bars = height(df_metrics);
                bar_colors_rep = zeros(num_bars, 3);
                
                % Find the index of each cluster_id in the *full* cluster list (which defines leicolors)
                % We need the index of 'df_metrics.cluster_id' relative to 'all_unique_clusters'
                
                % We need a reference list of ALL cluster IDs in the data, including 0,
                % to use the indices of the leicolors array.
                % Since we don't have the original 'unique_clusters' here, let's assume
                % that C1 maps to leicolors(2,:), C2 maps to leicolors(3,:), etc., 
                % *if* C0 is excluded from df_metrics.
                
                % Get the indices relative to the cluster ID (1 is C1, 2 is C2, etc.)
                % Since C0 (ID 0) is index 1 (black), C1 (ID 1) is index 2.
                % The actual color index in leicolors is (cluster_id + 1).
                
                color_indices = df_metrics.cluster_id + 1;
                
                % Ensure no index is out of bounds
                valid_indices = color_indices(color_indices <= size(leicolors, 1));
                bar_colors_rep(1:numel(valid_indices), :) = leicolors(valid_indices, :);

                h = bar(ax, (1:num_bars), y, 'FaceColor', 'flat', 'BarWidth', 0.9);
                h.CData = bar_colors_rep; % This sets the individual bar colors

                if r == n_rows
                    xticks(ax, 1:num_bars);
                    xticklabels(ax, arrayfun(@(x)sprintf('c%d', df_metrics.cluster_id(x)), 1:num_bars, 'UniformOutput',false));
                    xtickangle(ax, 90);
                else
                    set(ax, 'XTick', []);
                end
                ylabel(ax, metrics_cols{idx});
                title(ax, metrics_cols{idx}, 'Interpreter', 'none');
                grid(ax, 'on');

                % Style Update (Match Python)
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

    % Silhouette heatmap in the next available slot
    sil_slot = n_metrics + 1;
    if sil_slot <= n_rows * cols_per_row
        ax_sil = axes_handles(ceil(sil_slot/cols_per_row), mod(sil_slot-1, cols_per_row) + 1);
        if isempty(SS)
            text(0.5,0.5,'No silhouette matrix', 'HorizontalAlignment','center','Parent', ax_sil);
            axis(ax_sil,'off');
        else
            S = SS;
            
            %    Ensures symmetry by filling NaNs with the transpose value.
            for i = 1:size(S, 1)
                for j = 1:size(S, 2)
                    if isnan(S(i, j)) && ~isnan(S(j, i))
                        S(i, j) = S(j, i);
                    end
                end
            end
            
            S(1:size(S,1)+1:end) = NaN; 
            
            %    In MATLAB's imagesc, we must explicitly set the masked area 
            %    (the upper triangle) to NaN.
            mask_upper_tri = triu(true(size(S)), 1); % Upper triangle, excluding diagonal
            S(mask_upper_tri) = NaN; % Set the upper triangle to NaN
            
            cluster_labels = arrayfun(@(x)sprintf('c%d', x), df_metrics.cluster_id, 'UniformOutput', false);

            % Plotting the masked matrix
            imagesc(ax_sil, S, [-1 1]);
            colormap(ax_sil, 'parula'); % 'parula' is MATLAB's 'viridis'
            cb = colorbar(ax_sil);
            cb.Label.String = 'Silhouette Score';

            ax_sil.XTick = 1:size(S,2);
            ax_sil.YTick = 1:size(S,1);
            ax_sil.XTickLabel = cluster_labels;
            ax_sil.YTickLabel = cluster_labels;
            xtickangle(ax_sil, 90);
            title(ax_sil, 'Silhouette Score Heatmap');
            axis(ax_sil, 'equal', 'tight'); % Match Python 'square=True'
            box(ax_sil, 'off');
        end
    end

    sgtitle('Cluster Metrics Overview', 'FontSize', 14);
end


function density_image_matlab(W, ax, samplerate_hz, varargin)
    p = inputParser;
    addParameter(p, 'w', 400, @isscalar);
    addParameter(p, 'h', 240, @isscalar);
    addParameter(p, 'cmap', 'inferno', @ischar); % Use 'inferno' to match Python
    parse(p, varargin{:});

    if isempty(W)
        axis(ax,'off'); return;
    end

    [n, T] = size(W);
    x_min = 0; x_max = T-1;
    y_min = min(W(:)); y_max = max(W(:));
    
    % Add padding to y-axis (matches Python)
    pad = 1e-6 * (y_max - y_min);
    if pad == 0, pad = 1e-6; end
    y_min = y_min - pad; y_max = y_max + pad;

    x_edges = linspace(x_min, x_max, p.Results.w+1);
    y_edges = linspace(y_min, y_max, p.Results.h+1);

    % Use the 200x interpolation factor from the Python default
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
    
    % Transpose H to H' to match Python's H.T
    D = H' ./ (numel(t_flat) * mean(diff(x_edges)) * mean(diff(y_edges)));
    
    
    %    plotting D with a LogNorm color scale.
    D_log = log10(D + eps); 

    %    vmin = log10( max( smallest non-zero D, 1e-12 ) )
    %    vmax = log10( max(D) )
    
    D_pos = D(D>0);
    if isempty(D_pos), D_pos = 1; end % Safety for empty arrays
    
    vmin_log = log10(max(min(D_pos), 1e-12));
    vmax_log = log10(max(D(:)));

    % Safety check if all data is zero
    if vmax_log <= vmin_log, vmax_log = vmin_log + 1; end 
    
    axes(ax);
    % Plot the log-scaled data
    imagesc(ax, [x_min x_max], [y_min y_max], D_log);
    set(ax, 'YDir', 'normal');

    try
        % 'inferno' is available in MATLAB R2017a+
        colormap(ax, p.Results.cmap); 
    catch
        % 'hot' is a great Black-Red-Yellow-White fallback
        colormap(ax, 'hot'); 
    end

    % Apply the caxis limits derived from Python's LogNorm
    caxis(ax, [vmin_log, vmax_log]);

    % Set background to black to match ax.set_facecolor("black")
    set(ax, 'Color', 'black');
    
    if ~isempty(samplerate_hz)
        xlabel(ax, 'Time (ms)');
        xticks_data = get(ax,'XTick');
        set(ax,'XTickLabel', arrayfun(@(x) sprintf('%.0f', x/samplerate_hz*1e3), xticks_data, 'UniformOutput', false));
    else
        xlabel(ax, 'Sample');
    end

    axis(ax, 'tight');
    
    % Add grid to match Python ax.grid()
   % grid(ax, 'on');
    %set(ax, 'GridColor', [1 1 1], 'GridAlpha', 0.2, 'LineWidth', 0.5);
    % Hide top/right spines
    box(ax, 'off');
    set(ax, 'YColor', [1 1 1], 'XColor', [1 1 1]); % Make ticks/labels white
end
function cluster_activity_kde_ax_mat(spike_times_ms, cluster_ids, recording_duration_ms, ax, time_pixels, cmapname)
    if nargin < 4, ax = gca; end
    if nargin < 5 || isempty(time_pixels), time_pixels = 100; end
    if nargin < 6, cmapname = 'inferno'; end % Default to 'inferno'

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
    
    try
        colormap(ax, cmapname); 
    catch
        colormap(ax, 'hot'); % Fallback
    end

    %    Set min to 0 (black) and max to the 98th percentile
    %    to prevent a few hot spots from washing out the colormap.
    vmax = quantile(kde_matrix(:), 0.98);
    if vmax <= 0, vmax = max(kde_matrix(:)); end % Handle sparse data
    if vmax == 0, vmax = 1; end % Handle all-zero data

    caxis(ax, [0, vmax]);
    
    set(ax, 'Color', 'black');
    set(ax, 'YColor', [1 1 1], 'XColor', [1 1 1]);
    

    set(ax, 'YTick', 1:K, 'YTickLabel', arrayfun(@(c)sprintf('Cl: %d', c), clusters, 'UniformOutput', false));
    xlabel(ax, 'Time (min)');
    ylabel(ax, 'Presence Plot');
    set(ax, 'YDir', 'reverse');
end

function fig = create_cross_correlogram_page(cross_correlograms, l1_distances, show_figures, leicolors)
% CREATE_CROSS_CORRELOGRAM_PAGE - Create page with cross-correlograms and L1 distances
    if show_figures
        fig = figure('Visible', 'on', 'Units','normalized','Position',[0.1 0.1 0.9 0.8]);
    else
        fig = figure('Visible', 'off');
    end

    % Get unique clusters for colors
    all_clusters = unique([[cross_correlograms.ref_cluster], [cross_correlograms.target_cluster]]);
    
    % Plot first 4 cross-correlograms (left side)
    n_to_plot = min(4, length(cross_correlograms));
    for i = 1:n_to_plot
        subplot(2, 3, i);
        ccg_data = cross_correlograms(i);
        
        % Get colors for the clusters
        ref_color_idx = mod(find(all_clusters == ccg_data.ref_cluster, 1) - 1, size(leicolors, 1)) + 1;
        target_color_idx = mod(find(all_clusters == ccg_data.target_cluster, 1) - 1, size(leicolors, 1)) + 1;
        
        ref_color = leicolors(ref_color_idx, :);
        target_color = leicolors(target_color_idx, :);
        
        % Plot cross-correlogram
        bar(ccg_data.bin_centers, ccg_data.ccg, 1, 'FaceColor', target_color, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
        hold on;
        plot([0 0], ylim, 'k--', 'LineWidth', 1);
        hold off;
        
        title(sprintf('C%d → C%d', ccg_data.ref_cluster, ccg_data.target_cluster));
        xlabel('Time Lag (ms)');
        ylabel('Rate (spikes/ref)');
        grid on;
        box off;
    end

    % Plot L1 distance matrix (right side - spans positions 5-6)
    if ~isempty(l1_distances)
        ax_dist = subplot(2, 3, [5, 6]);
        
        imagesc(l1_distances);
        
        % Set labels
        cluster_labels = arrayfun(@(x) sprintf('C%d', x), all_clusters, 'UniformOutput', false);
        set(ax_dist, 'XTick', 1:length(all_clusters), 'XTickLabel', cluster_labels);
        set(ax_dist, 'YTick', 1:length(all_clusters), 'YTickLabel', cluster_labels);
        xtickangle(ax_dist, 45);
        
        % Add colorbar and title
        colorbar(ax_dist);
        title('L1 Distance Matrix', 'FontSize', 12);
        xlabel('Target Cluster');
        ylabel('Reference Cluster');
        
        % Add distance values as text
        for i = 1:size(l1_distances, 1)
            for j = 1:size(l1_distances, 2)
                if i ~= j && ~isnan(l1_distances(i,j))
                    text(j, i, sprintf('%.2f', l1_distances(i,j)), ...
                        'HorizontalAlignment', 'center', 'FontSize', 8, ...
                        'Color', 'white', 'FontWeight', 'bold');
                end
            end
        end
        
        axis(ax_dist, 'image');
        colormap(ax_dist, 'hot');
    else
        subplot(2, 3, [5, 6]);
        text(0.5, 0.5, 'No L1 distances computed', 'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    end

    sgtitle('Cross-Correlogram Analysis with L1 Distances', 'FontSize', 14, 'FontWeight', 'bold');
end