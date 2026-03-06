function results = compareSpikeMorphology(correlate_results, spikes_t1, spikes_t2, varargin)
    % compareSpikeMorphology - Compare spike waveform shapes using Euclidean distance
    %
    % Inputs:
    %   correlate_results  : Output struct from correlateTimelines()
    %   spikes_t1          : Spike waveform matrix (num_spikes x 64) for new set
    %   spikes_t2          : Spike waveform matrix (num_spikes x 64) for old set
    %
    % Optional name-value pairs:
    %   'plot_examples'    : Number of spike pair examples to plot (default: 10)
    %
    % Output:
    %   results struct with:
    %     .shared_distances    : table [idx1, idx2, euclidean_dist]
    %     .dist_stats          : mean, median, std of shared distances
    %     .unique_t1_stats     : stats for unique-to-t1 spikes (for reference)
    %     .unique_t2_stats     : stats for unique-to-t2 spikes (for reference)

    p = inputParser;
    addParameter(p, 'plot_examples', 10, @isnumeric);
    parse(p, varargin{:});
    plot_examples = p.Results.plot_examples;

    % ── Extract spike indices and timestamps from correlate_results ───────
    shared_t = correlate_results.shared;
    unique_t1_t = correlate_results.unique_t1;
    unique_t2_t = correlate_results.unique_t2;

    n_shared = height(shared_t);

    % ── Compute Euclidean distances for shared spike pairs ────────────────
    shared_distances = zeros(n_shared, 1);

    for i = 1:n_shared
        idx1 = shared_t.idx1(i);
        idx2 = shared_t.idx2(i);

        % Extract waveforms
        spike1 = spikes_t1(idx1, :);
        spike2 = spikes_t2(idx2, :);

        % Handle case where spikes have different lengths (pad with zeros)
        len1 = length(spike1);
        len2 = length(spike2);
        max_len = max(len1, len2);

        if len1 < max_len
            spike1 = [spike1, zeros(1, max_len - len1)];
        end
        if len2 < max_len
            spike2 = [spike2, zeros(1, max_len - len2)];
        end

        % Compute Euclidean distance
        shared_distances(i) = norm(spike1 - spike2);
    end

    results.shared_distances = table(...
        shared_t.idx1, shared_t.idx2, shared_distances, ...
        'VariableNames', {'idx1', 'idx2', 'dist'});

    results.dist_stats = struct(...
        'mean', mean(shared_distances), ...
        'median', median(shared_distances), ...
        'std', std(shared_distances), ...
        'min', min(shared_distances), ...
        'max', max(shared_distances));

    % ── Store unique spike stats for reference ────────────────────────────
    results.n_shared = n_shared;
    results.n_unique_t1 = height(unique_t1_t);
    results.n_unique_t2 = height(unique_t2_t);

    % ── VISUALIZATION ──────────────────────────────────────────────────────
    figure('Color', 'w', 'Name', 'Spike Morphology Comparison', 'Position', [200, 200, 1200, 700]);

    % Plot 1: Distance histogram
    subplot(2,3,1);
    histogram(shared_distances, 30, 'FaceColor', [0.4 0.2 0.7], 'EdgeColor', 'none');
    xlabel('Euclidean Distance');
    ylabel('Count');
    title(sprintf('Shared Spike Shape Distances (n=%d)', n_shared));
    hold on;
    xline(results.dist_stats.mean, 'r--', 'LineWidth', 1.5);
    xline(results.dist_stats.median, 'g--', 'LineWidth', 1.5);
    legend(sprintf('μ = %.2f', results.dist_stats.mean), sprintf('med = %.2f', results.dist_stats.median));
    grid on;

    % Plot 2: Distance vs time1
    subplot(2,3,2);
    scatter(shared_t.t1, shared_distances, 20, shared_distances, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');
    colorbar;
    xlabel('Time t1 (ms)');
    ylabel('Euclidean Distance');
    title('Distance vs Time (t1)');
    grid on;

    % Plot 3: Distance vs time2
    subplot(2,3,3);
    scatter(shared_t.t2, shared_distances, 20, shared_distances, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');
    colorbar;
    xlabel('Time t2 (ms)');
    ylabel('Euclidean Distance');
    title('Distance vs Time (t2)');
    grid on;

    % Plot 4: Distance vs offset (dt)
    subplot(2,3,4);
    scatter(shared_t.dt * 1000, shared_distances, 20, shared_distances, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');
    colorbar;
    xlabel('Time Offset (µs)');
    ylabel('Euclidean Distance');
    title('Distance vs Time Offset');
    grid on;

    % Plot 5: Cumulative distribution
    subplot(2,3,5);
    [f, x] = ecdf(shared_distances);
    plot(x, f, 'LineWidth', 2, 'Color', [0.4 0.2 0.7]);
    xlabel('Euclidean Distance');
    ylabel('Cumulative Fraction');
    title('CDF of Spike Shape Distances');
    grid on;

    % Plot 6: Summary stats text
    subplot(2,3,6);
    axis off;
    stats_text = sprintf(...
        ['Spike Morphology Comparison\n\n' ...
         'Shared pairs:      %d\n' ...
         'Unique (t1 only):  %d\n' ...
         'Unique (t2 only):  %d\n\n' ...
         'Distance Statistics:\n' ...
         '  Mean:     %.3f\n' ...
         '  Median:   %.3f\n' ...
         '  Std:      %.3f\n' ...
         '  Min:      %.3f\n' ...
         '  Max:      %.3f\n'], ...
        results.n_shared, results.n_unique_t1, results.n_unique_t2, ...
        results.dist_stats.mean, results.dist_stats.median, ...
        results.dist_stats.std, results.dist_stats.min, results.dist_stats.max);
    text(0.1, 0.5, stats_text, 'FontName', 'Courier', 'FontSize', 10, 'VerticalAlignment', 'middle');

    % ── Example overlays: plot lowest and highest distance pairs ──────────
    n_examples = min(plot_examples, n_shared);
    if n_examples > 0
        [~, sort_idx] = sort(shared_distances);

        % Lowest distance examples
        fig_low = figure('Color', 'w', 'Name', 'Closest Spike Pairs', ...
            'Position', [100, 100, 1000, 400 + 100*ceil(n_examples/3)]);

        for k = 1:min(ceil(n_examples/2), n_examples)
            i = sort_idx(k);
            idx1 = shared_t.idx1(i);
            idx2 = shared_t.idx2(i);

            subplot(ceil(n_examples/2), 3, k);
            spike1 = spikes_t1(idx1, :);
            spike2 = spikes_t2(idx2, :);

            % Normalize lengths
            max_len = max(length(spike1), length(spike2));
            spike1 = [spike1, zeros(1, max(0, max_len - length(spike1)))];
            spike2 = [spike2, zeros(1, max(0, max_len - length(spike2)))];

            plot(spike1, 'b', 'LineWidth', 1.5); hold on;
            plot(spike2, 'r--', 'LineWidth', 1.5);
            xlabel('Sample');
            ylabel('Amplitude (µV)');
            dt_us = shared_t.dt(i) * 1000;  % convert ms to µs
            title(sprintf('Pair %d: dist=%.2f, dt=%.2f µs\n(idx1=%d, idx2=%d)', ...
                k, shared_distances(i), dt_us, idx1, idx2));
            legend('t1 (new)', 't2 (old)', 'Location', 'best');
            grid on;
        end

        % Highest distance examples
        fig_high = figure('Color', 'w', 'Name', 'Most Dissimilar Spike Pairs', ...
            'Position', [150, 150, 1000, 400 + 100*ceil(n_examples/3)]);

        for k = 1:min(ceil(n_examples/2), n_examples)
            i = sort_idx(n_shared - k + 1);
            idx1 = shared_t.idx1(i);
            idx2 = shared_t.idx2(i);

            subplot(ceil(n_examples/2), 3, k);
            spike1 = spikes_t1(idx1, :);
            spike2 = spikes_t2(idx2, :);

            % Normalize lengths
            max_len = max(length(spike1), length(spike2));
            spike1 = [spike1, zeros(1, max(0, max_len - length(spike1)))];
            spike2 = [spike2, zeros(1, max(0, max_len - length(spike2)))];

            plot(spike1, 'b', 'LineWidth', 1.5); hold on;
            plot(spike2, 'r--', 'LineWidth', 1.5);
            xlabel('Sample');
            ylabel('Amplitude (µV)');
            dt_us = shared_t.dt(i) * 1000;  % convert ms to µs
            title(sprintf('Pair %d: dist=%.2f, dt=%.2f µs\n(idx1=%d, idx2=%d)', ...
                k, shared_distances(i), dt_us, idx1, idx2));
            legend('t1 (new)', 't2 (old)', 'Location', 'best');
            grid on;
        end

        % ── Figure: Highest distance pairs binned by time ──────────────────
        % Bin shared spikes by time and find max distance in each bin
        t_recording = max(shared_t.t1);
        n_bins = min(12, max(5, floor(sqrt(n_shared))));  % adaptive bin count
        bin_edges = linspace(0, t_recording, n_bins + 1);
        
        fig_binned = figure('Color', 'w', 'Name', 'Worst Spikes per Time Bin', ...
            'Position', [200, 200, 1200, 500 + 100*ceil(n_bins/4)]);
        
        subplot_idx = 1;
        for b = 1:n_bins
            % Find spikes in this time bin
            in_bin = (shared_t.t1 >= bin_edges(b)) & (shared_t.t1 < bin_edges(b+1));
            
            if ~any(in_bin)
                subplot(ceil(n_bins/4), 4, subplot_idx);
                axis off;
                text(0.5, 0.5, 'No spikes', 'HorizontalAlignment', 'center', 'Units', 'normalized');
                subplot_idx = subplot_idx + 1;
                continue;
            end
            
            % Find spike with max distance in this bin
            bin_distances = shared_distances(in_bin);
            [~, max_idx_in_bin] = max(bin_distances);
            
            % Get actual indices
            bin_indices = find(in_bin);
            actual_idx = bin_indices(max_idx_in_bin);
            
            idx1 = shared_t.idx1(actual_idx);
            idx2 = shared_t.idx2(actual_idx);
            max_dist_in_bin = shared_distances(actual_idx);
            
            subplot(ceil(n_bins/4), 4, subplot_idx);
            spike1 = spikes_t1(idx1, :);
            spike2 = spikes_t2(idx2, :);
            
            % Normalize lengths
            max_len = max(length(spike1), length(spike2));
            spike1 = [spike1, zeros(1, max(0, max_len - length(spike1)))];
            spike2 = [spike2, zeros(1, max(0, max_len - length(spike2)))];
            
            plot(spike1, 'b', 'LineWidth', 1.5); hold on;
            plot(spike2, 'r--', 'LineWidth', 1.5);
            xlabel('Sample');
            ylabel('Amplitude (µV)');
            dt_us = shared_t.dt(actual_idx) * 1000;  % convert ms to µs
            title(sprintf('Bin %.1f-%.1f s | dist=%.2f, dt=%.2f µs\n(idx1=%d, idx2=%d)', ...
                bin_edges(b)/1000, bin_edges(b+1)/1000, max_dist_in_bin, dt_us, idx1, idx2));
            legend('t1 (new)', 't2 (old)', 'Location', 'best', 'FontSize', 8);
            grid on;
            
            subplot_idx = subplot_idx + 1;
        end
    end

    % ── Console output ────────────────────────────────────────────────────
    fprintf('\n--- Spike Morphology Comparison ---\n');
    fprintf('  Shared spike pairs     : %d\n', results.n_shared);
    fprintf('  Unique to t1           : %d\n', results.n_unique_t1);
    fprintf('  Unique to t2           : %d\n', results.n_unique_t2);
    fprintf('\nEuclidean Distance Statistics:\n');
    fprintf('  Mean                   : %.4f\n', results.dist_stats.mean);
    fprintf('  Median                 : %.4f\n', results.dist_stats.median);
    fprintf('  Std Dev                : %.4f\n', results.dist_stats.std);
    fprintf('  Range                  : [%.4f, %.4f]\n', results.dist_stats.min, results.dist_stats.max);
    fprintf('------------------------------------\n\n');

end
