function results = correlateTimelines(t1, t2, binSize)
    % correlateTimelines - Compare two spike timestamp vectors.
    % t1, t2   : Raw timestamps (ms) from 30k and 20k sets
    % binSize  : Bin window for spike-density correlation (default 50 ms)
    %
    % Shared spikes are identified with a tolerance of 1 sample at 30 kHz
    % (1/30000 s = 1/30 ms ≈ 0.0333 ms). Outputs include shared/unique
    % spike sets and a correlogram of timing offsets for shared pairs.

    if nargin < 3, binSize = 50; end

    tol_ms = (22/30000) * 1000;  % 1 sample at 30 kHz, in ms (~0.0333 ms)

    % ── 1. SHARED / UNIQUE SPIKE MATCHING ────────────────────────────────
    [t1s, idx1_sort] = sort(t1(:));
    [t2s, idx2_sort] = sort(t2(:));

    matched1 = false(size(t1s));
    matched2 = false(size(t2s));
    dt_shared = zeros(min(numel(t1s), numel(t2s)), 1);
    n_match = 0;

    j = 1;
    for i = 1:numel(t1s)
        % advance j past spikes that are definitely too early
        while j <= numel(t2s) && t2s(j) < t1s(i) - tol_ms
            j = j + 1;
        end
        if j <= numel(t2s) && abs(t2s(j) - t1s(i)) <= tol_ms
            matched1(i) = true;
            matched2(j) = true;
            n_match = n_match + 1;
            dt_shared(n_match) = t2s(j) - t1s(i);  % offset in ms
            j = j + 1;
        end
    end
    dt_shared = dt_shared(1:n_match);

    results.shared = table(...
        idx1_sort(matched1),  idx2_sort(matched2), ...
        t1s(matched1),        t2s(matched2), ...
        dt_shared, ...
        'VariableNames', {'idx1','idx2','t1','t2','dt'});

    results.unique_t1 = table(...
        idx1_sort(~matched1), t1s(~matched1), ...
        'VariableNames', {'idx1','t1'});

    results.unique_t2 = table(...
        idx2_sort(~matched2), t2s(~matched2), ...
        'VariableNames', {'idx2','t2'});

    results.n_shared    = n_match;
    results.n_unique_t1 = sum(~matched1);
    results.n_unique_t2 = sum(~matched2);

    % ── 2. TEMPORAL CORRELATION (binned spike density) ───────────────────
    maxT  = max(max(t1s), max(t2s));
    edges = 0:binSize:maxT;

    counts1 = histcounts(t1s, edges);
    counts2 = histcounts(t2s, edges);

    [R_matrix, p_val]  = corrcoef(counts1, counts2);
    results.R_value    = R_matrix(1,2);
    results.p_value    = p_val(1,2);

    % ── 3. FIGURE 1: Global Temporal Correlation ─────────────────────────
    figure('Color', 'w', 'Name', 'Global Temporal Correlation', 'Position', [100, 100, 900, 600]);

    % Plot 1: Event Density Scatter
    subplot(2,2,1);
    scatter(counts1, counts2, 25, 'filled', 'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
    hold on;
    p_fit = polyfit(counts1, counts2, 1);
    plot(counts1, polyval(p_fit, counts1), 'r-', 'LineWidth', 2);
    xlabel('Spikes / Bin (30k Set)');
    ylabel('Spikes / Bin (20k Set)');
    title(['Global Event Density (R = ', num2str(results.R_value, 4), ')']);
    grid on;

    % Plot 2: Firing Rate Envelopes
    subplot(2,2,2);
    t_axis = edges(1:end-1) + binSize/2;
    plot(t_axis/1000, counts1, 'b', 'LineWidth', 1); hold on;
    plot(t_axis/1000, counts2, 'r--', 'LineWidth', 1);
    xlabel('Time (seconds)');
    ylabel(['Spikes per ', num2str(binSize), 'ms']);
    title('Firing Rate Envelope Alignment');
    legend({['30k Set (R=',num2str(results.R_value,3),')'], '20k Set'}, 'Location', 'northeast');
    grid on;

    % Plot 3: Cumulative Distribution
    subplot(2,2,3:4);
    [f1, x1] = ecdf(t1s);
    [f2, x2] = ecdf(t2s);
    plot(x1/1000, f1, 'b', 'LineWidth', 2); hold on;
    plot(x2/1000, f2, 'r--', 'LineWidth', 2);
    xlabel('Time (seconds)');
    ylabel('Normalized Cumulative Events');
    title('Cumulative Temporal Alignment');
    legend('30k Distribution', '20k Distribution', 'Location', 'southeast');
    grid on;

    % ── 4. FIGURE 2: Shared / Unique Spike Comparison ────────────────────
    figure('Color', 'w', 'Name', 'Shared & Unique Spike Comparison', 'Position', [120, 120, 1000, 700]);

    % Plot 1: Spike count breakdown
    subplot(2,2,1);
    bar_vals = [results.n_shared, results.n_unique_t1, results.n_unique_t2];
    bar_h = bar(bar_vals, 'FaceColor', 'flat');
    bar_h.CData = [0.2 0.6 0.2; 0.2 0.4 0.8; 0.8 0.3 0.2];
    set(gca, 'XTickLabel', {'Shared', 'Unique 30k', 'Unique 20k'});
    ylabel('Spike Count');
    title('Spike Membership');
    grid on;
    text(1:3, bar_vals + max(bar_vals)*0.02, num2str(bar_vals(:)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);

    % Plot 2: Shared spikes as fraction of each set
    subplot(2,2,2);
    pct1 = 100 * results.n_shared / numel(t1s);
    pct2 = 100 * results.n_shared / numel(t2s);
    bar([pct1, pct2], 'FaceColor', 'flat', 'CData', [0.2 0.6 0.2; 0.2 0.6 0.2]);
    set(gca, 'XTickLabel', {'30k Set', '20k Set'});
    ylabel('% of Set Shared');
    title(sprintf('Shared Fraction (tol = 1/30k s)\nn = %d shared', results.n_shared));
    ylim([0, 110]);
    grid on;
    text(1:2, [pct1, pct2] + 2, {sprintf('%.1f%%', pct1), sprintf('%.1f%%', pct2)}, ...
        'HorizontalAlignment', 'center', 'FontSize', 9);

    % Plot 3: Cross-correlogram of shared spikes
    subplot(2,2,3);
    if n_match > 1
        % For shared spikes: t2 time - t1 time
        shared_offsets = results.shared.t2 - results.shared.t1;
        nbins = max(100, round(sqrt(n_match) * 3));
        histogram(shared_offsets, nbins, 'FaceColor', [0.2 0.6 0.2], 'EdgeColor', 'none');
        xlabel('t2 − t1 (ms)');
        ylabel('Count');
        title(sprintf('Cross-Correlogram: Shared Spikes\nmedian = %.4f ms, SD = %.4f ms', ...
            median(shared_offsets), std(shared_offsets)));
        xline(0, 'k--', 'LineWidth', 1);
    else
        text(0.5, 0.5, 'No shared spikes', 'HorizontalAlignment', 'center', 'Units', 'normalized');
        title('Cross-Correlogram: Shared Spikes');
    end
    grid on;

    % Plot 4: Cross-correlogram of unique spikes
    subplot(2,2,4);
    unique_offsets = [];
    
    % For each unique t1 spike, find nearest t2 spike
    if results.n_unique_t1 > 0 && numel(t2s) > 0
        for ut1 = results.unique_t1.t1'
            [~, nearest_idx] = min(abs(t2s - ut1));
            unique_offsets = [unique_offsets; t2s(nearest_idx) - ut1];
        end
    end
    
    % For each unique t2 spike, find nearest t1 spike
    if results.n_unique_t2 > 0 && numel(t1s) > 0
        for ut2 = results.unique_t2.t2'
            [~, nearest_idx] = min(abs(t1s - ut2));
            unique_offsets = [unique_offsets; ut2 - t1s(nearest_idx)];
        end
    end
    
    if ~isempty(unique_offsets)
        nbins = max(100, round(sqrt(length(unique_offsets)) * 3));
        histogram(unique_offsets, nbins, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'none');
        xlabel('Nearest neighbor offset (ms)');
        ylabel('Count');
        title(sprintf('Cross-Correlogram: Unique Spikes\nmedian = %.4f ms, SD = %.4f ms', ...
            median(unique_offsets), std(unique_offsets)));
        xline(0, 'k--', 'LineWidth', 1);
    else
        text(0.5, 0.5, 'No unique spikes', 'HorizontalAlignment', 'center', 'Units', 'normalized');
        title('Cross-Correlogram: Unique Spikes');
    end
    grid on;

    % ── 5. CONSOLE SUMMARY ───────────────────────────────────────────────
    fprintf('\n--- correlateTimelines Summary ---\n');
    fprintf('  Tolerance        : 1/30000 s = %.4f ms\n', tol_ms);
    fprintf('  30k spikes       : %d\n', numel(t1s));
    fprintf('  20k spikes       : %d\n', numel(t2s));
    fprintf('  Shared           : %d  (%.1f%% of 30k | %.1f%% of 20k)\n', ...
        results.n_shared, pct1, pct2);
    fprintf('  Unique to 30k    : %d\n', results.n_unique_t1);
    fprintf('  Unique to 20k    : %d\n', results.n_unique_t2);
    fprintf('  Density R        : %.4f  (p = %.2e)\n', results.R_value, results.p_value);
    fprintf('----------------------------------\n\n');
end