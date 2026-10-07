function results = compareSpikeMorphology(correlate_results, spikes_t1, spikes_t2, varargin)
    % compareSpikeMorphology - Compare spike waveform shapes using Euclidean distance
    %
    % Inputs:
    %   correlate_results  : Output from correlateSpikeTrains (shared_overlap must be a table)
    %   spikes_t1          : Waveform matrix (n_spikes x n_samples) for dataset 1
    %   spikes_t2          : Waveform matrix (n_spikes x n_samples) for dataset 2
    %
    % Optional name-value pairs:
    %   'plot_examples'    : Number of example pairs to plot (default: 10)
    %   'label1'           : Label for dataset 1 (default: 't1 (MATLAB)')
    %   'label2'           : Label for dataset 2 (default: 't2 (Python)')
    %
    % Added diagnostics vs original:
    %   - Amplitude ratio per matched pair (catches scaling/filter differences)
    %   - Peak alignment check (catches timing shift within waveform)
    %   - Unique spike waveform inspection
    %   - Per-bin amplitude ratio to detect time-varying filter drift

    p = inputParser;
    addParameter(p, 'plot_examples', 10, @isnumeric);
    addParameter(p, 'label1', 't1 (MATLAB)', @ischar);
    addParameter(p, 'label2', 't2 (Python)',  @ischar);
    parse(p, varargin{:});
    plot_examples = p.Results.plot_examples;
    label1 = p.Results.label1;
    label2 = p.Results.label2;

    shared_t   = correlate_results.shared_overlap;
    unique_t1_t = correlate_results.unique_data1;
    unique_t2_t = correlate_results.unique_data2;
    n_shared   = height(shared_t);

    % ── Euclidean distance + amplitude ratio for every shared pair ────────
    shared_distances  = zeros(n_shared, 1);
    amplitude_ratios  = zeros(n_shared, 1);   % peak(t1)/peak(t2)
    peak_offset_samp  = zeros(n_shared, 1);   % sample offset of minimum

    for i = 1:n_shared
        idx1 = shared_t.idx1(i);
        idx2 = shared_t.idx2(i);

        s1 = spikes_t1(idx1, :);
        s2 = spikes_t2(idx2, :);

        % Pad to equal length if needed
        ml = max(length(s1), length(s2));
        s1 = [s1, zeros(1, ml-length(s1))];
        s2 = [s2, zeros(1, ml-length(s2))];

        shared_distances(i) = norm(s1 - s2);

        % Amplitude ratio — peak absolute value
        p1 = max(abs(s1));
        p2 = max(abs(s2));
        if p2 > 0
            amplitude_ratios(i) = p1 / p2;
        else
            amplitude_ratios(i) = NaN;
        end

        % Peak (minimum) position offset between t1 and t2
        [~, loc1] = min(s1);
        [~, loc2] = min(s2);
        peak_offset_samp(i) = loc1 - loc2;
    end

    results.shared_distances = table(...
        shared_t.idx1, shared_t.idx2, shared_distances, ...
        amplitude_ratios, peak_offset_samp, ...
        'VariableNames', {'idx1','idx2','dist','amp_ratio','peak_offset_samp'});

    results.dist_stats = struct(...
        'mean',   mean(shared_distances), ...
        'median', median(shared_distances), ...
        'std',    std(shared_distances), ...
        'min',    min(shared_distances), ...
        'max',    max(shared_distances));

    results.amp_ratio_stats = struct(...
        'mean',   nanmean(amplitude_ratios), ...
        'median', nanmedian(amplitude_ratios), ...
        'std',    nanstd(amplitude_ratios));

    results.n_shared   = n_shared;
    results.n_unique_t1 = height(unique_t1_t);
    results.n_unique_t2 = height(unique_t2_t);

    % ── Console output ────────────────────────────────────────────────────
    fprintf('\n--- Spike Morphology Comparison ---\n');
    fprintf('  Shared spike pairs     : %d\n', n_shared);
    fprintf('  Unique to t1 (%s): %d\n', label1, results.n_unique_t1);
    fprintf('  Unique to t2 (%s): %d\n', label2, results.n_unique_t2);
    fprintf('\nEuclidean Distance Statistics:\n');
    fprintf('  Mean=%.4f  Median=%.4f  Std=%.4f  [%.4f, %.4f]\n', ...
        results.dist_stats.mean, results.dist_stats.median, ...
        results.dist_stats.std, results.dist_stats.min, results.dist_stats.max);
    fprintf('\nAmplitude Ratio (%s/%s) Statistics:\n', label1, label2);
    fprintf('  Mean=%.4f  Median=%.4f  Std=%.4f\n', ...
        results.amp_ratio_stats.mean, results.amp_ratio_stats.median, results.amp_ratio_stats.std);
    if abs(results.amp_ratio_stats.median - 1) > 0.05
        fprintf('  *** Median ratio %.3f is far from 1.0 — likely a FILTER DIFFERENCE\n', ...
            results.amp_ratio_stats.median);
        if results.amp_ratio_stats.median > 1
            fprintf('      %s spikes have LARGER amplitudes than %s\n', label1, label2);
        else
            fprintf('      %s spikes have LARGER amplitudes than %s\n', label2, label1);
        end
    else
        fprintf('  Amplitudes are consistent between pipelines (ratio ≈ 1).\n');
    end
    fprintf('\nPeak sample offset (t1 peak pos - t2 peak pos):\n');
    fprintf('  Mean=%.2f  Median=%.2f  Std=%.2f samples\n', ...
        mean(peak_offset_samp), median(peak_offset_samp), std(peak_offset_samp));
    if abs(median(peak_offset_samp)) > 1
        fprintf('  *** Median offset %+.1f samples — peaks are not aligned\n', ...
            median(peak_offset_samp));
    else
        fprintf('  Peak positions are well aligned.\n');
    end
    fprintf('------------------------------------\n\n');

    % ── Main summary figure ───────────────────────────────────────────────
    figure('Color','w','Name','Spike Morphology Comparison','Position',[200 200 1400 800]);

    subplot(2,4,1);
    histogram(shared_distances, 30, 'FaceColor',[0.4 0.2 0.7],'EdgeColor','none');
    xline(results.dist_stats.mean,  'r--','LineWidth',1.5);
    xline(results.dist_stats.median,'g--','LineWidth',1.5);
    xlabel('Euclidean Distance'); ylabel('Count');
    title(sprintf('Shape Distances (n=%d)', n_shared));
    legend(sprintf('μ=%.1f',results.dist_stats.mean), ...
           sprintf('med=%.1f',results.dist_stats.median));
    grid on;

    subplot(2,4,2);
    histogram(amplitude_ratios, 30, 'FaceColor',[0.2 0.6 0.4],'EdgeColor','none');
    xline(1,'k--','LineWidth',1.5);
    xline(results.amp_ratio_stats.median,'r--','LineWidth',1.5);
    xlabel(sprintf('Amplitude Ratio (%s/%s)', label1, label2));
    ylabel('Count');
    title(sprintf('Amplitude Ratio (med=%.3f)', results.amp_ratio_stats.median));
    grid on;

    subplot(2,4,3);
    histogram(peak_offset_samp, -10:10, 'FaceColor',[0.8 0.4 0.2],'EdgeColor','none');
    xline(0,'k--','LineWidth',1.5);
    xlabel('Peak Offset (samples)'); ylabel('Count');
    title(sprintf('Peak Position Offset (med=%+.1f)', median(peak_offset_samp)));
    grid on;

    subplot(2,4,4);
    scatter(shared_t.dt*1000, shared_distances, 15, amplitude_ratios, 'filled',...
        'MarkerFaceAlpha',0.5,'MarkerEdgeColor','none');
    colorbar; clim([0.5 1.5]);
    xlabel('Time Offset (µs)'); ylabel('Euclidean Distance');
    title('Distance vs dt (colour=amp ratio)');
    grid on;

    subplot(2,4,5);
    scatter(shared_t.t1, amplitude_ratios, 15, shared_distances, 'filled',...
        'MarkerFaceAlpha',0.5,'MarkerEdgeColor','none');
    colorbar;
    yline(1,'k--'); yline(results.amp_ratio_stats.median,'r--');
    xlabel('Time t1 (ms)'); ylabel('Amp Ratio');
    title('Amplitude Ratio over Time — flat=filter OK, drift=problem');
    grid on;

    subplot(2,4,6);
    scatter(shared_t.t1, shared_distances, 15, amplitude_ratios, 'filled',...
        'MarkerFaceAlpha',0.5,'MarkerEdgeColor','none');
    colorbar; clim([0.5 1.5]);
    xlabel('Time t1 (ms)'); ylabel('Euclidean Distance');
    title('Distance vs Time (colour=amp ratio)');
    grid on;

    subplot(2,4,7);
    [f,x] = ecdf(shared_distances);
    plot(x, f, 'LineWidth',2,'Color',[0.4 0.2 0.7]);
    xlabel('Euclidean Distance'); ylabel('Cumulative Fraction');
    title('CDF of Distances'); grid on;

    subplot(2,4,8);
    axis off;
    txt = sprintf([...
        'Spike Morphology Summary\n\n'...
        'Shared:         %d\n'...
        'Unique t1:      %d\n'...
        'Unique t2:      %d\n\n'...
        'Distance:\n'...
        '  Mean   %.3f\n'...
        '  Median %.3f\n'...
        '  Std    %.3f\n\n'...
        'Amp Ratio (%s/%s):\n'...
        '  Mean   %.4f\n'...
        '  Median %.4f\n'...
        '  Std    %.4f\n\n'...
        'Peak offset (samp):\n'...
        '  Median %+.2f\n'], ...
        n_shared, results.n_unique_t1, results.n_unique_t2,...
        results.dist_stats.mean, results.dist_stats.median, results.dist_stats.std,...
        label1, label2,...
        results.amp_ratio_stats.mean, results.amp_ratio_stats.median, results.amp_ratio_stats.std,...
        median(peak_offset_samp));
    text(0.05,0.95,txt,'FontName','Courier','FontSize',9,...
        'VerticalAlignment','top','Units','normalized');

    % ── Mean waveforms comparison ─────────────────────────────────────────
    % Most useful single plot: if filter differs, mean shapes will diverge
    figure('Color','w','Name','Mean Waveform Comparison');
    n_samp1 = size(spikes_t1,2);
    n_samp2 = size(spikes_t2,2);
    ml = max(n_samp1, n_samp2);

    % Use only shared spikes for fair comparison
    idx1s = shared_t.idx1;
    idx2s = shared_t.idx2;
    s1_mat = [spikes_t1(idx1s,:), zeros(n_shared, ml-n_samp1)];
    s2_mat = [spikes_t2(idx2s,:), zeros(n_shared, ml-n_samp2)];

    m1 = mean(s1_mat,1);
    m2 = mean(s2_mat,1);
    sd1 = std(s1_mat,0,1);
    sd2 = std(s2_mat,0,1);
    t_ax = 1:ml;

    subplot(1,2,1);
    fill([t_ax, fliplr(t_ax)], [m1+sd1, fliplr(m1-sd1)], ...
        'b','FaceAlpha',0.15,'EdgeColor','none'); hold on;
    fill([t_ax, fliplr(t_ax)], [m2+sd2, fliplr(m2-sd2)], ...
        'r','FaceAlpha',0.15,'EdgeColor','none');
    plot(t_ax, m1, 'b', 'LineWidth',2);
    plot(t_ax, m2, 'r--', 'LineWidth',2);
    legend(label1, label2, 'Location','best');
    xlabel('Sample'); ylabel('Amplitude (µV)');
    title('Mean ± SD of shared spike waveforms');
    grid on;

    subplot(1,2,2);
    diff_mean = m1 - m2;
    plot(t_ax, diff_mean, 'k', 'LineWidth',2);
    yline(0,'r--');
    xlabel('Sample'); ylabel('Amplitude difference (µV)');
    title(sprintf('Mean waveform difference (%s - %s)', label1, label2));
    grid on;
    sgtitle('If shapes match: filter OK. If scaled: filter gain differs. If shifted: alignment differs.');

    % ── Example pairs: worst (highest distance) ───────────────────────────
    n_examples = min(plot_examples, n_shared);
    if n_examples > 0
        [~, sort_idx] = sort(shared_distances, 'descend');

        figure('Color','w','Name','Most Dissimilar Spike Pairs',...
            'Position',[100 100 1200 400+100*ceil(n_examples/4)]);
        for k = 1:n_examples
            i    = sort_idx(k);
            idx1 = shared_t.idx1(i);
            idx2 = shared_t.idx2(i);
            s1   = spikes_t1(idx1,:);
            s2   = spikes_t2(idx2,:);
            ml2  = max(length(s1),length(s2));
            s1   = [s1, zeros(1,ml2-length(s1))];
            s2   = [s2, zeros(1,ml2-length(s2))];

            subplot(ceil(n_examples/4), 4, k);
            plot(s1,'b','LineWidth',1.5); hold on;
            plot(s2,'r--','LineWidth',1.5);
            xlabel('Sample'); ylabel('µV');
            dt_us = shared_t.dt(i)*1000;
            title(sprintf('dist=%.1f amp_r=%.2f dt=%.2fµs\n(idx1=%d idx2=%d)',...
                shared_distances(i), amplitude_ratios(i), dt_us, idx1, idx2),...
                'FontSize',8);
            legend(label1,label2,'Location','best','FontSize',7);
            grid on;
        end
        sgtitle('Most dissimilar shared spike pairs');

        % Best matches
        [~, sort_asc] = sort(shared_distances);
        figure('Color','w','Name','Closest Spike Pairs',...
            'Position',[150 150 1200 400+100*ceil(n_examples/4)]);
        for k = 1:n_examples
            i    = sort_asc(k);
            idx1 = shared_t.idx1(i);
            idx2 = shared_t.idx2(i);
            s1   = spikes_t1(idx1,:);
            s2   = spikes_t2(idx2,:);
            ml2  = max(length(s1),length(s2));
            s1   = [s1, zeros(1,ml2-length(s1))];
            s2   = [s2, zeros(1,ml2-length(s2))];

            subplot(ceil(n_examples/4), 4, k);
            plot(s1,'b','LineWidth',1.5); hold on;
            plot(s2,'r--','LineWidth',1.5);
            xlabel('Sample'); ylabel('µV');
            dt_us = shared_t.dt(i)*1000;
            title(sprintf('dist=%.1f amp_r=%.2f dt=%.2fµs\n(idx1=%d idx2=%d)',...
                shared_distances(i), amplitude_ratios(i), dt_us, idx1, idx2),...
                'FontSize',8);
            legend(label1,label2,'Location','best','FontSize',7);
            grid on;
        end
        sgtitle('Most similar shared spike pairs');
    end

    % ── Unique spike waveforms ────────────────────────────────────────────
    if height(unique_t1_t) > 0
        n_show = min(12, height(unique_t1_t));
        figure('Color','w','Name',sprintf('Unique to %s', label1),...
            'Position',[200 200 1200 400+100*ceil(n_show/4)]);
        for k = 1:n_show
            subplot(ceil(n_show/4), 4, k);
            plot(spikes_t1(unique_t1_t.orig_idx(k),:), 'b','LineWidth',1.5);
            title(sprintf('t=%.2fms', unique_t1_t.t(k)),'FontSize',8);
            xlabel('Sample'); ylabel('µV'); grid on;
        end
        sgtitle(sprintf('Spikes unique to %s — are these real spikes or noise?', label1));
    end

    if height(unique_t2_t) > 0
        n_show = min(12, height(unique_t2_t));
        figure('Color','w','Name',sprintf('Unique to %s', label2),...
            'Position',[250 250 1200 400+100*ceil(n_show/4)]);
        for k = 1:n_show
            subplot(ceil(n_show/4), 4, k);
            plot(spikes_t2(unique_t2_t.orig_idx(k),:), 'r--','LineWidth',1.5);
            title(sprintf('t=%.2fms', unique_t2_t.t(k)),'FontSize',8);
            xlabel('Sample'); ylabel('µV'); grid on;
        end
        sgtitle(sprintf('Spikes unique to %s — are these real spikes or noise?', label2));
    end

end