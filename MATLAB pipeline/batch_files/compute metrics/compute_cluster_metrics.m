function [df_metrics, SS, figs] = compute_cluster_metrics(data, varargin)
% COMPUTE_CLUSTER_METRICS - Compute quality metrics for spike sorting (MATLAB port of Python)
% Returns df_metrics (table), SS (silhouette matrix), figs (if plotting requested)

    p = inputParser;
    addParameter(p, 'exclude_cluster_0', true, @islogical);
    addParameter(p, 'n_neighbors', 5, @isscalar);
    addParameter(p, 'bin_duration', 60000.0, @isscalar);
    addParameter(p, 'make_plots', false, @islogical);
    addParameter(p, 'save_plots', false, @islogical);
    addParameter(p, 'plot_params', struct(), @isstruct);
    addParameter(p, 'n_jobs', 1, @isscalar); % not heavily used in MATLAB port
    addParameter(p, 'compute_cross_correlograms', true, @islogical); %  PARAMETER
    addParameter(p, 'ccg_time_window', [-50, 50], @(x) numel(x)==2); %  time window in ms
    addParameter(p, 'ccg_bin_size', 1.0, @isscalar); %  bin size in ms
    parse(p, varargin{:});

    % Input validation and extract fields
    if ~isstruct(data) || ~isfield(data,'cluster_class') || ~isfield(data,'spikes') || ~isfield(data,'inspk')
        error('data must be struct with fields cluster_class, spikes, inspk');
    end
    cluster_class = data.cluster_class;
    waveforms = double(data.spikes);
    features = double(data.inspk);
    if ~isempty(features)
    
        % mean(..., 1) calculates the mean down the columns (per feature)
        feat_mean = mean(features, 1);
        feat_std = std(features, 0, 1);
        
        % Replace 0 std with a tiny value (eps) to prevent division by zero
        feat_std(feat_std == 0) = eps;
        
        % MATLAB's bsxfun is ideal for broadcasting the subtraction and division
        features = bsxfun(@rdivide, bsxfun(@minus, features, feat_mean), feat_std);
        
        % Alternative for R2017a+ (more readable)
        % features = (features - feat_mean) ./ feat_std;
        
        fprintf('DEBUG: Successfully Z-Score normalized %d features.\n', size(features, 2));
    end

    if ~isfield(data,'filename') || isempty(data.filename)
        if isfield(data,'fullpath') && ~isempty(data.fullpath)
            [~,fn,ext] = fileparts(data.fullpath); data.filename = [fn ext];
        else
            data.filename = 'unknown_channel';
        end
    end

    cluster_ids = int32(cluster_class(:,1));
    spike_times_ms = double(cluster_class(:,2));
    unique_clusters = unique(cluster_ids);
    recording_duration_ms = max(spike_times_ms);

    % Select clusters to compute
    if p.Results.exclude_cluster_0
        metric_clusters = unique_clusters(unique_clusters ~= 0);
    else
        metric_clusters = unique_clusters;
    end
    
    cross_correlograms = [];
    l1_matrix = [];
    if p.Results.compute_cross_correlograms && numel(unique_clusters) > 1
        try
            [cross_correlograms, l1_matrix] = compute_cross_correlograms(spike_times_ms, cluster_ids, ...
                'time_window', p.Results.ccg_time_window, ...
                'bin_size', p.Results.ccg_bin_size, ...
                'exclude_cluster_0', p.Results.exclude_cluster_0);
            fprintf('Computed %d cross-correlogram pairs\n', length(cross_correlograms));
        catch ME
            warning('Failed to compute cross-correlograms: %s', ME.message);
            cross_correlograms = [];
        end
    end

    % Preallocate
    results = cell(length(metric_clusters),1);

    % Compute metrics per cluster (serial loop to keep code simple / robust)
    for ii = 1:length(metric_clusters)
        cid = metric_clusters(ii);
        mask = (cluster_ids == cid);
        times_c = spike_times_ms(mask);
        W = waveforms(mask, :);

        m = struct();
        m.cluster_id = double(cid);
        m.num_spikes = numel(times_c);
        m.firing_rate = firing_rate(times_c, recording_duration_ms); % Hz
        % safe snr
        try
            m.snr = snr(W);
        catch
            m.snr = NaN;
        end
        m.presence_ratio = presence_ratio(times_c, recording_duration_ms, p.Results.bin_duration);

        % amplitude cutoff
        try
            m.amplitude_cutoff = amplitude_cutoff(W);
        catch
            m.amplitude_cutoff = NaN;
        end

        % ISI / CV2 / violations
        try
            isis = isi(times_c);
            m.cv2 = (numel(isis) > 1) * cv2(isis);
            [vr, fp, nv] = isi_violations(times_c, 'recording_duration', recording_duration_ms);
            m.isi_violation_rate = vr;
            m.isi_violations_count = nv;
            m.isi_fp_rate = fp;
        catch
            m.cv2 = NaN; m.isi_violation_rate = NaN; m.isi_violations_count = 0; m.isi_fp_rate = NaN;
        end

        % isolation / l_ratio / dprime / nearest-neighbor
        if numel(unique_clusters) > 1
            try
                labels_full = int32(cluster_ids);
                [iso, lval] = mahalanobis_metrics(features, labels_full, cid);
                m.isolation_distance = iso;
                m.l_ratio = lval;
            catch
                m.isolation_distance = NaN; m.l_ratio = NaN;
            end
            try
                m.d_prime = d_prime_lda(features, labels_full, cid);
            catch
                m.d_prime = NaN;
            end
            try
                [h, mm] = nearest_neighbor_metrics(features, labels_full, cid, 10000, p.Results.n_neighbors);
                m.nn_hit_rate = h; m.nn_miss_rate = mm;
            catch
                m.nn_hit_rate = NaN; m.nn_miss_rate = NaN;
            end
        else
            m.isolation_distance = NaN; m.l_ratio = NaN; m.d_prime = NaN; m.nn_hit_rate = NaN; m.nn_miss_rate = NaN;
        end

        results{ii} = m;
    end

    if ~isempty(results)
        df_metrics = struct2table([results{:}]);
    else
        df_metrics = table();
    end

    % compute silhouette scores matrix SS and per-cluster scores
    if numel(unique_clusters) > 1
        if p.Results.exclude_cluster_0
            valid_mask = (cluster_ids ~= 0);
            feat_valid = features(valid_mask, :);
            labels_valid = cluster_ids(valid_mask);
        else
            feat_valid = features;
            labels_valid = cluster_ids;
        end
        try
            [scores, SS] = silhouette_score(feat_valid, labels_valid, 'return_matrix', true);
            % Map scores back into df_metrics (order: metric_clusters)
            if ~isempty(df_metrics)
                % silhouette_score returns per-cluster scores aligned to unique(labels_valid)
                uniq = unique(labels_valid);
                scores_map = containers.Map(double(uniq), scores);
                sc = zeros(height(df_metrics),1);
                for r=1:height(df_metrics)
                    cid = df_metrics.cluster_id(r);
                    if isKey(scores_map,double(cid))
                        sc(r) = scores_map(double(cid));
                    else
                        sc(r) = NaN;
                    end
                end
                df_metrics.silhouette_score = sc;
            end

            % --- NEW: remap SS matrix to match df_metrics.cluster_id order ---
            % Build SS_mapped with rows/cols in the same order as df_metrics.cluster_id
            if ~isempty(SS) && ~isempty(df_metrics)
                uniq = unique(labels_valid); % order used by SS
                n_out = height(df_metrics);
                SS_mapped = NaN(n_out, n_out);
                ids_out = double(df_metrics.cluster_id);
                for i_out = 1:n_out
                    for j_out = 1:n_out
                        % find positions in uniq
                        pos_i = find(uniq == ids_out(i_out), 1);
                        pos_j = find(uniq == ids_out(j_out), 1);
                        if ~isempty(pos_i) && ~isempty(pos_j) && pos_i <= size(SS,1) && pos_j <= size(SS,2)
                            SS_mapped(i_out, j_out) = SS(pos_i, pos_j);
                        else
                            SS_mapped(i_out, j_out) = NaN;
                        end
                    end
                end
                SS = SS_mapped;
            end
            % --- end remap ---
        catch
            SS = []; 
            if ~isempty(df_metrics)
                df_metrics.silhouette_score = NaN(height(df_metrics),1);
            end
        end
    else
        SS = [];
        if ~isempty(df_metrics)
            df_metrics.silhouette_score = NaN(height(df_metrics),1);
        end
    end

    % Generate plots if requested (delegate to make_cluster_report)
    figs = [];
            if p.Results.make_plots || p.Results.save_plots
                % Ensure plot_params is a struct with expected fields
                plot_params = p.Results.plot_params;
                if isempty(plot_params) || ~isstruct(plot_params)
                    plot_params = struct();
                end

                % Provide sensible defaults if not present
                if ~isfield(plot_params, 'outdir') || isempty(plot_params.outdir)
                    plot_params.outdir = '.';
                end
                if ~isfield(plot_params, 'test')
                    plot_params.test = false;
                end
                if ~isfield(plot_params, 'test_suffix')
                    plot_params.test_suffix = '_testMerge';
                end
                if ~isfield(plot_params, 'apply_report')
                    plot_params.apply_report = false;
                end
                if ~isfield(plot_params, 'backup_original')
                    plot_params.backup_original = false;
                end
                if ~isfield(plot_params, 'save_figs')
                    % map compute_cluster_metrics 'save_plots' to make_cluster_report 'save_figs'
                    plot_params.save_figs = p.Results.save_plots;
                end

                try
                    figs = make_cluster_report(data, ...
                        'calc_metrics', false, ...
                        'metrics_df', df_metrics, ...
                        'SS', SS, ...
                        'cross_correlograms', cross_correlograms, ...
                        'l1_distances', l1_matrix, ...
                        'exclude_cluster_0', p.Results.exclude_cluster_0, ...
                        'samplerate_hz', [], ...
                        'clusters_per_page', 6, ...
                        'bin_duration_ms', p.Results.bin_duration, ...
                        'refractory_ms', 3.0, ...
                        'n_neighbors', p.Results.n_neighbors, ...
                        'max_waveforms_per_cluster', 1000, ...
                        'show_figures', p.Results.make_plots, ...
                        'test', plot_params.test, ...
                        'apply_report', plot_params.apply_report, ...
                        'backup_original', plot_params.backup_original, ...
                        'test_suffix', plot_params.test_suffix, ...
                        'outdir', plot_params.outdir, ...
                        'save_figs', plot_params.save_figs);
                catch ME_plot
                    fprintf('Failed to generate cluster report: %s\n', ME_plot.message);
                    figs = [];
                end
            end
        plot_params = p.Results.plot_params;
        plot_params.calc_metrics = false;
        plot_params.metrics_df = df_metrics;
        plot_params.SS = SS;
        plot_params.cross_correlograms = cross_correlograms; 
        % Ensure max_waveforms_per_cluster is defined
        if isfield(plot_params, 'max_waveforms_per_cluster') && ~isempty(plot_params.max_waveforms_per_cluster)
            max_w = plot_params.max_waveforms_per_cluster;
        else
            max_w = 1000; % Default value
        end

        try
            % Determine samplerate_hz from several likely locations:
            % Priority: plot_params.samplerate_hz -> data.samplerate_hz -> data.par.sr
            % -> data.a.par.sr -> data.times.par.sr
            samplerate_hz = [];
            if isstruct(plot_params) && isfield(plot_params, 'samplerate_hz') && ...
                    ~isempty(plot_params.samplerate_hz) && isnumeric(plot_params.samplerate_hz) && isscalar(plot_params.samplerate_hz)
                samplerate_hz = plot_params.samplerate_hz;
            elseif isfield(data, 'samplerate_hz') && ~isempty(data.samplerate_hz) && isnumeric(data.samplerate_hz) && isscalar(data.samplerate_hz)
                samplerate_hz = data.samplerate_hz;
            elseif isfield(data, 'par') && isstruct(data.par) && isfield(data.par, 'sr') && ...
                    ~isempty(data.par.sr) && isnumeric(data.par.sr) && isscalar(data.par.sr)
                samplerate_hz = data.par.sr;
            elseif isfield(data, 'a') && isstruct(data.a) && isfield(data.a, 'par') && isfield(data.a.par, 'sr') && ...
                    ~isempty(data.a.par.sr) && isnumeric(data.a.par.sr) && isscalar(data.a.par.sr)
                samplerate_hz = data.a.par.sr;
            elseif isfield(data, 'times') && isstruct(data.times) && isfield(data.times, 'par') && isfield(data.times.par, 'sr') && ...
                    ~isempty(data.times.par.sr) && isnumeric(data.times.par.sr) && isscalar(data.times.par.sr)
                samplerate_hz = data.times.par.sr;
            else
                samplerate_hz = []; % fallback: use sample indices if unknown
            end

            % Defensive: ensure scalar numeric or empty
            if ~(isempty(samplerate_hz) || (isnumeric(samplerate_hz) && isscalar(samplerate_hz)))
                samplerate_hz = [];
            end

            % Call make_cluster_report with explicit name/value args
            figs = make_cluster_report( ...
                data, ...
                'calc_metrics', false, ...
                'metrics_df', df_metrics, ...
                'SS', SS, ...
                'cross_correlograms', cross_correlograms, ...
                'l1_distances',l1_matrix,...
                'exclude_cluster_0', p.Results.exclude_cluster_0, ...
                'samplerate_hz', samplerate_hz, ...
                'clusters_per_page', 6, ...
                'bin_duration_ms', p.Results.bin_duration, ...
                'refractory_ms', 3.0, ...
                'n_neighbors', p.Results.n_neighbors, ...
                'max_waveforms_per_cluster', max_w, ...
                'show_figures', p.Results.make_plots ...
            );
        catch ME
            warning('make_cluster_report failed: %s', ME.message);
            figs = [];
        end

        if p.Results.save_plots && ~isempty(figs)
            save_plot_files(figs, data, plot_params);
        end
    end

end

function save_plot_files(figs, data, plot_params)
% SAVE_PLOT_FILES - Save generated figures to files (robust to cell / empty inputs)
    if isempty(figs)
        return;
    end

    % Normalize to cell array
    if ~iscell(figs)
        figs = {figs};
    end

    % Ensure data.filename exists
    if ~isfield(data, 'filename') || isempty(data.filename)
        if isfield(data, 'fullpath') && ~isempty(data.fullpath)
            [~, fn, ext] = fileparts(data.fullpath);
            data.filename = [fn ext];
        else
            data.filename = 'unknown_channel';
        end
    end

    outdir = '.';
    if isstruct(plot_params) && isfield(plot_params, 'outdir') && ~isempty(plot_params.outdir)
        outdir = plot_params.outdir;
    end

    for i = 1:numel(figs)
        f = figs{i};
        if isgraphics(f, 'figure') && isvalid(f)
            try
                fname = fullfile(outdir, sprintf('%s_plot_%03d.png', data.filename, i));
                saveas(f, fname);
            catch ME
                warning('Failed to save figure %d: %s', i, ME.message);
            end
        end
    end
end

function n = num_spikes(spike_times)
% NUM_SPIKES - Calculate number of spikes
    n = length(spike_times);
end

function rate = firing_rate(spike_times, recording_duration, start_time, end_time)
% FIRING_RATE - Calculate average firing rate in Hz
    
    if nargin < 3
        start_time = 0.0;
    end
    if nargin < 4
        end_time = recording_duration;
    end
    
    if recording_duration <= 0
        error('Recording duration must be positive');
    end
    
    if isempty(spike_times)
        rate = 0.0;
        return;
    end
    
    if any(spike_times < 0)
        error('Spike times cannot be negative');
    end
    
    if end_time <= start_time
        error('end_time must be greater than start_time');
    end
    
    % Filter spike times within the specified window
    mask = (spike_times >= start_time) & (spike_times <= end_time);
    n_spikes_in_window = sum(mask);
    
    % Calculate duration of the window (convert to seconds for Hz)
    window_duration = (end_time - start_time) / 1000.0; % convert ms to seconds
    
    if window_duration == 0
        rate = 0.0;
        return;
    end
    
    % Calculate firing rate (spikes per second)
    rate = n_spikes_in_window / window_duration;
end

function snr_val = snr(waveforms, noise_samples_n)
% SNR - Calculate Signal-to-Noise Ratio for waveforms
    
    if nargin < 2
        noise_samples_n = 5;
    end
    
    if ndims(waveforms) ~= 2
        error('Input must be 2D (n_spikes, n_samples)');
    end
    
    [n_spikes, n_samples] = size(waveforms);
    
    if n_spikes == 0
        error('Waveforms array cannot be empty');
    end
    
    if n_samples < noise_samples_n * 2
        error('Waveforms are too short to estimate noise from %d samples at each end', noise_samples_n);
    end
    
    % 1. Calculate the mean template
    template = mean(waveforms, 1);
    
    % 2. Calculate the peak-to-peak amplitude of the template
    peak_to_peak = max(template) - min(template);
    
    % 3. Estimate noise from the first and last N samples
    residuals = waveforms - template; % broadcasts
    noise_residuals = [residuals(:, 1:noise_samples_n), residuals(:, end-noise_samples_n+1:end)];
    noise_level = std(noise_residuals(:));
    
    % 4. Handle division by zero
    if noise_level == 0
        % If noise is zero, return infinity if signal exists, else 0
        if peak_to_peak > 0
            snr_val = Inf;
        else
            snr_val = 0.0;
        end
        return;
    end
    
    % 5. Return SNR
    snr_val = peak_to_peak / noise_level;
end

function ratio = presence_ratio(spike_times, recording_duration, bin_duration)
% PRESENCE_RATIO - Calculate fraction of time bins with activity
    
    if nargin < 3
        bin_duration = 60000.0; % 1 minute in ms
    end
    
    if recording_duration <= 0
        error('Recording duration must be positive');
    end
    
    if bin_duration <= 0
        error('Bin duration must be positive');
    end
    
    if isempty(spike_times)
        ratio = 0.0;
        return;
    end
    
    % Calculate number of bins
    n_bins = ceil(recording_duration / bin_duration);
    
    if n_bins == 0
        ratio = 0.0;
        return;
    end
    
    % Create time bins
    bins = 0:bin_duration:recording_duration;
    
    % Count spikes in each bin
    spike_counts = histcounts(spike_times, bins);
    
    % Calculate presence ratio (fraction of bins with at least one spike)
    bins_with_spikes = sum(spike_counts > 0);
    ratio = bins_with_spikes / n_bins;
end

function fraction_missing = amplitude_cutoff(waveforms, num_histogram_bins, histogram_smoothing_value)
% AMPLITUDE_CUTOFF - Calculate fraction of spikes missing from amplitude distribution
    
    if nargin < 2
        num_histogram_bins = 500;
    end
    if nargin < 3
        histogram_smoothing_value = 3;
    end
    
    % Calculate amplitude cutoff in milliseconds
    amplitudes = range(waveforms, 2); % peak-to-peak for each waveform
    
    % 1. histogram
    [hist_density, bin_edges] = histcounts(amplitudes, num_histogram_bins, 'Normalization', 'pdf');
    
    % 2. smooth with Gaussian filter
    pdf = imgaussfilt(hist_density, histogram_smoothing_value);
    support = bin_edges(1:end-1);
    
    % 3. find peak
    [~, peak_idx] = max(pdf);
    
    % 4. point where pdf (to the right) ~ left-edge density
    right_segment = pdf(peak_idx:end);
    target = pdf(1);
    [~, rel_idx] = min(abs(right_segment - target));
    G = peak_idx + rel_idx - 1; % -1 because MATLAB indexing
    
    % 5. integrate tail
    bin_size = mean(diff(support));
    fraction_missing = sum(pdf(G:end)) * bin_size;
    
    % cap at 0.5
    fraction_missing = min(fraction_missing, 0.5);
end
function [isolation_distance, l_ratio] = mahalanobis_metrics(features, labels, target_cluster)
    % ... (Initial checks and splitting are correct) ...
    
    target_mask = (labels == target_cluster);
    pcs_for_this = features(target_mask, :);
    pcs_for_other = features(~target_mask, :);
    
    n_self = size(pcs_for_this, 1);
    n_other = size(pcs_for_other, 1);
    
    if n_self < 2 || n_other < 1
        isolation_distance = NaN;
        l_ratio = NaN;
        fprintf('DEBUG C%d: Too few spikes. n_self=%d, n_other=%d\n', target_cluster, n_self, n_other);
        return;
    end
    
    mean_val = mean(pcs_for_this, 1);
    dof = size(pcs_for_this, 2); % #features
    
    % --- Robust Covariance Calculation and Regularization ---
    try
        cov_matrix = cov(pcs_for_this);
        
        % CRITICAL FIX: Diagonal Loading 
        min_variance_threshold = 1e-6; % Must be > 0.
        
        diag_cov = diag(cov_matrix);
        
        % PRINT 1: Check the minimum variance before regularization
        fprintf('DEBUG C%d: Min variance before regularization: %e\n', target_cluster, min(diag_cov));

        small_variance_mask = (diag_cov < min_variance_threshold);
        
        if any(small_variance_mask)
             % Add the threshold to all small-variance dimensions
             % Note: This is a simplified way to apply diagonal loading.
             cov_matrix(small_variance_mask, small_variance_mask) = ...
                 cov_matrix(small_variance_mask, small_variance_mask) + ...
                 min_variance_threshold * eye(sum(small_variance_mask));
             fprintf('DEBUG C%d: Applied diagonal loading to %d features.\n', target_cluster, sum(small_variance_mask));
        end

        VI = pinv(cov_matrix);
        
        % PRINT 2: Check the condition number of the final matrix
        cond_num = cond(cov_matrix);
        %fprintf('DEBUG C%d: Condition Number (cond()): %e\n', target_cluster, cond_num);

    catch ME
        % PRINT 3: Report crash during matrix inversion
        fprintf('DEBUG C%d: CRASH in pinv/cov. Error: %s\n', target_cluster, ME.message);
        isolation_distance = NaN;
        l_ratio = NaN;
        return;
    end
    
    % --- Metric Calculation ---

    % Calculate Mahalanobis DISTANCE (D) to *other* spikes
    mahal_other_dist = pdist2(mean_val, pcs_for_other, 'mahalanobis', VI);
    
    % Convert to SQUARED Mahalanobis Distance (delta^2)
    delta2_other = mahal_other_dist.^2;

    delta2_other_sorted = sort(delta2_other);
    
    n = min(n_self, n_other);
    
    % L-ratio: sum over OTHER, divided by size of THIS cluster
    p_values = chi2cdf(delta2_other, dof, 'upper');
    l_ratio = sum(p_values) / double(n_self);
    
    % Isolation distance: n-th element of the sorted SQUARED distances
    if n >= 1
        isolation_distance = delta2_other_sorted(n);
    else
        isolation_distance = NaN;
    end
    
    % PRINT 4: Report calculated metrics
    fprintf('DEBUG C%d: FINAL ID: %e | FINAL L_ratio: %e\n', target_cluster, isolation_distance, l_ratio);

end

function dprime = d_prime_lda(features, cluster_labels, target_cluster)
% D_PRIME_LDA - LDA-based d-prime for cluster separation
    
    if size(features, 1) ~= length(cluster_labels)
        error('features and cluster_labels must have same length');
    end
    
    if ~any(cluster_labels == target_cluster)
        error('Target cluster %d not found', target_cluster);
    end
    
    X = features;
    y = (cluster_labels == target_cluster);
    
    % need both classes
    if sum(y) < 2 || sum(~y) < 2
        dprime = 0.0;
        return;
    end
    
    try
        % Perform LDA
        Mdl = fitcdiscr(X, y, 'DiscrimType', 'linear');
        X_lda = predict(Mdl, X); % This gives class predictions, need scores
        
        % For proper LDA projection, we need the discriminant coefficients
        coef = Mdl.Coeffs(1,2).Linear;
        X_lda_proj = X * coef;
        
        this_proj = X_lda_proj(y);
        other_proj = X_lda_proj(~y);
        
        mu1 = mean(this_proj);
        mu2 = mean(other_proj);
        s1 = std(this_proj);
        s2 = std(other_proj);
        
        pooled = sqrt(0.5 * (s1^2 + s2^2));
        if pooled == 0
            dprime = 0.0;
        else
            dprime = abs(mu1 - mu2) / pooled;
        end
    catch
        dprime = 0.0;
    end
end


function [scores, SS] = silhouette_score(features, labels, varargin)
% SILHOUETTE_SCORE - Compute silhouette scores between cluster pairs
    
    p = inputParser;
    addParameter(p, 'return_matrix', false, @islogical);
    addParameter(p, 'metric', 'euclidean', @ischar);
    parse(p, varargin{:});
    
    if size(features, 1) ~= length(labels)
        error('features and labels must have same length');
    end
    
    if ~ismember(p.Results.metric, {'euclidean', 'cityblock', 'cosine'})
        error('Unsupported metric: %s', p.Results.metric);
    end
    
    unique_labels = unique(labels);
    if length(unique_labels) < 2
        error('At least 2 clusters are required');
    end
    
    K = length(unique_labels);
    SS = NaN(K, K);
    
    % For each pair of clusters
    for i = 1:K
        for j = i+1:K
            i_lab = unique_labels(i);
            j_lab = unique_labels(j);
            
            mask = (labels == i_lab) | (labels == j_lab);
            if sum(mask) > 2 && length(unique(labels(mask))) > 1
                try
                    % Use MATLAB's built-in silhouette function
                    s = silhouette(features(mask, :), labels(mask), p.Results.metric);
                    SS(i, j) = mean(s);
                catch
                    SS(i, j) = NaN;
                end
            end
        end
    end
    
    % Per-cluster score = min( min over row, min over col )
    min_by_col = min(SS, [], 1, 'omitnan');
    min_by_row = min(SS, [], 2, 'omitnan');
    
    scores = zeros(K, 1);
    for k = 1:K
        scores(k) = min([min_by_col(k), min_by_row(k)], [], 'omitnan');
    end
    
    if ~p.Results.return_matrix
        scores = scores;
    end
end

function isis = isi(spike_times_ms)
% ISI - Calculate Inter-Spike Intervals from spike times
%
% Args:
%   spike_times_ms: Array of spike times in milliseconds (may be unsorted / contain NaN)
%
% Returns:
%   isis: Array of inter-spike intervals in milliseconds

    if isempty(spike_times_ms) || numel(spike_times_ms) < 2
        isis = [];
        return;
    end

    % Sanitize: remove NaN/Inf and sort so we always compute correct consecutive ISIs
    spike_times_ms = spike_times_ms(~isnan(spike_times_ms) & isfinite(spike_times_ms));
    if isempty(spike_times_ms) || numel(spike_times_ms) < 2
        isis = [];
        return;
    end

    % Sort times (defensive: many inputs can be unsorted)
    spike_times_ms = sort(spike_times_ms(:));

    % Calculate ISIs
    isis = diff(spike_times_ms);
end

function cv2_val = cv2(isi_ms)
% CV2 - Calculate CV2 (Coefficient of Variation 2) - local irregularity index
%
% Args:
%   isi_ms: Array of inter-spike intervals in milliseconds
%
% Returns:
%   cv2_val: CV2 value (0=regular, 1=irregular)

    if length(isi_ms) < 2
        cv2_val = 0.0;
        return;
    end
    
    % Check for invalid values
    if any(isnan(isi_ms)) || any(isinf(isi_ms))
        error('isi_ms contains NaN or Inf values');
    end
    
    if any(isi_ms < 0)
        error('isi_ms cannot contain negative values');
    end
    
    % Calculate absolute differences between consecutive ISIs
    diff_isi = abs(diff(isi_ms));
    
    % Calculate sum of consecutive ISI pairs
    denom = isi_ms(1:end-1) + isi_ms(2:end);
    
    % Avoid division by zero
    denom(denom == 0) = eps;
    
    % Calculate CV2 as mean of normalized local variations
    cv2_values = 2.0 * diff_isi ./ denom;
    cv2_val = mean(cv2_values);
end

function [violation_rate, fp_rate, num_violations] = isi_violations(spike_times, varargin)
% ISI_VIOLATIONS - Calculate rate of Inter-Spike Interval violations
%
% Args:
%   spike_times: Array of spike times in seconds (must be sorted)
%   Optional:
%     'refractory_period', 3.0 (ms)
%     'censored_period', 0.0 (ms) 
%     'recording_duration', [] (ms)
%
% Returns:
%   violation_rate: Ratio of ISI violations
%   fp_rate: False positive rate (%)
%   num_violations: Absolute count of violations

    % Parse inputs
    p = inputParser;
    addParameter(p, 'refractory_period', 3.0, @isscalar);   % ISI threshold (ms)
    addParameter(p, 'censored_period', 0.0, @isscalar);     % min ISI threshold (ms)
    addParameter(p, 'recording_duration', [], @isscalar);   % total duration (ms)
    parse(p, varargin{:});
    
    refractory_period = p.Results.refractory_period;
    censored_period = p.Results.censored_period;
    recording_duration = p.Results.recording_duration;
    
    if isempty(spike_times)
        violation_rate = 0.0;
        fp_rate = 0.0;
        num_violations = 0;
        return;
    end
    
    % Check if sorted
    if ~issorted(spike_times)
        error('spike_times must be sorted ascending');
    end
    
    if censored_period < 0
        error('censored_period cannot be negative');
    end
    
    if censored_period >= refractory_period
        error('censored_period must be smaller than refractory_period');
    end
    
    % 1) Remove duplicate/too-close spikes (≤ censored_period)
    spike_times_clean = spike_times;
    dup_idx = find(diff(spike_times_clean) <= censored_period);
    if ~isempty(dup_idx)
        % Remove the second spike in each duplicate pair
        spike_times_clean(dup_idx + 1) = [];
    end
    
    % 2) Calculate ISIs
    isis = diff(spike_times_clean);
    
    % 3) Count violations strictly inside (censored, refractory)
    num_violations = sum((isis > censored_period) & (isis < refractory_period));
    
    % 4) Get recording duration
    if isempty(recording_duration)
        recording_duration = spike_times_clean(end) - spike_times_clean(1);
    end
    
    n_spikes = length(spike_times_clean);
    if n_spikes <= 1 || recording_duration <= 0
        violation_rate = 0.0;
        fp_rate = 0.0;
        return;
    end
    
    % 5) Unit firing rate (spikes/ms)
    total_rate = n_spikes / recording_duration;
    
    % 6) Violation time = total time in which violations could have happened
    violation_time = 2.0 * n_spikes * (refractory_period - censored_period);
    
    if violation_time <= 0 || total_rate == 0
        violation_rate = 0.0;
        fp_rate = 0.0;
        return;
    end
    
    % 7) Violation rate (violations/ms)
    violation_rate = num_violations / violation_time;
    
    % 8) False positive rate (contamination fraction in %)
    fp_rate = (violation_rate / total_rate) * 100.0;
end

function [hit_rate, miss_rate] = nearest_neighbor_metrics(features, labels, target_cluster, max_spikes_for_nn, n_neighbors)
% NEAREST_NEIGHBOR_METRICS - kNN-based hit/miss rates for a target cluster

    if nargin < 4
        max_spikes_for_nn = 10000;
    end
    if nargin < 5
        n_neighbors = 5;
    end
    
    if size(features, 1) ~= length(labels)
        error('features and labels must have same length');
    end
    if ~any(labels == target_cluster)
        error('target_cluster %d not found in labels', target_cluster);
    end
    if n_neighbors < 1
        error('n_neighbors must be >= 1');
    end
    
    % 1) Separate target and other spikes
    is_target = (labels == target_cluster);
    X_target = features(is_target, :);
    X_other = features(~is_target, :);
    X = [X_target; X_other];
    
    n_target = size(X_target, 1);
    total_spikes = size(X, 1);
    
    if n_target < 2
        % nothing to measure
        hit_rate = 1.0;
        miss_rate = 0.0;
        return;
    end
    
    % 2) Subsample if too many spikes
    if total_spikes > max_spikes_for_nn
        ratio = max_spikes_for_nn / total_spikes;
        % Create indices for deterministic subsampling
        inds = 1:round(1/ratio):total_spikes;
        X = X(inds, :);
        % Update counts
        n_target_subsampled = round(n_target * ratio);
        total_spikes = length(inds);
    else
        n_target_subsampled = n_target;
    end
    
    % 3) k-NN using MATLAB's knnsearch
    [~, dists] = knnsearch(X, X, 'K', n_neighbors + 1); % +1 to exclude self
    neighbor_indices = knnsearch(X, X, 'K', n_neighbors + 1);
    neighbor_indices = neighbor_indices(:, 2:end); % Remove self
    
    % 4) Calculate hit and miss rates
    target_rows = 1:n_target_subsampled;
    other_rows = (n_target_subsampled+1):total_spikes;
    
    % Neighbors of target spikes (excluding self)
    target_neighbors = neighbor_indices(target_rows, :);
    target_neighbors = target_neighbors(:);
    
    % Neighbors of non-target spikes
    other_neighbors = neighbor_indices(other_rows, :);
    other_neighbors = other_neighbors(:);
    
    % Hit rate: fraction of target spike neighbors that are also target
    if ~isempty(target_neighbors)
        hit_rate = mean(target_neighbors <= n_target_subsampled);
    else
        hit_rate = 1.0;
    end
    
    % Miss rate: fraction of non-target spike neighbors that are target
    if ~isempty(other_neighbors)
        miss_rate = mean(other_neighbors <= n_target_subsampled);
    else
        miss_rate = 0.0;
    end
end

% -----------------------
% Helper: safely close/delete figure handles (cell or single)
function close_figs_safe(figs)
    if isempty(figs), return; end
    if ~iscell(figs), figs = {figs}; end
    for k = 1:numel(figs)
        f = figs{k};
        if isgraphics(f, 'figure') && isvalid(f)
            try
                close(f);
            catch
                % ignore
            end
        end
    end
end
