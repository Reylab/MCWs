function plot_ind_spike_dist(orig_spikes, resc_spikes, ch, cluster_num, set_plots)
    % plot_ind_spike_dist - Plot individual spikes with multiple weighting variations
    % 
    % Inputs:
    %   orig_spikes  - Matrix of original spike waveforms (n_spikes x n_samples)
    %   resc_spikes  - Matrix of rescued spike waveforms (n_spikes x n_samples)
    %   ch           - Channel number or label
    %   cluster_num  - Cluster number to analyze
    %
    % Creates detailed plots for each spike showing:
    %   - Distance comparisons under 5 different weighting schemes
    %   - Waveform visualization with template overlay
    %   - Peak width markers and std bands
    
    if nargin < 4
        error('plot_ind_spike_dist requires 4 arguments: orig_spikes, resc_spikes, ch, cluster_num');
    end
    if nargin < 5
        set_plots = true;
    end
    if nargin < 5
        set_plots = true;
    end
    
    % Create output folder
    folder_base = sprintf('wave%d_clust%d_individual', ch, cluster_num);
    if ~exist(folder_base, 'dir')
        mkdir(folder_base);
    end
    
    % Load data for distance computation
    ch_lbl = get_channel_label(ch);
    fname_spk = sprintf('%s_spikes.mat', ch_lbl);
    fname_times = sprintf('times_%s.mat', ch_lbl);
    
    if ~exist(fname_spk, 'file') || ~exist(fname_times, 'file')
        error('Data files not found: %s and %s', fname_spk, fname_times);
    end
    
    SPK = load(fname_spk);
    S = load(fname_times);
    
    % Load good spikes and build templates
    cluster_class = S.cluster_class;
    spikes_good = S.spikes;
    class_good_mask = cluster_class(:,1) ~= 0;
    class_good = cluster_class(class_good_mask, 1);
    spikes_good_classified = spikes_good(class_good_mask, :);
    
    % Exclude rescued spikes from template building
    if isfield(SPK, 'rescue_mask') && ~isempty(SPK.rescue_mask) && any(SPK.rescue_mask)
        rescued_timestamps = SPK.index_all(SPK.rescue_mask);
        good_timestamps = SPK.index(class_good_mask);
        original_good_mask = ~ismember(good_timestamps, rescued_timestamps);
        fprintf('Building templates from %d original spikes (excluding %d rescued)\n', ...
            sum(original_good_mask), sum(~original_good_mask));
    else
        original_good_mask = true(size(class_good));
        fprintf('Building templates from all %d spikes\n', length(class_good));
    end
    
    [centers, maxdist, ~] = build_templates(class_good(original_good_mask), spikes_good_classified(original_good_mask, :));
    
    par = SPK.par;
    par.amp_dir = 'neg';
    
    % Define weighting function variations
    weight_variants = {
        'spike_width_only', ...
        'template_width_only', ...
        'spike_and_template_width', ...
        'spike_or_template_width', ...
        'spike_AND_vs_XOR'
    };
    
    % Combine both spike sets for processing
    all_spikes = [orig_spikes; resc_spikes];
    spike_labels = [repmat({'orig'}, size(orig_spikes, 1), 1); repmat({'resc'}, size(resc_spikes, 1), 1)];
    n_spikes_orig = size(orig_spikes, 1);
    n_spikes_resc = size(resc_spikes, 1);
    n_spikes_total = n_spikes_orig + n_spikes_resc;
    
    n_clusters = size(centers, 1);
    
    % Precompute template metrics
    template_widths = cell(n_clusters, 1);
    template_stds = maxdist;
    for c = 1:n_clusters
        template_widths{c} = compute_peak_width(centers(c, :), par.amp_dir);
    end
    
    % Calculate common y-axis limits for waveform plots
    all_waveforms = [all_spikes(:); centers(:)];
    y_min = min(all_waveforms);
    y_max = max(all_waveforms);
    y_limits = [y_min, y_max];
    offset = (y_max - y_min) * 0.05;
    
    % Plot each spike
    for spike_idx = 1:n_spikes_total
        spike_waveform = all_spikes(spike_idx, :);
        spike_type = spike_labels{spike_idx};
        
        % Determine spike number within its set
        if strcmp(spike_type, 'orig')
            spike_num_in_set = spike_idx;
        else
            spike_num_in_set = spike_idx - n_spikes_orig;
        end
        
        spike_width = compute_peak_width(spike_waveform, par.amp_dir);
        
        % Define weight parameters for spike_AND_vs_XOR variant
        xor_weights_phase1 = [1, 1, 1];        % XOR weight held constant at 1
        spike_weights_phase1 = [1, 5, 50];     % Spike weight varies: 1, 5, 50
        
        xor_weights_phase2 = 50;   % XOR weight constant at 50
        spike_weights_phase2 = 50; % Spike weight constant at 50
        
         
        % Compute distances under all weight variants
        distances_by_variant = cell(1, length(weight_variants));
        
        for var_idx = 1:length(weight_variants)
            variant = weight_variants{var_idx};
            
            % All variants use parametric spike weights
            n_weight_points = length(xor_weights_phase1) + length(xor_weights_phase2);
            distances = zeros(n_clusters, n_weight_points);
            
            % Phase 1: vary spike weight, hold XOR at 1
            for pt = 1:length(spike_weights_phase1)
                for c = 1:n_clusters
                    template = centers(c, :);
                    distance = compute_weighted_distance(spike_waveform, template, ...
                        spike_width, template_widths{c}, variant, par, ...
                        spike_weights_phase1(pt), xor_weights_phase1(pt));
                    distances(c, pt) = distance / template_stds(c);
                end
            end
            
            % Phase 2: vary XOR weight (or keep same for non-XOR variants)
            for pt = 1:length(xor_weights_phase2)
                for c = 1:n_clusters
                    template = centers(c, :);
                    distance = compute_weighted_distance(spike_waveform, template, ...
                        spike_width, template_widths{c}, variant, par, ...
                        spike_weights_phase2(pt), xor_weights_phase2(pt));
                    distances(c, length(spike_weights_phase1) + pt) = distance / template_stds(c);
                end
            end
            
            distances_by_variant{var_idx} = distances;
        end
        
        % Create figure with 2 plots: all variants overlaid (left) + waveform (right)
        fig = figure('Color', 'w', 'Position', [100, 100, 1400, 600], 'Visible', 'off');
        if set_plots
            set(fig, 'Visible', 'on');
        end
        
        % Left plot: all 5 variants overlaid on same axes
        axes('Position', [0.1, 0.15, 0.35, 0.75]);
        hold on;
        
        colors = lines(length(weight_variants));
        for var_idx = 1:length(weight_variants)
            variant = weight_variants{var_idx};
            distances = distances_by_variant{var_idx};
            
            % Plot target cluster distance (matching reference: dist_w(cluster_num) / maxdist(cluster_num))
            target_distances = distances(cluster_num, :);
            plot(1:size(distances, 2), target_distances, 'o-', 'LineWidth', 2, ...
                'MarkerSize', 6, 'Color', colors(var_idx, :), 'DisplayName', strrep(variant, '_', ' '));
        end
        
        hold off;
        xlabel('Weight Point', 'FontSize', 11);
        ylabel('Normalized Distance', 'FontSize', 11);
        xticks(1:4);
        xticklabels({'(1,1)', '(5,1)', '(50,1)', '(50,50)'});
        xtickangle(45);
        title('All Weight Variants', 'FontSize', 12);
        grid on;
        legend('Location', 'best', 'FontSize', 9);
        
        % Add phase divider line at position 3.5
        xline(3.5, 'k--', 'LineWidth', 1.5, 'Alpha', 0.5);
        hold off;
        
        % Right plot: Waveform with template overlay
        axes('Position', [0.55, 0.15, 0.35, 0.75]);
        hold on;
        
        % Plot spike waveform
        plot(spike_waveform, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Spike');
        
        % Plot template for winning cluster
        all_distances = distances_by_variant{1}(:);  % flatten all distances
        [~, winning_idx] = min(all_distances);
        winning_cluster = mod(winning_idx - 1, n_clusters) + 1;  % convert flat index back to cluster
        template = centers(winning_cluster, :);
        plot(template, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Template C%d', winning_cluster));
        
        % Plot template std bands
        plot(template + template_stds(winning_cluster), 'r--', 'LineWidth', 1, 'DisplayName', 'Std±');
        plot(template - template_stds(winning_cluster), 'r--', 'LineWidth', 1);
        
        % Plot spike width markers
        if ~isnan(spike_width.left) && ~isnan(spike_width.right)
            plot([spike_width.left spike_width.left], y_limits, 'b:', 'LineWidth', 2, 'DisplayName', 'Spike width');
            plot([spike_width.right spike_width.right], y_limits, 'b:', 'LineWidth', 2);
        end
        
        % Plot template width markers
        if ~isnan(template_widths{winning_cluster}.left) && ~isnan(template_widths{winning_cluster}.right)
            plot([template_widths{winning_cluster}.left template_widths{winning_cluster}.left], y_limits, 'r:', 'LineWidth', 2, 'DisplayName', 'Template width');
            plot([template_widths{winning_cluster}.right template_widths{winning_cluster}.right], y_limits, 'r:', 'LineWidth', 2);
        end
        
        hold off;
        xlabel('Sample');
        ylabel('Amplitude');
        title(sprintf('Spike %d Waveform & Winning Template (C%d)', spike_num_in_set, winning_cluster), ...
            'FontSize', 10, 'Interpreter', 'none');
        ylim(y_limits);
        legend('Location', 'northeast', 'FontSize', 8);
        grid on;
        
        % Overall title with spike type label
        spike_type_label = upper(spike_type);
        sgtitle(sprintf('Spike %d [%s] (Channel %s, Target Cluster %d)', spike_num_in_set, spike_type_label, ch_lbl, cluster_num), ...
            'FontSize', 12, 'Interpreter', 'none');
        
        % Save figure (always, regardless of set_plots)
        save_path = sprintf('%s/spike_%s_%04d_wave%s_clust%d.png', folder_base, spike_type, spike_num_in_set, ch_lbl, cluster_num);
        saveas(fig, save_path);
        close(fig);
        
        if mod(spike_idx, 10) == 0
            fprintf('Processed %d/%d spikes\n', spike_idx, n_spikes_total);
        end
    end
    
    fprintf('Individual spike plots saved to: %s\n', folder_base);
end

function distance = compute_weighted_distance(spike, template, spike_width, template_width, variant, par, varargin)
    % Compute distance using same approach as get_weight_matrix in plot_spike_distances
    % This ensures normalized distances match between both functions
    
    % Parse optional weight parameters
    if length(varargin) >= 2
        spike_wt = varargin{1};  % weight for spike region
        xor_wt = varargin{2};    % weight for XOR region (only used in spike_AND_vs_XOR)
    else
        spike_wt = 1;
        xor_wt = 3;
    end
    
    n = length(spike);
    spike_mask = false(1, n);
    template_mask = false(1, n);
    
    % Create masks from width struct
    if isstruct(spike_width) && ~isnan(spike_width.left) && ~isnan(spike_width.right)
        spike_mask(spike_width.left:min(spike_width.right, n)) = true;
    end
    if isstruct(template_width) && ~isnan(template_width.left) && ~isnan(template_width.right)
        template_mask(template_width.left:min(template_width.right, n)) = true;
    end
    
    % Initialize weights
    weights = ones(1, n);
    
    % Apply weighting based on variant
    % Key: each variant applies weights differently, but ALL use the same approach as get_weight_matrix
    switch variant
        case 'spike_width_only'
            % Only weight spike region
            weights(spike_mask) = spike_wt;
            
        case 'template_width_only'
            % Only weight template region
            weights(template_mask) = spike_wt;
            
        case 'spike_and_template_width'
            % Only weight overlap region
            overlap = spike_mask & template_mask;
            weights(overlap) = spike_wt;
            
        case 'spike_or_template_width'
            % Weight union of spike and template regions
            weights(spike_mask | template_mask) = spike_wt;
            
        case 'spike_AND_vs_XOR'
            % Different weights for overlap (AND) vs non-overlap (XOR)
            overlap = spike_mask & template_mask;
            xor_region = (spike_mask | template_mask) & ~overlap;
            weights(overlap) = spike_wt;       % AND region
            weights(xor_region) = xor_wt;      % XOR region
            
        otherwise
            % No weighting
    end
    
    % CRITICAL: compute normalization constant based on THIS template's weights
    % This matches get_weight_matrix which computes normConst per template
    sum_weights = sum(weights);
    normConst = sqrt(n) / sqrt(sum_weights);
    
    % Compute distance
    diff = (spike - template) .^ 2;
    distance = normConst * sqrt(sum(diff .* weights));
end

function width_struct = compute_peak_width(spike_x, amp_dir)
    if strcmp(amp_dir, 'neg')
        wav = -spike_x;
    else
        wav = spike_x;
    end
    [pks, locs, w, p] = findpeaks(wav);
    
    if ~isempty(pks)
        [pk, peak_loc] = max(pks);
        pk_loc = locs(peak_loc);
        prom_max = p(peak_loc);

        level = pk - prom_max/2;
        above_level = find(wav >= level);
        if ~isempty(above_level)
            % Find connected components
            diff_above = diff(above_level);
            breaks = find(diff_above > 1);
            segments = {};
            start_idx = 1;
            for b = 1:length(breaks)
                segments{end+1} = above_level(start_idx:breaks(b));
                start_idx = breaks(b) + 1;
            end
            segments{end+1} = above_level(start_idx:end);
            % Find segment containing peak_loc
            peak_segment = [];
            for s = 1:length(segments)
                if any(segments{s} == pk_loc)
                    peak_segment = segments{s};
                    break;
                end
            end
            if ~isempty(peak_segment)
                left_width = min(peak_segment);
                right_width = max(peak_segment);
            else
                left_width = NaN;
                right_width = NaN;
            end
        else
            left_width = NaN;
            right_width = NaN;
        end
        if left_width <= 0
            left_width = 1;
        end
        if right_width > length(spike_x)
            right_width = length(spike_x);
        end
    else
        left_width = NaN;
        right_width = NaN;
    end
    
    width_struct.left = left_width;
    width_struct.right = right_width;
end

function ch_lbl = get_channel_label(ch)
    if isnumeric(ch)
        files = dir(sprintf('*_%d_spikes.mat', ch));
        if ~isempty(files)
            [~, name, ~] = fileparts(files(1).name);
            if endsWith(name, '_spikes')
                ch_lbl = name(1:end-length('_spikes'));
            else
                ch_lbl = name;
            end
        else
            ch_lbl = num2str(ch);
        end
    elseif ischar(ch) || isstring(ch)
        [~, name, ~] = fileparts(char(ch));
        if endsWith(name, '_spikes')
            ch_lbl = name(1:end-length('_spikes'));
        else
            ch_lbl = name;
        end
    else
        error('Channel input must be integer or string filename');
    end
end