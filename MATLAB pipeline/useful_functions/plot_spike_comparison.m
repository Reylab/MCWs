function out = plot_spike_comparison(orig_spikes, resc_spikes, ch, cluster_num)
% plot_spike_comparison - Generate comparison plots of original vs rescued spikes
% Inputs:
%   orig_spikes - Matrix of original spike waveforms (n_spikes x n_samples)
%   resc_spikes - Matrix of rescued spike waveforms (n_spikes x n_samples)
%   ch - Channel number or label
%   cluster_num - Cluster number
% Outputs:
%   out - Structure with fields:
%         .weights - Array of weight values
%         .orig - Structure with table for each weight (W1, W5, W10, etc.)
%         .resc - Structure with table for each weight

    % Create folder
    folder_name = sprintf('wave%d_clust%d', ch, cluster_num);
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    
    % Load data for distance computation
    ch_lbl = get_channel_label(ch);
    fname_spk = sprintf('%s_spikes.mat', ch_lbl);
    fname_times = sprintf('times_%s.mat', ch_lbl);
    
    if exist(fname_spk, 'file') && exist(fname_times, 'file')
        SPK = load(fname_spk);
        S = load(fname_times);
        
        % Load good spikes and build templates
        cluster_class = S.cluster_class;
        spikes_good = S.spikes;
        class_good_mask = cluster_class(:,1) ~= 0;
        class_good = cluster_class(class_good_mask, 1);
        spikes_good_classified = spikes_good(class_good_mask, :);
        
        % Exclude rescued spikes from template building to match rescue process
        if isfield(SPK, 'rescue_mask') && ~isempty(SPK.rescue_mask) && any(SPK.rescue_mask)
            % Get timestamps of rescued spikes from index_all
            rescued_timestamps = SPK.index_all(SPK.rescue_mask);
            
            % Get timestamps of good classified spikes from SPK.index
            good_timestamps = SPK.index(class_good_mask);
            
            % Find which good spikes are rescued by matching timestamps
            original_good_mask = ~ismember(good_timestamps, rescued_timestamps);
            
            fprintf('Building templates from %d original spikes (excluding %d rescued)\n', ...
                sum(original_good_mask), sum(~original_good_mask));
        else
            original_good_mask = true(size(class_good));
            fprintf('Building templates from all %d spikes (no rescue_mask found)\n', length(class_good));
        end
        
        % Debug: print template info
        [centers, maxdist, ~] = build_templates(class_good(original_good_mask), spikes_good_classified(original_good_mask, :));
        fprintf('Built %d templates for clusters: ', size(centers, 1));
        fprintf('%d ', 1:size(centers, 1));
        fprintf('\n');
        fprintf('maxdist values: ');
        fprintf('%.2f ', maxdist);
        fprintf('\n');
        
        par = SPK.par;
        par.amp_dir = 'neg';
        weights = [1, 5, 10, 25, 50, 100, 200, 300, 500, 1000];
        
        % Compute distances for orig_spikes
        n_orig = size(orig_spikes, 1);
        norm_dist_orig = zeros(n_orig, length(weights));
        raw_dist_orig = zeros(n_orig, length(weights));
        win_cls_orig = zeros(n_orig, length(weights));
        
        % Create tables for all weights
        dist_tables_orig = cell(1, length(weights));
        table_weight = 300;  % Weight to display in console
        
        for w_idx = 1:length(weights)
            par.pk_weight = weights(w_idx);
            dist_table_w = zeros(n_orig, size(centers, 1) + 1);
            
            for i = 1:n_orig
                spike = orig_spikes(i, :);
                [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, par.pk_weight, par.amp_dir);
                dist_w = (normConst_vec .* sqrt(sum(w_matrix .* (repmat(spike, size(centers,1),1) - centers).^2, 2)))';
                
                % Store distances for this weight
                dist_table_w(i, 1:size(centers, 1)) = dist_w;
                
                % Find conforming clusters (within maxdist) like nearest_neighbor does
                conforming = find(dist_w < maxdist * par.template_sdnum);
                if isempty(conforming)
                    win_cls_orig(i, w_idx) = 0;  % No cluster within threshold
                else
                    [~, idx] = min(dist_w(conforming));
                    win_cls_orig(i, w_idx) = conforming(idx);  % Actual cluster number
                end
                
                % Store winning cluster
                dist_table_w(i, end) = win_cls_orig(i, w_idx);
                
                raw_dist_orig(i, w_idx) = dist_w(cluster_num);
                norm_dist_orig(i, w_idx) = dist_w(cluster_num) / maxdist(cluster_num);
            end
            
            % Create table for this weight
            col_names = cell(1, size(centers, 1) + 1);
            for cls = 1:size(centers, 1)
                col_names{cls} = sprintf('Dist_C%d', cls);
            end
            col_names{end} = 'Winner';
            dist_tables_orig{w_idx} = array2table(dist_table_w, 'VariableNames', col_names);
        end
        
        % Display table for one weight
        [~, display_idx] = min(abs(weights - table_weight));
        fprintf('\n=== ORIGINAL SPIKES (Weight=%d) ===\n', weights(display_idx));
        disp(dist_tables_orig{display_idx});
        
        % Compute distances for resc_spikes
        n_resc = size(resc_spikes, 1);
        norm_dist_resc = zeros(n_resc, length(weights));
        raw_dist_resc = zeros(n_resc, length(weights));
        win_cls_resc = zeros(n_resc, length(weights));
        
        % Create tables for all weights
        dist_tables_resc = cell(1, length(weights));
        
        for w_idx = 1:length(weights)
            par.pk_weight = weights(w_idx);
            dist_table_w = zeros(n_resc, size(centers, 1) + 1);
            
            for i = 1:n_resc
                spike = resc_spikes(i, :);
                [normConst_vec, w_matrix] = get_weight_matrix(spike, centers, par.pk_weight, par.amp_dir);
                dist_w = (normConst_vec .* sqrt(sum(w_matrix .* (repmat(spike, size(centers,1),1) - centers).^2, 2)))';
                
                % Store distances for this weight
                dist_table_w(i, 1:size(centers, 1)) = dist_w;
                
                % Find conforming clusters (within maxdist) like nearest_neighbor does
                conforming = find(dist_w < maxdist * par.template_sdnum);
                if isempty(conforming)
                    win_cls_resc(i, w_idx) = 0;  % No cluster within threshold
                else
                    [~, idx] = min(dist_w(conforming));
                    win_cls_resc(i, w_idx) = conforming(idx);  % Actual cluster number
                end
                
                % Store winning cluster
                dist_table_w(i, end) = win_cls_resc(i, w_idx);
                
                raw_dist_resc(i, w_idx) = dist_w(cluster_num);
                norm_dist_resc(i, w_idx) = dist_w(cluster_num) / maxdist(cluster_num);
            end
            
            % Create table for this weight
            col_names = cell(1, size(centers, 1) + 1);
            for cls = 1:size(centers, 1)
                col_names{cls} = sprintf('Dist_C%d', cls);
            end
            col_names{end} = 'Winner';
            dist_tables_resc{w_idx} = array2table(dist_table_w, 'VariableNames', col_names);
        end
        
        % Display table for one weight
        fprintf('\n=== RESCUED SPIKES (Weight=%d) ===\n', weights(display_idx));
        disp(dist_tables_resc{display_idx});
        
        % Create output structure with all tables
        out.weights = weights;
        out.orig = struct();
        out.resc = struct();
        for w_idx = 1:length(weights)
            field_name = sprintf('W%d', weights(w_idx));
            out.orig.(field_name) = dist_tables_orig{w_idx};
            out.resc.(field_name) = dist_tables_resc{w_idx};
        end
        
        out_table = fullfile(folder_name, 'distance_tables.mat');
        save(out_table, 'out');
        % Create figure for waveforms
        figure;
        
        % Subplot 1: Original spikes
        subplot(2, 1, 1);
        plot(orig_spikes', 'b-', 'LineWidth', 0.5);
        title(sprintf('Original Spikes - Channel %d, Cluster %d', ch, cluster_num));
        xlabel('Sample');
        ylabel('Amplitude');
        
        % Subplot 2: Rescued spikes
        subplot(2, 1, 2);
        plot(resc_spikes', 'r-', 'LineWidth', 0.5);
        title(sprintf('Rescued Spikes - Channel %d, Cluster %d', ch, cluster_num));
        xlabel('Sample');
        ylabel('Amplitude');
        
        % Save the waveform plot
        save_path_wave = sprintf('%s/waveforms_wave%d_clust%d.png', folder_name, ch, cluster_num);
        saveas(gcf, save_path_wave);
        close(gcf);
        
        % Figure for normalized distances
        figure;
        hold on;
        h1 = scatter(nan, nan, 'bo', 'filled'); % Dummy for legend - Original assigned
        h2 = scatter(nan, nan, 'go', 'filled'); % Dummy for legend - Original switched
        h3 = scatter(nan, nan, '^b', 'filled'); % Dummy for legend - Original unclassified
        h4 = scatter(nan, nan, 'ro', 'filled'); % Dummy for legend - Rescued assigned
        h5 = scatter(nan, nan, 'ko', 'filled'); % Dummy for legend - Rescued switched
        h6 = scatter(nan, nan, '^r', 'filled'); % Dummy for legend - Rescued unclassified
        for i = 1:n_orig
            for w_idx = 1:length(weights)
                if win_cls_orig(i, w_idx) == cluster_num
                    scatter(weights(w_idx), norm_dist_orig(i, w_idx), 'bo', 'filled');
                elseif win_cls_orig(i, w_idx) == 0
                    scatter(weights(w_idx), norm_dist_orig(i, w_idx), '^b', 'filled');
                else
                    scatter(weights(w_idx), norm_dist_orig(i, w_idx), 'go', 'filled');
                end
            end
        end
        for i = 1:n_resc
            for w_idx = 1:length(weights)
                if win_cls_resc(i, w_idx) == cluster_num
                    scatter(weights(w_idx), norm_dist_resc(i, w_idx), 'ro', 'filled');
                elseif win_cls_resc(i, w_idx) == 0
                    scatter(weights(w_idx), norm_dist_resc(i, w_idx), '^r', 'filled');
                else
                    scatter(weights(w_idx), norm_dist_resc(i, w_idx), 'ko', 'filled');
                end
            end
        end
        set(gca, 'XScale', 'log');
        xticks(weights);
        xticklabels(string(weights));
        title(sprintf('Normalized Distances vs Weight - Cluster %d', cluster_num));
        xlabel('Weight');
        ylabel('Normalized Distance');
        legend([h1 h2 h3 h4 h5 h6], 'Original (assigned)', 'Original (switched)', 'Original (unclassified)', 'Rescued (assigned)', 'Rescued (switched)', 'Rescued (unclassified)', 'Location', 'eastoutside');
        grid on;
        hold off;
        
        % Save normalized distances
        save_path_norm = sprintf('%s/norm_distances_wave%d_clust%d.png', folder_name, ch, cluster_num);
        saveas(gcf, save_path_norm);
        close(gcf);
        
        % Figure for raw distances
        figure;
        hold on;
        h1 = scatter(nan, nan, 'bo', 'filled'); % Dummy for legend - Original assigned
        h2 = scatter(nan, nan, 'go', 'filled'); % Dummy for legend - Original switched
        h3 = scatter(nan, nan, '^b', 'filled'); % Dummy for legend - Original unclassified
        h4 = scatter(nan, nan, 'ro', 'filled'); % Dummy for legend - Rescued assigned
        h5 = scatter(nan, nan, 'ko', 'filled'); % Dummy for legend - Rescued switched
        h6 = scatter(nan, nan, '^r', 'filled'); % Dummy for legend - Rescued unclassified
        for i = 1:n_orig
            for w_idx = 1:length(weights)
                if win_cls_orig(i, w_idx) == cluster_num
                    scatter(weights(w_idx), raw_dist_orig(i, w_idx), 'bo', 'filled');
                elseif win_cls_orig(i, w_idx) == 0
                    scatter(weights(w_idx), raw_dist_orig(i, w_idx), '^b', 'filled');
                else
                    scatter(weights(w_idx), raw_dist_orig(i, w_idx), 'go', 'filled');
                end
            end
        end
        for i = 1:n_resc
            for w_idx = 1:length(weights)
                if win_cls_resc(i, w_idx) == cluster_num
                    scatter(weights(w_idx), raw_dist_resc(i, w_idx), 'ro', 'filled');
                elseif win_cls_resc(i, w_idx) == 0
                    scatter(weights(w_idx), raw_dist_resc(i, w_idx), '^r', 'filled');
                else
                    scatter(weights(w_idx), raw_dist_resc(i, w_idx), 'ko', 'filled');
                end
            end
        end
        set(gca, 'XScale', 'log');
        xticks(weights);
        xticklabels(string(weights));
        title(sprintf('Raw Distances vs Weight - Cluster %d', cluster_num));
        xlabel('Weight');
        ylabel('Raw Distance');
        legend([h1 h2 h3 h4 h5 h6], 'Original (assigned)', 'Original (switched)', 'Original (unclassified)', 'Rescued (assigned)', 'Rescued (switched)', 'Rescued (unclassified)', 'Location', 'eastoutside');
        grid on;
        hold off;
        
        % Save raw distances
        save_path_raw = sprintf('%s/raw_distances_wave%d_clust%d.png', folder_name, ch, cluster_num);
        saveas(gcf, save_path_raw);
        close(gcf);
    else
        warning('Data files not found for distance computation. Plotting waveforms only.');
        
        % Create figure for waveforms
        figure;
        
        % Subplot 1: Original spikes
        subplot(2, 1, 1);
        plot(orig_spikes', 'b-', 'LineWidth', 0.5);
        title(sprintf('Original Spikes - Channel %d, Cluster %d', ch, cluster_num));
        xlabel('Sample');
        ylabel('Amplitude');
        
        % Subplot 2: Rescued spikes
        subplot(2, 1, 2);
        plot(resc_spikes', 'r-', 'LineWidth', 0.5);
        title(sprintf('Rescued Spikes - Channel %d, Cluster %d', ch, cluster_num));
        xlabel('Sample');
        ylabel('Amplitude');
        
        % Save the waveform plot
        save_path_wave = sprintf('%s/waveforms_wave%d_clust%d.png', folder_name, ch, cluster_num);
        saveas(gcf, save_path_wave);
        
        close(gcf);
    end
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

function [normConst, weights] = get_weight_matrix(spike_x, tmplt_vect, pk_weight, amp_dir)
    % Compute peak width for spike_x
    spike_width = get_peak_width(spike_x, amp_dir);
    
    n_templates = size(tmplt_vect, 1);
    n_samples = size(tmplt_vect, 2);
    weights = ones(n_templates, n_samples);
    
    % Create spike mask once (static across templates)
    spike_mask = false(1, n_samples);
    if ~isnan(spike_width.left) && ~isnan(spike_width.right)
        spike_mask(spike_width.left:spike_width.right) = true;
    end
    
    for i = 1:n_templates
        template = tmplt_vect(i, :);
        template_width = get_peak_width(template, amp_dir);
        
        % Create template mask
        template_mask = false(1, n_samples);
        if ~isnan(template_width.left) && ~isnan(template_width.right)
            template_mask(template_width.left:template_width.right) = true;
        end
        
        % Overlap: both have width
        overlap = spike_mask & template_mask;
        % Symmetric difference: only one has width
        symmetric_diff = xor(spike_mask, template_mask);
        
        % Set weights: pk_weight in overlap, max(1, pk_weight/2) in symmetric difference
        weights(i, symmetric_diff) = max(1, pk_weight / 2);
        weights(i, overlap) = pk_weight;
    end
    
    normConst = sqrt(n_samples) ./ sqrt(sum(weights, 2));
end

function width_struct = get_peak_width(spike_x, amp_dir)
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