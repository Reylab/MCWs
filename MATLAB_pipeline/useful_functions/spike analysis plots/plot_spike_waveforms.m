function plot_spike_waveforms(orig_spikes, resc_spikes, ch, cluster_num)
% plot_spike_waveforms - Plot comparison of original vs rescued spike waveforms
% Inputs:
%   orig_spikes - Matrix of original spike waveforms (n_spikes x n_samples)
%   resc_spikes - Matrix of rescued spike waveforms (n_spikes x n_samples)
%   ch - Channel number or label
%   cluster_num - Cluster number

    % Create folder
    folder_name = sprintf('wave%d_clust%d', ch, cluster_num);
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    
    % Load data for template
    ch_lbl = get_channel_label(ch);
    fname_spk = sprintf('%s_spikes.mat', ch_lbl);
    fname_times = sprintf('times_%s.mat', ch_lbl);
    
    template_center = [];
    template_std = [];
    
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
            rescued_timestamps = SPK.index_all(SPK.rescue_mask);
            good_timestamps = SPK.index(class_good_mask);
            original_good_mask = ~ismember(good_timestamps, rescued_timestamps);
        else
            original_good_mask = true(size(class_good));
        end
        
        % Build templates
        [centers, maxdist, ~] = build_templates(class_good(original_good_mask), spikes_good_classified(original_good_mask, :));
        
        if cluster_num <= size(centers, 1)
            template_center = centers(cluster_num, :);
            template_std = maxdist(cluster_num);
        end
    end
    
    % Calculate common y-axis limits across all spikes
    all_spikes = [orig_spikes(:); resc_spikes(:)];
    y_min = min(all_spikes);
    y_max = max(all_spikes);
    y_limits = [y_min, y_max];
    
    % PLOT 1: Standard color-coded plot (blue = good, red = poor_rescue)
    figure;
    
    % Subplot 1: Good spikes
    subplot(2, 1, 1);
    hold on;
    plot(orig_spikes', 'b-', 'LineWidth', 0.5);
    if ~isempty(template_center)
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good Spikes - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    if ~isempty(template_center)
        legend({'Spikes', 'Template', '+1 STD', '-1 STD'}, 'Location', 'best');
    end
    
    % Subplot 2: Poor_rescue spikes
    subplot(2, 1, 2);
    hold on;
    plot(resc_spikes', 'r-', 'LineWidth', 0.5);
    if ~isempty(template_center)
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Poor_Rescue Spikes - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    if ~isempty(template_center)
        legend({'Spikes', 'Template', '+1 STD', '-1 STD'}, 'Location', 'best');
    end
    
    % Save the standard waveform plot
    save_path_standard = sprintf('%s/waveforms_standard_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path_standard);
    close(gcf);
    fprintf('Standard waveform plot saved to: %s\n', save_path_standard);
    
    % PLOT 2: Individual color for each spike
    figure;
    
    % Subplot 1: Good spikes with individual colors
    subplot(2, 1, 1);
    n_orig = size(orig_spikes, 1);
    colors_orig = lines(n_orig);  % Generate distinct colors
    hold on;
    for i = 1:n_orig
        plot(orig_spikes(i, :), 'Color', colors_orig(i, :), 'LineWidth', 0.5);
    end
    if ~isempty(template_center)
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good Spikes (Individual Colors) - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    
    % Subplot 2: Poor_rescue spikes with individual colors
    subplot(2, 1, 2);
    n_resc = size(resc_spikes, 1);
    colors_resc = lines(n_resc);  % Generate distinct colors
    hold on;
    for i = 1:n_resc
        plot(resc_spikes(i, :), 'Color', colors_resc(i, :), 'LineWidth', 0.5);
    end
    if ~isempty(template_center)
        plot(template_center, 'k-', 'LineWidth', 2.5);
        plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Poor_Rescue Spikes (Individual Colors) - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    
    % Save the individual colors plot
    save_path_colors = sprintf('%s/waveforms_individual_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path_colors);
    close(gcf);
    fprintf('Individual colors plot saved to: %s\n', save_path_colors);
    
    % PLOT 3: Overlay of both good (blue) and poor_rescue (red) spikes
    figure;
    hold on;
    h1 = plot(orig_spikes(1, :)', 'b-', 'LineWidth', 0.5);
    plot(orig_spikes(2:end, :)', 'b-', 'LineWidth', 0.5);
    h2 = plot(resc_spikes(1, :)', 'r-', 'LineWidth', 0.5);
    plot(resc_spikes(2:end, :)', 'r-', 'LineWidth', 0.5);
    if ~isempty(template_center)
        h3 = plot(template_center, 'k-', 'LineWidth', 2.5);
        h4 = plot(template_center + template_std, 'k--', 'LineWidth', 1.5);
        plot(template_center - template_std, 'k--', 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('Good vs Poor_Rescue Overlay - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    if ~isempty(template_center)
        legend([h1, h2, h3, h4], {'Good', 'Poor_Rescue', 'Template', '+/-1 STD'}, 'Location', 'best', 'Interpreter', 'none');
    else
        legend([h1, h2], {'Good', 'Poor_Rescue'}, 'Location', 'best', 'Interpreter', 'none');
    end
    
    % Save the overlay plot
    save_path_overlay = sprintf('%s/waveforms_overlay_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path_overlay);
    close(gcf);
    fprintf('Overlay plot saved to: %s\n', save_path_overlay);
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
