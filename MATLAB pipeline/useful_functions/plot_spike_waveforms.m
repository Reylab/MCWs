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
    
    % Calculate common y-axis limits across all spikes
    all_spikes = [orig_spikes(:); resc_spikes(:)];
    y_min = min(all_spikes);
    y_max = max(all_spikes);
    y_limits = [y_min, y_max];
    
    % PLOT 1: Standard color-coded plot (blue = good, red = poor_rescue)
    figure;
    
    % Subplot 1: Good spikes
    subplot(2, 1, 1);
    plot(orig_spikes', 'b-', 'LineWidth', 0.5);
    title(sprintf('Good Spikes - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    
    % Subplot 2: Poor_rescue spikes
    subplot(2, 1, 2);
    plot(resc_spikes', 'r-', 'LineWidth', 0.5);
    title(sprintf('Poor_Rescue Spikes - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    
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
    hold off;
    title(sprintf('Good vs Poor_Rescue Overlay - Channel %d, Cluster %d', ch, cluster_num), 'Interpreter', 'none');
    xlabel('Sample');
    ylabel('Amplitude');
    ylim(y_limits);
    legend([h1, h2], {'Good', 'Poor_Rescue'}, 'Location', 'best', 'Interpreter', 'none');
    
    % Save the overlay plot
    save_path_overlay = sprintf('%s/waveforms_overlay_wave%d_clust%d.png', folder_name, ch, cluster_num);
    saveas(gcf, save_path_overlay);
    close(gcf);
    fprintf('Overlay plot saved to: %s\n', save_path_overlay);
end
