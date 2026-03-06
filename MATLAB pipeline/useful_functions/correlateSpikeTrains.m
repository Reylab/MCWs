function results = correlateSpikeTrains(data1, data2, tolerance)
% correlateSpikeTrains: Correlates two sets of [ClusterID, TimeMS]
% Inputs:
%   data1 - [ClusterID, TimeMS] (N x 2)
%   data2 - [ClusterID, TimeMS] (M x 2)
%   tolerance - Window in ms (e.g., 0.025)

    results = struct();

    % --- 1. Nearest Neighbor Search ---
    [idx2_in_1, dist12] = knnsearch(data2(:,2), data1(:,2));
    [idx1_in_2, dist21] = knnsearch(data1(:,2), data2(:,2));

    % --- 2. Logical Matching ---
    mask1 = dist12 <= tolerance;
    mask2 = dist21 <= tolerance;

    % --- 3. SHARED OVERLAP DATA ---
    % This table contains everything needed to verify spikes
    % Col 1: Index in Data 1 | Col 2: Index in Data 2
    % Col 3: Cluster in Data 1 | Col 4: Cluster in Data 2
    shared_idx1 = find(mask1);
    shared_idx2 = idx2_in_1(mask1);
    
    results.shared_overlap = [shared_idx1, shared_idx2, ...
                              data1(shared_idx1, 1), data2(shared_idx2, 1)];

    % --- 4. UNIQUE DATA ---
    % Format: [OriginalIndex, ClusterID, TimeMS]
    results.unique_data1 = [find(~mask1), data1(~mask1, :)];
    results.unique_data2 = [find(~mask2), data2(~mask2, :)];

    % --- 5. PRINTED REPORT ---
    fprintf('\n==============================================\n');
    fprintf('           SPIKE CORRELATION REPORT          \n');
    fprintf('==============================================\n');
    fprintf('%-15s | %-10s | %-10s\n', 'Dataset', 'Shared', 'Unique');
    fprintf('----------------------------------------------\n');
    fprintf('%-15s | %-10d | %-10d\n', 'Dataset 1', size(results.shared_overlap, 1), size(results.unique_data1, 1));
    fprintf('%-15s | %-10d | %-10d\n', 'Dataset 2', size(results.unique_data2, 1), size(results.unique_data2, 1));
    fprintf('==============================================\n');

    % --- 6. CLUSTER CORRELATION PLOTS ---
    if ~isempty(results.shared_overlap)
        % Row 3 and 4 of our overlap matrix are the Cluster IDs
        [results.overlapMatrix, ~, results.clusterLabels] = ...
            crosstab(results.shared_overlap(:,3), results.shared_overlap(:,4));
        
        figure('Color', 'w', 'Name', 'Cluster Mapping Analysis');
        
        % Plot 1: Mapping Counts
        subplot(1,2,1);
        h1 = heatmap(results.overlapMatrix);
        h1.Title = 'Spike Match Counts';
        h1.XLabel = 'Dataset 2 Clusters'; h1.YLabel = 'Dataset 1 Clusters';

        % Plot 2: Probability Mapping
        subplot(1,2,2);
        probMatrix = results.overlapMatrix ./ sum(results.overlapMatrix, 2);
        h2 = heatmap(probMatrix);
        h2.Title = 'Cluster Mapping Probability (Normalized)';
        h2.XLabel = 'Dataset 2 Clusters'; h2.YLabel = 'Dataset 1 Clusters';
        colormap(parula);
    end

    % --- Global Correlation (Ignoring Clusters) ---

    % Calculate global match percentage
    totalSpikes1 = size(data1, 1);
    totalShared = size(results.shared_overlap, 1);
    matchRate = (totalShared / totalSpikes1) * 100;
    
    fprintf('Global Match Rate: %.2f%%\n', matchRate);
    
    % Calculate the actual time differences (Jitter)
    % shared_overlap(:,1) is index in data1, shared_overlap(:,2) is index in data2
    time1 = data1(results.shared_overlap(:,1), 2);
    time2 = data2(results.shared_overlap(:,2), 2);
    jitter = time1 - time2; 
    
    % Visualization of Global Correlation
    figure('Color', 'w');
    subplot(1,2,1);
    histogram(jitter, 50, 'FaceColor', [0.4 0.6 0.8]);
    title('Global Timing Jitter (All Spikes)');
    xlabel('Time Difference (ms)');
    ylabel('Count');
    
    subplot(1,2,2);
    % Cross-correlogram (Simple version)
    [counts, centers] = histcounts(jitter, 100);
    stem(centers(1:end-1), counts, 'Marker', 'none');
    title('Global Cross-Correlogram');
    xlabel('Lag (ms)');
end