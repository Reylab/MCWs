function [cross_correlograms, l1_matrix] = compute_cross_correlograms(spike_times_ms, cluster_ids, varargin)
% COMPUTE_CROSS_CORRELOGRAMS - Calculate cross-correlograms between cluster pairs
%
% Args:
%   spike_times_ms: Array of spike times in milliseconds
%   cluster_ids: Array of cluster IDs for each spike
%   Optional:
%     'time_window', [-50, 50] (ms) - window around reference spike
%     'bin_size', 1.0 (ms) - bin width for histogram
%     'exclude_cluster_0', true - whether to exclude noise cluster
%     'normalize', true - normalize by reference spike count
%
% Returns:
%   cross_correlograms: Struct array with CCG data for each pair

    p = inputParser;
    addParameter(p, 'time_window', [-50, 50], @(x) numel(x)==2);
    addParameter(p, 'bin_size', 1.0, @isscalar);
    addParameter(p, 'exclude_cluster_0', true, @islogical);
    addParameter(p, 'normalize', true, @islogical);
    parse(p, varargin{:});
    
    time_window = p.Results.time_window;
    bin_size = p.Results.bin_size;
    
    % Get unique clusters
    unique_clusters = unique(cluster_ids);
    if p.Results.exclude_cluster_0
        unique_clusters = unique_clusters(unique_clusters ~= 0);
    end
    
    n_clusters = length(unique_clusters);
    
    % Create time bins
    bin_edges = time_window(1):bin_size:time_window(2);
    n_bins = length(bin_edges) - 1;
    
    % Preallocate results
    cross_correlograms = [];
    pair_count = 0;
    
    % Calculate all pairwise cross-correlograms
    for i = 1:n_clusters
        ref_cluster = unique_clusters(i);
        ref_spikes = spike_times_ms(cluster_ids == ref_cluster);
        
        for j = 1:n_clusters
            if i == j
                continue; % Skip auto-correlograms for now
            end
            
            target_cluster = unique_clusters(j);
            target_spikes = spike_times_ms(cluster_ids == target_cluster);
            
            pair_count = pair_count + 1;
            
            % Calculate cross-correlogram
            ccg = zeros(1, n_bins);
            
            for k = 1:length(ref_spikes)
                ref_time = ref_spikes(k);
                
                % Find target spikes within time window
                time_diffs = target_spikes - ref_time;
                in_window = (time_diffs >= time_window(1)) & (time_diffs <= time_window(2));
                
                if any(in_window)
                    % Bin the time differences
                    bin_indices = discretize(time_diffs(in_window), bin_edges);
                    valid_bins = bin_indices(~isnan(bin_indices));
                    
                    for bin_idx = valid_bins
                        ccg(bin_idx) = ccg(bin_idx) + 1;
                    end
                end
            end
            
            % Normalize by number of reference spikes
            if p.Results.normalize && length(ref_spikes) > 0
                ccg = ccg / length(ref_spikes);
            end
            
            % Store results
            cross_correlograms(pair_count).ref_cluster = ref_cluster;
            cross_correlograms(pair_count).target_cluster = target_cluster;
            cross_correlograms(pair_count).ccg = ccg;
            cross_correlograms(pair_count).bin_centers = bin_edges(1:end-1) + bin_size/2;
            cross_correlograms(pair_count).time_window = time_window;
            cross_correlograms(pair_count).bin_size = bin_size;

             l1_matrix = [];
    if ~isempty(cross_correlograms)
        % Get unique clusters from cross-correlograms
        all_ref_clusters = [cross_correlograms.ref_cluster];
        all_target_clusters = [cross_correlograms.target_cluster];
        unique_clusters = unique([all_ref_clusters, all_target_clusters]);
        n_clusters = length(unique_clusters);
        
        l1_matrix = NaN(n_clusters, n_clusters);
        
        % Create a lookup for CCG data
        ccg_lookup = containers.Map('KeyType', 'char', 'ValueType', 'any');
        for i = 1:length(cross_correlograms)
            key = sprintf('%d_%d', cross_correlograms(i).ref_cluster, cross_correlograms(i).target_cluster);
            ccg_lookup(key) = cross_correlograms(i).ccg;
        end
        
        % Compute L1 distances between all pairs
        for i = 1:n_clusters
            for j = 1:n_clusters
                if i == j
                    l1_matrix(i,j) = 0; % Distance to self is 0
                else
                    % Try both directions
                    key1 = sprintf('%d_%d', unique_clusters(i), unique_clusters(j));
                    key2 = sprintf('%d_%d', unique_clusters(j), unique_clusters(i));
                    
                    if isKey(ccg_lookup, key1) && isKey(ccg_lookup, key2)
                        ccg1 = ccg_lookup(key1);
                        ccg2 = ccg_lookup(key2);
                        % Use symmetric average
                        l1_matrix(i,j) = 0.5 * (cross_correlogram_distance(ccg1, ccg2) + cross_correlogram_distance(ccg2, ccg1));
                    elseif isKey(ccg_lookup, key1)
                        ccg1 = ccg_lookup(key1);
                        l1_matrix(i,j) = cross_correlogram_distance(ccg1, ccg1);
                    elseif isKey(ccg_lookup, key2)
                        ccg2 = ccg_lookup(key2);
                        l1_matrix(i,j) = cross_correlogram_distance(ccg2, ccg2);
                    else
                        l1_matrix(i,j) = NaN;
                    end
                end
            end
        end
    end
end

function l1_distance = cross_correlogram_distance(ccg1, ccg2)
% CROSS_CORRELOGRAM_DISTANCE - Calculate L1 distance between two cross-correlograms
%
% Args:
%   ccg1, ccg2: Cross-correlogram vectors (must be same length)
%
% Returns:
%   l1_distance: L1 (Manhattan) distance between the two CCGs

    if length(ccg1) ~= length(ccg2)
        error('CCG vectors must have the same length');
    end
    
    % Calculate L1 distance
    l1_distance = sum(abs(ccg1 - ccg2));
end

function significant_pairs = find_significant_ccg_pairs(cross_correlograms, varargin)
% FIND_SIGNIFICANT_CCG_PAIRS - Identify significant cross-correlogram interactions
%
% Args:
%   cross_correlograms: Output from compute_cross_correlograms
%   Optional:
%     'significance_threshold', 3.0 - z-score threshold for significance
%     'baseline_window', [-50, -10] - window for baseline calculation (ms)
%
% Returns:
%   significant_pairs: Struct array of significant interactions

    p = inputParser;
    addParameter(p, 'significance_threshold', 3.0, @isscalar);
    addParameter(p, 'baseline_window', [-50, -10], @(x) numel(x)==2);
    parse(p, varargin{:});
    
    significant_pairs = [];
    count = 0;
    
    for i = 1:length(cross_correlograms)
        ccg_data = cross_correlograms(i);
        ccg = ccg_data.ccg;
        bin_centers = ccg_data.bin_centers;
        
        % Find bins in baseline window
        baseline_mask = (bin_centers >= p.Results.baseline_window(1)) & ...
                       (bin_centers <= p.Results.baseline_window(2));
        
        if sum(baseline_mask) < 2
            continue;
        end
        
        baseline_mean = mean(ccg(baseline_mask));
        baseline_std = std(ccg(baseline_mask));
        
        if baseline_std == 0
            continue;
        end
        
        % Find significant peaks/troughs
        z_scores = (ccg - baseline_mean) / baseline_std;
        significant_bins = find(abs(z_scores) > p.Results.significance_threshold);
        
        if ~isempty(significant_bins)
            count = count + 1;
            significant_pairs(count).ref_cluster = ccg_data.ref_cluster;
            significant_pairs(count).target_cluster = ccg_data.target_cluster;
            significant_pairs(count).z_scores = z_scores;
            significant_pairs(count).significant_bins = significant_bins;
            significant_pairs(count).max_abs_zscore = max(abs(z_scores));
        end
    end
end