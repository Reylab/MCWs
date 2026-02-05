function [random_indices, random_spikes] = get_random_original_spikes(spikes_struct, times_struct, cluster_num, num_to_select)
% get_random_original_spikes - Extract random original (non-rescued) spikes from a specific cluster
% Inputs:
%   spikes_struct - Struct from spikes file (e.g., spk325)
%   times_struct - Struct from times file (e.g., times325)
%   cluster_num - Cluster ID to filter by
%   num_to_select - Number of random spikes to select
% Outputs:
%   random_indices - Random indices from selected spikes
%   random_spikes - Corresponding spike waveforms

    % Get rescued timestamps
    rescued_timestamps = spikes_struct.index_all(spikes_struct.rescue_mask);
    
    % Find original (non-rescued) spikes in the clustered data
    is_rescued = ismember(spikes_struct.index, rescued_timestamps);
    is_original = ~is_rescued;
    
    % Filter by cluster
    in_cluster = times_struct.cluster_class(:, 1) == cluster_num;
    
    % Get selected indices (original spikes in cluster)
    selected_indices = find(is_original & in_cluster);
    
    % Select random num_to_select (or all if fewer)
    if length(selected_indices) <= num_to_select
        random_indices = selected_indices;
    else
        random_indices = selected_indices(randperm(length(selected_indices), num_to_select));
    end
    
    % Get corresponding spikes
    random_spikes = times_struct.spikes(random_indices, :);
end