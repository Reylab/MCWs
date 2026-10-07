function [random_indices, random_spikes] = get_random_rescued_spikes(spikes_struct, times_struct, cluster_num, sample_condition_func)
% get_random_rescued_spikes - Extract random 20 rescued spikes from a specific cluster meeting a condition
% Inputs:
%   spikes_struct - Struct from spikes file (e.g., spk325)
%   times_struct - Struct from times file (e.g., times325)
%   cluster_num - Cluster ID to filter by
%   sample_condition_func - Function handle for sample condition, e.g., @(spikes) spikes(:,20) > -40
% Outputs:
%   random_indices - Random 20 (or fewer) indices from selected spikes
%   random_spikes - Corresponding spike waveforms

    % Get rescued timestamps
    rescued_timestamps = spikes_struct.index_all(spikes_struct.rescue_mask);
    
    % Find rescued spikes in the clustered data
    is_rescued = ismember(spikes_struct.index, rescued_timestamps);
    
    % Filter by cluster
    in_cluster = times_struct.cluster_class(:, 1) == cluster_num;
    
    % Apply sample condition
    sample_condition = sample_condition_func(times_struct.spikes);
    
    % Get selected indices
    selected_indices = find(is_rescued & in_cluster & sample_condition);
    
    % Select random 20 (or all if fewer)
    if length(selected_indices) <= 20
        random_indices = selected_indices;
    else
        random_indices = selected_indices(randperm(length(selected_indices), 20));
    end
    
    % Get corresponding spikes
    random_spikes = times_struct.spikes(random_indices, :);
end