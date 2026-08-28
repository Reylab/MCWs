function [cluster_summary, cluster_masks] = get_cluster_quarantine_reasons(spikes_struct, times_struct)
% get_cluster_quarantine_reasons - Break down within_channel quarantine reasons by cluster
% Inputs:
%   spikes_struct - Struct from spikes file (e.g., SPK = load('CSC1_spikes.mat'))
%                   must contain index_all and quarantine_properties.reason
%   times_struct  - Struct from a clustering run on all spikes (e.g., TIMES = load('times_CSC1_spikes.mat'))
%                   must contain cluster_class (col1 = cluster id, col2 = spike time)
% Outputs:
%   cluster_summary - table: cluster_id, n_spikes, n_quarantined, then one count column per reason
%   cluster_masks   - struct keyed by cluster_<id>, each with:
%                       .all          - logical mask into index_all/spikes_all: all members of that cluster
%                       .quarantined  - logical mask into index_all/spikes_all: only the quarantined members

    index_all = spikes_struct.index_all(:);
    reason_all = spikes_struct.quarantine_properties.reason(:);

    if isfield(spikes_struct, 'mask_non_quarantine')
        quarantined_all = ~logical(spikes_struct.mask_non_quarantine(:));
    else
        quarantined_all = ~strcmp(reason_all, 'pass');
    end

    cluster_ids = times_struct.cluster_class(:, 1);
    spike_times = times_struct.cluster_class(:, 2);

    % Match clustered spikes back to their position in index_all by timestamp
    [tf, loc] = ismember(spike_times, index_all);
    if any(~tf)
        warning('get_cluster_quarantine_reasons:UnmatchedSpikes', ...
            '%d/%d clustered spikes were not found in index_all (timestamp mismatch).', ...
            sum(~tf), numel(tf));
    end

    reason_categories = unique(reason_all);
    unique_clusters = unique(cluster_ids(cluster_ids > 0)); % skip class 0 (unclassified)

    cluster_masks = struct();
    summary_rows = cell(numel(unique_clusters), 3 + numel(reason_categories));

    for ci = 1:numel(unique_clusters)
        c = unique_clusters(ci);
        in_c = (cluster_ids == c);

        matched_in_c = in_c & tf;
        reason_in_c = repmat({'not_found'}, sum(in_c), 1);
        reason_in_c(matched_in_c(in_c)) = reason_all(loc(matched_in_c));
        quarantined_in_c = false(sum(in_c), 1);
        quarantined_in_c(matched_in_c(in_c)) = quarantined_all(loc(matched_in_c));

        mask_field = sprintf('cluster_%d', c);
        mask_all_members = false(size(index_all));
        mask_all_members(loc(matched_in_c)) = true;

        mask_quarantined_members = false(size(index_all));
        q_idx = loc(matched_in_c);
        q_idx = q_idx(quarantined_all(q_idx));
        mask_quarantined_members(q_idx) = true;

        cluster_masks.(mask_field).all = mask_all_members;
        cluster_masks.(mask_field).quarantined = mask_quarantined_members;

        row = {c, sum(in_c), sum(quarantined_in_c)};
        for r = 1:numel(reason_categories)
            row{end+1} = sum(strcmp(reason_in_c, reason_categories{r})); %#ok<AGROW>
        end
        summary_rows(ci, :) = row;
    end

    varnames = [{'cluster_id', 'n_spikes', 'n_quarantined'}, reason_categories(:)'];
    cluster_summary = cell2table(summary_rows, 'VariableNames', varnames);
end
