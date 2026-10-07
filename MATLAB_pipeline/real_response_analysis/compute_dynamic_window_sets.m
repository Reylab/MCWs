function results = compute_dynamic_window_sets(arm_class_ids, grapes_list, cc_list, onset_map, per_stim_list, stim_list, ch, dyn_buffer)
% COMPUTE_DYNAMIC_WINDOW_SETS  Runs find_response_spikes_dynamic for an
% arbitrary number of arms against a shared onset_map/stim_list. Generalizes
% the duplicated per-arm calls that used to be hand-copied 3x in
% test_rescue_grape_metrics.m (once for the stim_list-scoped summary, again
% for the all-stim overall summary) so adding/removing an arm is a list
% change, not a copy-pasted new call.
%
% VARIABLES
%   arm_class_ids - input: numeric vector or cell array, resolved class number per arm
%                   (arm 1 is whichever arm the caller wants first, e.g. GT)
%   grapes_list   - input: cell array, one arm's loaded grapes_*.mat struct per arm
%   cc_list       - input: cell array, one arm's full cluster_class matrix per arm
%   onset_map     - input: containers.Map, stim ID -> onset times (ms), shared by all arms
%   per_stim_list - input: cell array, one arm's per_stim table (from get_class_metrics) per arm
%   stim_list     - input: which stim IDs to score
%   ch            - input: channel number
%   dyn_buffer    - input: ms padding on each side of a stimulus's own [onset, onset+dura] window
%   results       - output: struct array (one per arm) with .inside/.outside/.stim_of_inside
    if isnumeric(arm_class_ids)
        arm_class_ids = num2cell(arm_class_ids);
    end
    n = numel(arm_class_ids);
    results = struct('inside',{},'outside',{},'stim_of_inside',{});
    for k = 1:n
        [in_k, out_k, stim_of_k] = find_response_spikes_dynamic( ...
            arm_class_ids{k}, grapes_list{k}, cc_list{k}, onset_map, per_stim_list{k}, stim_list, ch, dyn_buffer);
        results(k).inside = in_k;
        results(k).outside = out_k;
        results(k).stim_of_inside = stim_of_k;
    end
end
