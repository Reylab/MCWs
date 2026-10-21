function results = compute_response_window_sets(arm_class_ids, grapes_list, cc_list, all_onset_map, ...
        stimlist_per_stim_list, all_per_stim_list, stim_list, all_stim_list, ch, dyn_buffer)
% COMPUTE_RESPONSE_WINDOW_SETS  Splits each arm's class spikes into THREE
% non-overlapping buckets in one pass: "stim list" responses (each
% stim_list stimulus's own measured onset+dura), "other" responses (every
% OTHER scored stimulus, using a window borrowed from stim_list's own
% median onset+dura, since most non-target stimuli were never expected to
% elicit a response so their own latency estimate is meaningless/absent),
% and "outside" (not in either).
%
% This replaces separately calling find_response_spikes_dynamic once for
% stim_list and again for all_stim_list -- that older pattern double-counted
% stim_list's own spikes into the "all stimuli" bucket (all_stim_list is a
% superset of stim_list). Doing it in ONE call over all_stim_list, with
% other_stim_list-specific stimuli routed to the borrowed window, guarantees
% each spike lands in exactly one bucket, and "outside" is a true "not in
% ANY window" baseline (see find_response_spikes_dynamic's other_stim_list
% argument for how the single unified window set is built).
%
% VARIABLES
%   arm_class_ids   - input: numeric vector or cell array, resolved class number per arm
%   grapes_list/cc_list - input: cell arrays, one arm's grapes struct / cluster_class per arm
%   all_onset_map   - input: containers.Map covering every scored stimulus (stim_list + other)
%   stimlist_per_stim_list - input: cell array, one arm's per_stim table SCOPED TO stim_list
%                     (get_class_metrics's m.per_stim) -- source of the borrowed median onset/dura
%   all_per_stim_list - input: cell array, one arm's per_stim table for EVERY scored stimulus
%                     (get_class_metrics's m.per_stim_all) -- needed so stim_list members still
%                     get their own row inside the unified call
%   stim_list/all_stim_list - input: the "real" trusted stim IDs, and every scored stim ID
%   ch/dyn_buffer   - input: channel number, ms padding on each side of every window
%   results         - output: struct array (one per arm) with:
%                     .inside_stimlist  - spikes inside a stim_list stimulus's own window
%                     .inside_other     - spikes inside another (borrowed-window) stimulus's window
%                     .outside          - spikes inside neither
%                     .stim_of_inside   - which stim ID matched each inside spike (both kinds, in
%                                         the original find_response_spikes_dynamic order)
%                     .other_stim_list  - setdiff(all_stim_list, stim_list)
%                     .fixed_onset/.fixed_dura - the borrowed window actually used (NaN if
%                                         stim_list had no good_lat rows to take a median from)
    if isnumeric(arm_class_ids)
        arm_class_ids = num2cell(arm_class_ids);
    end
    other_stim_list = setdiff(all_stim_list, stim_list);
    n = numel(arm_class_ids);
    results = struct('inside_stimlist',{},'inside_other',{},'outside',{},'stim_of_inside',{}, ...
                      'other_stim_list',{},'fixed_onset',{},'fixed_dura',{});
    for k = 1:n
        ps = stimlist_per_stim_list{k};
        good = ps.good_lat;
        fixed_onset = median(ps.onset(good), 'omitnan');
        fixed_dura  = median(ps.dura(good), 'omitnan');

        [in_k, out_k, stim_of_k] = find_response_spikes_dynamic( ...
            arm_class_ids{k}, grapes_list{k}, cc_list{k}, all_onset_map, all_per_stim_list{k}, ...
            all_stim_list, ch, dyn_buffer, [], other_stim_list, fixed_onset, fixed_dura);

        is_other = ismember(stim_of_k, other_stim_list);
        results(k).inside_stimlist = in_k(~is_other);
        results(k).inside_other = in_k(is_other);
        results(k).outside = out_k;
        results(k).stim_of_inside = stim_of_k;
        results(k).other_stim_list = other_stim_list;
        results(k).fixed_onset = fixed_onset;
        results(k).fixed_dura = fixed_dura;
    end
end
