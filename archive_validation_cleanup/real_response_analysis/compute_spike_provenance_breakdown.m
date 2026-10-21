function pv = compute_spike_provenance_breakdown(times_struct, spikes_struct, subset_times, tol)
% COMPUTE_SPIKE_PROVENANCE_BREAKDOWN  For a given subset of a rescue arm's
% own spike times, splits them into original / forced / rescued counts, and
% (for the rescued portion) tallies which quarantine mask(s) they originally
% failed. Thin wrapper over classify_spike_provenance.m -- see that file for
% the actual per-spike classification (and why it uses tolerance-based
% matching instead of exact equality). compute_spike_waveform_groups.m uses
% the SAME classifier to get actual spike times per category instead of
% counts, so the two can't disagree.
%
% Only meaningful for RESCUE ARMS (wave_clus + rescue_spikes metadata) --
% GT/hand-selected data has none of these fields, so don't call this on it.
%
% VARIABLES
%   times_struct/spikes_struct/subset_times/tol - input: see classify_spike_provenance.m
%   pv            - output: struct with n_orig/n_forced/n_resc. n_forced
%                   EXCLUDES rescued spikes (rescue takes priority: a spike
%                   that's both force-classified and rescued counts under
%                   n_resc, not n_forced) so n_orig+n_forced+n_resc always
%                   sums to n -- a true partition, not "involvement" counts.
%                   mask_labels (cellstr, only masks actually present),
%                   mask_counts (per label, how many of the RESCUED subset
%                   spikes failed it -- non-disjoint FROM EACH OTHER, a
%                   spike failing 2 masks counts under both, same
%                   "involvement" convention as responses/mask_audit.m and
%                   rescue_mask_breakdown.m -- but every rescued spike is
%                   counted in n_resc exactly once regardless of how many
%                   masks it failed)
    n = numel(subset_times);
    pv = struct('n_orig', n, 'n_forced', 0, 'n_resc', 0, 'mask_labels', {{}}, 'mask_counts', []);
    if n == 0
        return;
    end

    pc = classify_spike_provenance(times_struct, spikes_struct, subset_times, tol);
    pv.n_resc = sum(pc.is_rescued);
    pv.n_forced = sum(pc.is_forced & ~pc.is_rescued);
    pv.n_orig = sum(~pc.is_forced & ~pc.is_rescued);

    if pv.n_resc == 0 || isempty(pc.mask_labels)
        return;
    end
    pv.mask_labels = pc.mask_labels;
    pv.mask_counts = sum(pc.mask_fail(pc.is_rescued, :), 1);
end
