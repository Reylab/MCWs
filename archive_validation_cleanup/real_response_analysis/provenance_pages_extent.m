function [lo, hi] = provenance_pages_extent(view_sets, times_struct, spikes_struct, tol, overlays)
% PROVENANCE_PAGES_EXTENT  Raw [lo hi] (no padding) of everything
% plot_provenance_waveform_pages.m would draw for these views: every
% category panel's mean+/-std band plus every overlay curve. Lets the caller
% fold all breakdown pages (every arm, entire + not-GT, every view) into ONE
% fixed y-limit before drawing any of them. Must stay in step with that
% function's own per-panel waveform lookup.
    lo = Inf; hi = -Inf;
    for vi = 1:numel(view_sets)
        groups = compute_spike_waveform_groups(times_struct, spikes_struct, view_sets{vi}, tol);
        for gi = 1:numel(groups)
            W = get_waveforms_for_times(groups(gi).times, times_struct.cluster_class(:,2), times_struct.spikes, tol);
            if isempty(W), continue; end
            mu = mean(W,1); sd = std(W,0,1);
            lo = min(lo, min(mu-sd)); hi = max(hi, max(mu+sd));
        end
    end
    for oi = 1:numel(overlays)
        lo = min(lo, min(overlays(oi).wave)); hi = max(hi, max(overlays(oi).wave));
    end
end
