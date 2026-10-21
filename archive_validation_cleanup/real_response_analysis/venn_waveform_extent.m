function yl = venn_waveform_extent(gt_times, arm_times, matched_gt, ~, gt_only, arm_only, tol, overlays)
% VENN_WAVEFORM_EXTENT  [lo hi] data extent (every region's mean+/-std band,
% plus every overlay curve) for ONE venn's waveform group. Used to fold
% many groups' extents into ONE shared y-limit across every venn diagram's
% waveform page in a run (see run_rescue_grape_analysis.m's two-pass
% collect-then-draw loop) -- same idea as plot_waveform_comparison_n.m's
% compute_shared_ylim, but pooled ACROSS groups instead of just within one
% page's views. Returns [] if this group has no data at all (caller should
% skip empty results when folding into the overall min/max).
    W_gt_only  = get_waveforms_for_times(gt_only,  gt_times.cluster_class(:,2),  gt_times.spikes,  tol);
    W_arm_only = get_waveforms_for_times(arm_only, arm_times.cluster_class(:,2), arm_times.spikes, tol);
    % Must mirror plot_venn_waveform_group.m exactly: one GT-side waveform per matched pair.
    W_matched  = get_waveforms_for_times(matched_gt, gt_times.cluster_class(:,2), gt_times.spikes, tol);

    lo = Inf; hi = -Inf;
    groups = {W_gt_only, W_arm_only, W_matched};
    for gi = 1:numel(groups)
        W = groups{gi};
        if isempty(W), continue; end
        mu = mean(W,1); sd = std(W,0,1);
        lo = min(lo, min(mu-sd)); hi = max(hi, max(mu+sd));
    end
    for oi = 1:numel(overlays)
        lo = min(lo, min(overlays(oi).wave)); hi = max(hi, max(overlays(oi).wave));
    end
    if ~isfinite(lo) || ~isfinite(hi) || lo == hi
        yl = [];
    else
        yl = [lo hi];
    end
end
