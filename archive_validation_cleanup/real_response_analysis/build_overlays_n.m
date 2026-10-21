function overlays = build_overlays_n(ref_name, ref_template, arm_name, arm_template, matched_name, matched_template)
% BUILD_OVERLAYS_N  Standalone reference-template overlay list: {ref
% template, arm template, matched template}, same colors/styles/widths
% used on every waveform page in this pipeline (plot_waveform_comparison_n.m,
% plot_venn_waveform_group.m). ref_name/arm_name label the first two
% slots (ref_name is GT for most groups, or armK for the consecutive-pair
% groups); matched_name labels who the matched template was paired with.
% Any wave left empty (or all-NaN) is silently skipped.
%
% Colors deliberately avoid the red/blue/green already used elsewhere for
% the gt_only/arm_only/matched GROUP shading (plot_venn_waveform_group.m),
% so a template line is never confused for a group's own mean+/-std band.
% Each overlay also carries a WIDTH -- matched (the important one, see
% run_rescue_grape_analysis.m) is thickest, so it stays legible where
% templates cross; the drawing code (draw_overlays in
% plot_venn_waveform_group.m, and plot_waveform_group's own overlay loop
% in plot_waveform_comparison_n.m) draws thickest-first/thinnest-last so
% the thin lines always render ON TOP of the thick one instead of a
% draw-order coin-flip deciding which line is visible where they overlap.
% Solid lines with clearly separated widths (4 / 2.2 / 1.0): when the
% templates coincide (GT vs matched typically differ by <1% of peak), each
% thinner line shows as a stripe INSIDE the thicker one below it. Dashed
% styles broke this -- the gaps in the top line let the lower one through.
    overlays = struct('name',{},'wave',{},'color',{},'style',{},'width',{});
    if ~isempty(ref_template) && ~all(isnan(ref_template))
        overlays(end+1) = struct('name',sprintf('%s template',ref_name),'wave',ref_template,'color',[0 0 0],'style','-','width',1.0);
    end
    if ~isempty(arm_template) && ~all(isnan(arm_template))
        overlays(end+1) = struct('name',sprintf('%s template',arm_name),'wave',arm_template,'color',[0.95 0.6 0.1],'style','-','width',2.2);
    end
    if ~isempty(matched_template) && ~all(isnan(matched_template))
        overlays(end+1) = struct('name',sprintf('matched (%s+%s)',ref_name,matched_name),'wave',matched_template,'color',[0.85 0.5 0.85],'style','-','width',4.0);
    end
end
