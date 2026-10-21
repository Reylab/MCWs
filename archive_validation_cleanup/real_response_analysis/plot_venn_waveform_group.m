function plot_venn_waveform_group(gt_name, gt_times, arm_name, arm_times, gtc, arm_class, ch, ...
    matched_gt, matched_arm, gt_only, arm_only, tol, ...
    gt_template, arm_template, matched_template_global, ...
    group_title, out_png, shared_ylim)
% PLOT_VENN_WAVEFORM_GROUP  One composite page per venn diagram (see
% plot_membership_set.m / draw_membership_venn.m, the venn embedded in this
% same page): top panel pools every spike in the venn (matched + gt_only +
% arm_only) color-coded by which region it came from (green = matched, same
% convention as the venn's own overlap lens), the venn diagram itself in the
% middle, and three panels below it for each region on its own -- gt_only /
% matched / arm_only, the venn's own left-to-right order -- each carrying the same {GT template, arm template,
% matched template} overlay scheme the rest of the waveform pages use
% (build_overlays_n, shared with plot_waveform_comparison_n.m).
%
% matched_template_global is ALWAYS the whole-session (all_spikes-scope)
% matched template, even when THIS call's own matched_gt/matched_arm are a
% narrower subset (the response/non_response/inside-total/outside/
% pre_rescue venns are all scoped subsets of the full population) -- the
% overall population's matched template is the more reliable reference, not
% whatever a possibly-small subset happens to average to. Build it ONCE per
% arm (see run_rescue_grape_analysis.m) and pass the same struct into every
% one of that arm's venn calls.
%
% shared_ylim: precomputed [lo hi] (see venn_waveform_extent.m), shared
% across EVERY venn's waveform page in this run, so amplitude is visually
% comparable across every venn diagram, not just within one page. Pass []
% to fall back to each panel auto-scaling independently.
%
% VARIABLES
%   gt_name/arm_name     - labels (e.g. 'gt', 'rescue_sd1')
%   gt_times/arm_times   - loaded times_*.mat structs (.cluster_class, .spikes)
%   gtc/arm_class        - GT/arm class numbers (venn title labeling only)
%   matched_gt/matched_arm - this venn's matched spike times, GT side / arm side
%   gt_only/arm_only     - this venn's unmatched spike times, each side
%   tol                  - spike-time match tolerance (ms)
%   gt_template/arm_template - each side's own mean waveform (whole class)
%   group_title/out_png  - figure title / save path

    col_gt_only  = [0.85 0.3 0.3];   % matches draw_membership_venn's left-circle red
    col_arm_only = [0.3 0.4 0.85];   % matches draw_membership_venn's right-circle blue
    col_matched  = [0.2 0.7 0.3];    % matches draw_membership_venn's overlap-lens green

    % matched_name = arm_name (not a compound string) -- build_overlays_n's
    % own label format is 'matched (%s+%s)', ref_name, matched_name, so
    % passing anything other than the plain arm name here double-prints
    % gt_name (e.g. "matched (gt+gt+rescue_sd1)").
    overlays = build_overlays_n(gt_name, gt_template, arm_name, arm_template, ...
        arm_name, matched_template_global);

    W_gt_only  = get_waveforms_for_times(gt_only,  gt_times.cluster_class(:,2),  gt_times.spikes,  tol);
    W_arm_only = get_waveforms_for_times(arm_only, arm_times.cluster_class(:,2), arm_times.spikes, tol);
    % One waveform per matched PAIR (GT side), so n equals the venn's matched
    % count -- stacking GT + arm copies of the same spike double-counted it.
    W_matched  = get_waveforms_for_times(matched_gt, gt_times.cluster_class(:,2), gt_times.spikes, tol);

    f = figure('Visible','off','Position',[100 100 1000 1150],'Color','w');
    tl = tiledlayout(f,3,3,'Padding','loose','TileSpacing','loose');
    title(tl, group_title, 'Interpreter','none');

    % ---- row 1: all three regions pooled, color-coded by venn membership ----
    ax_top = nexttile(tl,1,[1 3]); hold(ax_top,'on');
    draw_mean_band(ax_top, W_gt_only,  col_gt_only,  sprintf('%s-only (n=%d)', gt_name, size(W_gt_only,1)));
    draw_mean_band(ax_top, W_arm_only, col_arm_only, sprintf('%s-only (n=%d)', arm_name, size(W_arm_only,1)));
    draw_mean_band(ax_top, W_matched,  col_matched,  sprintf('matched (n=%d)', size(W_matched,1)));
    draw_overlays(ax_top, overlays);
    title(ax_top, 'all spikes, by venn region', 'Interpreter','none','FontSize',9);
    xlabel(ax_top,'sample'); ylabel(ax_top,'amplitude'); grid(ax_top,'on');
    legend(ax_top, 'show', 'FontSize',7, 'Interpreter','none', 'Location','eastoutside');

    % ---- row 2: the venn diagram itself ----
    ax_venn = nexttile(tl,4,[1 3]);
    draw_membership_venn(ax_venn, gtc, arm_name, arm_class, ...
        numel(matched_gt)+numel(gt_only), numel(matched_arm)+numel(arm_only), ...
        numel(matched_gt), numel(gt_only), numel(arm_only), gt_name, ch);

    % ---- row 3: gt_only / matched / arm_only -- same left-to-right order as the venn ----
    sub_W      = {W_gt_only, W_matched, W_arm_only};
    sub_titles = {sprintf('%s-only (n=%d)', gt_name, size(W_gt_only,1)), ...
                  sprintf('matched (n=%d)', size(W_matched,1)), ...
                  sprintf('%s-only (n=%d)', arm_name, size(W_arm_only,1))};
    sub_colors = {col_gt_only, col_matched, col_arm_only};
    ax_bot = gobjects(1,3);
    for si = 1:3
        ax = nexttile(tl, 6+si); hold(ax,'on');
        ax_bot(si) = ax;
        draw_mean_band(ax, sub_W{si}, sub_colors{si}, 'mean +/- std (this group)');
        draw_overlays(ax, overlays);
        title(ax, sub_titles{si}, 'Interpreter','none','FontSize',9);
        xlabel(ax,'sample'); ylabel(ax,'amplitude'); grid(ax,'on');
        if si == 1
            lgd = legend(ax, 'show', 'FontSize',7, 'Interpreter','none', 'Orientation','horizontal');
            lgd.Layout.Tile = 'south';
        end
    end

    if ~isempty(shared_ylim)
        ylim(ax_top, shared_ylim);
        for si = 1:3, ylim(ax_bot(si), shared_ylim); end
    end

    save_fig_local(f, out_png);
end

function draw_mean_band(ax, W, color, display_name)
    if isempty(W)
        text(ax,0.5,0.5,'no spikes','HorizontalAlignment','center','Units','normalized');
        return;
    end
    mu = mean(W,1); sd = std(W,0,1); xx = 1:numel(mu);
    fill(ax,[xx fliplr(xx)],[mu+sd fliplr(mu-sd)],color,'FaceAlpha',0.2,'EdgeColor','none','HandleVisibility','off');
    plot(ax, xx, mu, 'Color', color, 'LineWidth', 1.6, 'DisplayName', display_name);
end

function draw_overlays(ax, overlays)
% Thickest overlay drawn FIRST (so it sits at the back) and thinnest last
% (so it renders on top, staying visible wherever the lines cross) --
% independent of whatever order build_overlays_n happened to build the
% struct array in.
    if isempty(overlays), return; end
    [~, order] = sort([overlays.width], 'descend');
    for oi = order
        ov = overlays(oi);
        plot(ax, 1:numel(ov.wave), ov.wave, ov.style, 'Color', ov.color, 'LineWidth', ov.width, 'DisplayName', ov.name);
    end
end

function save_fig_local(f, out_png)
    set(f, 'Color', 'w');
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
    end
    close(f);
end
