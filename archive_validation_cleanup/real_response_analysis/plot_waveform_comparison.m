function plot_waveform_comparison(times_gt, times_sd1, times_sd2, class_num, arm_class_sd1, arm_class_sd2, ...
        gt_inside_dyn, sd1_inside_dyn, sd2_inside_dyn, ...
        gt_inside_other, sd1_inside_other, sd2_inside_other, ...
        gt_outside_all, sd1_outside_all, sd2_outside_all, ...
        tol, ch, out_dir, varargin)
% PLOT_WAVEFORM_COMPARISON  15 waveform panels (mean +/- std), 5 groups x 3
% views, packaged as 5 composite PNGs (one per group):
%   Group A: GT spikes missed by BOTH SD1 and SD2 (full-session populations,
%            not response-window-restricted) -- the strongest "rescue can't
%            recover this" case
%   Group B: SD1-not-GT (what rescue adds beyond GT, full populations)
%   Group C: SD2-not-SD1 (what the higher sdnum adds beyond SD1, full populations)
%   Group D: GT-not-SD1, regardless of SD2 status (full populations)
%   Group E: GT-not-SD2, regardless of SD1 status (full populations)
% Each group split into 3 views by where its spikes actually fall:
%   1) in-response, stim_list scope   2) in-response, OTHER (non-stim_list) scope
%   3) outside-response, all-scored-stimuli scope (the only trustworthy
%      "outside" definition -- see the overall summary's comment on why
%      stim_list-scoped outside is contaminated)
%   View 2 is deliberately "other", not "all" -- pass sets that are already
%   disjoint from view 1 (e.g. compute_response_window_sets's .inside_other,
%   which is scored against a window BORROWED from stim_list's own median
%   onset/dura, since most non-stim_list stimuli were never expected to
%   elicit a response and so have no trustworthy latency of their own).
%   Passing the old "all scored stimuli" set here instead would double-count
%   stim_list's own spikes into this view.
%
% VARIABLES
%   times_gt/times_sd1/times_sd2 - loaded times_*.mat structs per arm (need
%                   .cluster_class and .spikes, i.e. the FULL times file, not a subset)
%   class_num/arm_class_sd1/arm_class_sd2 - matched class number per arm
%   gt_inside_dyn/sd1_inside_dyn/sd2_inside_dyn - stim_list-scoped dynamic in-window sets
%   gt_inside_other/sd1_inside_other/sd2_inside_other - OTHER (non-stim_list)-scoped in-window
%                   sets, disjoint from *_inside_dyn (see compute_response_window_sets)
%   gt_outside_all/sd1_outside_all/sd2_outside_all - all-stimuli-scoped outside sets
%   tol           - spike-time match tolerance (ms), same convention as elsewhere
%   ch/out_dir    - channel (labels only), output folder
%
%   gt_full_t/sd1_full_t/sd2_full_t - each arm's FULL matched-class population,
%                   whole session, not response-window-restricted at all
%   in_sd1/in_sd2 - per-GT-spike: does it have a full-population counterpart in sd1/sd2
%   groupA/groupB/groupC/groupD/groupE - each group's base spike-time set (still absolute time)
%   viewA1/viewA2/viewA3 (and B/C/D/E equivalents) - group base set intersected
%                   with the relevant in/out-window scope
%
% OPTIONAL NAME/VALUE ARGS (both default to '', which skips that arm's
% stored-template overlay -- existing callers that don't pass them still
% work exactly as before, just without that one overlay line):
%   'sd1_template_file'  path to sd1's templates_wc_offline.mat
%   'sd2_template_file'  path to sd2's templates_wc_offline.mat
%
% TEMPLATE OVERLAYS (added on top of each panel's own mean+/-std band):
%   Every group panel gets the GT template (mean waveform of GT's full
%   class_num population) as a reference line. Groups B/C/D/E additionally
%   get the ONE arm they're actually about -- B/C from their own plotted
%   arm, D/E from the arm named in "GT-not-<arm>" -- as a stored template
%   (from that arm's templates_wc_offline.mat) plus a "matched" template
%   (pooled from GT's + that arm's own waveform snippets at the spike times
%   they agree on). Group A (GT missed by BOTH sd1 and sd2) has no single
%   arm to pair with, so it only gets the GT-template reference -- no
%   arm/matched overlay.

    p = inputParser;
    addParameter(p, 'sd1_template_file', '');
    addParameter(p, 'sd2_template_file', '');
    parse(p, varargin{:});
    sd1_template_file = p.Results.sd1_template_file;
    sd2_template_file = p.Results.sd2_template_file;

    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    gt_full_t  = times_gt.cluster_class(times_gt.cluster_class(:,1)==class_num, 2);
    sd1_full_t = times_sd1.cluster_class(times_sd1.cluster_class(:,1)==arm_class_sd1, 2);
    sd2_full_t = times_sd2.cluster_class(times_sd2.cluster_class(:,1)==arm_class_sd2, 2);

    % ---- reference templates for overlay: GT template (mean of GT's full
    % class population), each arm's stored wave_clus template (from
    % templates_wc_offline, if a path was given), and a "matched" template
    % pooling GT's + that arm's own waveform snippets at the spike times
    % they agree on (membership_sets(gt_full_t, arm_full_t, ...) run fresh
    % here against GT specifically -- Group C's own arm-vs-arm
    % membership_sets call below is for a different purpose, selecting
    % which spikes to PLOT, not this reference) ----
    gt_template = mean(times_gt.spikes(times_gt.cluster_class(:,1)==class_num, :), 1);

    [sd1_matched_gt, sd1_matched_arm] = membership_sets(gt_full_t, sd1_full_t, tol);
    sd1_matched_template = mean([ ...
        get_waveforms_for_times(sd1_matched_gt, times_gt.cluster_class(:,2), times_gt.spikes, tol); ...
        get_waveforms_for_times(sd1_matched_arm, times_sd1.cluster_class(:,2), times_sd1.spikes, tol)], 1);
    sd1_stored_template = load_stored_template(sd1_template_file, ch, arm_class_sd1);

    [sd2_matched_gt, sd2_matched_arm] = membership_sets(gt_full_t, sd2_full_t, tol);
    sd2_matched_template = mean([ ...
        get_waveforms_for_times(sd2_matched_gt, times_gt.cluster_class(:,2), times_gt.spikes, tol); ...
        get_waveforms_for_times(sd2_matched_arm, times_sd2.cluster_class(:,2), times_sd2.spikes, tol)], 1);
    sd2_stored_template = load_stored_template(sd2_template_file, ch, arm_class_sd2);

    % Group A has no single arm to pair with (it's about BOTH), so it only
    % gets the GT-template reference. B/D pair with sd1, C/E pair with sd2
    % -- whichever arm each group's exclusion is actually defined against,
    % regardless of whose spikes.mat the group's own waveforms come from.
    overlays_A = build_template_overlays(gt_template, [], '', []);
    overlays_sd1_pair = build_template_overlays(gt_template, sd1_stored_template, 'sd1', sd1_matched_template);
    overlays_sd2_pair = build_template_overlays(gt_template, sd2_stored_template, 'sd2', sd2_matched_template);

    % ---- Group A: GT missed by BOTH SD1 and SD2 ----
    in_sd1 = nearest_match(gt_full_t, sd1_full_t, tol) > 0;
    in_sd2 = nearest_match(gt_full_t, sd2_full_t, tol) > 0;
    groupA = gt_full_t(~in_sd1 & ~in_sd2);

    % ---- Group B: SD1-not-GT (gained by rescue) ----
    [~,~,~,sd1_gained] = membership_sets(gt_full_t, sd1_full_t, tol);
    groupB = sd1_gained;

    % ---- Group C: SD2-not-SD1 (gained by the higher sdnum) ----
    [~,~,~,sd2_gained] = membership_sets(sd1_full_t, sd2_full_t, tol);
    groupC = sd2_gained;

    view_labels = {'in-response (stim list)', 'in-response (other stimuli)', 'outside-response (baseline)'};

    viewA1 = groupA(nearest_match(groupA, gt_inside_dyn, tol) > 0);
    viewA2 = groupA(nearest_match(groupA, gt_inside_other, tol) > 0);
    viewA3 = groupA(nearest_match(groupA, gt_outside_all, tol) > 0);
    plot_waveform_group({viewA1,viewA2,viewA3}, view_labels, times_gt.cluster_class(:,2), times_gt.spikes, tol, ...
        sprintf('ch%d -- Group A: GT missed by BOTH SD1 and SD2 (n=%d total)', ch, numel(groupA)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_groupA_gt_missed_by_both.png', ch)), overlays_A);

    viewB1 = groupB(nearest_match(groupB, sd1_inside_dyn, tol) > 0);
    viewB2 = groupB(nearest_match(groupB, sd1_inside_other, tol) > 0);
    viewB3 = groupB(nearest_match(groupB, sd1_outside_all, tol) > 0);
    plot_waveform_group({viewB1,viewB2,viewB3}, view_labels, times_sd1.cluster_class(:,2), times_sd1.spikes, tol, ...
        sprintf('ch%d -- Group B: SD1 - not - GT (n=%d total)', ch, numel(groupB)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_groupB_sd1_not_gt.png', ch)), overlays_sd1_pair);

    viewC1 = groupC(nearest_match(groupC, sd2_inside_dyn, tol) > 0);
    viewC2 = groupC(nearest_match(groupC, sd2_inside_other, tol) > 0);
    viewC3 = groupC(nearest_match(groupC, sd2_outside_all, tol) > 0);
    plot_waveform_group({viewC1,viewC2,viewC3}, view_labels, times_sd2.cluster_class(:,2), times_sd2.spikes, tol, ...
        sprintf('ch%d -- Group C: SD2 - not - SD1 (n=%d total)', ch, numel(groupC)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_groupC_sd2_not_sd1.png', ch)), overlays_sd2_pair);

    % ---- Group E: GT-not-SD2 (skipping SD2-not-GT and SD1-not-SD2 -- the
    % latter is trivially near-empty since SD1's raw population is a proven
    % subset of SD2's, so it wouldn't be a meaningful comparison; SD1's
    % "out of window but still counted" spikes still get matched here and
    % contribute to the waveform average like any other member) ----
    groupE = gt_full_t(~in_sd2);   % reuses in_sd2 from Group A above, no recompute needed

    viewE1 = groupE(nearest_match(groupE, gt_inside_dyn, tol) > 0);
    viewE2 = groupE(nearest_match(groupE, gt_inside_other, tol) > 0);
    viewE3 = groupE(nearest_match(groupE, gt_outside_all, tol) > 0);
    plot_waveform_group({viewE1,viewE2,viewE3}, view_labels, times_gt.cluster_class(:,2), times_gt.spikes, tol, ...
        sprintf('ch%d -- Group E: GT - not - SD2 (n=%d total)', ch, numel(groupE)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_groupE_gt_not_sd2.png', ch)), overlays_sd2_pair);

    fprintf('[waveforms] group sizes: A(gt missed by both)=%d  B(sd1-not-gt)=%d  C(sd2-not-sd1)=%d  E(gt-not-sd2)=%d\n', ...
            numel(groupA), numel(groupB), numel(groupC), numel(groupE));

    % ---- Group D: GT-not-SD1 (regardless of SD2) ----
    % Same 3-window-view single-trace format as A/B/C/E.
    groupD = gt_full_t(~in_sd1);   % GT not in SD1 -- same in_sd1 mask already computed above

    viewD1 = groupD(nearest_match(groupD, gt_inside_dyn, tol) > 0);
    viewD2 = groupD(nearest_match(groupD, gt_inside_other, tol) > 0);
    viewD3 = groupD(nearest_match(groupD, gt_outside_all, tol) > 0);
    plot_waveform_group({viewD1,viewD2,viewD3}, view_labels, times_gt.cluster_class(:,2), times_gt.spikes, tol, ...
        sprintf('ch%d -- Group D: GT - not - SD1 (n=%d total)', ch, numel(groupD)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_groupD_gt_not_sd1.png', ch)), overlays_sd1_pair);

    fprintf('[waveforms] group sizes: D(gt-not-sd1)=%d\n', numel(groupD));
end

% ==========================================================================
function overlays = build_template_overlays(gt_template, arm_template, arm_name, matched_template)
% BUILD_TEMPLATE_OVERLAYS  Packages up to 3 reference waveforms (GT
% template, one arm's stored template, and the GT+that-arm matched
% template) into the struct array plot_waveform_group draws on every panel.
% Any input that's empty or all-NaN (e.g. no templates_wc_offline.mat given,
% or Group A's arm_template=[]/matched_template=[]) is silently skipped.
    overlays = struct('name',{},'wave',{},'color',{},'style',{});
    if ~isempty(gt_template) && ~all(isnan(gt_template))
        overlays(end+1) = struct('name','GT template','wave',gt_template,'color',[0.1 0.6 0.1],'style','--');
    end
    if ~isempty(arm_template) && ~all(isnan(arm_template))
        overlays(end+1) = struct('name',sprintf('%s template',arm_name),'wave',arm_template,'color',[0.85 0.55 0.1],'style','--');
    end
    if ~isempty(matched_template) && ~all(isnan(matched_template))
        overlays(end+1) = struct('name',sprintf('matched (gt+%s)',arm_name),'wave',matched_template,'color',[0.5 0.2 0.7],'style','-.');
    end
end

function plot_waveform_group(view_sets, view_labels, times_cc_col2, spikes_matrix, tol, group_title, out_png, overlays)
    f = figure('Visible','off','Position',[100 100 1300 420],'Color','w');
    tl = tiledlayout(f,1,3,'Padding','loose','TileSpacing','loose');
    title(tl, group_title, 'Interpreter','none');
    for vi = 1:3
        ax = nexttile(tl); hold(ax,'on');
        W = get_waveforms_for_times(view_sets{vi}, times_cc_col2, spikes_matrix, tol);
        if isempty(W)
            text(ax,0.5,0.5,'no spikes','HorizontalAlignment','center','Units','normalized');
            n_str = '0';
        else
            mu = mean(W,1); sd = std(W,0,1); xx = 1:numel(mu);
            fill(ax,[xx fliplr(xx)],[mu+sd fliplr(mu-sd)],[0.2 0.4 0.8],'FaceAlpha',0.25,'EdgeColor','none');
            plot(ax, xx, mu, 'Color',[0.1 0.2 0.6], 'LineWidth',1.6, 'DisplayName','mean (this group)');
            n_str = num2str(size(W,1));
        end
        for oi = 1:numel(overlays)
            ov = overlays(oi);
            plot(ax, 1:numel(ov.wave), ov.wave, ov.style, 'Color', ov.color, 'LineWidth', 1.3, 'DisplayName', ov.name);
        end
        title(ax, sprintf('%s (n=%s)', view_labels{vi}, n_str), 'Interpreter','none','FontSize',9);
        xlabel(ax,'sample'); ylabel(ax,'amplitude');
        grid(ax,'on');
        if vi == 1 && (~isempty(W) || ~isempty(overlays))
            legend(ax, 'show', 'Location','best', 'FontSize',7, 'Interpreter','none');
        end
    end
    save_fig_local(f, out_png);
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
