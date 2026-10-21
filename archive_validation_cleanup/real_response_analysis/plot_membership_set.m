function plot_membership_set(gtc,trial_name,arm_class,gt_t,arm_t,matched_gt,gt_only,arm_only,out_dir,ch,left_name)
% PLOT_MEMBERSHIP_SET  Two-circle venn of GT-vs-arm spike-time membership.
% Standalone copy of the local function already used in
% test_rescue_grape_metrics.m -- pulled out (unchanged) so run scripts other
% than that one script can call it too. Actual drawing lives in
% draw_membership_venn.m, shared with plot_venn_waveform_group.m's embedded
% copy of the same venn.
    % VARIABLES
    %   gtc         - left-side class number (for labeling the plot)
    %   trial_name  - right-side arm name (for labeling)
    %   arm_class   - right-side matched arm class number (for labeling)
    %   gt_t/arm_t  - full left-side / right-side spike time lists (drive circle sizes)
    %   matched_gt  - paired spikes (drives the "matched" count in the middle)
    %   gt_only/arm_only - unpaired spikes on each side
    %   out_dir     - output folder
    %   ch          - channel number (for labeling/filename)
    %   left_name   - OPTIONAL. Label for the LEFT side (default 'GT'). Pass something
    %                 else (e.g. 'sd1') when the comparison isn't GT-vs-arm at all --
    %                 e.g. an sd1-vs-sd2 sanity check -- so the plot doesn't say "gt"
    %                 when GT isn't part of the comparison.
    %   n_gt/n_arm/n_m - total left-side spikes, total right-side spikes, matched-pair count
    %   tag         - filename fragment (channel/left-class/arm-name)
    if nargin < 11 || isempty(left_name), left_name = 'GT'; end
    n_gt = numel(gt_t); n_arm = numel(arm_t); n_m = numel(matched_gt);
    n_gt_only = numel(gt_only); n_arm_only = numel(arm_only);
    tag = sprintf('ch%d_%s%d_%s', ch, left_name, gtc, trial_name);
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    f = figure('Visible','off','Position',[100 100 520 460],'Color','w');
    ax = axes(f);
    draw_membership_venn(ax, gtc, trial_name, arm_class, n_gt, n_arm, n_m, n_gt_only, n_arm_only, left_name, ch);
    save_fig_local(f, fullfile(out_dir, ['venn_' tag '.png']));
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
