function [matched_gt, matched_arm,gt_only,arm_only] = membership_sets(gt_t,arm_t,tol)
% MEMBERSHIP_SETS  Standalone copy of the local function already used
% throughout test_rescue_grape_metrics.m -- pulled out for
% plot_stim_comparison_page.m's gained/lost table (deduplicated one-to-one
% matching, unlike classify_three_way.m's per-arm raster classification).
    % VARIABLES
    %   gt_t        - input: GT spike times
    %   arm_t       - input: arm spike times
    %   tol         - input: match tolerance (ms)
    %   gm          - raw greedy nearest-match (index into arm_t per GT spike, 0=none)
    %   gm_u        - gm after enforce_one_to_one (each arm spike claimed once)
    %   gt_order    - GT spike index order returned by enforce_one_to_one
    %   used        - boolean per arm spike: was it claimed by some GT spike?
    %   matched_gt/matched_arm - output: paired-up GT/arm spike times
    %   gt_only/arm_only       - output: unpaired spikes on each side
    gm =  nearest_match(gt_t,arm_t,tol);
    [gm_u,gt_order] = enforce_one_to_one(gt_t,arm_t,gm,tol);
    matched_gt = gt_t(gt_order(gm_u>0));
    matched_arm = arm_t(gm_u(gm_u>0));
    gt_only = gt_t(gt_order(gm_u==0));
    used = false(numel(arm_t),1);
    used(gm_u(gm_u>0)) = true;
    arm_only = arm_t(~used);
end
