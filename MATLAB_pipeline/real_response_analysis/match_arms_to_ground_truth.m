function [arm_classes, T_all, DET_all, COMP_all, matches] = match_arms_to_ground_truth(ccg, gt_abs_times, arm_names, arm_cc_list, ch, tol, min_recall)
% MATCH_ARMS_TO_GROUND_TRUTH  Matches each arm's cluster_class to the GT
% class encoded in ccg, for an arbitrary number of arms. Generalizes the
% class-matching loop that used to be hand-written per arm (with hardcoded
% rescue_sd1/rescue_sd2 branches) in test_rescue_grape_metrics.m so a
% different dataset/arm-count doesn't require copy-pasting the loop body.
%
% VARIABLES
%   ccg          - input: GT spikes as [class time_ms], one row per GT spike (match_cc's reference)
%   gt_abs_times - input: GT spike times (for building each arm's membership-set venn inputs)
%   arm_names    - input: cellstr, one name per arm
%   arm_cc_list  - input: cell array, one arm's full cluster_class matrix per arm, same order as arm_names
%   ch           - input: channel number
%   tol          - input: spike-time match tolerance (ms)
%   min_recall   - input: match_cc's threshold for calling a class "matched"
%   arm_classes  - output: resolved class number in each arm, one per arm_names entry
%   T_all/DET_all/COMP_all - output: match_cc results, accumulated across arms, with an .arm column
%   matches      - output: struct array (one per arm) with arm_t/matched_gt/gt_only/arm_only,
%                  ready to hand to plot_membership_set
    arm_classes = nan(numel(arm_names),1);
    T_all = table(); DET_all = table(); COMP_all = table();
    matches = struct('arm_t',{},'matched_gt',{},'gt_only',{},'arm_only',{});

    for arm = 1:numel(arm_names)
        arm_c = arm_names{arm};
        cca = arm_cc_list{arm};
        [T, DET, COMP] = match_cc(ccg, cca, ch, tol, min_recall);
        T.arm = repmat(string(arm_c), height(T), 1);
        DET.arm = repmat(string(arm_c), height(DET), 1);
        COMP.arm = repmat(string(arm_c), height(COMP), 1);
        T_all = [T_all; T];
        DET_all = [DET_all; DET];
        COMP_all = [COMP_all; COMP];

        arm_class = T.arm_class(1);
        arm_classes(arm) = arm_class;
        fprintf('[%s] matched -> arm class%d (recall=%.2f, precision=%.2f, status=%s)\n', ...
                arm_c, arm_class, T.recall(1), T.precision(1), T.status(1));

        arm_t = cca(cca(:,1)==arm_class, 2);
        [matched_gt, ~, gt_only, arm_only] = membership_sets(gt_abs_times, arm_t, tol);
        matches(arm).arm_t = arm_t;
        matches(arm).matched_gt = matched_gt;
        matches(arm).gt_only = gt_only;
        matches(arm).arm_only = arm_only;
    end
end
