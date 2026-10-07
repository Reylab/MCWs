function arm_classes = resolve_arm_classes(cfg, times_gt, times_arms)
% RESOLVE_ARM_CLASSES  Class in each arm that matches cfg.gt.class (by spike
% time, match_arms_to_ground_truth), unless cfg.arms(k).class forces one.
% Shared by the validation stages so they can run on their own.
    gt_cc = times_gt.cluster_class;
    gt_abs_times = sort(gt_cc(gt_cc(:,1) == cfg.gt.class, 2));
    ccg = [repmat(cfg.gt.class, numel(gt_abs_times), 1), gt_abs_times];
    arm_classes = match_arms_to_ground_truth(ccg, gt_abs_times, {cfg.arms.name}, ...
        cellfun(@(t) t.cluster_class, times_arms, 'UniformOutput', false), cfg.ch, cfg.tol, cfg.min_recall);
    for k = 1:numel(cfg.arms)
        if ~isempty(cfg.arms(k).class), arm_classes(k) = cfg.arms(k).class; end
    end
end
