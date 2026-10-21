addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
cfgs = {validation_cfg_pt10_emu7_ch368_norefract(), validation_cfg_pt11_emu17_ch328_norefract()};
for ci = 1:2
    cfg = cfgs{ci};
    g = load(fullfile(cfg.gt.dir, cfg.gt.times_file), 'cluster_class');
    gt_t = g.cluster_class(g.cluster_class(:,1) == cfg.gt.class, 2);
    for k = 1:2
        T = load(fullfile(cfg.arms(k).dir, cfg.arms(k).times_file), 'cluster_class', 'quarantine');
        cls = arm_cls(cfg.ch);
        in_c = T.cluster_class(:,1) == cls;
        S = three_set_membership(gt_t, T.cluster_class(:,1), T.cluster_class(:,2), T.quarantine.all, in_c, cfg.tol);
        [mg, ~, go, ao] = membership_sets(sort(gt_t), sort(T.cluster_class(in_c,2)), cfg.tol);
        c = S.counts;
        fprintf('ch%d %s: matched %d vs 2-way %d | GT not in arm %d vs %d | arm not in GT %d vs %d\n', cfg.ch, cfg.arms(k).name, ...
            c.gt_arm + c.gt_arm_q, numel(mg), c.gt_only + c.gt_q, numel(go), c.arm_only + c.arm_q, numel(ao));
    end
end
function c = arm_cls(ch)
    if ch == 368, c = 6; else, c = 1; end
end
