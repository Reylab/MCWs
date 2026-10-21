addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
PY = 'python_recluster_20261012_second_pass';
J = {'pt6_ch333', 1:5; 'pt4_ch259', 1:5; 'pt12_ch300_class1', [1 3 4]; 'pt12_ch300_class2', [1 3 4]; 'pt12_ch299', 1:8};
for i = 1:size(J, 1)
    cfg = validation_sessions(J{i, 1});
    tok = regexp(J{i, 1}, '_class\d+$', 'match', 'once');
    for w = J{i, 2}
        wc = sprintf('wave_clus%d', w);
        out = fullfile(cfg.root, wc, ['class_match' tok]);
        T = quick_class_match(J{i, 1}, wc, 'out_dir', out);                          % GT vs wave_clus
        quick_class_match(J{i, 1}, PY, 'ref', wc, 'gt_class', T.matched_class(1), 'out_dir', out);   % wave_clus vs python sd1
        fprintf('[cm] %s %s: GT class %d -> %s class %d\n', J{i, 1}, wc, cfg.gt.class, wc, T.matched_class(1));
    end
end
