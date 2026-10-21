addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files/clustering');
cfg = validation_cfg_pt10_emu7_ch368_norefract();
for k = 1:2
    T = load(fullfile(cfg.arms(k).dir, cfg.arms(k).times_file));
    A = audit_rescue(T, 6);
    [~, ~, info] = sd1_pass_templates(T, 6, 1, [1 99]);
    fprintf('%s: candidates %d, rescued python %d / replay %d, mismatches %d | rescue pass-1 core for class 6: %d\n', ...
        cfg.arms(k).name, A.n_cand, A.n_rescued_py, A.n_rescued_replay, A.n_mismatch, info.res_core);
end
