addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files/clustering');
cfg = validation_cfg_pt10_emu7_ch368();
for k = 1:2
    T = load(fullfile(cfg.arms(k).dir, cfg.arms(k).times_file));
    A = audit_rescue(T, 6);
    fprintf('\n== %s (sdnum %g): candidates %d, rescued python %d / replay %d, label mismatches %d\n', ...
        cfg.arms(k).name, T.par.template_sdnum, A.n_cand, A.n_rescued_py, A.n_rescued_replay, A.n_mismatch);
    disp(A.how)
    e = A.cls_euclid_label;
    fprintf('class 6 rescued: %d | euclid->6 %d | euclid->other class then mahal->6 %d\n', numel(e), nnz(e==6), nnz(e~=6));
    fprintf('  dist to class-6 refined template (sd units): median %.2f, max %.2f\n', median(A.cls_dist_sd), max(A.cls_dist_sd));
    w = A.cls_fail_width; rf = A.cls_fail_refract;
    fprintf('  bad width: %d (by mahal %d) | refractory: %d (by mahal %d)\n', nnz(w), nnz(w & e~=6), nnz(rf), nnz(rf & e~=6));
    fprintf('  refractory rescues within %.1f ms of a kept class-6 spike: %d of %d\n', A.ref_ms, nnz(rf & A.cls_isi_ms < A.ref_ms), nnz(rf));
    fprintf('  all class-6 rescues within %.1f ms of a kept class-6 spike: %d of %d\n', A.ref_ms, nnz(A.cls_isi_ms < A.ref_ms), numel(e));
end
