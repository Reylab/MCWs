addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files/clustering');
T = load('/mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07/times_mRFUS07 raw_368.mat');
fprintf('saved: template_sdnum=%g rescue_first_pass_sdnum=%g\n', T.par.template_sdnum, T.par.rescue_first_pass_sdnum);
for sd = [1 2 3]
    T.par.template_sdnum = sd;
    A = audit_rescue(T, 5);
    fprintf('replay sd1 -> sd%d: rescued %d (python %d), label mismatches %d of %d candidates\n', ...
        sd, A.n_rescued_replay, A.n_rescued_py, A.n_mismatch, A.n_cand);
end
