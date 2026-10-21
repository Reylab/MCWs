addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline', ...
        '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files/clustering', ...
        '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
P = load('/mnt/data0/python_p10EMU07CH368/times_2026-10-06_13-46-07/times_mRFUS07 raw_368.mat');
fprintf('fields: %s\n', strjoin(fieldnames(P)', ' '));
if isfield(P,'par'), fprintf('par: %s\n', strjoin(fieldnames(P.par)', ' ')); end
if isfield(P,'quarantine'), fprintf('quarantine: %s\n', strjoin(fieldnames(P.quarantine)', ' ')); end
cfg = validation_cfg_pt10_emu7_ch368_recluster_20261006();
T = load(fullfile(cfg.arms(1).dir, cfg.arms(1).times_file));
lab = T.cluster_class(:,1); resc = logical(T.quarantine.rescued(:)); q = logical(T.quarantine.all(:));
forced = logical(T.forced(:));
pre = lab .* ~resc;
fprintf('\nper class: final n | rescued | forced | not forced/not rescued (SPC)\n');
for c = 0:max(lab)
    fprintf('  class %d: %6d | %5d | %5d | %6d\n', c, nnz(lab==c), nnz(lab==c & resc), nnz(lab==c & forced), nnz(lab==c & ~forced & ~resc));
end
fprintf('quarantined %d, of them label>0 before rescue (should be 0): %d\n', nnz(q), nnz(q & pre > 0));
p = T.par;
fprintf('rescue par: template_sdnum %g first_pass_sdnum %g amp %s mahal %d\n', p.template_sdnum, p.first_pass_sdnum, mat2str(p.amp_pct_range), p.mahal_reassign);
% replay
valid = pre > 0; cand = q & pre == 0;
W = double(T.spikes); F = double(T.features);
par.template_type = 'center'; par.template_sdnum = p.template_sdnum;
par.template_first_pass_sdnum = p.first_pass_sdnum; par.template_amp_pct_range = double(p.amp_pct_range(:)');
par.template_mahal_reassign = logical(p.mahal_reassign);
rep = force_membership_wc(W(valid,:), pre(valid)', W(cand,:), par, F(valid,:), F(cand,:));
py = lab(cand)';
K = max([rep py]);
C = zeros(K+1); for i = 1:numel(py), C(py(i)+1, rep(i)+1) = C(py(i)+1, rep(i)+1) + 1; end
fprintf('\nrows = python class of candidates, cols = replay class (0..%d)\n', K);
disp(array2table(C, 'RowNames', compose('py%d', 0:K), 'VariableNames', compose('rep%d', 0:K)));
