D = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 10 emu 7 ch 368/validation_gt_vs_python_recluster_20261006_sd2';
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline');
st = load_validation_state(D); cfg = st.cfg;
V = readtable(fullfile(D, 'venn_three_set', 'venn3_counts.csv'));
v = V(strcmp(V.scope, 'cluster'), :);
S = load(fullfile(D, 'subsets', 'subsets_python_recluster_sd2.mat')); s = S.subsets;
T = load(fullfile(cfg.arms(1).dir, cfg.arms(1).times_file), 'cluster_class', 'quarantine');
n = @(x) numel(s.(x).rows);
fprintf('\n%-22s %8s %8s\n', 'check', 'subset', 'venn');
chk = {'gt (n_gt)', n('gt'), v.n_gt; 'arm (n_arm)', n('arm'), v.n_arm;
       'gt_only', n('gt_only'), v.gt_only; 'gt_lost (gt_q)', n('gt_lost'), v.gt_q;
       'gt_not_in_arm', n('gt_not_in_arm'), v.gt_only + v.gt_q;
       'matched (gt_arm+all3)', n('matched'), v.gt_arm + v.gt_arm_q;
       'matched_rescued (all3)', n('matched_rescued'), v.gt_arm_q;
       'arm_only', n('arm_only'), v.arm_only; 'arm_rescued_not_gt', n('arm_rescued_not_gt'), v.arm_q;
       'gt: only+lost+matched', n('gt_only')+n('gt_lost')+n('matched'), v.n_gt;
       'arm: matched+only+resc', n('matched')+n('arm_only')+n('arm_rescued_not_gt'), v.n_arm;
       'pie I+II-I+III-II+IV', n('pie_I')+n('pie_II_I')+n('pie_III_II')+n('pie_IV'), v.n_arm;
       'rescued (IV)', n('rescued'), v.gt_arm_q + v.arm_q;
       'q_not_rescued', n('q_not_rescued'), v.n_quarantined - nnz(T.quarantine.rescued)};
fn = fieldnames(s); qn = fn(startsWith(fn, 'qnr_'));
chk(end+1,:) = {'sum of qnr_<prop>', sum(cellfun(n, qn)), n('q_not_rescued')};
ok = true;
for i = 1:size(chk,1)
    flag = ''; if chk{i,2} ~= chk{i,3}, flag = '  <-- MISMATCH'; ok = false; end
    fprintf('%-22s %8d %8d%s\n', chk{i,:}, flag);
end
% independent spike-time checks (tol = cfg.tol), not using three_set_membership
near = @(a, b) any(abs(a(:) - b(:)') <= cfg.tol, 2);
gt_t = s.gt.times; arm_t = s.arm.times;
fprintf('\nspike-time checks (any spike within %.1f ms):\n', cfg.tol);
fprintf('  arm_only spikes near a GT spike:            %d of %d\n', nnz(near(s.arm_only.times, gt_t)), n('arm_only'));
fprintf('  arm_rescued_not_gt spikes near a GT spike:  %d of %d\n', nnz(near(s.arm_rescued_not_gt.times, gt_t)), n('arm_rescued_not_gt'));
fprintf('  gt_only spikes near an arm spike:           %d of %d\n', nnz(near(s.gt_only.times, arm_t)), n('gt_only'));
fprintf('  gt_lost spikes near an arm spike:           %d of %d\n', nnz(near(s.gt_lost.times, arm_t)), n('gt_lost'));
fprintf('  matched spikes with NO GT spike nearby:     %d of %d\n', nnz(~near(s.matched.times, gt_t)), n('matched'));
q_out = T.cluster_class(T.quarantine.all(:) & T.cluster_class(:,1) ~= st.arm_classes(1), 2);
fprintf('  gt_lost with NO non-arm quarantined nearby: %d of %d\n', nnz(~near(s.gt_lost.times, q_out)), n('gt_lost'));
fprintf('  q_not_rescued flagged rescued / not quarantined: %d / %d\n', ...
    nnz(T.quarantine.rescued(s.q_not_rescued.rows)), nnz(~T.quarantine.all(s.q_not_rescued.rows)));
fprintf('  arm subsets all in class %d: %d\n', st.arm_classes(1), all(T.cluster_class([s.matched.rows; s.arm_only.rows; s.arm_rescued_not_gt.rows],1) == st.arm_classes(1)));
fprintf('\nall counts match venn: %d\n', ok);
