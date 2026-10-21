addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline', '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/real_response_analysis');
R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 11 emu 17 ch 328';
G = load(fullfile(R, 'hand selected', 'times_mLFUS08 raw_328.mat'), 'cluster_class');
A = load(fullfile(R, 'python_recluster_20261008_second_pass', 'times_mLFUS08 raw_328.mat'), 'cluster_class', 'quarantine');
gl = G.cluster_class(:,1); gt = G.cluster_class(:,2); al = A.cluster_class(:,1); at = A.cluster_class(:,2);
q = logical(A.quarantine.all(:)); resc = logical(A.quarantine.rescued(:));
Sc = three_set_membership(gt(gl == 1), al, at, q, al == 2, 0.5);
Sf = three_set_membership(gt(gl > 0), al, at, q, al > 0, 0.5);
c_ao = Sc.rows.arm_only; f_ao = Sf.rows.arm_only;
fprintf('cluster arm only: %d | full arm only: %d\n', numel(c_ao), numel(f_ao));
fprintf('  of the cluster arm-only (class-2) spikes: still arm only in full %d, matched to another GT class in full %d, other %d\n', ...
    nnz(ismember(c_ao, f_ao)), nnz(ismember(c_ao, Sf.rows.gt_arm)), nnz(~ismember(c_ao, [f_ao; Sf.rows.gt_arm])));
fprintf('  full arm only = class-2 spikes %d + other arm classes %d\n', nnz(al(f_ao) == 2), nnz(al(f_ao) ~= 2));
% which GT classes the class-2 arm-only spikes match in the full venn
m = c_ao(ismember(c_ao, Sf.rows.gt_arm));
[~, j] = min(abs(gt(gl > 0)' - at(m)), [], 2); gg = gl(gl > 0); gc = gg(j);
fprintf('  cluster arm-only spikes that match GT in full, by GT class: %s\n', mat2str(arrayfun(@(c) nnz(gc == c), 1:max(gl))));
fprintf('  arm sets: cluster %d (class 2), full %d (class>0); GT sets: cluster %d (class 1), full %d (class>0)\n', nnz(al == 2), nnz(al > 0), nnz(gl == 1), nnz(gl > 0));
