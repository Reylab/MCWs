P = '/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline';
addpath(fullfile(P, 'batch_files', 'clustering'), fullfile(P, 'real_response_analysis'));
D = '/mnt/data0/MCWs Sessions/matlab validation/analysis trials old/new trial pt 9 emu 3 ch 331/RECLUSTER_VALIDATION_20260910_122564/handselect';
R = '/mnt/data0/MCWs Sessions/matlab validation/single channel/pt 9 emu 3 ch 331 check';
T = load(fullfile(D, 'times_mLFUS02 raw_331.mat'));
S = load(fullfile(D, 'mLFUS02 raw_331_spikes.mat'), 'index', 'spikes');
idx = S.index(:); W = double(S.spikes);
% rows of spikes_all already in the rebuilt clusters
[tf, loc] = ismembertol(T.cluster_class(:,2), idx, 0.5, 'DataScale', 1);
fprintf('rebuilt spikes found in spikes_all: %d of %d\n', nnz(tf), numel(tf));
lab = zeros(numel(idx), 1); lab(loc(tf)) = T.cluster_class(tf, 1);
target = [6797 1756 5709 3538];
fprintf('rebuilt class sizes: %s   (original grapes: %s)\n', mat2str(arrayfun(@(c) nnz(lab==c), 1:4)), mat2str(target));
cand = lab == 0;
% trial windows (-1000..2000 ms around every onset): where the raster rebuild could see spikes
S2 = load(fullfile(R, 'stimulus.mat')); F = load(fullfile(R, 'finalevents.mat')); E = load(fullfile(R, 'experiment_properties_online3.mat'));
N = min(numel(S2.stimulus), numel(E.scr_end_cell));
allstim = unique(cell2mat(cellfun(@(c) c.pics2use(:)', E.scr_config_cell(1:N), 'UniformOutput', false)));
om = build_all_stim_onset_map(S2, F, N, allstim, E);
on = sort(cell2mat(om.values));
in_trial = false(size(idx));
for k = 1:numel(on), in_trial = in_trial | (idx >= on(k) - 1000 & idx <= on(k) + 2000); end
fprintf('spikes_all in a trial window: %d of %d; unassigned in a window: %d, outside: %d\n', ...
    nnz(in_trial), numel(idx), nnz(cand & in_trial), nnz(cand & ~in_trial));
fprintf('rebuilt (assigned) spikes outside every window: %d\n', nnz(~cand & ~in_trial));
for v = 1:2
    par = struct('template_type', 'center', 'template_sdnum', T.par.template_sdnum, ...
                 'template_first_pass_sdnum', 1, 'template_amp_pct_range', [1 99], 'template_mahal_reassign', false);
    name = 'two-pass sd1->sd3, amp gate [1 99]';
    if v == 2
        par.template_first_pass_sdnum = T.par.template_sdnum; par.template_amp_pct_range = [];
        name = 'single pass sd3, no amp gate';
    end
    new = force_membership_wc(W(~cand, :), lab(~cand)', W(cand, :), par);
    full = lab; full(cand) = new(:);
    n_all = arrayfun(@(c) nnz(full == c), 1:4);
    n_out = arrayfun(@(c) nnz(full == c & cand & ~in_trial), 1:4);
    n_in = arrayfun(@(c) nnz(full == c & cand & in_trial), 1:4);
    fprintf('\n%s:\n  recovered class sizes %s  vs original %s  (diff %s)\n  newly assigned outside windows %s, inside windows %s, still 0: %d\n', ...
        name, mat2str(n_all), mat2str(target), mat2str(n_all - target), mat2str(n_out), mat2str(n_in), nnz(full == 0));
end
