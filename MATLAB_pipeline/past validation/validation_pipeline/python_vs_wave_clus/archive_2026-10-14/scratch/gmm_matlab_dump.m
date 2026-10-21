function gmm_matlab_dump(chdir_)
% Run MATLAB's GMM feature selection (gmm_extract.m, unchanged) on clean_spikes.mat
% in chdir_, and dump the fits + every intermediate (same steps as gmm_extract)
% to matlab_gmm.mat (-v7) for the python comparison.
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/batch_files/feature extraction');
cd(chdir_);
S = load('clean_spikes.mat'); spikes = S.spikes;
par.scales = 4; par.features = 'gmm'; par.filename = 'chan_spikes.mat';
rng('shuffle');
tic; [inspk, coeffs] = gmm_extract(spikes, par); t_extract = toc;      % caches gmm/chan_gmm.mat
L = load(fullfile('gmm', 'chan_gmm.mat'));                              % g, xg, pg, wd_coeffs, summary_table
g = L.g; m = numel(g);
mu = zeros(16, m); sd = zeros(16, m); a = zeros(16, m); kj = nan(16, m); md = nan(16, m); Mc = zeros(16, m);
for i = 1:m
    mu(:, i) = g{i}.mu(:); sd(:, i) = sqrt(squeeze(g{i}.Sigma(:))); a(:, i) = g{i}.ComponentProportion(:);
    kj(:, i) = L.summary_table.kj{i}(:); md(:, i) = L.summary_table.("med idist"){i}(:);
    mc = L.summary_table.M_comp{i}; Mc(1:numel(mc), i) = mc(:);
end
% intermediates, exactly as gmm_extract computes them
g_init = g; kj_mat = cell(1, m); medDist_cell = cell(1, m); M_comp = cell(1, m); polyID = cell(1, m);
for i = 1:m
    keep = a(:, i) > 0.005; pid = (1:16)'; polyID{i} = pid(keep);
    medDist_cell{i} = L.summary_table.("med idist"){i}(keep); kj_mat{i} = L.summary_table.kj{i}(keep);
    mc = L.summary_table.M_comp{i}; M_comp{i} = mc(ismember(mc, polyID{i}));
end
[k_sort, k_sortIdx, md_sort, md_sortIdx] = deal(cell(1, m));
for r = 1:m
    msk = ~isnan(kj_mat{r}); v = kj_mat{r}(msk); ids = polyID{r}(msk); [v, o] = sort(v, 'descend'); ids = ids(o);
    keep = ~ismember(ids, M_comp{r}); k_sort{r} = v(keep); k_sortIdx{r} = ids(keep);
    msk = ~isnan(medDist_cell{r}); v = medDist_cell{r}(msk); ids = polyID{r}(msk); [v, o] = sort(v, 'descend'); ids = ids(o);
    keep = ~ismember(ids, M_comp{r}); md_sort{r} = v(keep); md_sortIdx{r} = ids(keep);
end
[k_select, ~, kDist_vec] = processKvKnee(k_sort, k_sortIdx);
[md_select, ~, md_vec] = processMedDistKnee(md_sort, md_sortIdx);
[~, ~, ~, ~, sel10, select_all] = lineExclusion(md_vec, md_select, k_select, kDist_vec, 'chan');
sm = spikeMatch(sel10, select_all.trim10_0pct.threshQ2, select_all.trim10_0pct.threshQ4, g_init, L.wd_coeffs, 'chan', 5, 0.9);
coeffs_replay = unique(sm(:, 3));
wd = L.wd_coeffs;
save('matlab_gmm.mat', 'mu', 'sd', 'a', 'kj', 'md', 'Mc', 'wd', 'coeffs', 'coeffs_replay', 'k_select', 'md_select', ...
     'kDist_vec', 'md_vec', 'sel10', 'sm', 't_extract', '-v7');
fprintf('%s: gmm_extract coeffs [%s] (%d), replay [%s], %.0f s\n', chdir_, num2str(coeffs(:)'), numel(coeffs), num2str(coeffs_replay(:)'), t_extract);
end
