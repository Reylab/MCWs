function ok = ppt_check_unit(session, multi)
% PPT_CHECK_UNIT  Bookkeeping check of one for ppt report. For each comparison (wave_clus notch vs python,
% GT vs python, GT vs wave_clus notch) the venn regions are rebuilt from the membership file and checked:
%   1. each region's spike count = the venn's number for it
%   2. the regions of each class add up to the whole class (every spike in exactly one region)
%   3. the regions' in-window counts (each sort's spikes in that sort's own full-run windows -- the windows the
%      subset raster pages use) add up to the in-window count on that sort's full-run raster page
% Writes <out>/accounting.csv and prints PASS / FAIL lines. OK = all checks pass.
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; ch = cfg.ch; root = cfg.root;
out = fullfile(root, 'for ppt'); if multi, out = fullfile(out, sprintf('GT c%d', g)); end
M = load(fullfile(out, sprintf('membership_ch%d.mat', ch)));
wc = fullfile(root, 'wave_clus notch');
d = dir(fullfile(root, 'python_recluster_*_second_pass')); py = d(end).name;
py_c = M.classes.python; wc_c = M.classes.wc_notch;
A = readtable(fullfile(wc, sprintf('venn_gt_vs_wave_clus_class%d', g), 'arm_classes.csv')); wg = A.class(1);
B = readtable(fullfile(root, sprintf('venn_gt_vs_python_class%d', g), 'arm_classes.csv')); pg = B.class(1);
C = {'wave_clus notch vs python', 'wc_notch', wc_c, 'python', py_c, true, fullfile(wc, sprintf('validation_wave_clus_vs_python_c%d', py_c)), ...
        fullfile(wc, sprintf('venn_wave_clus_vs_python_c%d', py_c)); ...
     'GT vs python', 'gt', g, 'python', pg, true, fullfile(root, sprintf('validation_gt_vs_%s_class%d', py, g)), ...
        fullfile(root, sprintf('venn_gt_vs_python_class%d', g)); ...
     'GT vs wave_clus notch', 'gt', g, 'wc_notch', wg, false, fullfile(wc, sprintf('validation_gt_vs_wave_clus_notch_class%d', g)), ...
        fullfile(wc, sprintf('venn_gt_vs_wave_clus_class%d', g))};
rows = cell(0, 7); ok = true;
for i = 1:size(C, 1)
    [nmc, rn, rc, an, ac, hasq, sd, vd] = C{i, :};
    st = load_validation_state(sd); Tr = st.ref_tables{1}; Ta = st.ref_tables{2};
    inr = @(t) count_in_windows(t, Tr.stim_number, Tr.win_lo, Tr.win_hi, st.onset_map);
    ina = @(t) count_in_windows(t, Ta.stim_number, Ta.win_lo, Ta.win_hi, st.onset_map);
    R = M.(rn); X = M.(an);
    xc = R.([an '_class']); rc_of_x = X.([rn '_class']);
    if hasq, xq_r = R.([an '_quarantined']); xq = X.quarantined; else, xq_r = false(height(R), 1); xq = false(height(X), 1); end
    in_r = R.class == rc; in_a = X.class == ac; not_r = isnan(rc_of_x) | rc_of_x ~= rc;
    V = readtable(fullfile(vd, 'venn_three_set', 'venn3_counts.csv')); V = V(strcmp(V.scope, 'cluster'), :);
    reg = {'ref only', R.t(in_r & (isnan(xc) | (xc ~= ac & ~xq_r))), V.gt_only, 'ref'; ...
           'ref x quarantine', R.t(in_r & xq_r & xc ~= ac), V.gt_q, 'ref'; ...
           'matched (ref spikes)', R.t(in_r & xc == ac), V.gt_arm + V.gt_arm_q, 'ref'; ...
           'arm only', X.t(in_a & ~xq & not_r), V.arm_only, 'arm'; ...
           'rescued only', X.t(in_a & xq & not_r), V.arm_q, 'arm'; ...
           'matched (arm spikes)', X.t(in_a & ~not_r), V.gt_arm + V.gt_arm_q, 'arm'; ...
           'quarantine only', X.t(xq & ~in_a & not_r), V.q_only, 'arm'};
    for k = 1:size(reg, 1)
        if strcmp(reg{k, 4}, 'ref'), n_in = inr(reg{k, 2}); else, n_in = ina(reg{k, 2}); end
        pass = numel(reg{k, 2}) == reg{k, 3};
        rows(end + 1, :) = {nmc, reg{k, 1}, numel(reg{k, 2}), reg{k, 3}, n_in, '', ternary(pass, 'PASS', 'FAIL')}; %#ok<AGROW>
        ok = ok && pass;
    end
    % class totals and in-window totals (in-window total = the number on that sort's full-run raster page)
    ref_tot = sum([rows{end - 6:end - 4, 3}]); arm_tot = sum([rows{end - 3:end - 1, 3}]);
    ref_in = sum([rows{end - 6:end - 4, 5}]); arm_in = sum([rows{end - 3:end - 1, 5}]);
    page = st.n_in_window;
    chk = {sprintf('%s class c%d: regions sum', rn, rc), ref_tot, nnz(in_r), ref_in, page(1); ...
           sprintf('%s class c%d: regions sum', an, ac), arm_tot, nnz(in_a), arm_in, page(2)};
    for k = 1:2
        pass = chk{k, 2} == chk{k, 3} && chk{k, 4} == chk{k, 5};
        rows(end + 1, :) = {nmc, chk{k, 1}, chk{k, 2}, chk{k, 3}, chk{k, 4}, sprintf('raster page in-window %d', chk{k, 5}), ...
            ternary(pass, 'PASS', 'FAIL')}; %#ok<AGROW>
        ok = ok && pass;
    end
end
T = cell2table(rows, 'VariableNames', {'comparison', 'region', 'n_spikes', 'expected', 'n_in_window', 'page', 'check'});
writetable(T, fullfile(out, 'accounting.csv'));
fprintf('[ppt_check] %s: %s (%d checks, %d FAIL)\n', session, ternary(ok, 'PASS', 'FAIL'), height(T), nnz(strcmp(T.check, 'FAIL')));
if ~ok, disp(T(strcmp(T.check, 'FAIL'), :)); end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
