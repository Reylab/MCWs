function ppt_compare_classes(session, multi, py_k)
% PPT_COMPARE_CLASSES  The for ppt comparisons with a CHOSEN python class instead of the matched one ->
% <for ppt>/alt_python_c<k>/ (nothing else in the report changes):
%   gt_vs_python/          GT class g vs python c<k>: venn + class matches + full raster pages + per-stimulus
%                          summary (run_validation, stages class_match / venn / rasters)
%   wave_clus_vs_python/   wave_clus notch class holding most of python c<k> vs python c<k>: the same
%   <comparison>/regions/  every venn region: waveforms (templates on top), rasters in the windows of the sort
%                          the spikes come from (the comparison run's reference / arm table, + default-window
%                          version), by_destination/ (where the spikes are in the other sort), csv
%   <comparison>/regions/02_*_x_quarantine|06_quarantine_only/not_rescued_by_mask/  never-rescued spikes by mask
%   accounting.csv         regions = venn numbers, regions add up to each class, in-window sums = raster pages
%   ppt_compare_classes('pt6emu13_ch264_class3', false, 1)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'), fullfile(fileparts(here), 'responses'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root; ch = cfg.ch; tf = cfg.gt.times_file;
pp = ppt_paths(cfg, multi); rep = pp.rep;   % report folder (run tag: ppt_paths)
out = fullfile(rep, sprintf('alt_python_c%d', py_k)); if ~exist(out, 'dir'), mkdir(out); end
M = load(fullfile(rep, sprintf('membership_ch%d.mat', ch)));
py = pp.py;
wcd = fullfile(root, 'wave_clus notch');

% ---- the two full comparison runs with python class k ----
S_gp = fullfile(out, 'gt_vs_python');
c = cfg; c.out_dir = S_gp; c.stages = {'class_match', 'venn', 'rasters'}; c.venn_detections = true;
c.arms = struct('name', 'python_sd1', 'dir', fullfile(root, py), 'times_file', tf, 'grapes_file', 'grapes_blanks_rescue.mat', 'class', py_k);
run_validation(c);
S_wp = fullfile(out, 'wave_clus_vs_python');
python_cluster_validation(session, 'wave_clus notch', py_k, 'python', py, 'stages', {'class_match', 'venn', 'rasters'}, ...
    'out_dir', S_wp, 'tol', cfg.tol);
st = load_validation_state(S_wp); wc_k = st.cfg.gt.class;            % the wave_clus class holding most of python c<k>
fprintf('[ppt_compare_classes] GT c%d | python c%d | wave_clus notch c%d\n', g, py_k, wc_k);

tm = struct('name', {sprintf('GT c%d', g), sprintf('wave_clus notch c%d', wc_k), sprintf('python c%d', py_k)}, ...
    'wave', {M.templates.gt(g, :), M.templates.wc_notch(wc_k, :), M.templates.python(py_k, :)}, ...
    'color', {[0 0 0], [0 .45 .7], [.9 .6 0]}, 'style', {'-', '--', ':'});
C = {'GT vs python', S_gp, 'gt', g, 'GT', [0 0 0]; 'wave_clus notch vs python', S_wp, 'wc_notch', wc_k, 'wave_clus notch', [0 .45 .7]};
acc = cell(0, 7);
for i = 1:size(C, 1)
    [cmp, sd, rn, rc, rlab, rcol] = C{i, :};
    st = load_validation_state(sd); Tr = st.ref_tables{1}; Ta = st.ref_tables{2};
    two = isfield(st, 'ref_tables_default') && ~isempty(st.ref_tables_default);
    inr = @(t) count_in_windows(t, Tr.stim_number, Tr.win_lo, Tr.win_hi, st.onset_map);
    ina = @(t) count_in_windows(t, Ta.stim_number, Ta.win_lo, Ta.win_hi, st.onset_map);
    R = M.(rn); P = M.python; xc = R.python_class; xq = R.python_quarantined; rcx = P.([rn '_class']);
    in_r = R.class == rc; in_a = P.class == py_k; not_r = isnan(rcx) | rcx ~= rc;
    V = readtable(fullfile(sd, 'venn_three_set', 'venn3_counts.csv')); V = V(strcmp(V.scope, 'cluster'), :);
    % region | ref side? | mask | venn number | other sort's class (by destination) | matched class there
    reg = {sprintf('01_%s_only', rn), true, in_r & (isnan(xc) | (xc ~= py_k & ~xq)), V.gt_only; ...
           sprintf('02_%s_x_quarantine', rn), true, in_r & xq & xc ~= py_k, V.gt_q; ...
           '03_matched', true, in_r & xc == py_k, V.gt_arm + V.gt_arm_q; ...
           '04_python_only', false, in_a & ~P.quarantined & not_r, V.arm_only; ...
           '05_rescued_only', false, in_a & P.quarantined & not_r, V.arm_q; ...
           '06_quarantine_only', false, P.quarantined & ~in_a & not_r, V.q_only};
    rd = fullfile(sd, 'regions'); if ~exist(rd, 'dir'), mkdir(rd); end
    grp = struct('name', {}, 'W', {}, 'color', {});
    for j = 1:size(reg, 1)
        [nm, rside, m, vn] = reg{j, :};
        if rside, t = R.t(m); Wv = M.waves.(rn)(m, :); n_in = inr(t); side = 'gt'; oc = xc(m); om = py_k; onm = 'python'; cc = rcol;
        else, t = P.t(m); Wv = M.waves.python(m, :); n_in = ina(t); side = 'arm'; oc = rcx(m); om = rc; onm = rlab; cc = [.9 .6 0]; end
        acc(end + 1, :) = {cmp, nm(4:end), nnz(m), vn, n_in, '', ternary(nnz(m) == vn, 'PASS', 'FAIL')}; %#ok<AGROW>
        grp(end + 1) = struct('name', strrep(nm(4:end), '_', ' '), 'W', Wv, 'color', cc); %#ok<AGROW>
        if isempty(t), continue; end
        d1 = fullfile(rd, nm); if ~exist(d1, 'dir'), mkdir(d1); end
        if rside, writetable(R(m, :), fullfile(d1, 'spikes.csv')); else, writetable(P(m, :), fullfile(d1, 'spikes.csv')); end
        plot_subset_rasters([], 'dir', sd, 'times', t, 'name', nm(4:end), 'side', side, 'out_dir', d1);
        if two, plot_subset_rasters([], 'dir', sd, 'times', t, 'name', [nm(4:end) '_default_window'], 'side', side, 'out_dir', d1, 'window', 'default'); end
        if ~strcmp(nm, '03_matched')
            ppt_by_destination(t, Wv, oc, om, onm, tm, fullfile(d1, 'by_destination'), sd, two, side, cc, ...
                sprintf('ch%d: %s (%s c%d vs python c%d), by where they are in %s', ch, strrep(nm(4:end), '_', ' '), rlab, rc, py_k, onm));
        end
    end
    plot_subset_waveforms(grp, tm, sprintf('ch%d: %s c%d vs python c%d -- the venn regions', ch, rlab, rc, py_k), fullfile(rd, 'waveforms_regions.png'));
    % class sums: regions of each class = the class; in-window sums = the full raster pages
    n = cell2mat(acc(end - 5:end, 3)); nin = cell2mat(acc(end - 5:end, 5)); page = st.n_in_window;
    acc(end + 1, :) = {cmp, sprintf('%s class c%d: regions sum', rn, rc), sum(n(1:3)), nnz(in_r), sum(nin(1:3)), ...
        sprintf('raster page in-window %d', page(1)), ternary(sum(n(1:3)) == nnz(in_r) && sum(nin(1:3)) == page(1), 'PASS', 'FAIL')}; %#ok<AGROW>
    na = n(4) + n(5) + nnz(in_a & ~not_r); nina = nin(4) + nin(5) + ina(P.t(in_a & ~not_r));
    acc(end + 1, :) = {cmp, sprintf('python class c%d: regions sum', py_k), na, nnz(in_a), nina, ...
        sprintf('raster page in-window %d', page(2)), ternary(na == nnz(in_a) && nina == page(2), 'PASS', 'FAIL')}; %#ok<AGROW>
end
ppt_compare_classes_masks(session, multi, py_k);    % never-rescued spikes by mask (02 / 06 regions)
A = cell2table(acc, 'VariableNames', {'comparison', 'region', 'n_spikes', 'expected', 'n_in_window', 'page', 'check'});
writetable(A, fullfile(out, 'accounting.csv'));
fprintf('[ppt_compare_classes] %s python c%d: %s (%d FAIL) -> %s\n', session, py_k, ternary(all(A.check == "PASS" | strcmp(A.check, 'PASS')), 'PASS', 'FAIL'), ...
    nnz(strcmp(A.check, 'FAIL')), out);
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
