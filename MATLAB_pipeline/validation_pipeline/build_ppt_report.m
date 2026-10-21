function M = build_ppt_report(o)
% BUILD_PPT_REPORT  wave_clus vs MCWs (python) vs GT comparison figures for one channel, one GT unit,
% into <root>/for ppt/ (or o.out). Only the wave_clus runs in o.runs are compared (the notch run);
% the no-notch run is only used for the notch vs no notch detection comparison (07).
%
%   o.session  = 'pt4_ch259';                               % validation_sessions id (root, GT class, stim list)
%   o.runs     = struct('label', 'notch', 'folder', 'wave_clus notch');
%   o.no_notch = 'wave_clus no notch';                      % for 07 (optional)
%   M = build_ppt_report(o);
%
% Optional: o.python (default: newest python_recluster_*_second_pass in the channel), o.state (raster
% state folder with GT's response windows; default validation_gt_vs_<python>_class<g>, made with
% run_validation's class_match + rasters stages when missing), o.out (default <root>/for ppt),
% o.tol (0.5). Before: each run folder has grapes_blanks.mat (build_wave_clus_arm) and its venns
% (venn_wave_clus, with 'gt_vs_python', true so <root>/venn_gt_vs_python_class<g> is this python arm).
%
% Layout of <out>:
%   01_metrics/{gt, python}/        GT metric pages | python report cluster page
%   02_venns/gt_vs_python/          GT vs python class matches + venns
%   06_all_classes/                 rasters of every GT and python class (windows of the sort's matched class)
%   07_no_notch_vs_notch/           compare_detections pages
%   membership_ch<ch>.mat           every spike of every sort, where it is in every other (build_sort_membership)
%   <run folder>/                   per compared wave_clus run:
%     01_metrics/                   fig2print
%     02_venns/wave_clus_vs_python/, 02_venns/gt_vs_wave_clus/
%     03_wave_clus_only/            wave_clus class spikes python has neither in its class nor in quarantine
%     04_subsets/                   python class / wave_clus class / matched: waveforms + rasters
%     05_wave_clus_x_quarantine/    wave_clus class spikes python quarantined and not in its class,
%                                   then by quarantine property (waveforms, rasters, counts)
%     06_all_classes/               rasters of every class of this run (windows of its matched class)
%   (every raster folder <name> gets a <name>_default_window twin when the run has late-latency or
%    long-duration stimuli; each page's footer says which window it uses)
%     08_rasters_wave_clus_vs_python/  full comparison (run class = reference, python class = arm): both raster
%                                   pages + the per-stimulus summary (z-score, sign-test p, ... paired tests)
% Classes: GT class g from the session; python's = the one holding most of GT class g; each run's =
% the one holding most of that python class (as venn_wave_clus / python_cluster_validation pick them).
% Rasters: GT's response windows and the session's stim list, in / out percentages in the title.
if ~isfield(o, 'tol'), o.tol = 0.5; end
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(o.session); g = cfg.gt.class; tf = cfg.gt.times_file; ch = cfg.ch;
o.root = cfg.root; stem = tf(7:end - 4);
pp = ppt_paths(cfg);   % python arm, validation and report folders (run tag: ppt_paths)
if ~isfield(o, 'python') || isempty(o.python)
    o.python = pp.py;
end
[~, pyn] = fileparts(o.python);
if ~isfield(o, 'state') || isempty(o.state), state = fullfile(pp.val, sprintf('validation_gt_vs_%s_class%d', pyn, g));
else, state = fullfile(o.root, o.state); end
if ~isfield(o, 'out'), o.out = pp.rep; end
out = o.out; if ~exist(out, 'dir'), mkdir(out); end
st = fullfile(state, 'validation_state.mat'); have = false;
if exist(st, 'file')                                      % tables made with the current window rule (cfg.max_dura)
    w = load(st, 'ref_tables', 'cfg'); have = isfield(w, 'ref_tables') && ~isempty(w.ref_tables) && isfield(w.cfg, 'max_lat');
end
if ~have                                                  % GT's response windows (raster tables) for this python arm
    c = cfg; c.out_dir = state; c.stages = {'class_match', 'rasters'};
    c.arms = struct('name', 'python_sd1', 'dir', fullfile(o.root, o.python), 'times_file', tf, ...
                    'grapes_file', 'grapes_blanks_rescue.mat', 'class', []);
    run_validation(c);
end
col.wc = [0.00 0.45 0.70]; col.py = [0.90 0.60 0.00]; col.both = [0.35 0.35 0.35]; col.q = [0.80 0.15 0.15];

%% shared: GT / python metrics, GT vs python venns
copy_into(dir(fullfile(o.root, 'hand selected', 'metrics', '*.png')), fullfile(out, '01_metrics', 'gt'));
pdf_pages(fullfile(o.root, o.python, 'metrics'), fullfile(out, '01_metrics', 'python'));
copy_venns(fullfile(pp.val, sprintf('venn_gt_vs_python_class%d', g)), fullfile(out, '02_venns', 'gt_vs_python'));

%% membership: GT, python and every wave_clus run
rn = arrayfun(@(r) matlab.lang.makeValidName(['wc_' r.label]), o.runs, 'UniformOutput', false);
sorts = struct('name', [{'gt', 'python'}, rn], 'file', [{fullfile(o.root, 'hand selected', tf), ...
    fullfile(o.root, o.python, tf)}, arrayfun(@(r) fullfile(o.root, r.folder, tf), o.runs, 'UniformOutput', false)]);
M = build_sort_membership(sorts, 'tol', o.tol, 'out_file', fullfile(out, sprintf('membership_ch%d.mat', ch)));
P = M.python; G = M.gt;
py_c = mode(G.python_class(G.class == g & G.python_class > 0));
% 06: every GT and python class, all with the windows of that sort's MATCHED class (GT's class g, python's
% matched class): the GT vs python run's reference table for GT spikes, its arm table for python spikes
w = load(fullfile(state, 'validation_state.mat'), 'ref_tables_default');
two_g = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
for s = {'gt', 'gt'; 'python', 'arm'}'
    T = M.(s{1});
    for k = unique(T.class(T.class > 0))'
        subset_rasters(state, T.t(T.class == k), sprintf('%s_c%d', s{1}, k), fullfile(out, '06_all_classes'), two_g, s{2});
    end
end
M.classes = struct('gt', g, 'python', py_c);

%% per wave_clus run
for r = 1:numel(o.runs)
    run = o.runs(r); nm = rn{r}; R = run.folder; wcf = fullfile(o.root, R);
    wco = wcf; if ~isempty(cfg.tag), wco = fullfile(pp.val, R); end   % its validation runs / venns
    W = M.(nm); wave = M.waves.(nm);
    wc_c = mode(P.([nm '_class'])(P.class == py_c & P.([nm '_class']) > 0));
    M.classes.(nm) = wc_c;
    lab = sprintf('wave_clus %s c%d', run.label, wc_c);
    tm = struct('name', {sprintf('GT c%d', g), lab, sprintf('python c%d', py_c)}, ...
        'wave', {M.templates.gt(g, :), M.templates.(nm)(wc_c, :), M.templates.python(py_c, :)}, ...
        'color', {[0 0 0], col.wc, col.py}, 'style', {'-', '--', ':'});

    % 08 (made first: its windows are the subsets' windows): full raster comparison, this run's class (reference) vs python's class: both raster pages and the
    % per-stimulus summary (n response, median / mean post, z-score, baseline, sign-test p; paired tests)
    vd = fullfile(wco, sprintf('validation_wave_clus_vs_python_c%d', py_c));
    sm = dir(fullfile(vd, 'stimulus_rasters', '**', '*_gt_vs_arm_summary.png')); ok = false;
    if ~isempty(sm), w = load(fullfile(vd, 'validation_state.mat'), 'cfg'); ok = isfield(w.cfg, 'max_lat'); end
    if ~ok
        python_cluster_validation(o.session, R, py_c, 'python', o.python, 'stages', {'class_match', 'rasters'}, ...
            'out_dir', vd, 'tol', o.tol);
    end
    copy_into(dir(fullfile(vd, 'stimulus_rasters', '**', '*.png')), fullfile(out, R, '08_rasters_wave_clus_vs_python'));
    copy_into(dir(fullfile(vd, 'stimulus_rasters_default_window', '**', '*.png')), ...
        fullfile(out, R, '08_rasters_wave_clus_vs_python', 'default_window'));

    % every wave_clus subset uses the wave_clus class's own full-run windows, every python subset python's
    % (the reference / arm tables of this 08 run) -- the same windows as the 08 pages
    w = load(fullfile(vd, 'validation_state.mat'), 'ref_tables_default');
    two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
    rw = @(t, name, sub) subset_rasters(vd, t, name, fullfile(out, sub), two, 'gt');    % wave_clus spikes
    rp = @(t, name, sub) subset_rasters(vd, t, name, fullfile(out, sub), two, 'arm');   % python spikes

    % 01 / 02: copies
    copy_into(dir(fullfile(wcf, 'fig2print*.png')), fullfile(out, R, '01_metrics'));
    copy_venns(fullfile(wco, sprintf('venn_wave_clus_vs_python_c%d', py_c)), fullfile(out, R, '02_venns', 'wave_clus_vs_python'));
    copy_venns(fullfile(wco, sprintf('venn_gt_vs_wave_clus_class%d', g)), fullfile(out, R, '02_venns', 'gt_vs_wave_clus'));

    % 03: wave_clus class spikes python has neither in its class nor in quarantine (the venn's wave_clus-only)
    in_wc = W.class == wc_c;
    only = in_wc & (isnan(W.python_class) | (W.python_class ~= py_c & ~W.python_quarantined));
    where = {sprintf('no python spike within %g ms', o.tol), isnan(W.python_class); ...
             'python cluster 0 (not quarantined)', W.python_class == 0 & ~W.python_quarantined};
    for k = setdiff(unique(W.python_class(only & ~isnan(W.python_class)))', [0 py_c])
        where(end + 1, :) = {sprintf('python class %d', k), W.python_class == k}; %#ok<AGROW>
    end
    grp = struct('name', sprintf('%s only', lab), 'W', wave(only, :), 'color', col.wc);
    for j = 1:size(where, 1)
        grp(end + 1) = struct('name', ['  ... ' where{j, 1}], 'W', wave(only & where{j, 2}, :), 'color', col.wc * 0.6); %#ok<AGROW>
    end
    s3 = fullfile(R, '03_wave_clus_only'); mk(fullfile(out, s3));
    plot_subset_waveforms(grp, tm, sprintf(['ch%d: %s spikes python has neither in c%d nor in quarantine, ' ...
        'and where they are in python'], ch, lab, py_c), fullfile(out, s3, 'waveforms_wave_clus_only.png'));
    rw(W.t(only), sprintf('wave_clus_c%d_only', wc_c), s3);
    writetable(W(only, :), fullfile(out, s3, 'wave_clus_only_spikes.csv'));

    % 04: python class, wave_clus class, matched
    in_py = P.class == py_c; matched = in_wc & W.python_class == py_c;
    subs = {sprintf('python_c%d', py_c), P.t(in_py), M.waves.python(in_py, :), col.py, rp; ...
            sprintf('wave_clus_c%d', wc_c), W.t(in_wc), wave(in_wc, :), col.wc, rw; ...
            sprintf('matched_wave_clus_c%d_python_c%d', wc_c, py_c), W.t(matched), wave(matched, :), col.both, rw};
    s4 = fullfile(R, '04_subsets'); mk(fullfile(out, s4));
    grp = struct('name', strrep(subs(:, 1)', '_', ' '), 'W', subs(:, 3)', 'color', subs(:, 4)');
    plot_subset_waveforms(grp, tm, sprintf('ch%d: python c%d, %s and the matched spikes', ch, py_c, lab), ...
        fullfile(out, s4, 'waveforms_subsets.png'));
    for i = 1:size(subs, 1), subs{i, 5}(subs{i, 2}, subs{i, 1}, s4); end

    % 05: wave_clus class spikes python quarantined and NOT in python's class, then by quarantine property
    % (python's flags on the paired python spike)
    rows = find(in_wc & W.python_quarantined & W.python_class ~= py_c); idx = W.python_idx(rows);
    s5 = fullfile(R, '05_wave_clus_x_quarantine'); mk(fullfile(out, s5));
    rw(W.t(rows), sprintf('00_wave_clus_c%d_x_python_quarantine_not_python_c%d', wc_c, py_c), s5);
    pq = P.Properties.VariableNames(startsWith(P.Properties.VariableNames, 'q_'));
    grp = struct('name', sprintf('%s x quarantine, not python c%d', lab, py_c), 'W', wave(rows, :), 'color', col.q);
    cnt = {sprintf('all (%s x quarantine, not python c%d)', lab, py_c), numel(rows)};
    for j = 1:numel(pq)
        hit = P.(pq{j})(idx); pn = strrep(pq{j}(3:end), '_', ' ');
        cnt(end + 1, :) = {pn, nnz(hit)}; %#ok<AGROW>
        if ~any(hit), continue; end                           % 0-spike properties: only in counts.csv
        grp(end + 1) = struct('name', ['quarantined for ' pn], 'W', wave(rows(hit), :), 'color', col.q * 0.75); %#ok<AGROW>
        rw(W.t(rows(hit)), sprintf('%02d_%s', j, pq{j}(3:end)), s5);
    end
    resc = P.rescued(idx);                                    % rescued, but into a class other than python's
    cnt(end + 1:end + 2, :) = {'rescued into another python class', nnz(resc); 'not rescued', nnz(~resc)};
    if any(resc)
        grp(end + 1) = struct('name', 'rescued into another python class', 'W', wave(rows(resc), :), 'color', col.both);
        grp(end + 1) = struct('name', 'not rescued', 'W', wave(rows(~resc), :), 'color', col.q * 0.6);
        rw(W.t(rows(resc)), '90_rescued_other_class', s5); rw(W.t(rows(~resc)), '91_not_rescued', s5);
    end
    plot_subset_waveforms(grp, tm, sprintf(['ch%d: %s spikes python quarantined and not in python c%d, ' ...
        'then by quarantine property (a spike can fail several)'], ch, lab, py_c), fullfile(out, s5, 'waveforms_by_quarantine_property.png'));
    F = false(numel(idx), numel(pq)); for j = 1:numel(pq), F(:, j) = P.(pq{j})(idx); end
    [u, ~, k] = unique(F, 'rows');
    combo = arrayfun(@(x) strjoin(strrep(erase(pq(u(x, :)), 'q_'), '_', ' '), ' + '), (1:size(u, 1))', 'UniformOutput', false);
    combo(cellfun(@isempty, combo)) = {'(no property flag)'};
    writetable(cell2table(cnt, 'VariableNames', {'group', 'n_spikes'}), fullfile(out, s5, 'counts.csv'));
    writetable(table(combo, accumarray(k, 1), 'VariableNames', {'properties', 'n_spikes'}), fullfile(out, s5, 'counts_by_combination.csv'));

    % 06: every class of this run
    for k = unique(W.class(W.class > 0))'                  % every class with the matched wave_clus class's windows
        rw(W.t(W.class == k), sprintf('wave_clus_c%d', k), fullfile(R, '06_all_classes'));
    end
end

%% 07: no notch vs notch detections (the no-notch run is only used here)
d7 = fullfile(out, '07_no_notch_vs_notch');
nt = find(strcmp({o.runs.label}, 'notch'), 1);
if isfield(o, 'no_notch') && ~isempty(o.no_notch) && ~isempty(nt) && ~exist(fullfile(d7, 'detections_summary.csv'), 'file')
    compare_detections(fullfile(o.root, o.no_notch, [stem '_spikes.mat']), fullfile(o.root, o.runs(nt).folder, [stem '_spikes.mat']), ...
        'names', {'no notch', 'notch'}, 'tol', 0.7, 'raw', fullfile(o.root, [stem '.NC5']), 'out_dir', d7);
end
save(fullfile(out, sprintf('membership_ch%d.mat', ch)), '-struct', 'M', 'classes', '-append');
fprintf('[build_ppt_report] ch%d done -> %s\n', ch, out);
end

function subset_rasters(state, t, name, out_dir, two, side)
% a subset with the full run's windows of the sort it comes from (side 'gt' = the run's reference unit,
% 'arm' = its arm); with late-latency / long-duration stimuli also the default-window version
if isempty(t), fprintf('[build_ppt_report] %s: no spikes, no rasters\n', name); return; end
plot_subset_rasters([], 'dir', state, 'times', t, 'name', name, 'side', side, 'out_dir', out_dir);
if two
    plot_subset_rasters([], 'dir', state, 'times', t, 'name', [name '_default_window'], 'side', side, ...
        'out_dir', out_dir, 'window', 'default');
end
end

function mk(d)
if ~exist(d, 'dir'), mkdir(d); end
end

function copy_into(files, dst)
if isempty(files), return; end
mk(dst);
for f = files', copyfile(fullfile(f.folder, f.name), dst); end
end

function copy_venns(src, dst)
% the two class matches + the three venns of one run_validation venn folder
if ~exist(src, 'dir'), warning('build_ppt_report: no %s', src); return; end
copy_into([dir(fullfile(src, 'class_match_*.png')); dir(fullfile(src, 'venn_three_set', '*.png'))], dst);
end

function pdf_pages(metrics_dir, dst)
% the cluster page(s) of the python metrics report, as png (poppler needs MATLAB's library path cleared)
r = dir(fullfile(metrics_dir, 'report_*.pdf')); if isempty(r), return; end
mk(dst);
pdf = fullfile(r(1).folder, r(1).name);
[~, txt] = system(sprintf('LD_LIBRARY_PATH= pdftotext -f 1 -l 1 "%s" - | grep -o "Page 1/[0-9]*"', pdf));
n = str2double(regexp(txt, '\d+$', 'match', 'once')); if isnan(n), n = 1; end
system(sprintf('LD_LIBRARY_PATH= pdftoppm -r 110 -png -f 1 -l %d "%s" "%s"', n, pdf, fullfile(dst, 'python_cluster_page')));
end
