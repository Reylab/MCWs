function ppt_gt_subsets(session, multi)
% PPT_GT_SUBSETS  GT comparisons of a for ppt report (build_ppt_report must have run first), GT class g as
% the reference, each sort's class matched to it (the class in the GT venn's arm_classes.csv):
%   <out>/09_gt_vs_python/                    GT vs python
%     01_gt_only/  02_python_only/  03_matched/
%     04_gt_x_quarantine/   GT spikes python quarantined and not in its class, then by quarantine property
%     05_rescued_only/      python class spikes rescued into the class that GT lacks
%     06_quarantine_only/   python quarantined spikes in neither class, then by quarantine property
%     summary/ (+ default_window/)  the full GT vs python raster pages + per-stimulus summary + sign test
%   <out>/wave_clus notch/09_gt_vs_wave_clus/ GT vs wave_clus notch
%     01_gt_only/  02_wave_clus_only/  03_matched/  summary/ (+ default_window/)
% Each subset folder: waveforms (with the GT and matched-class templates), rasters with GT's full-run windows
% (+ the default-window version when the run has late-latency / long-duration stimuli), csv of the spikes.
% counts.csv per comparison: each region's size and in-window count, next to the venn's numbers.
%   ppt_gt_subsets('pt4_ch259')  |  ppt_gt_subsets('pt11emu17_ch321_class2', true)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; ch = cfg.ch; root = cfg.root; tf = cfg.gt.times_file;
out = fullfile(root, 'for ppt'); if multi, out = fullfile(out, sprintf('GT c%d', g)); end
M = load(fullfile(out, sprintf('membership_ch%d.mat', ch)));
d = dir(fullfile(root, 'python_recluster_*_second_pass')); py = d(end).name;
col.gt = [0 0 0]; col.py = [0.90 0.60 0.00]; col.wc = [0.00 0.45 0.70]; col.q = [0.80 0.15 0.15]; col.m = [0.35 0.35 0.35];
wc = fullfile(root, 'wave_clus notch');

cmp = struct('arm', {'python', 'wc_notch'}, 'label', {'python', 'wave_clus notch'}, ...
    'venn', {fullfile(root, sprintf('venn_gt_vs_python_class%d', g)), fullfile(wc, sprintf('venn_gt_vs_wave_clus_class%d', g))}, ...
    'dir', {fullfile(out, '09_gt_vs_python'), fullfile(out, 'wave_clus notch', '09_gt_vs_wave_clus')}, ...
    'color', {col.py, col.wc});
for c = cmp
    A = readtable(fullfile(c.venn, 'arm_classes.csv')); a = A.class(1);
    G = M.gt; R = M.(c.arm); Wg = M.waves.gt; Wa = M.waves.(c.arm);
    ac = G.([c.arm '_class']); gc = R.gt_class; isq = ismember('quarantined', R.Properties.VariableNames);
    if isq, aq = G.([c.arm '_quarantined']); rq = R.quarantined; else, aq = false(height(G), 1); rq = false(height(R), 1); end
    in_g = G.class == g; in_a = R.class == a; not_g = isnan(gc) | gc ~= g;
    lab_a = sprintf('%s c%d', c.label, a);
    tm = struct('name', {sprintf('GT c%d', g), lab_a}, 'wave', {M.templates.gt(g, :), M.templates.(c.arm)(a, :)}, ...
        'color', {col.gt, c.color}, 'style', {'-', '--'});
    % regions: name | GT side? | mask | color | venn column
    reg = {'01_gt_only', true, in_g & (isnan(ac) | (ac ~= a & ~aq)), col.gt, 'gt_only'; ...
           sprintf('02_%s_only', strrep(c.label, ' notch', '')), false, in_a & ~rq & not_g, c.color, 'arm_only'; ...
           '03_matched', true, in_g & ac == a, col.m, ''};
    if isq
        reg = [reg; {'04_gt_x_quarantine', true, in_g & aq & ac ~= a, col.q, 'gt_q'; ...
                     '05_rescued_only', false, in_a & rq & not_g, col.q, 'arm_q'; ...
                     '06_quarantine_only', false, rq & ~in_a & not_g, col.q, 'q_only'}]; %#ok<AGROW>
    end
    % the full GT vs <sort> run first: its windows are the subsets' windows (GT spikes: GT's table, the
    % sort's spikes: the sort's table) and its pages are the summary
    % summary: the full GT vs <sort> raster pages + per-stimulus summary + sign test
    if strcmp(c.arm, 'python')
        vd = fullfile(root, sprintf('validation_gt_vs_%s_class%d', py, g));
    else
        vd = fullfile(wc, sprintf('validation_gt_vs_wave_clus_notch_class%d', g)); ok = false;
        if exist(fullfile(vd, 'validation_state.mat'), 'file')
            w = load(fullfile(vd, 'validation_state.mat'), 'cfg', 'ref_tables'); ok = isfield(w.cfg, 'max_lat') && ~isempty(w.ref_tables);
        end
        if ~ok
            r = cfg; r.out_dir = vd; r.stages = {'class_match', 'rasters'};
            r.arms = struct('name', 'wave_clus notch', 'dir', wc, 'times_file', tf, 'grapes_file', 'grapes_blanks.mat', 'class', a);
            run_validation(r);
        end
    end
    state = vd; st = load_validation_state(state);
    two = isfield(st, 'ref_tables_default') && ~isempty(st.ref_tables_default);
    Tg = st.ref_tables{1}; Ta = st.ref_tables{2};
    inw_g = @(t) count_in_windows(t, Tg.stim_number, Tg.win_lo, Tg.win_hi, st.onset_map);
    inw_a = @(t) count_in_windows(t, Ta.stim_number, Ta.win_lo, Ta.win_hi, st.onset_map);
    V = readtable(fullfile(c.venn, 'venn_three_set', 'venn3_counts.csv')); V = V(strcmp(V.scope, 'cluster'), :);
    if ~exist(c.dir, 'dir'), mkdir(c.dir); end
    cnt = cell(0, 5); grp = struct('name', {}, 'W', {}, 'color', {});
    for i = 1:size(reg, 1)
        [nm, gside, m, cc, vcol] = reg{i, :};
        if gside, t = G.t(m); Wv = Wg(m, :); Tb = G(m, :); else, t = R.t(m); Wv = Wa(m, :); Tb = R(m, :); end
        vn = NaN;
        if ~isempty(vcol), vn = V.(vcol)(1); elseif ismember('gt_arm', V.Properties.VariableNames), vn = V.gt_arm(1) + V.gt_arm_q(1); end
        if gside, n_in = inw_g(t); else, n_in = inw_a(t); end
        cnt(end + 1, :) = {nm(4:end), nnz(m), n_in, vn, ternary(gside, 'GT spikes, GT windows', [c.label ' spikes, ' c.label ' windows'])}; %#ok<AGROW>
        sd = fullfile(c.dir, nm); if ~exist(sd, 'dir'), mkdir(sd); end
        writetable(Tb, fullfile(sd, [nm(4:end) '_spikes.csv']));
        grp(end + 1) = struct('name', strrep(nm(4:end), '_', ' '), 'W', Wv, 'color', cc); %#ok<AGROW>
        sub_rasters(state, t, nm(4:end), sd, two, ternary(gside, 'gt', 'arm'));
        if isq && any(strcmp(nm, {'04_gt_x_quarantine', '06_quarantine_only'}))   % by quarantine property
            if gside, idx = G.([c.arm '_idx'])(m); else, idx = find(m); end
            by_property(R, idx, t, Wv, tm, sd, state, two, ternary(gside, 'gt', 'arm'), col.q, sprintf('ch%d: %s, by quarantine property (GT c%d vs %s)', ...
                ch, strrep(nm(4:end), '_', ' '), g, lab_a));
        end
    end
    plot_subset_waveforms(grp, tm, sprintf('ch%d: GT c%d vs %s -- the venn regions', ch, g, lab_a), ...
        fullfile(c.dir, 'waveforms_regions.png'));
    writetable(cell2table(cnt, 'VariableNames', {'region', 'n_spikes', 'n_in_window', 'venn', 'counted_on'}), ...
        fullfile(c.dir, 'counts.csv'));

    copy_pngs(fullfile(vd, 'stimulus_rasters'), fullfile(c.dir, 'summary'));
    copy_pngs(fullfile(vd, 'stimulus_rasters_default_window'), fullfile(c.dir, 'summary', 'default_window'));
end
fprintf('[ppt_gt_subsets] %s done -> %s\n', session, out);
end

function by_property(R, idx, t, Wv, tm, sd, state, two, side, cq, ttl)
% the region split by python's quarantine flags (on the python spike itself, or the one paired with a GT spike)
pq = R.Properties.VariableNames(startsWith(R.Properties.VariableNames, 'q_'));
d = fullfile(sd, 'by_property'); if ~exist(d, 'dir'), mkdir(d); end
grp = struct('name', 'all', 'W', Wv, 'color', cq); cnt = {'all', numel(idx)};
F = false(numel(idx), numel(pq));
for j = 1:numel(pq)
    hit = R.(pq{j})(idx); F(:, j) = hit; pn = strrep(pq{j}(3:end), '_', ' ');
    cnt(end + 1, :) = {pn, nnz(hit)}; %#ok<AGROW>
    if ~any(hit), continue; end
    grp(end + 1) = struct('name', ['quarantined for ' pn], 'W', Wv(hit, :), 'color', cq * 0.75); %#ok<AGROW>
    sub_rasters(state, t(hit), sprintf('%02d_%s', j, pq{j}(3:end)), d, two, side);
end
plot_subset_waveforms(grp, tm, [ttl ' (a spike can fail several)'], fullfile(d, 'waveforms_by_quarantine_property.png'));
writetable(cell2table(cnt, 'VariableNames', {'property', 'n_spikes'}), fullfile(d, 'counts.csv'));
if ~isempty(idx)
    [u, ~, k] = unique(F, 'rows');
    combo = arrayfun(@(x) strjoin(strrep(erase(pq(u(x, :)), 'q_'), '_', ' '), ' + '), (1:size(u, 1))', 'UniformOutput', false);
    combo(cellfun(@isempty, combo)) = {'(no property flag)'};
    writetable(table(combo, accumarray(k, 1), 'VariableNames', {'properties', 'n_spikes'}), fullfile(d, 'counts_by_combination.csv'));
end
end

function sub_rasters(state, t, name, d, two, side)
if isempty(t), fprintf('[ppt_gt_subsets] %s: no spikes, no rasters\n', name); return; end
plot_subset_rasters([], 'dir', state, 'times', t, 'name', name, 'side', side, 'out_dir', d);
if two
    plot_subset_rasters([], 'dir', state, 'times', t, 'name', [name '_default_window'], 'side', side, 'out_dir', d, 'window', 'default');
end
end

function copy_pngs(src, dst)
f = dir(fullfile(src, '**', '*.png')); if isempty(f), return; end
if ~exist(dst, 'dir'), mkdir(dst); end
for k = 1:numel(f), copyfile(fullfile(f(k).folder, f(k).name), dst); end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
