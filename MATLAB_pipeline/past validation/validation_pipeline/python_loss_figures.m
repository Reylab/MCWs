function S = python_loss_figures(session, varargin)
% PYTHON_LOSS_FIGURES  Diagnostic figures: where the GT unit's spikes go in the
% python sorts (before / after the second pass) and in the wave_clus runs, and
% why python loses them. Read-only: loads the saved sorts, changes nothing.
%
%   python_loss_figures('pt6_ch333')
%   python_loss_figures('pt12_ch300_class2', 'python', 'python_recluster_20261012_second_pass')
%
% session - validation_sessions id (channel folder, GT times file and class)
% Name-value:
%   'python'    newest python arm folder   (default 'python_recluster_20261012_second_pass')
%   'before'    python arm before the second pass (default 'python_recluster_20261007_rescued')
%   'wave_clus' wave_clus run folders (default: every wave_clus* folder with the times file)
%   'out_dir'   default <root>/validation_gt_vs_<python>[_classK]/diagnostics
%   'tol'       spike-time match tolerance, ms (default 0.5)
%   'python_class'  fix the newest python sort's cluster (default: the one matched to GT)
%
% Figures (diag_ch<ch>_gt<g>_*.png) + diagnostics_summary.csv:
%   recall_purity   per sort: share of the GT unit in the matched cluster, and
%                   the share of that cluster that is GT (purity)
%   gt_fate         every GT spike in each python sort: matched cluster / other
%                   cluster / quarantined by reason (within-channel second
%                   peak = prominence ratio < 2, within-channel low amplitude =
%                   bottom 5 %, within-channel width outside 3-15 samples,
%                   other within-channel, refractory, bundle) /
%                   unassigned / not detected
%   lost_waveforms  python's own waveforms of the GT spikes each rule threw
%                   out, against the kept GT spikes
%   second_pass     GT spikes' cluster before -> after the second pass, how
%                   the movers were placed (Mahalanobis / template match), and
%                   each cluster's spread (sqrt of summed sample variance)
%   time_offset     nearest-spike time offset to GT for each sort (1 bin = 1 sample)

p = inputParser;
addParameter(p, 'python', 'python_recluster_20261012_second_pass');
addParameter(p, 'before', 'python_recluster_20261007_rescued');
addParameter(p, 'wave_clus', {});
addParameter(p, 'out_dir', '');
addParameter(p, 'tol', 0.5);
addParameter(p, 'python_class', []);   % fix the newest python sort's matched cluster (default: matched to GT)
parse(p, varargin{:});
o = p.Results;
here = fileparts(mfilename('fullpath'));
addpath(here);

cfg = validation_sessions(session);
root = cfg.root; tf = cfg.gt.times_file; g = cfg.gt.class; ch = cfg.ch;
tok = regexp(session, '_class\d+$', 'match', 'once');
out = o.out_dir;
if isempty(out), out = fullfile(root, ['validation_gt_vs_' o.python tok], 'diagnostics'); end
if ~exist(out, 'dir'), mkdir(out); end
tag = sprintf('ch%d GT class %d', ch, g);
pre = fullfile(out, sprintf('diag_ch%d_gt%d_', ch, g));

G = load(fullfile(cfg.gt.dir, tf), 'cluster_class');
gt = sort(G.cluster_class(G.cluster_class(:, 1) == g, 2));

wc = cellstr(o.wave_clus);
if isempty(o.wave_clus)
    d = dir(fullfile(root, 'wave_clus*'));
    d = d([d.isdir] & arrayfun(@(x) exist(fullfile(x.folder, x.name, tf), 'file') == 2, d)');
    n = cellfun(@(s) str2double(regexp(s, '\d+$', 'match', 'once')), {d.name});
    [~, k] = sort(n); wc = {d(k).name};
end
names = [wc, {o.before, o.python}];
kind  = [repmat({'wave_clus'}, 1, numel(wc)), {'python', 'python'}];
keep = cellfun(@(s) exist(fullfile(root, s, tf), 'file') == 2, names);
names = names(keep); kind = kind(keep);

R = struct([]);
for s = 1:numel(names)
    A = load(fullfile(root, names{s}, tf));
    fc = []; if strcmp(names{s}, o.python), fc = o.python_class; end
    r = match_sort(A, gt, o.tol, fc);
    r.name = names{s}; r.kind = kind{s}; r.A = A;
    R = [R, r]; %#ok<AGROW>
end
isPy = strcmp({R.kind}, 'python');

%% 1. recall / purity
f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 max(700, 110*numel(R)) 460]);
ax = axes(f);
b = bar(ax, 100*[[R.recall]; [R.purity]]', 'grouped');
b(1).FaceColor = [0.2 0.45 0.75]; b(2).FaceColor = [0.85 0.45 0.15];
set(ax, 'XTick', 1:numel(R), 'XTickLabel', short_names({R.name}), 'TickLabelInterpreter', 'none', ...
    'XTickLabelRotation', 30, 'YLim', [0 110]);
for s = 1:numel(R)
    text(ax, s, 104, sprintf('c%d\nn=%d', R(s).best, R(s).nbest), 'HorizontalAlignment', 'center', 'FontSize', 7);
end
ylabel(ax, '%'); grid(ax, 'on'); box(ax, 'off');
legend(ax, {'GT recovered (in matched cluster)', 'purity (matched cluster that is GT)'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
title(ax, sprintf('%s (%d spikes): recall vs purity of the matched cluster', tag, numel(gt)), 'Interpreter', 'none');
save_png(f, [pre 'recall_purity.png']);

%% 2. GT fate in each python sort
cats = {'matched cluster', 'other cluster', 'within-ch: second peak (prom ratio < 2)', ...
        'within-ch: low amplitude (bottom 5%)', 'within-ch: width outside 3-15 samples', 'within-ch: other', ...
        'refractory', 'bundle', 'unassigned (not quarantined)', 'not detected'};
cols = [0.20 0.60 0.30; 0.55 0.75 0.55; 0.85 0.25 0.20; 0.95 0.55 0.20; 0.70 0.40 0.10; 0.95 0.80 0.40; ...
        0.45 0.30 0.65; 0.55 0.55 0.85; 0.65 0.65 0.65; 0.25 0.25 0.25];
pyR = R(isPy);
M = zeros(numel(pyR), numel(cats));
for s = 1:numel(pyR)
    pyR(s).fate = gt_fate(pyR(s));
    M(s, :) = arrayfun(@(c) nnz(pyR(s).fate == c), 1:numel(cats));
end
R(isPy) = pyR;
f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1200 140 + 90*numel(pyR)]);
ax = axes(f);
h = barh(ax, 100*M./numel(gt), 'stacked');
for c = 1:numel(cats), h(c).FaceColor = cols(c, :); end
set(ax, 'YTick', 1:numel(pyR), 'YTickLabel', {pyR.name}, 'TickLabelInterpreter', 'none', 'YDir', 'reverse');
xlim(ax, [0 100]); xlabel(ax, '% of GT spikes');
for s = 1:numel(pyR)
    x0 = 0;
    for c = 1:numel(cats)
        w = 100*M(s, c)/numel(gt);
        if w >= 2.5, text(ax, x0 + w/2, s, sprintf('%d', M(s, c)), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', 'w'); end
        x0 = x0 + w;
    end
end
legend(ax, cats, 'Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none');
title(ax, {sprintf('%s: where every GT spike ends up in python', tag), ...
    'rescued spikes count under the cluster they ended in', ...
    'reason order: within-channel, refractory, bundle'}, 'Interpreter', 'none', 'FontSize', 9);
save_png(f, [pre 'gt_fate.png']);

%% 3. waveforms python threw out
P = pyR(end); W = double(P.A.spikes);
kept = P.j(P.fate(P.det) == 1);
groups = {3, 'within-ch: second peak|(prom ratio < 2)'; 4, 'within-ch: low amplitude|(bottom 5%)'; ...
          5, 'within-ch: width|outside 3-15 samples'; 7, 'refractory'};
f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1400 380]);
tl = tiledlayout(f, 1, 1 + size(groups, 1), 'TileSpacing', 'compact');
title(tl, sprintf('%s: python''s waveforms of GT spikes it quarantined and did not rescue (%s)', tag, P.name), 'Interpreter', 'none');
ref = mean(W(kept, :), 1);
ax = nexttile(tl); plot_wave(ax, W(kept, :), [0.2 0.6 0.3], ref, {'GT spikes python kept', '(dashed = their mean)', sprintf('n=%d', numel(kept))});
yl = ylim(ax);
for k = 1:size(groups, 1)
    jj = P.j(P.fate(P.det) == groups{k, 1});
    ax = nexttile(tl);
    plot_wave(ax, W(jj, :), cols(groups{k, 1}, :), ref, [strsplit(groups{k, 2}, '|'), {sprintf('n=%d', numel(jj))}]);
    yl = [min(yl(1), min(ylim(ax))), max(yl(2), max(ylim(ax)))];
end
set(findobj(f, 'Type', 'axes'), 'YLim', yl);
save_png(f, [pre 'lost_waveforms.png']);

%% 4. second pass: GT spikes' cluster before -> after
iB = find(strcmp({R.name}, o.before)); iA = find(strcmp({R.name}, o.python));
if ~isempty(iB) && ~isempty(iA)
    B = R(iB); A = R(iA);
    both = B.det & A.det;
    lb = B.A.cluster_class(B.idx(both), 1); la = A.A.cluster_class(A.idx(both), 1);
    ub = unique(lb); ua = unique(la);
    X = zeros(numel(ub), numel(ua));
    for a = 1:numel(ub), for c = 1:numel(ua), X(a, c) = nnz(lb == ub(a) & la == ua(c)); end, end
    mv = lb ~= la;
    jm = A.idx(both); jm = jm(mv);
    mah = isfield(A.A, 'mahal_reassigned') && any(A.A.mahal_reassigned(:));
    if mah, nm = nnz(logical(A.A.mahal_reassigned(jm))); else, nm = 0; end
    nf = nnz(logical(A.A.forced(jm))) - nm;
    lab = A.A.cluster_class(:, 1); WA = double(A.A.spikes);
    cl = setdiff(unique(lab), 0)';
    spread = arrayfun(@(c) sqrt(sum(var(WA(lab == c, :), 1, 1))), cl);
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1500 430]);
    tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact');
    title(tl, sprintf('%s: second pass (%s -> %s)', tag, o.before, o.python), 'Interpreter', 'none');
    ax = nexttile(tl);
    imagesc(ax, log10(X + 1)); colormap(ax, flipud(gray)); axis(ax, 'tight');
    for a = 1:numel(ub), for c = 1:numel(ua)
        if X(a, c), text(ax, c, a, sprintf('%d', X(a, c)), 'HorizontalAlignment', 'center', 'Color', [0.8 0.1 0.1], 'FontWeight', 'bold'); end
    end, end
    set(ax, 'XTick', 1:numel(ua), 'XTickLabel', "c" + string(ua), 'YTick', 1:numel(ub), 'YTickLabel', "c" + string(ub));
    xlabel(ax, 'cluster after'); ylabel(ax, 'cluster before');
    title(ax, sprintf('GT spikes: before -> after (%d changed cluster)', nnz(mv)));
    ax = nexttile(tl);
    bar(ax, [nm, nf, nnz(mv) - nm - nf], 'FaceColor', [0.5 0.5 0.7]);
    set(ax, 'XTick', 1:3, 'XTickLabel', {'Mahalanobis', 'template', 'core/split'}, 'XTickLabelRotation', 0);
    for k = 1:3, v = [nm, nf, nnz(mv) - nm - nf]; text(ax, k, v(k), sprintf('%d', v(k)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'); end
    ylabel(ax, 'GT spikes that changed cluster'); title(ax, 'how the movers were placed'); box(ax, 'off');
    ax = nexttile(tl);
    bar(ax, spread, 'FaceColor', [0.85 0.45 0.15]);
    set(ax, 'XTick', 1:numel(cl), 'XTickLabel', compose('c%d', cl(:)), 'XTickLabelRotation', 0);
    for k = 1:numel(cl)
        text(ax, k, spread(k), sprintf('n=%d', nnz(lab == cl(k))), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    ylabel(ax, 'spread: sqrt(sum of sample variances)');
    title(ax, 'cluster spread after (radius scales with it)'); box(ax, 'off');
    save_png(f, [pre 'second_pass.png']);
end

%% 5. spike-time offset to GT
sr = 30;   % samples per ms
f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 900 420]);
ax = axes(f); hold(ax, 'on');
ed = (-15.5:15.5) / sr;
show = unique([find(strcmp({R.kind}, 'wave_clus'), 1), find(isPy)]);
lg = {};
for s = show
    dd = R(s).dt(abs(R(s).dt) <= 0.5);
    histogram(ax, dd, ed, 'DisplayStyle', 'stairs', 'LineWidth', 1.5, 'Normalization', 'probability');
    lg{end+1} = sprintf('%s (median %.3f ms)', R(s).name, median(dd)); %#ok<AGROW>
end
xline(ax, 0, 'k:');
xlabel(ax, 'nearest spike time - GT time (ms); 1 bin = 1 sample'); ylabel(ax, 'fraction of GT spikes');
legend(ax, lg, 'Interpreter', 'none', 'Location', 'northeast'); grid(ax, 'on');
title(ax, sprintf('%s: spike-time offset to GT', tag));
save_png(f, [pre 'time_offset.png']);

%% summary
S = table({R.name}', {R.kind}', [R.best]', [R.nbest]', 100*[R.det_rate]', 100*[R.recall]', 100*[R.purity]', [R.dt_median]', ...
    'VariableNames', {'sort', 'kind', 'matched_cluster', 'cluster_size', 'detected_pct', 'recall_pct', 'purity_pct', 'dt_median_ms'});
for c = 1:numel(cats)
    v = nan(numel(R), 1); v(isPy) = M(:, c);
    S.(matlab.lang.makeValidName(['n_' cats{c}])) = v;
end
writetable(S, fullfile(out, sprintf('diagnostics_summary_ch%d_gt%d.csv', ch, g)));
fprintf('[python_loss_figures] %s -> %s\n', tag, out);
end

% -------------------------------------------------------------------------
function r = match_sort(A, gt, tol, force_class)
% nearest spike of this sort for every GT spike; matched cluster = most common
% non-zero label among detected GT spikes
    t = A.cluster_class(:, 2); lab = A.cluster_class(:, 1);
    [ts, ord] = sort(t);
    k = interp1(ts, 1:numel(ts), gt, 'nearest', 'extrap');
    r.idx = ord(k);
    r.dt = ts(k) - gt;
    r.det = abs(r.dt) <= tol;
    r.j = r.idx(r.det);
    L = lab(r.j);
    r.best = mode(L(L > 0));
    if nargin >= 4 && ~isempty(force_class), r.best = force_class; end
    r.nbest = nnz(lab == r.best);
    r.recall = nnz(L == r.best) / numel(gt);
    r.purity = nnz(L == r.best) / max(r.nbest, 1);
    r.det_rate = mean(r.det);
    r.dt_median = median(r.dt(abs(r.dt) <= 2));
    r.fate = [];
end

function fate = gt_fate(r)
% category (1..10, see cats in the caller) of every GT spike in a python sort
    A = r.A; lab = A.cluster_class(:, 1); n = numel(lab);
    Q = A.quarantine;
    flag = @(f) isfield(Q, f) && numel(Q.(f)) == n;
    get = @(f) logical(Q.(f)(:));
    wch = false(n, 1); ref = wch; bun = wch;
    if flag('within_channel'), wch = get('within_channel'); end
    if flag('refractory_period'), ref = get('refractory_period'); end
    if flag('bundle_collision'), bun = get('bundle_collision'); end
    fate = 10 * ones(size(r.idx));         % not detected
    j = r.idx; L = lab(j);
    d = r.det;
    fate(d & L == r.best) = 1;
    fate(d & L > 0 & L ~= r.best) = 2;
    z = find(d & L == 0);
    for m = z'
        k = j(m);
        if wch(k)
            [pr, la, wd] = props(A, k);
            if pr < 2, fate(m) = 3; elseif la, fate(m) = 4; elseif wd < 3 || wd > 15, fate(m) = 5; else, fate(m) = 6; end
        elseif ref(k), fate(m) = 7;
        elseif bun(k), fate(m) = 8;
        else, fate(m) = 9;
        end
    end
    fate = fate(:);
end

function [pr, la, wd] = props(A, k)
    pr = NaN; la = false; wd = NaN;
    if ~isfield(A, 'quarantine_properties') || k > numel(A.quarantine_properties), return; end
    P = A.quarantine_properties;          % cell of structs, or a struct array (older conversions)
    if iscell(P), s = P{k}; elseif isstruct(P), s = P(k); else, return; end
    if true
        if isstruct(s)
            if isfield(s, 'prominence_ratio'), pr = double(s.prominence_ratio); end
            if isfield(s, 'low_low_amp_spike'), la = double(s.low_low_amp_spike) > 0; end
            if isfield(s, 'width'), wd = double(s.width); end
        end
    end
end

function plot_wave(ax, W, col, ref, ttl)
    hold(ax, 'on');
    if ~isempty(W)
        k = 1:size(W, 1);
        if numel(k) > 300, k = k(round(linspace(1, numel(k), 300))); end
        hl = plot(ax, W(k, :)', 'Color', [col 0.12]); %#ok<NASGU>
        plot(ax, mean(W, 1), 'Color', col * 0.6, 'LineWidth', 2.2);
    end
    plot(ax, ref, 'k--', 'LineWidth', 1.2);
    title(ax, ttl, 'Interpreter', 'none', 'FontSize', 9);
    xlabel(ax, 'sample'); grid(ax, 'on'); box(ax, 'off');
end

function s = short_names(c)
    s = strrep(strrep(c, 'python_recluster_', 'py_'), '_second_pass', '_2ndpass');
end

function save_png(f, fn)
    print(f, fn, '-dpng', '-r150');
    close(f);
end
