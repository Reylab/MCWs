function R = compare_detections(file_a, file_b, varargin)
% COMPARE_DETECTIONS  Two detections of the same channel, spike by spike: which
% spikes both have, which only one has, and how far the shared ones moved.
% Only the spike times decide the matching; the waveforms are only drawn.
%
%   compare_detections(file_a, file_b)
%   compare_detections(file_a, file_b, 'names', {'no notch', 'notch'}, 'tol', 0.5, 'out_dir', '/some/folder')
%
% FILE_A / FILE_B: a path or the struct load() returned for it -- a *_spikes.mat (index = spike times in ms, spikes = waveforms;
% wave_clus, MCWs or python) or a times_*.mat (cluster_class(:,2) = times in ms,
% spikes). Waveforms are optional (no waveform page without them).
%
% Matching: one-to-one, closest pairs first; a pair counts as the same spike when
% the two times are within TOL ms (default 0.5, the validation pipeline's value).
%
% Name-value:
%   'names'     {name_a, name_b}   default: the two files' folder names
%   'tol'       matching tolerance in ms (default 0.5)
%   'max_shift' how far to look for a shifted partner, ms (default 3)
%   'out_dir'   default <folder of file_b>/compare_<name_a>_vs_<name_b>
%   'raw'       the channel's raw NC5 (int16, /4 = uV). With it, each run's detection signal is
%               rebuilt from its own par (bandpass, order, notch = par.process_info when
%               par.preprocessing), and every spike only one run has is looked up in the other
%               run's signal to tell WHY it is missing (below that run's threshold / crosses but a
%               spike of that run sits within its 1.5 ms dead time / crosses and nothing near)
%
% Writes to out_dir:
%   detections_overview.png  2-set venn at TOL; matched count vs tolerance (is TOL
%                            too small?); signed shift of every pair within MAX_SHIFT
%                            in samples, TOL marked
%   detections_timeline.png  "firing diagram": A only / both / B only along the
%                            recording, rate per 10 s, and the shift of each pair over time
%   (with 'raw') each run's detection is REPLAYED from the raw file (its own par) and its log says,
%   for every spike only the other run has, exactly why it was not detected:
%     threshold (never crosses) / refractory period (par.ref_ms after its own earlier spike, at -x ms) /
%     crossed but the peak was placed > tol away / artifact check (> stdmax) / its spike there is
%     paired with another one / segment edge / unresolved
%   detections_why_missing.png  the unique spikes: stored waveforms, the same times in their own and
%                            in the other run's detection signal (with thresholds), cause counts
%   detections_why_timeline.png each unique spike along the recording by cause, and its peak in the
%                            other run's signal against that run's threshold
%   detections_why_examples.png  +-5 ms of both detection signals around examples of every cause
%   detections_why_missing.csv  one row per unique spike: time, cause, detail (ms), peak in the other signal
%   detections_summary.csv   counts at TOL and at every tolerance on the curve
%   detections_match.mat     indices of A only / B only / pairs and their shifts
% Returns R with the same numbers.
p = inputParser;
addParameter(p, 'names', {});
addParameter(p, 'tol', 0.7);
addParameter(p, 'max_shift', 3);
addParameter(p, 'out_dir', '');
addParameter(p, 'raw', '');
parse(p, varargin{:}); o = p.Results;
addpath(fileparts(mfilename('fullpath')));    % plot_traces

A = read_spikes(file_a); B = read_spikes(file_b);
if isempty(o.names) && (isstruct(file_a) || isstruct(file_b)), o.names = {'A', 'B'}; end
if isempty(o.names)
    [~, na] = fileparts(fileparts(file_a)); [~, nb] = fileparts(fileparts(file_b));
    if strcmp(na, nb), na = [na ' (A)']; nb = [nb ' (B)']; end
    o.names = {na, nb};
end
[na, nb] = o.names{:};
if isempty(o.out_dir) && isstruct(file_b), o.out_dir = fullfile(pwd, sprintf('compare_%s_vs_%s', safe(o.names{1}), safe(o.names{2}))); end
if isempty(o.out_dir)
    o.out_dir = fullfile(fileparts(file_b), sprintf('compare_%s_vs_%s', safe(na), safe(nb)));
end
if ~exist(o.out_dir, 'dir'), mkdir(o.out_dir); end
sr = A.sr; dt = 1000 / sr;                    % ms per sample

% ---- one-to-one matching at every tolerance up to max_shift ----
[pa, pb, pd] = candidate_pairs(A.t, B.t, o.max_shift);         % pd = tB - tA
tols = unique([dt/2, dt:dt:o.max_shift, o.tol]);
n_match = zeros(size(tols));
for k = 1:numel(tols)
    n_match(k) = numel(greedy(pa, pb, pd, tols(k)));
end
keep = greedy(pa, pb, pd, o.tol);
ia = pa(keep); ib = pb(keep); shift = pd(keep);
only_a = setdiff((1:numel(A.t))', ia); only_b = setdiff((1:numel(B.t))', ib);
keep_max = greedy(pa, pb, pd, o.max_shift);                     % every pair a looser tolerance would accept
shift_max = pd(keep_max);
near_a = ismember(only_a, pa(keep_max)); near_b = ismember(only_b, pb(keep_max));

R = struct('names', {o.names}, 'tol', o.tol, 'n_a', numel(A.t), 'n_b', numel(B.t), 'n_both', numel(ia), ...
    'n_only_a', numel(only_a), 'n_only_b', numel(only_b), ...
    'n_exact', nnz(abs(shift) < dt/2), 'n_shifted', nnz(abs(shift) >= dt/2), ...
    'n_only_a_partner_within_max', nnz(near_a), 'n_only_b_partner_within_max', nnz(near_b), ...
    'tols', tols, 'n_match', n_match, 'out_dir', o.out_dir);
if isstruct(file_a), src_a = 'struct input'; else, src_a = file_a; end
if isstruct(file_b), src_b = 'struct input'; else, src_b = file_b; end
save(fullfile(o.out_dir, 'detections_match.mat'), 'R', 'ia', 'ib', 'shift', 'only_a', 'only_b', 'src_a', 'src_b');
writetable(table(tols(:), n_match(:), n_match(:) / numel(A.t), n_match(:) / numel(B.t), ...
    'VariableNames', {'tol_ms', 'n_both', ['frac_of_' safe(na)], ['frac_of_' safe(nb)]}), ...
    fullfile(o.out_dir, 'detections_summary.csv'));
fprintf(['[compare_detections] %s %d | %s %d | both %d (exact %d, shifted %d) | %s only %d | %s only %d  (tol %.2f ms)\n' ...
         '                     unmatched with a partner within %.1f ms: %s only %d, %s only %d\n'], ...
    na, R.n_a, nb, R.n_b, R.n_both, R.n_exact, R.n_shifted, na, R.n_only_a, nb, R.n_only_b, o.tol, ...
    o.max_shift, na, R.n_only_a_partner_within_max, nb, R.n_only_b_partner_within_max);

ca = [0.00 0.45 0.70]; cb = [0.90 0.60 0.00]; cm = [0.35 0.35 0.35];   % A only, B only, both

% ---- page 1: venn, tolerance curve, shift histogram ----
f = figure('Visible', 'off', 'Position', [50 50 1800 650], 'Color', 'w');
ax = axes(f, 'Position', [0.03 0.12 0.25 0.66]); draw_venn(ax, R, na, nb, ca, cb, cm);
ax = axes(f, 'Position', [0.36 0.13 0.28 0.68]); hold(ax, 'on');
plot(ax, tols, 100 * n_match / numel(A.t), '-', 'Color', ca, 'LineWidth', 2, 'DisplayName', sprintf('%% of %s', na));
plot(ax, tols, 100 * n_match / numel(B.t), '-', 'Color', cb, 'LineWidth', 2, 'DisplayName', sprintf('%% of %s', nb));
xline(ax, o.tol, '--k', sprintf('tol %.2f ms', o.tol), 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
k1 = find(tols >= o.tol, 1); gain = n_match(end) - n_match(k1);
xlabel(ax, 'matching tolerance (ms)'); ylabel(ax, 'spikes matched (%)'); grid(ax, 'on'); legend(ax, 'Location', 'southeast');
title(ax, {'matched spikes vs tolerance', sprintf('widening %.2f -> %.1f ms adds %d pairs (%.1f%% of %s)', ...
    o.tol, o.max_shift, gain, 100 * gain / numel(A.t), na)}, 'FontSize', 10);
ax = axes(f, 'Position', [0.70 0.13 0.28 0.68]); hold(ax, 'on');
e = ((-round(o.max_shift / dt) - 0.5):(round(o.max_shift / dt) + 0.5)) * dt;
histogram(ax, shift_max, e, 'FaceColor', cm, 'EdgeColor', 'none');
xline(ax, [-o.tol o.tol], '--k', 'HandleVisibility', 'off');
set(ax, 'YScale', 'log'); xlabel(ax, sprintf('shift %s - %s (ms; 1 bar = 1 sample)', nb, na)); ylabel(ax, 'pairs (log)'); grid(ax, 'on');
title(ax, {sprintf('how far shared spikes moved (all pairs within %.1f ms)', o.max_shift), ...
    sprintf('exact %d | shifted inside tol %d | only matchable beyond tol %d', R.n_exact, R.n_shifted, nnz(abs(shift_max) > o.tol))}, 'FontSize', 10);
sgtitle(f, sprintf('%s vs %s -- detections (same spike = within %.2f ms, one-to-one, closest first)', na, nb, o.tol), 'FontWeight', 'bold', 'Interpreter', 'none');
print(f, fullfile(o.out_dir, 'detections_overview.png'), '-dpng', '-r120'); close(f);

% ---- page 2: firing diagram ----
f = figure('Visible', 'off', 'Position', [50 50 1800 950], 'Color', 'w');
ax = subplot(3, 1, 1, 'Parent', f); hold(ax, 'on');
ev = {A.t(only_a), A.t(ia), B.t(only_b)}; cc = {ca, cm, cb};
lab = {sprintf('%s only (%d)', na, numel(only_a)), sprintf('both (%d)', numel(ia)), sprintf('%s only (%d)', nb, numel(only_b))};
for r = 1:3
    x = ev{r}(:)' / 1000;
    plot(ax, [x; x; nan(size(x))], repmat([r - 0.35; r + 0.35; nan], 1, numel(x)), '-', 'Color', cc{r}, 'LineWidth', 0.5);
end
set(ax, 'YTick', 1:3, 'YTickLabel', lab, 'YLim', [0.4 3.6]); xlabel(ax, 'time (s)'); box(ax, 'off');
title(ax, sprintf('every spike along the recording (matched at %.2f ms)', o.tol));
ax = subplot(3, 1, 2, 'Parent', f); hold(ax, 'on');
tend = max([A.t; B.t]) / 1000; be = 0:10:(tend + 10); bc = be(1:end - 1) + 5;
plot(ax, bc, histcounts(A.t(only_a) / 1000, be), '-', 'Color', ca, 'LineWidth', 1.4, 'DisplayName', [na ' only']);
plot(ax, bc, histcounts(B.t(only_b) / 1000, be), '-', 'Color', cb, 'LineWidth', 1.4, 'DisplayName', [nb ' only']);
plot(ax, bc, histcounts(A.t(ia) / 1000, be), '-', 'Color', cm, 'LineWidth', 1, 'DisplayName', 'both');
ylabel(ax, 'spikes / 10 s'); xlabel(ax, 'time (s)'); legend(ax, 'Location', 'northeast'); grid(ax, 'on');
ax = subplot(3, 1, 3, 'Parent', f); hold(ax, 'on');
sh = abs(shift) >= dt / 2;
plot(ax, A.t(ia(sh)) / 1000, shift(sh), '.', 'Color', cm, 'MarkerSize', 6);
far = setdiff(keep_max, keep); fa = A.t(pa(far)) / 1000;
plot(ax, fa, pd(far), '.', 'Color', [0.8 0.1 0.1], 'MarkerSize', 8);
yline(ax, [-o.tol o.tol], '--k');
xlabel(ax, 'time (s)'); ylabel(ax, sprintf('shift %s - %s (ms)', nb, na)); grid(ax, 'on'); ylim(ax, [-o.max_shift o.max_shift]);
title(ax, sprintf('shifted pairs over time: grey = matched (inside tol), red = %d more pairs only a tolerance up to %.1f ms would accept', ...
    numel(far), o.max_shift));
sgtitle(f, sprintf('%s vs %s -- firing diagram', na, nb), 'FontWeight', 'bold', 'Interpreter', 'none');
print(f, fullfile(o.out_dir, 'detections_timeline.png'), '-dpng', '-r110'); close(f);

% ---- pages 3-5 (with 'raw'): WHY each unique spike is missing from the other run ----
% Each run's detection is replayed step by step from the raw file with its own par (bandpass,
% order, notch = par.process_info when par.preprocessing, threshold per segment, 1.5 ms dead time,
% peak search, stdmax artifact check) -- the wave_clus / codes_emu amp_detect logic. For every spike
% only one run has, the OTHER run's own log says exactly what it did there.
if ~isempty(o.raw)
    x = read_raw(o.raw);
    La = replay_detection(x, A.par, sr); Lb = replay_detection(x, B.par, sr);
    for q = {{La, A, na}, {Lb, B, nb}}
        [L, S, nm] = q{1}{:};
        rep_ = mean(ismember(round(S.t * sr / 1000), L.idx(~L.art)));
        fprintf('[compare_detections] %s replay: thresholds %s (file %s); reproduces %.2f%% of its %d spikes\n', ...
            nm, mat2str(L.thr', 4), mat2str(S.thr', 4), 100 * rep_, numel(S.t));
    end
    tol_s = round(o.tol * sr / 1000);
    [ka, da, pka] = why_missing(round(A.t(only_a) * sr / 1000), Lb, tol_s, sr);
    [kb, db, pkb] = why_missing(round(B.t(only_b) * sr / 1000), La, tol_s, sr);
    cause_names = cause_list();
    R.why_only_a = ka; R.why_only_b = kb;
    T = [table(repmat(string(na), numel(only_a), 1), A.t(only_a), string(cause_names(ka))', da, pka, ...
               'VariableNames', {'only_in', 't_ms', 'why_missing_in_other', 'detail_ms', 'other_signal_peak_uV'}); ...
         table(repmat(string(nb), numel(only_b), 1), B.t(only_b), string(cause_names(kb))', db, pkb, ...
               'VariableNames', {'only_in', 't_ms', 'why_missing_in_other', 'detail_ms', 'other_signal_peak_uV'})];
    writetable(T, fullfile(o.out_dir, 'detections_why_missing.csv'));
    for q = {{ka, na, nb}, {kb, nb, na}}
        [kk, n1, n2] = q{1}{:};
        fprintf('[compare_detections] why %s lacks %s-only (%d): %s\n', n2, n1, numel(kk), strjoin(arrayfun(@(k) ...
            sprintf('%s %d', cause_names{k}, nnz(kk == k)), find(accumarray(kk, 1, [numel(cause_names) 1]))', 'UniformOutput', false), ' | '));
    end
    save(fullfile(o.out_dir, 'detections_match.mat'), 'ka', 'kb', 'da', 'db', 'cause_names', '-append');
    ccol = [0.12 0.47 0.71; 1.00 0.50 0.05; 0.58 0.40 0.74; 0.84 0.15 0.16; 0.55 0.34 0.29; 0.50 0.50 0.50; 0 0 0];

    % ---- page 3: the unique spikes in both runs' detection signals ----
    ea_a = epochs(La.xd, round(A.t(only_a) * sr / 1000)); ea_b = epochs(Lb.xd, round(A.t(only_a) * sr / 1000));
    eb_b = epochs(Lb.xd, round(B.t(only_b) * sr / 1000)); eb_a = epochs(La.xd, round(B.t(only_b) * sr / 1000));
    f = figure('Visible', 'off', 'Position', [50 50 1900 1000], 'Color', 'w');
    rows = {only_a, A, ea_a, La, ea_b, Lb, ka, na, nb, ca, cb; only_b, B, eb_b, Lb, eb_a, La, kb, nb, na, cb, ca};
    allW = [ea_a; ea_b; eb_a; eb_b]; yl = prctile(allW(:), [0.5 99.5]); yl = yl + 0.15 * diff(yl) * [-1 1];
    if isempty(allW) || ~all(isfinite(yl)) || diff(yl) <= 0, yl = [-100 50]; end   % no unique spikes on either side
    rs = RandStream('mt19937ar', 'Seed', 0);
    for r = 1:2
        [idx, S, E_own, L_own, E_oth, L_oth, kk, n1, n2, c1, c2] = rows{r, :};
        pnl = {S.W(idx, :), sprintf('%s only (n=%d): its stored waveforms', n1, numel(idx)), c1, []; ...
               E_own, sprintf('same spikes in %s''s detection signal', n1), c1, -mean(L_own.thr); ...
               E_oth, sprintf('same times in %s''s detection signal', n2), c2, -mean(L_oth.thr)};
        for q = 1:3
            ax = subplot(2, 4, 4 * (r - 1) + q, 'Parent', f); hold(ax, 'on');
            W = pnl{q, 1};
            if ~isempty(W)
                pick = randperm(rs, size(W, 1), min(size(W, 1), 5000));
                plot_traces(ax, W(pick, :), pnl{q, 3}, 0.15);
                plot(ax, mean(W, 1), 'Color', 'w', 'LineWidth', 3); plot(ax, mean(W, 1), 'Color', 'k', 'LineWidth', 1.5);
            end
            if ~isempty(pnl{q, 4}), yline(ax, pnl{q, 4}, '--', 'Color', pnl{q, 3}, 'LineWidth', 1.2); end
            ylim(ax, yl); xlim(ax, [1 64]); grid(ax, 'on'); xlabel(ax, 'sample (spike at 20)'); ylabel(ax, 'uV');
            title(ax, pnl{q, 2}, 'Interpreter', 'none', 'FontSize', 9);
        end
        ax = subplot(2, 4, 4 * r, 'Parent', f); axis(ax, 'off');
        cnt = accumarray(kk, 1, [numel(cause_names) 1]);
        lines_ = [{sprintf('why %s does not have them', n2), '(its own detection, replayed)', ''}, ...
                  arrayfun(@(k) sprintf('%5d  %s', cnt(k), cause_names{k}), find(cnt)', 'UniformOutput', false)];
        text(ax, -0.1, 0.85, lines_, 'FontSize', 9.5, 'VerticalAlignment', 'top', 'Interpreter', 'none', 'FontName', 'FixedWidth');
    end
    sgtitle(f, sprintf(['%s vs %s -- the spikes only one run has, in both runs'' detection signals ' ...
        '(dashed = that run''s threshold)'], na, nb), 'FontWeight', 'bold', 'Interpreter', 'none');
    print(f, fullfile(o.out_dir, 'detections_why_missing.png'), '-dpng', '-r110'); close(f);

    % ---- page 4: where in the recording, by cause, and how far each one was from the other run's threshold ----
    f = figure('Visible', 'off', 'Position', [50 50 1800 1100], 'Color', 'w');
    grp = {A.t(only_a), ka, pka, Lb, sprintf('%s only (%d)', na, numel(only_a)), nb; ...
           B.t(only_b), kb, pkb, La, sprintf('%s only (%d)', nb, numel(only_b)), na};
    for g = 1:2
        [tt, kk, pk, Lo, ttl, n2] = grp{g, :};
        present = find(accumarray(kk, 1, [numel(cause_names) 1]))';
        top = 0.93 - (g - 1) * 0.47;
        ax = axes(f, 'Position', [0.25 top - 0.17 0.72 0.15]); hold(ax, 'on');
        for j = 1:numel(present)
            xx = tt(kk == present(j))' / 1000;
            plot(ax, [xx; xx; nan(size(xx))], repmat([j - 0.35; j + 0.35; nan], 1, numel(xx)), '-', 'Color', ccol(present(j), :), 'LineWidth', 0.8);
        end
        set(ax, 'YTick', 1:numel(present), 'YTickLabel', arrayfun(@(k) sprintf('%s (%d)', cause_names{k}, nnz(kk == k)), present, ...
            'UniformOutput', false), 'YLim', [0.4 numel(present) + 0.6], 'TickLabelInterpreter', 'none'); box(ax, 'off'); grid(ax, 'on');
        title(ax, sprintf('%s: why %s does not have them, along the recording', ttl, n2), 'Interpreter', 'none'); xlim(ax, [0 tend]);
        ax = axes(f, 'Position', [0.25 top - 0.40 0.72 0.18]); hold(ax, 'on');
        for j = 1:numel(present)
            m = kk == present(j);
            scatter(ax, tt(m) / 1000, pk(m), 8, ccol(present(j), :), 'filled', 'DisplayName', cause_names{present(j)});
        end
        stairs(ax, ((0:numel(Lo.thr)) * Lo.seg) / sr, -[Lo.thr; Lo.thr(end)], 'k', 'LineWidth', 1.2, 'DisplayName', sprintf('%s threshold', n2));
        xlabel(ax, 'time (s)'); ylabel(ax, sprintf('peak in %s''s signal (uV)', n2), 'Interpreter', 'none'); grid(ax, 'on'); xlim(ax, [0 tend]);
        legend(ax, 'Location', 'southwest', 'Interpreter', 'none', 'FontSize', 8);
    end
    sgtitle(f, sprintf('%s vs %s -- the spikes only one run has, by cause', na, nb), 'FontWeight', 'bold', 'Interpreter', 'none');
    print(f, fullfile(o.out_dir, 'detections_why_timeline.png'), '-dpng', '-r110'); close(f);

    % ---- page 5: example sections, every cause that is not plain threshold shown in full (up to 4) ----
    ex = zeros(0, 3);                                   % [direction, row in only list, cause]
    for g = 1:2
        kk = grp{g, 2};
        for k = find(accumarray(kk, 1, [numel(cause_names) 1]))'
            m = find(kk == k); nshow = min(numel(m), 2 + 2 * (k ~= 1));
            ex = [ex; repmat([g 0 k], nshow, 1)]; ex(end - nshow + 1:end, 2) = m(round(linspace(1, numel(m), nshow))); %#ok<AGROW>
        end
    end
    f = figure('Visible', 'off', 'Position', [50 50 1600 240 * size(ex, 1) + 100], 'Color', 'w');
    for e_ = 1:size(ex, 1)
        g = ex(e_, 1); tt = grp{g, 1}(ex(e_, 2)); k = ex(e_, 3);
        if g == 1, dd = da(ex(e_, 2)); else, dd = db(ex(e_, 2)); end
        s0 = round(tt * sr / 1000); w = (-150:150);
        w = w(s0 + w >= 1 & s0 + w <= numel(La.xd));
        ax = subplot(size(ex, 1), 1, e_, 'Parent', f); hold(ax, 'on');
        plot(ax, w / sr * 1000, La.xd(s0 + w), '-', 'Color', ca, 'LineWidth', 1);
        plot(ax, w / sr * 1000, Lb.xd(s0 + w), '-', 'Color', cb, 'LineWidth', 1);
        sa = min(numel(La.thr), floor((s0 - 1) / La.seg) + 1); sb = min(numel(Lb.thr), floor((s0 - 1) / Lb.seg) + 1);
        yline(ax, -La.thr(sa), '--', 'Color', ca); yline(ax, -Lb.thr(sb), '--', 'Color', cb);
        yl_ = ylim(ax); y0 = yl_(1);
        ma = abs(A.t - tt) <= 5; mb = abs(B.t - tt) <= 5;
        plot(ax, A.t(ma) - tt, repmat(y0 + 0.08 * diff(yl_), nnz(ma), 1), 'v', 'Color', ca, 'MarkerFaceColor', ca, 'MarkerSize', 6);
        plot(ax, B.t(mb) - tt, repmat(y0 + 0.02 * diff(yl_), nnz(mb), 1), '^', 'Color', cb, 'MarkerFaceColor', cb, 'MarkerSize', 6);
        plot(ax, 0, y0 + 0.16 * diff(yl_), 'p', 'Color', ccol(k, :), 'MarkerFaceColor', ccol(k, :), 'MarkerSize', 12);
        xlim(ax, [-5 5]); grid(ax, 'on');
        own = grp{g, 5}; other = grp{g, 6};
        title(ax, sprintf('%s spike at %.3f s -- why %s misses it: %s%s', own, tt / 1000, other, cause_names{k}, ...
            ternary(isnan(dd), '', sprintf(' (%+.2f ms)', dd))), 'Interpreter', 'none', 'FontSize', 9);
    end
    xlabel(ax, sprintf('ms from the spike (blue = %s signal / spikes v, orange = %s signal / spikes ^, dashed = thresholds)', na, nb));
    print(f, fullfile(o.out_dir, 'detections_why_examples.png'), '-dpng', '-r100'); close(f);
end
fprintf('[compare_detections] -> %s\n', o.out_dir);
end

% --------------------------------------------------------------------------
function S = read_spikes(f)
% f = a file path, or the struct load() returned for one
if isstruct(f), L0 = f; v = fieldnames(f)'; else, w = whos('-file', f); v = {w.name}; L0 = []; end
getv = @(name) get_var(f, L0, name);
if ismember('index', v)
    S.t = double(getv('index')); S.t = S.t(:);
elseif ismember('cluster_class', v)
    cc = getv('cluster_class'); S.t = double(cc(:, 2));
else
    error('compare_detections: input has neither index nor cluster_class');
end
S.W = [];
if ismember('spikes', v)
    W = getv('spikes'); if size(W, 1) == numel(S.t), S.W = double(W); end
end
S.thr = [];
if ismember('threshold', v), S.thr = double(getv('threshold')); S.thr = S.thr(:); end
S.sr = 30000; S.par = struct();
if ismember('par', v), S.par = getv('par'); if isfield(S.par, 'sr'), S.sr = double(S.par.sr); end, end
[S.t, o] = sort(S.t); if ~isempty(S.W), S.W = S.W(o, :); end
end

function x = get_var(f, L0, name)
if ~isempty(L0), x = L0.(name); else, L = load(f, name); x = L.(name); end
end

function [pa, pb, pd] = candidate_pairs(ta, tb, maxd)
% every (a, b) with |tb - ta| <= maxd; ta, tb sorted
pa = zeros(0, 1); pb = pa; pd = pa; j0 = 1; nb = numel(tb);
ca = cell(numel(ta), 1); cbb = ca;
for i = 1:numel(ta)
    while j0 <= nb && tb(j0) < ta(i) - maxd, j0 = j0 + 1; end
    j = j0;
    while j <= nb && tb(j) <= ta(i) + maxd, j = j + 1; end
    if j > j0, ca{i} = repmat(i, j - j0, 1); cbb{i} = (j0:j - 1)'; end
end
pa = vertcat(pa, ca{:}); pb = vertcat(pb, cbb{:}); pd = tb(pb) - ta(pa);
end

function keep = greedy(pa, pb, pd, tol)
% one-to-one, closest pairs first, among pairs within tol; returns indices into pa/pb
c = find(abs(pd) <= tol + 1e-9);
[~, o] = sort(abs(pd(c))); c = c(o);
ua = false(max([pa; 0]), 1); ub = false(max([pb; 0]), 1); keep = false(numel(c), 1);
for k = 1:numel(c)
    a = pa(c(k)); b = pb(c(k));
    if ~ua(a) && ~ub(b), ua(a) = true; ub(b) = true; keep(k) = true; end
end
keep = sort(c(keep));
end

function draw_venn(ax, R, na, nb, ca, cb, cm)
hold(ax, 'on'); axis(ax, 'equal'); axis(ax, 'off');
r1 = sqrt(R.n_a); r2 = sqrt(R.n_b); s = 1 / max([r1 r2 1]); r1 = r1 * s; r2 = r2 * s;
d = (r1 + r2) * max(0.15, 1 - R.n_both / max(min(R.n_a, R.n_b), 1));
th = linspace(0, 2 * pi, 200);
fill(ax, -d/2 + r1 * cos(th), r1 * sin(th), ca, 'FaceAlpha', 0.25, 'EdgeColor', ca, 'LineWidth', 1.5);
fill(ax, d/2 + r2 * cos(th), r2 * sin(th), cb, 'FaceAlpha', 0.25, 'EdgeColor', cb, 'LineWidth', 1.5);
u = max(R.n_a + R.n_b - R.n_both, 1);
text(ax, -d/2 - r1/2, 0, sprintf('%d\n(%.1f%%)', R.n_only_a, 100 * R.n_only_a / u), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(ax, d/2 + r2/2, 0, sprintf('%d\n(%.1f%%)', R.n_only_b, 100 * R.n_only_b / u), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(ax, (r1 - r2) / 2 * 0, 0, sprintf('%d\n(%.1f%%)\nboth', R.n_both, 100 * R.n_both / u), 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Color', cm * 0.6);
text(ax, -d/2, r1 + 0.12, sprintf('%s  n=%d', na, R.n_a), 'HorizontalAlignment', 'center', 'Color', ca, 'FontWeight', 'bold', 'Interpreter', 'none');
text(ax, d/2, -r2 - 0.12, sprintf('%s  n=%d', nb, R.n_b), 'HorizontalAlignment', 'center', 'Color', cb, 'FontWeight', 'bold', 'Interpreter', 'none');
end

function d = nearest_signed(t, other)
% signed distance (other - t) to the nearest spike of OTHER, for every t
d = nan(size(t)); if isempty(other) || isempty(t), return; end
j = discretize(t, [-Inf; other(:); Inf]);           % other(j-1) <= t < other(j)
lo = max(j - 1, 1); hi = min(j, numel(other));
dl = other(lo) - t; dh = other(hi) - t;
d = dl; pick = abs(dh) < abs(dl); d(pick) = dh(pick);
end

function x = read_raw(f)
fid = fopen(f, 'r', 'l'); x = fread(fid, Inf, 'int16=>double') / 4; fclose(fid);
end

function c = cause_list()
c = {'threshold: never crosses it here', 'refractory period (1.5 ms after its own spike)', ...
     'crossed, but its peak was placed > tol away', 'artifact check removed it (> stdmax)', ...
     'its spike here is paired with another one (one-to-one)', 'segment edge (not searched)', 'unresolved'};
end

function L = replay_detection(x, par, sr)
% wave_clus amp_detect (incl. the filt_signal notch cascade: [process_info.SOS; tf2sos(ellip)], gain
% G, fast_filtfilt), segment by segment, keeping a log: every accepted detection
% (crossing sample, index, removed by the artifact check) and every below-threshold sample that the
% dead time skipped (with the detection that blocked it). Global 1-based samples (= index ms * sr/1000).
for f = {'segments_length', 'w_pre', 'w_post', 'ref_ms', 'detect_order', 'detect_fmin', 'detect_fmax', ...
         'sort_order', 'sort_fmin', 'sort_fmax', 'stdmin', 'stdmax'}      % python's par stores some as integers
    if isfield(par, f{1}), par.(f{1}) = double(par.(f{1})); end
end
seg = round(par.segments_length * 60 * sr); n = numel(x);
w_pre = par.w_pre; w_post = par.w_post; ref = floor(par.ref_ms * sr / 1000); sref = floor(ref / 2);
[b, a] = ellip(par.detect_order, 0.1, 40, [par.detect_fmin par.detect_fmax] * 2 / sr);
[b2, a2] = ellip(par.sort_order, 0.1, 40, [par.sort_fmin par.sort_fmax] * 2 / sr);
nt = isfield(par, 'preprocessing') && par.preprocessing && isfield(par, 'process_info') && ~isempty(par.process_info);
if nt
    [sd, gd] = tf2sos(b, a); sd = [par.process_info.SOS; sd]; gd = gd * par.process_info.G;
    [ss, gs] = tf2sos(b2, a2); ss = [par.process_info.SOS; ss]; gs = gs * par.process_info.G;
end
% same filtering call as wave_clus filt_signal: fast_filtfilt (MCWs useful_functions) when on the path
if exist('fast_filtfilt', 'file') ~= 2
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'useful_functions'));
end
if exist('fast_filtfilt', 'file') == 2, ff = @fast_filtfilt; else, ff = @filtfilt; end
L.xd = zeros(n, 1, 'single'); L.seg = seg; nsg = ceil(n / seg); L.thr = zeros(nsg, 1); L.edge = false(n, 1);
idx = cell(nsg, 1); cr = idx; art = idx; bs = idx; bb = idx;
for k = 1:nsg
    r0 = (k - 1) * seg; r = r0 + 1:min(k * seg, n); xx = x(r);
    if nt, xd = ff(sd, gd, xx); xs = ff(ss, gs, xx); else, xd = ff(b, a, xx); xs = ff(b2, a2, xx); end
    thr = par.stdmin * median(abs(xd)) / 0.6745; thrmax = par.stdmax * median(abs(xs)) / 0.6745;
    L.thr(k) = thr; L.xd(r) = xd;
    m = numel(xd); L.edge(r0 + [1:w_pre + 1, m - w_post - 2 - sref + 1:m]) = true;
    xaux = find(xd(w_pre + 2:end - w_post - 2 - sref) < -thr) + w_pre + 1;
    I = zeros(numel(xaux), 1); C = I; nI = 0; skip = false(numel(xaux), 1); blk = zeros(numel(xaux), 1); last = 0;
    for i = 1:numel(xaux)
        if xaux(i) >= last + ref
            [~, iaux] = min(xs(xaux(i):xaux(i) + sref - 1));
            nI = nI + 1; I(nI) = iaux + xaux(i) - 1; C(nI) = xaux(i); last = I(nI);
        else
            skip(i) = true; blk(i) = last;
        end
    end
    I = I(1:nI); C = C(1:nI);
    xs2 = [xs; zeros(w_post, 1)];
    A_ = arrayfun(@(i) max(abs(xs2(max(i - w_pre, 1):i + w_post))) >= thrmax, I);
    idx{k} = r0 + I; cr{k} = r0 + C; art{k} = A_(:); bs{k} = r0 + xaux(skip); bb{k} = r0 + blk(skip);
end
L.idx = vertcat(idx{:}); L.cross = vertcat(cr{:}); L.art = logical(vertcat(art{:}));
L.blocked = vertcat(bs{:}); L.blocker = vertcat(bb{:});
end

function [kind, detail, pk] = why_missing(s, L, tol_s, sr)
% s = global samples of spikes the run L does not have. Looks at L's own log around each one.
kind = 7 * ones(numel(s), 1); detail = nan(numel(s), 1); pk = nan(numel(s), 1);
pre = 22; post = 15;                           % a crossing up to 22 samples before the peak can own it
for i = 1:numel(s)
    w = max(s(i) - pre, 1):min(s(i) + post, numel(L.xd));
    pk(i) = min(L.xd(w));
    if any(L.edge(w)), kind(i) = 6; continue; end
    seg = min(numel(L.thr), floor((s(i) - 1) / L.seg) + 1);
    if pk(i) > -L.thr(seg), kind(i) = 1; continue; end
    started = find(L.cross >= w(1) & L.cross <= w(end));
    if any(started)
        near_ = started(abs(L.idx(started) - s(i)) <= tol_s);
        if ~isempty(near_) && all(L.art(near_)), kind(i) = 4; detail(i) = (L.idx(near_(1)) - s(i)) / sr * 1000; continue; end
        if ~isempty(near_), kind(i) = 5; detail(i) = (L.idx(near_(1)) - s(i)) / sr * 1000; continue; end
        [~, j] = min(abs(L.idx(started) - s(i))); j = started(j);
        if L.art(j), kind(i) = 4; else, kind(i) = 3; end
        detail(i) = (L.idx(j) - s(i)) / sr * 1000; continue;
    end
    bl = find(L.blocked >= w(1) & L.blocked <= w(end), 1);
    if ~isempty(bl), kind(i) = 2; detail(i) = (L.blocker(bl) - s(i)) / sr * 1000; continue; end
end
end

function E = epochs(xd, s)
s = s(s > 19 & s + 44 <= numel(xd)); E = double(xd(s + (-19:44)));
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end

function s = safe(s)
s = regexprep(s, '[^\w]+', '_');
end
