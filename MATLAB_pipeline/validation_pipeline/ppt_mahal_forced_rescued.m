function ppt_mahal_forced_rescued(session, multi)
% PPT_MAHAL_FORCED_RESCUED  How python's matched class was built: spikes the Mahalanobis step moved in or out
% of it, and its forced / rescued spikes -> <for ppt>/10_mahal_forced_rescued/.
%
% Python (MicroWireSorter force_membership) first accepts a spike by Euclidean template matching (inside a
% class's radius template_sdnum * sqrt(sum per-sample variance) around its mean waveform, the nearest such
% class wins), then -- mahal_reassign -- moves it to the class with the smallest feature-space Mahalanobis
% distance AMONG THE CLASSES WHOSE RADIUS ACCEPTED IT (shrunk covariance 0.7*full + 0.3*diag). It never sends a
% spike to cluster 0. The times file flags moved spikes (mahal_reassigned) but does not store where they came
% from, so the ORIGIN IS ESTIMATED here: templates / radii / feature models are rebuilt from python's own
% clustered (not forced, not quarantined) spikes, and the origin = the Euclidean-nearest accepting class.
% Check of the estimate: the share of moved spikes whose final class is also the Mahalanobis winner among
% the accepting classes (printed in the figure / counts.csv). Origin = final class means "not resolved".
%
% Outputs:
%   mahal_transitions.png     origin (estimated) -> final class counts of the moved spikes, forcing / rescue pass
%   waveforms_mahal.png       spikes the matched class LOST (by destination) and GAINED (by origin), templates on
%   mahal_distances.png       lost spikes: Euclidean distance (in radii) and Mahalanobis d2, origin vs final
%   where_in_wave_clus_gt.png lost / gained / forced / rescued / clustered spikes of the class: where they are
%                             in wave_clus notch and in GT
%   waveforms_forced_rescued.png  the class split into clustered / forced / rescued spikes
%   <subset>/ rasters (python's own full-run windows, + default-window version), counts.csv, moved_spikes.csv
%   ppt_mahal_forced_rescued('pt4emu7_ch322_class3', false)
if nargin < 2, multi = false; end
here = fileparts(mfilename('fullpath')); addpath(here, fullfile(fileparts(here), 'real_response_analysis'));
cfg = validation_sessions(session); g = cfg.gt.class; root = cfg.root; tf = cfg.gt.times_file; ch = cfg.ch;
pp = ppt_paths(cfg, multi); rep = pp.rep;
out = fullfile(rep, '10_mahal_forced_rescued'); if ~exist(out, 'dir'), mkdir(out); end
M = load(fullfile(rep, sprintf('membership_ch%d.mat', ch)));
py = pp.py;
T = load(fullfile(root, py, tf), 'features', 'spikes', 'par');
P = M.python; W = M.waves.python; X = T.features; c = P.class; py_c = M.classes.python; wc_c = M.classes.wc_notch;
K = max(c); sdnum = getfield_or(T.par, 'template_sdnum', 3);

% ---- rebuild python's templates, radii and feature models from its clustered spikes ----
core = c > 0 & ~P.forced & ~P.quarantined;
tmpl = nan(K, size(W, 2)); rad = nan(K, 1); mu = nan(K, size(X, 2)); icov = nan(size(X, 2), size(X, 2), K);
for k = 1:K
    m = core & c == k; if nnz(m) < 2, continue; end
    tmpl(k, :) = mean(W(m, :), 1); rad(k) = sdnum * sqrt(sum(var(W(m, :), 1, 1)));
    mu(k, :) = mean(X(m, :), 1); C = cov(X(m, :)); C = 0.7 * C + 0.3 * diag(diag(C)) + 1e-6 * eye(size(C));
    icov(:, :, k) = pinv(C);
end

% ---- moved spikes: estimated origin, distances, and the self-check ----
mv = find(P.mahal_reassigned > 0);
de = pdist2(W(mv, :), tmpl);                                % Euclidean distance to every template
acc = de < rad';                                            % classes whose radius accepts the spike
d2 = nan(numel(mv), K);
for k = 1:K, if all(isfinite(mu(k, :))), z = X(mv, :) - mu(k, :); d2(:, k) = sum((z * icov(:, :, k)) .* z, 2); end, end
deA = de; deA(~acc) = inf; [~, orig] = min(deA, [], 2); orig(all(~acc, 2)) = 0;   % 0 = no class accepts it
d2A = d2; d2A(~acc) = inf; [~, mwin] = min(d2A, [], 2);
fin = c(mv); passname = repmat("forcing", numel(mv), 1); passname(P.rescued(mv)) = "rescue";
consistent = mean(mwin(orig > 0) == fin(orig > 0));
resolved = orig > 0 & orig ~= fin;
Mv = table(P.t(mv), fin, orig, passname, de(sub2ind(size(de), (1:numel(mv))', max(orig, 1))) ./ rad(max(orig, 1)), ...
    de(sub2ind(size(de), (1:numel(mv))', max(fin, 1))) ./ rad(max(fin, 1)), ...
    d2(sub2ind(size(d2), (1:numel(mv))', max(orig, 1))), d2(sub2ind(size(d2), (1:numel(mv))', max(fin, 1))), ...
    P.wc_notch_class(mv), P.gt_class(mv), 'VariableNames', {'t', 'final_class', 'origin_est', 'pass', ...
    'euclid_origin_radii', 'euclid_final_radii', 'mahal_d2_origin', 'mahal_d2_final', 'wc_notch_class', 'gt_class'});
writetable(Mv, fullfile(out, 'moved_spikes.csv'));
lost = mv(resolved & orig == py_c & fin ~= py_c); gain = mv(resolved & fin == py_c & orig ~= py_c);

% ---- 1. transition matrix ----
f = figure('Visible', 'off', 'Position', [50 50 1100 480], 'Color', 'w');
for p = 1:2
    nm = ["forcing", "rescue"]; s = passname == nm(p);
    A = accumarray([orig(s) + 1, fin(s)], 1, [K + 1, K]);
    ax = subplot(1, 2, p); imagesc(ax, A); colormap(ax, flipud(gray)); axis(ax, 'equal', 'tight');
    for i = 1:K + 1, for j = 1:K, if A(i, j), text(ax, j, i, num2str(A(i, j)), 'HorizontalAlignment', 'center', ...
            'Color', ternary(A(i, j) > max(A(:)) / 2, 'w', 'k'), 'FontWeight', ternary(i - 1 == py_c | j == py_c, 'bold', 'normal')); end, end, end
    set(ax, 'XTick', 1:K, 'YTick', 1:K + 1, 'YTickLabel', [{'none'}, arrayfun(@(k) sprintf('c%d', k), 1:K, 'UniformOutput', false)], ...
        'XTickLabel', arrayfun(@(k) sprintf('c%d', k), 1:K, 'UniformOutput', false));
    xlabel(ax, 'final class (after Mahalanobis)'); ylabel(ax, 'origin (estimated: Euclidean choice)');
    title(ax, sprintf('%s pass: %d moved (diagonal = origin not resolved)', nm(p), nnz(s)));
end
sgtitle(f, {sprintf('ch%d python: Mahalanobis-moved spikes (matched class c%d in bold)', ch, py_c), ...
    sprintf('origin estimate check: final = Mahalanobis winner among accepting classes for %.1f%%', 100 * consistent)}, 'FontWeight', 'bold');
print(f, fullfile(out, 'mahal_transitions.png'), '-dpng', '-r110'); close(f);

% ---- 2. waveforms lost / gained ----
tm = struct('name', {sprintf('python c%d', py_c), sprintf('GT c%d', g), sprintf('wave_clus notch c%d', wc_c)}, ...
    'wave', {M.templates.python(py_c, :), M.templates.gt(g, :), M.templates.wc_notch(wc_c, :)}, ...
    'color', {[.9 .6 0], [0 0 0], [0 .45 .7]}, 'style', {'-', '--', ':'});
grp = struct('name', {sprintf('LOST from c%d (all)', py_c), '  forcing pass', '  rescue pass'}, ...
    'W', {W(lost, :), W(lost(~P.rescued(lost)), :), W(lost(P.rescued(lost)), :)}, 'color', {[.8 .15 .15], [.6 .1 .1], [.45 .05 .05]});
for k = setdiff(unique(c(lost))', py_c)
    grp(end + 1) = struct('name', sprintf('  ... moved c%d -> c%d', py_c, k), 'W', W(lost(c(lost) == k), :), 'color', [.6 .1 .1]); %#ok<AGROW>
end
grp(end + 1) = struct('name', sprintf('GAINED into c%d (all)', py_c), 'W', W(gain, :), 'color', [.2 .6 .2]);
grp(end + 1) = struct('name', '  forcing pass', 'W', W(gain(~P.rescued(gain)), :), 'color', [.1 .4 .1]);
grp(end + 1) = struct('name', '  rescue pass', 'W', W(gain(P.rescued(gain)), :), 'color', [.05 .3 .05]);
og = orig(ismember(mv, gain));
for k = setdiff(unique(og)', [0 py_c])
    grp(end + 1) = struct('name', sprintf('  ... moved c%d -> c%d', k, py_c), 'W', W(gain(og == k), :), 'color', [.1 .4 .1]); %#ok<AGROW>
end
plot_subset_waveforms(grp, tm, sprintf('ch%d: spikes the Mahalanobis step moved out of / into python c%d (origin estimated)', ch, py_c), ...
    fullfile(out, 'waveforms_mahal.png'));

% ---- 3. distances of the lost spikes ----
f = figure('Visible', 'off', 'Position', [50 50 1100 480], 'Color', 'w'); L = ismember(mv, lost);
ax = subplot(1, 2, 1); scatter(ax, Mv.euclid_origin_radii(L), Mv.euclid_final_radii(L), 8, 'filled', 'MarkerFaceAlpha', .4);
hold(ax, 'on'); plot(ax, [0 1], [0 1], 'k--'); xline(ax, 1, ':'); yline(ax, 1, ':'); axis(ax, [0 1.1 0 1.1]); grid(ax, 'on');
xlabel(ax, sprintf('Euclidean distance to c%d (origin) / its radius', py_c)); ylabel(ax, 'to final class / its radius');
title(ax, 'above the diagonal: Euclidean preferred the origin');
ax = subplot(1, 2, 2); scatter(ax, Mv.mahal_d2_origin(L), Mv.mahal_d2_final(L), 8, 'filled', 'MarkerFaceAlpha', .4);
hold(ax, 'on'); lim = prctile([Mv.mahal_d2_origin(L); Mv.mahal_d2_final(L)], 99); if isempty(lim) || ~isfinite(lim), lim = 1; end
plot(ax, [0 lim], [0 lim], 'k--'); axis(ax, [0 lim 0 lim]); grid(ax, 'on');
xlabel(ax, sprintf('Mahalanobis d^2 to c%d (origin)', py_c)); ylabel(ax, 'to final class');
title(ax, 'below the diagonal: Mahalanobis preferred the final class');
sgtitle(f, sprintf('ch%d: the %d spikes moved out of python c%d', ch, numel(lost), py_c), 'FontWeight', 'bold');
print(f, fullfile(out, 'mahal_distances.png'), '-dpng', '-r110'); close(f);

% ---- 4. class composition: clustered / forced / rescued, and where every group is in wave_clus and GT ----
in_c = c == py_c; resc = P.rescued;
sets = {'clustered (not forced)', find(in_c & ~P.forced & ~resc); 'forced', find(in_c & P.forced & ~resc); ...
        'rescued', find(in_c & resc); ...
        'lost to Mahalanobis, forcing pass', lost(~resc(lost)); 'lost to Mahalanobis, rescue pass', lost(resc(lost)); ...
        'gained by Mahalanobis, forcing pass', gain(~resc(gain)); 'gained by Mahalanobis, rescue pass', gain(resc(gain))};
f = figure('Visible', 'off', 'Position', [50 50 1200 460], 'Color', 'w');
for s = 1:2
    other = ternary(s == 1, P.wc_notch_class, P.gt_class); oc = ternary(s == 1, wc_c, g);
    B = zeros(size(sets, 1), 3);
    for i = 1:size(sets, 1)
        o = other(sets{i, 2}); B(i, :) = [nnz(o == oc), nnz(~isnan(o) & o ~= oc), nnz(isnan(o))];
    end
    ax = subplot(1, 2, s); b = bar(ax, 100 * B ./ max(sum(B, 2), 1), 'stacked'); grid(ax, 'on');
    set(ax, 'XTickLabel', cellfun(@(x, y) sprintf('%s (%d)', x, numel(y)), sets(:, 1), sets(:, 2), 'UniformOutput', false), 'XTickLabelRotation', 25);
    ylabel(ax, '% of the group'); legend(b, {sprintf('in %s c%d', ternary(s == 1, 'wave_clus', 'GT'), oc), 'other class / cluster 0', 'no spike within tol'}, ...
        'Location', 'southoutside', 'Interpreter', 'none');
    title(ax, sprintf('where python c%d''s spikes are in %s', py_c, ternary(s == 1, 'wave_clus notch', 'GT')), 'Interpreter', 'none');
end
print(f, fullfile(out, 'where_in_wave_clus_gt.png'), '-dpng', '-r110'); close(f);
grp = struct('name', sets(:, 1)', 'W', cellfun(@(r) W(r, :), sets(:, 2)', 'UniformOutput', false), ...
    'color', {[.9 .6 0], [.6 .4 0], [.8 .15 .15], [.6 .1 .1], [.45 .05 .05], [.2 .6 .2], [.1 .4 .1]});
plot_subset_waveforms(grp, tm, sprintf('ch%d: python c%d by how its spikes got there', ch, py_c), fullfile(out, 'waveforms_forced_rescued.png'));

% ---- 5. rasters, python's own full-run windows ----
state = fullfile(pp.wcval, sprintf('validation_wave_clus_vs_python_c%d', py_c));
w = load(fullfile(state, 'validation_state.mat'), 'ref_tables_default'); two = isfield(w, 'ref_tables_default') && ~isempty(w.ref_tables_default);
nm = {'01_clustered', '02_forced', '03_rescued', '04_lost_to_mahal_forcing', '05_lost_to_mahal_rescue', ...
      '06_gained_by_mahal_forcing', '07_gained_by_mahal_rescue'};
for i = 1:numel(nm)
    t = P.t(sets{i, 2}); if isempty(t), continue; end
    plot_subset_rasters([], 'dir', state, 'times', t, 'name', nm{i}, 'side', 'arm', 'out_dir', out);
    if two, plot_subset_rasters([], 'dir', state, 'times', t, 'name', [nm{i} '_default_window'], 'side', 'arm', 'out_dir', out, 'window', 'default'); end
end

% ---- counts ----
cnt = [cellfun(@(x) string(x), sets(:, 1)), cellfun(@numel, sets(:, 2))];
cnt = [cnt; "moved by Mahalanobis (all classes)", numel(mv); "  forcing pass", nnz(passname == "forcing"); ...
       "  rescue pass", nnz(passname == "rescue"); "  origin resolved (origin ~= final)", nnz(resolved); ...
       "  estimate check: final = Mahalanobis winner (%)", sprintf('%.1f', 100 * consistent)];
writetable(array2table(cnt, 'VariableNames', {'group', 'n'}), fullfile(out, 'counts.csv'));
fprintf(['[ppt_mahal_forced_rescued] %s: c%d lost %d (forcing %d, rescue %d) / gained %d (forcing %d, rescue %d) to ' ...
    'Mahalanobis, %d forced, %d rescued (check %.1f%%) -> %s\n'], session, py_c, numel(lost), numel(sets{4, 2}), numel(sets{5, 2}), ...
    numel(gain), numel(sets{6, 2}), numel(sets{7, 2}), numel(sets{2, 2}), numel(sets{3, 2}), 100 * consistent, out);
end

function v = getfield_or(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = double(s.(f)); else, v = d; end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
