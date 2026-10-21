function ppt_quarantine_pages(pidx, P, T, out, ttl)
% PPT_QUARANTINE_PAGES  Two pages for a set of never-rescued quarantined spikes, split by mask:
%   distance_to_templates_by_mask.png  each spike's Euclidean distance to the NEAREST python class template, in
%       units of that class's rescue radius (rescue_template_sdnum x sqrt(sum per-sample variance), templates and
%       spread from the class's clustered spikes -- not forced, not quarantined). <= 1 = inside a rescue radius.
%       Only bundle-collision / within-channel spikes are rescue candidates (par.rescue_artifact_types).
%   quarantine_values_by_mask.png      the value that quarantined each spike, against the mask's threshold:
%       low amplitude  |trough at sample 20| vs the channel's 5th percentile (qc min_amplitude_percentile)
%       bad width      width (samples) vs [min_width_idx, max_width_idx]
%       prominence     prominence ratio vs final_prominence_ratio_pass (fail <= it)
%       refractory     time to the nearest other detection (ms) vs par.ref_ms
%       opposite peak  smaller of the two side ratios (opposite-polarity / peak-to-peak) vs
%                      qc opposite_peak_ratio_both_sides (fails when BOTH sides are above it)
%   Each panel says how many flagged spikes the reconstructed threshold reproduces.
% PIDX: rows of the python table P (membership) of the spikes; T: the python times file
% (quarantine_properties, par, spikes, cluster_class, forced, quarantine).
if ~exist(out, 'dir'), mkdir(out); end
pq = {'q_low_amplitude', 'low amplitude'; 'q_bad_width', 'bad width'; 'q_bad_prominence_ratio', 'bad prominence ratio'; ...
      'q_refractory', 'refractory'; 'q_structural', 'structural'; 'q_bundle_collision', 'bundle collision'; ...
      'q_opposite_peak', 'opposite peak'};
pq = pq(ismember(pq(:, 1), P.Properties.VariableNames), :);
pq = pq(cellfun(@(c) any(P.(c)(pidx)), pq(:, 1)), :);          % masks with spikes in this set only
if isempty(pq), return; end
qp = T.quarantine_properties; qc = T.par.qc_params;
if iscell(qp), qp = [qp{:}]; end                               % older conversions store it as a cell of structs
getq = @(f) arrayfun(@(s) double_or_nan(s, f), qp(:));
p20 = abs(getq('peak_sample_20')); w = getq('width'); pr = getq('prominence_ratio');
t_all = T.cluster_class(:, 2); [ts, o] = sort(t_all); gap = inf(size(ts));
gap(2:end) = diff(ts); gap(1:end - 1) = min(gap(1:end - 1), diff(ts)); nn = nan(size(t_all)); nn(o) = gap;   % nearest other detection
amp_thr = prctile(p20(~isnan(p20)), double(qc.min_amplitude_percentile));
ref_ms = 1.5; if isfield(T.par, 'ref_ms'), ref_ms = double(T.par.ref_ms); end
opp = min(getq('left_ratio_used'), getq('right_ratio_used'));          % both sides above <=> the smaller one above
opp_thr = 0.5; if isfield(qc, 'opposite_peak_ratio_both_sides'), opp_thr = double(qc.opposite_peak_ratio_both_sides); end

% ---- distance to the nearest class template, in rescue radii ----
cc = T.cluster_class(:, 1); W = double(T.spikes); K = max(cc);
sd = 1; if isfield(T.par, 'rescue_template_sdnum'), sd = double(T.par.rescue_template_sdnum); end
core = cc > 0 & ~logical(T.forced(:)) & ~logical(T.quarantine.all(:));
tm = nan(K, size(W, 2)); rad = nan(K, 1);
for k = 1:K, m = core & cc == k; if nnz(m) > 2, tm(k, :) = mean(W(m, :), 1); rad(k) = sd * sqrt(sum(var(W(m, :), 1, 1))); end, end
D = pdist2(W(pidx, :), tm) ./ rad'; [dmin, kmin] = min(D, [], 2);
elig = {}; if isfield(T.par, 'rescue_artifact_types'), elig = cellstr(string(T.par.rescue_artifact_types)); end
f = figure('Visible', 'off', 'Position', [50 50 1600 420 * ceil(size(pq, 1) / 3)], 'Color', 'w');
for j = 1:size(pq, 1)
    hit = P.(pq{j, 1})(pidx); ax = subplot(ceil(size(pq, 1) / 3), 3, j, 'Parent', f);
    if any(hit)
        histogram(ax, dmin(hit), 'BinEdges', linspace(0, max(4, prctile(dmin(hit), 99)), 50), 'FaceColor', [.8 .3 .3]);
    end
    xline(ax, 1, 'k--', 'rescue radius'); grid(ax, 'on');
    can = ~strcmp(pq{j, 2}, 'refractory');
    title(ax, {sprintf('%s: %d spikes, %.0f%% inside a radius', pq{j, 2}, nnz(hit), 100 * mean(dmin(hit) <= 1)), ...
        ternary(can, '', '(refractory spikes are never rescue candidates)')}, 'FontSize', 9);
    xlabel(ax, 'distance / rescue radius');
end
sgtitle(f, {ttl, sprintf('distance to the nearest python class template, in rescue radii (sdnum %g); rescued types: %s', ...
    sd, strjoin(elig, ', '))}, 'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 10);
print(f, fullfile(out, 'distance_to_templates_by_mask.png'), '-dpng', '-r110'); close(f);

% ---- the value that quarantined each spike vs its threshold ----
V = {'low amplitude', p20, amp_thr, 'below', '|trough at sample 20| (uV)'; ...
     'bad width', w, [double(qc.min_width_idx) double(qc.max_width_idx)], 'outside', 'width (samples)'; ...
     'bad prominence ratio', pr, double(qc.final_prominence_ratio_pass), 'at or below', 'prominence ratio'; ...
     'refractory', nn, ref_ms, 'below', 'time to the nearest other detection (ms)'; ...
     'opposite peak', opp, opp_thr, 'above', 'smaller of the two side ratios (opposite polarity / peak-to-peak)'};
f = figure('Visible', 'off', 'Position', [50 50 2100 800], 'Color', 'w');
for j = 1:size(V, 1)
    ax = subplot(2, 3, j, 'Parent', f); row = strcmp(pq(:, 2), V{j, 1});
    if ~any(row), axis(ax, 'off'); continue; end
    hit = P.(pq{row, 1})(pidx); x = V{j, 2}(pidx(hit)); thr = V{j, 3};
    if isempty(x) || all(isnan(x)), axis(ax, 'off'); title(ax, sprintf('%s: no spikes', V{j, 1})); continue; end
    x = x(~isnan(x)); hi = prctile(x, 99.5); if strcmp(V{j, 1}, 'refractory'), hi = max(3 * thr, prctile(x, 95)); end
    histogram(ax, x(x <= hi), 'NumBins', 50, 'FaceColor', [.8 .3 .3]); hold(ax, 'on'); grid(ax, 'on');
    for th = thr, xline(ax, th, 'k--', 'LineWidth', 1.5); end
    switch V{j, 4}
        case 'below', rep = mean(x < thr(1));
        case 'outside', rep = mean(x < thr(1) | x > thr(end));
        case 'above', rep = mean(x > thr(1));
        otherwise, rep = mean(x <= thr(1));
    end
    title(ax, {sprintf('%s (%d spikes): fails when %s %s', V{j, 1}, nnz(hit), V{j, 4}, mat2str(round(thr, 2))), ...
        sprintf('%.0f%% of these are on the failing side (threshold check)', 100 * rep)}, 'FontSize', 9);
    xlabel(ax, V{j, 5});
end
sgtitle(f, {ttl, 'the value that quarantined each spike vs the threshold (dashed)'}, 'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 10);
print(f, fullfile(out, 'quarantine_values_by_mask.png'), '-dpng', '-r110'); close(f);
end

function v = double_or_nan(s, f)
if isfield(s, f) && ~isempty(s.(f)), v = double(s.(f)); else, v = NaN; end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
