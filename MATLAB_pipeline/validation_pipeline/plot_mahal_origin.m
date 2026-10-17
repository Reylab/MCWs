function rows = plot_mahal_origin(times_path, cls, arm_name, ch, templates, yl, out_png)
% PLOT_MAHAL_ORIGIN  Where the spikes the Mahalanobis step moved into (and
% out of) class CLS came from, read from the run's own template_match
% records (Python times file, one record per template-matching step:
% clustering / rescue / second_pass_rematch / second_pass_rescue). Each
% record holds, per matched spike, the cluster the Euclidean match chose
% (euclid_class), the final cluster, and the squared feature-space
% Mahalanobis distance to every cluster model (mahal_d2), so nothing here is
% estimated. Without template_match (older runs) there is no page.
%
% Page: top row = waveforms of the class's spikes the Mahalanobis step did
% not move, then the moved-in spikes grouped by the cluster the Euclidean
% match had chosen, then the moved-out spikes grouped by where they went;
% bottom = squared Mahalanobis distance of the moved-in spikes to this class
% vs to their Euclidean cluster, and the per-step counts with each cluster
% model's spread (log det covariance) and radius.
%
% rows = {arm, class, step, direction ('into'/'out of'/'before second pass'),
%         other class, n} -- one row per (step, direction, other class).
rows = {};
w = whos('-file', times_path);
if ~any(strcmp({w.name}, 'template_match')), return; end
T = load(times_path, 'template_match', 'spikes', 'cluster_class');
has_pre = any(strcmp({w.name}, 'cluster_class_pre_second_pass'));
if has_pre, P = load(times_path, 'cluster_class_pre_second_pass'); pre = P.cluster_class_pre_second_pass(:, 1); end
W = double(T.spikes); final_all = T.cluster_class(:, 1);
steps = fieldnames(T.template_match);

in_idx = []; in_from = []; in_d_cls = []; in_d_eu = [];
out_idx = []; out_to = [];
info = {};
for s = 1:numel(steps)
    r = T.template_match.(steps{s});
    idx = double(r.idx(:)); eu = double(r.euclid_class(:)); fin = double(r.final_class(:));
    mv = logical(r.mahal_moved(:));
    into = mv & fin == cls; outof = mv & eu == cls & fin ~= cls;
    for c = unique(eu(into))'
        rows(end+1, :) = {arm_name, cls, steps{s}, 'into', c, nnz(into & eu == c)}; %#ok<AGROW>
    end
    for c = unique(fin(outof))'
        rows(end+1, :) = {arm_name, cls, steps{s}, 'out of', c, nnz(outof & fin == c)}; %#ok<AGROW>
    end
    md = double(r.mahal_d2);
    if any(into) && size(md, 2) >= cls
        sub = find(into);
        in_d_cls = [in_d_cls; md(sub + (cls - 1) * size(md, 1))]; %#ok<AGROW>
        in_d_eu = [in_d_eu; md(sub + (eu(sub) - 1) * size(md, 1))]; %#ok<AGROW>
    end
    in_idx = [in_idx; idx(into)]; in_from = [in_from; eu(into)]; %#ok<AGROW>
    out_idx = [out_idx; idx(outof)]; out_to = [out_to; fin(outof)]; %#ok<AGROW>
    K = numel(r.radius);
    ld = nan(1, K); if isfield(r, 'model_logdet'), ld = double(r.model_logdet(:))'; end
    nm = zeros(1, K); if isfield(r, 'model_n'), nm = double(r.model_n(:))'; end
    info{end+1} = sprintf('%s: %d matched, %d moved by Mahalanobis (%d into c%d, %d out of c%d)', ...
        steps{s}, numel(idx), nnz(mv), nnz(into), cls, nnz(outof), cls); %#ok<AGROW>
    info{end+1} = ['   per cluster  radius: ' strjoin(arrayfun(@(k) sprintf('c%d %.0f', k, r.radius(k)), 1:K, 'UniformOutput', false), '  ')]; %#ok<AGROW>
    info{end+1} = ['   log det cov: ' strjoin(arrayfun(@(k) sprintf('c%d %.1f (n=%d)', k, ld(k), nm(k)), 1:K, 'UniformOutput', false), '  ')]; %#ok<AGROW>
end
if has_pre && ~isempty(in_idx)
    for c = unique(pre(in_idx))'
        rows(end+1, :) = {arm_name, cls, 'all steps', 'before second pass', c, nnz(pre(in_idx) == c)}; %#ok<AGROW>
    end
end

% ---- waveform groups ----
cols = lines(8);
moved_any = false(size(final_all)); moved_any([in_idx; out_idx]) = true;
G = struct('name', sprintf('c%d, not moved', cls), 'W', W(final_all == cls & ~moved_any, :), 'color', [0 0 0]);
for c = unique(in_from)'
    G(end+1) = struct('name', sprintf('INTO c%d (Euclid: c%d)', cls, c), ...
        'W', W(in_idx(in_from == c), :), 'color', cols(mod(c - 1, 8) + 1, :)); %#ok<AGROW>
end
for c = unique(out_to)'
    G(end+1) = struct('name', sprintf('OUT of c%d to c%d', cls, c), ...
        'W', W(out_idx(out_to == c), :), 'color', cols(mod(c - 1, 8) + 1, :) * 0.6); %#ok<AGROW>
end

f = figure('Visible', 'off', 'Position', [100 100 max(1400, 330 * numel(G)) 1000], 'Color', 'w');
annotation(f, 'textbox', [0 0.95 1 0.045], 'EdgeColor', 'none', 'FontSize', 13, 'FontWeight', 'bold', ...
    'Interpreter', 'none', 'HorizontalAlignment', 'center', 'String', ...
    sprintf('ch%d  %s class %d -- Mahalanobis reassignment: %d spikes moved in, %d moved out (from the run''s template_match records)', ...
    ch, arm_name, cls, numel(in_idx), numel(out_idx)));
ng = numel(G); wd = 0.92 / ng;
for i = 1:ng
    draw_wave_group(axes(f, 'Position', [0.04 + (i - 1) * wd, 0.55, wd - 0.03, 0.34]), G(i), templates, yl, i == 1);
end
ax = axes(f, 'Position', [0.06 0.08 0.40 0.36]); hold(ax, 'on');
if ~isempty(in_d_cls)
    e = logspace(0, log10(max([in_d_cls; in_d_eu; 10])), 70);
    histogram(ax, in_d_cls, e, 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', [0.1 0.3 0.8], ...
        'DisplayName', sprintf('moved-in spikes -> class %d', cls));
    histogram(ax, in_d_eu, e, 'DisplayStyle', 'stairs', 'LineWidth', 1.8, 'EdgeColor', [0.85 0.2 0.2], ...
        'DisplayName', 'same spikes -> their Euclidean cluster');
    set(ax, 'XScale', 'log'); legend(ax, 'Location', 'northeast', 'FontSize', 8);
end
xlabel(ax, 'squared Mahalanobis distance (feature space)'); ylabel(ax, 'spikes'); grid(ax, 'on');
title(ax, {'moved-in spikes: the step takes the SMALLER distance (no threshold),', ...
           'among the clusters whose Euclidean radius the spike is inside'}, 'FontSize', 9);
annotation(f, 'textbox', [0.50 0.06 0.48 0.40], 'EdgeColor', 'none', 'FontSize', 8.5, 'Interpreter', 'none', ...
    'FontName', 'FixedWidth', 'VerticalAlignment', 'top', 'String', info);
print(f, out_png, '-dpng', '-r120');
close(f);
end
