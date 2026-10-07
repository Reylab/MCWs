function plot_mask_bar(ax, mb)
% PLOT_MASK_BAR  Spike counts by the quarantine property they failed, on AX,
% as overlapping bars: one bar per denominator, all at the same x, widest
% (smallest denominator) behind. Every series shares the numerator
% mb.counts, so with denominators in increasing order the bars always nest.
% Callers pass exclusive counts (one mask per spike, or 'multiple masks'),
% so the first series sums to 100.
%
%   mb.labels  - property names
%   mb.active  - (optional) logical per property: did it feed this run's
%                quarantine? Inactive ones are labeled 'off' instead of 0
%   mb.counts  - spikes of the group being broken down that failed each property
%   mb.series  - struct array, denominators in increasing order:
%                  .name  - legend text (n is appended)
%                  .tag   - short prefix for the per-bar value labels
%                  .n     - denominator
%                  .color - face color
%   mb.title   - title text (cell or char)
    hold(ax, 'on');
    S = mb.series;
    if isempty(mb.labels) || S(1).n == 0
        axis(ax, 'off');
        text(ax, 0.5, 0.5, 'no spikes', 'HorizontalAlignment', 'center', 'Units', 'normalized');
        title(ax, mb.title, 'FontSize', 10, 'Interpreter', 'none');
        return
    end
    x = 1:numel(mb.labels);
    c = mb.counts(:)';
    P = zeros(numel(S), numel(x));
    widths = linspace(0.78, 0.26, numel(S));
    h = gobjects(1, numel(S));
    for si = 1:numel(S)
        P(si,:) = 100*c / max(S(si).n, 1);
        if si == 1
            h(si) = bar(ax, x, P(si,:), widths(si), 'FaceColor', S(si).color, 'EdgeColor', S(si).color*0.75);
        else
            h(si) = bar(ax, x, P(si,:), widths(si), 'FaceColor', S(si).color, 'EdgeColor', 'none');
        end
    end
    top = max(P, [], 1);
    y_max = min(100, max(top))*(1.2 + 0.12*numel(S)) + 3;
    if ~isfield(mb, 'active') || isempty(mb.active), mb.active = true(size(x)); end
    for i = x
        if ~mb.active(i)
            lbl = 'off';
        elseif c(i) == 0
            lbl = '0';
        else
            parts = arrayfun(@(si) sprintf('%s %.1f%%', S(si).tag, P(si,i)), 1:numel(S), 'UniformOutput', false);
            parts{1} = sprintf('%s (%d)', parts{1}, c(i));
            lbl = strjoin(parts, newline);
        end
        text(ax, i, top(i) + 0.02*y_max, lbl, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    short = strrep(mb.labels, 'bad prominence ratio', 'prom. ratio');
    short = strrep(short, 'low amplitude', 'low amp.');
    short = strrep(short, 'bundle collision', 'bundle');
    short = strrep(short, 'multiple masks', 'multiple');
    short = strrep(short, 'task-specific', 'task');
    short = strrep(short, 'refractory', 'refract.');
    set(ax, 'XTick', x, 'XTickLabel', short, 'FontSize', 10, 'XLim', [0.4 numel(x)+0.6], 'XTickLabelRotation', 0);
    ylim(ax, [0 y_max]); grid(ax, 'on'); box(ax, 'off');
    ylabel(ax, '%');
    legend(ax, h, arrayfun(@(s) sprintf('%s (n=%d)', s.name, s.n), S, 'UniformOutput', false), ...
        'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 10, 'Interpreter', 'none', 'Box', 'off');
    title(ax, mb.title, 'FontSize', 12, 'Interpreter', 'none');
end
