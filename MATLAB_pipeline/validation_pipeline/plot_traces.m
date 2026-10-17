function h = plot_traces(ax, W, color, alpha, lw)
% PLOT_TRACES  Draw every row of W (spikes x samples) as faint lines, as ONE
% NaN-separated line object: thousands of separate line objects make
% print() take minutes per page, one object draws in seconds.
if nargin < 5, lw = 0.4; end
h = gobjects(0);
if isempty(W), return; end
[n, m] = size(W);
x = repmat([1:m NaN], 1, n);
y = [W NaN(n, 1)]'; y = y(:)';
h = plot(ax, x, y, 'Color', [color alpha], 'LineWidth', lw, 'HandleVisibility', 'off');
end
