function axs = draw_wave_grid(f, rect, groups, tmpls, yl, ncol)
% DRAW_WAVE_GRID  Waveform panels for GROUPS laid out inside RECT
% ([x y w h], figure-normalized) on F: at most NCOL (default 3) per row, as
% many rows as needed. All panels share the fixed y range YL and carry the
% same templates; the first panel holds the legend. Returns the axes.
    if nargin < 6, ncol = 3; end
    n = numel(groups);
    ncol = min(ncol, max(n, 1));
    nrow = max(1, ceil(n / ncol));
    gx = 0.025*rect(3)/0.4; gy = 0.30*rect(4)/nrow;
    w = (rect(3) - (ncol-1)*gx) / ncol;
    h = (rect(4) - (nrow-1)*gy) / nrow;
    axs = gobjects(1, n);
    for i = 1:n
        r = ceil(i / ncol); c = mod(i-1, ncol) + 1;
        pos = [rect(1) + (c-1)*(w+gx), rect(2) + rect(4) - r*h - (r-1)*gy, w, h];
        axs(i) = axes(f, 'Position', pos);
        draw_wave_group(axs(i), groups(i), tmpls, yl, i == 1);
        if r == nrow || i + ncol > n, xlabel(axs(i), 'sample'); end
        if c == 1, ylabel(axs(i), 'amplitude'); end
    end
end
