function plot_wave_page(groups, tmpls, yl, ttl, out_png)
% PLOT_WAVE_PAGE  A standalone page of waveform panels (draw_wave_grid, 3
% per row), fixed y range YL and the same templates on every panel.
    n = numel(groups);
    nrow = max(1, ceil(n / 3));
    f = figure('Visible', 'off', 'Position', [100 100 1500 120 + 330*nrow], 'Color', 'w');
    draw_wave_grid(f, [0.06 0.06 0.90 0.84 - 0.02*(nrow == 1)], groups, tmpls, yl, 3);
    annotation(f, 'textbox', [0 0.93 1 0.06], 'String', ttl, 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'EdgeColor', 'none', 'Interpreter', 'none');
    out_dir = fileparts(out_png);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    print(f, out_png, '-dpng', '-r130');
    close(f);
end
