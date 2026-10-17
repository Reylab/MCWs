function draw_wave_group(ax, g, tmpls, yl, show_legend)
% DRAW_WAVE_GROUP  One waveform panel: the group's single spikes (up to 5000), the
% group's mean (no std band), then the reference templates, on the
% fixed y range YL. g = struct with .name, .W, .color.
    if nargin < 5, show_legend = false; end
    hold(ax, 'on');
    n = size(g.W, 1);
    if n > 0
        rs = RandStream('mt19937ar', 'Seed', 0);
        pick = randperm(rs, n, min(n, 5000));
        plot_traces(ax, g.W(pick,:), g.color, 0.06);
        mu = mean(g.W,1); xx = 1:numel(mu);
        plot(ax, xx, mu, 'Color', g.color, 'LineWidth', 2, 'DisplayName', 'mean (this group)');
    else
        text(ax, 0.5, 0.5, 'no spikes', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    for t = tmpls(:)'
        plot(ax, 1:numel(t.wave), t.wave, t.style, 'Color', t.color, 'LineWidth', 1.6, 'DisplayName', t.name);
    end
    ylim(ax, yl); xlim(ax, [1 numel(tmpls(1).wave)]);
    grid(ax, 'on'); box(ax, 'off'); set(ax, 'FontSize', 8);
    ttl = sprintf('%s (n=%d)', g.name, n);
    if n > 0
        mu = mean(g.W,1);
        if min(mu) < yl(1) || max(mu) > yl(2)
            ttl = [ttl ' -- clipped'];
        end
    end
    title(ax, ttl, 'FontSize', 9, 'Interpreter', 'none');
    if show_legend
        legend(ax, 'show', 'Location', 'southeast', 'FontSize', 7, 'Interpreter', 'none');
    end
end
