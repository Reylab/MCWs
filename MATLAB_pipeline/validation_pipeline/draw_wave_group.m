function draw_wave_group(ax, g, tmpls, yl, show_legend)
% DRAW_WAVE_GROUP  One waveform panel: up to 150 faint single spikes, the
% group's mean +/- std band and mean, then the reference templates, on the
% fixed y range YL. g = struct with .name, .W, .color.
    if nargin < 5, show_legend = false; end
    hold(ax, 'on');
    n = size(g.W, 1);
    if n > 0
        rs = RandStream('mt19937ar', 'Seed', 0);
        pick = randperm(rs, n, min(n, 150));
        plot(ax, g.W(pick,:)', 'Color', [g.color 0.06], 'LineWidth', 0.4, 'HandleVisibility', 'off');
        mu = mean(g.W,1); sd = std(g.W,0,1); xx = 1:numel(mu);
        fill(ax, [xx fliplr(xx)], [mu+sd fliplr(mu-sd)], g.color, 'FaceAlpha', 0.22, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(ax, xx, mu, 'Color', g.color, 'LineWidth', 2, 'DisplayName', 'mean +/- std (this group)');
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
        mu = mean(g.W,1); sd = std(g.W,0,1);
        if min(mu) < yl(1) || max(mu) > yl(2)
            ttl = [ttl ' -- clipped'];
        elseif min(mu - sd) < yl(1) || max(mu + sd) > yl(2)
            ttl = [ttl ' -- std band clipped'];
        end
    end
    title(ax, ttl, 'FontSize', 9, 'Interpreter', 'none');
    if show_legend
        legend(ax, 'show', 'Location', 'southeast', 'FontSize', 7, 'Interpreter', 'none');
    end
end
