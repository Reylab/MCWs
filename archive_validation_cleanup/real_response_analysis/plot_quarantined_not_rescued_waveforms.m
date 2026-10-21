function plot_quarantined_not_rescued_waveforms(groups, overlays, title_prefix, out_png, shared_ylim)
% PLOT_QUARANTINED_NOT_RESCUED_WAVEFORMS  One page, one panel per quarantine
% mask (see classify_quarantined_not_rescued.m), each showing the mean+/-std
% waveform of that mask's spikes that were quarantined and NEVER rescued --
% the spikes that never made it into any cluster at all, as opposed to
% plot_provenance_waveform_pages.m's "rescued: <mask>" panels (spikes that
% failed a mask but WERE rescued back in).
%
% No venn on this page -- it's a single arm's own excluded population. The
% reference templates are the same {GT, arm, matched} set as every other
% waveform page (build_overlays_n), drawn thickest-first.
%
% shared_ylim: precomputed [lo hi] shared by every arm's quarantined page in
% this run (see run_rescue_grape_analysis.m). Pass [] to auto-scale.
%
% VARIABLES
%   groups         - input: classify_quarantined_not_rescued.m's output
%   overlays       - input: build_overlays_n output (GT / arm / matched templates)
%   title_prefix/out_png - input: figure title / save path

    if isempty(groups)
        f = figure('Visible','off','Position',[100 100 400 300],'Color','w');
        ax = axes(f);
        text(ax,0.5,0.5,'no quarantine masks available','HorizontalAlignment','center','Units','normalized');
        axis(ax,'off');
        save_fig_local(f, out_png);
        return;
    end

    n_panels = numel(groups);
    ncols = max(1, min(3, n_panels));
    nrows = ceil(n_panels / ncols);

    f = figure('Visible','off','Position',[100 100 380*ncols 320*nrows],'Color','w');
    tl = tiledlayout(f, nrows, ncols, 'Padding','loose','TileSpacing','loose');
    title(tl, title_prefix, 'Interpreter','none');

    axs = gobjects(1, n_panels);
    for gi = 1:n_panels
        ax = nexttile(tl); hold(ax,'on');
        axs(gi) = ax;
        g = groups(gi);
        W = g.waveforms;
        if isempty(W)
            text(ax,0.5,0.5,'no spikes','HorizontalAlignment','center','Units','normalized');
        else
            mu = mean(W,1); sd = std(W,0,1); xx = 1:numel(mu);
            fill(ax,[xx fliplr(xx)],[mu+sd fliplr(mu-sd)],[0.75 0.4 0.3],'FaceAlpha',0.25,'EdgeColor','none', ...
                'DisplayName','mean +/- std (this mask)');
            plot(ax, xx, mu, 'Color',[0.55 0.2 0.1], 'LineWidth',1.6, 'DisplayName','mean (this mask)');
        end
        if ~isempty(overlays)
            [~, ov_order] = sort([overlays.width], 'descend');
            for oi = ov_order
                ov = overlays(oi);
                plot(ax, 1:numel(ov.wave), ov.wave, ov.style, 'Color', ov.color, ...
                    'LineWidth', ov.width, 'DisplayName', ov.name);
            end
        end
        if isfield(g, 'pct_of_total')
            title(ax, sprintf('%s (n=%d, %.1f%% of quarantined)', g.name, size(W,1), g.pct_of_total), ...
                'Interpreter','none','FontSize',9);
        else
            title(ax, sprintf('%s (n=%d)', g.name, size(W,1)), 'Interpreter','none','FontSize',9);
        end
        xlabel(ax,'sample'); ylabel(ax,'amplitude');
        grid(ax,'on');
        if gi == 1
            lgd = legend(ax, 'show', 'FontSize',7, 'Interpreter','none', 'Orientation','horizontal');
            lgd.Layout.Tile = 'south';
        end
    end
    if ~isempty(shared_ylim)
        for gi = 1:n_panels, ylim(axs(gi), shared_ylim); end
    end

    save_fig_local(f, out_png);
end

function save_fig_local(f, out_png)
    set(f, 'Color', 'w');
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
    end
    close(f);
end
