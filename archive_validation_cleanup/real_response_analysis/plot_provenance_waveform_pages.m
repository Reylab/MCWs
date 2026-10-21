function plot_provenance_waveform_pages(view_sets, view_labels, times_struct, spikes_struct, tol, title_prefix, out_dir, overlays, fixed_ylim)
% fixed_ylim (optional): one [lo hi] for every page -- see
% provenance_pages_extent.m. Omit/[] to scale each view page to its own data.
% PLOT_PROVENANCE_WAVEFORM_PAGES  One PNG page per view (3 total, matching
% view_sets/view_labels), each showing a mean +/- std waveform panel per
% provenance category from compute_spike_waveform_groups.m ('original',
% 'forced', 'rescued: <mask>' for every mask actually present) -- lets you
% visually sanity-check whether e.g. spikes rescued from a specific mask
% actually look like real class waveforms or like noise. Each panel also
% overlays the SAME reference templates (GT template, this arm's stored
% template, matched(GT,arm) template) as the main waveform_comparison
% figures, via the same `overlays` struct build_overlays_n already built for
% that group -- so a category's shape can be judged against the same
% reference lines, not in isolation.
%
% Only meaningful for RESCUE ARMS (wave_clus + rescue_spikes metadata) --
% don't call this for a GT-sourced group.
%
% VARIABLES
%   view_sets/view_labels - input: the 3 view spike-time subsets + names,
%                   same convention as plot_waveform_comparison_n.m
%   times_struct/spikes_struct/tol - input: see classify_spike_provenance.m
%   title_prefix    - input: figure title prefix (e.g. arm name + group)
%   out_dir         - input: subfolder to write the 3 pages into
%   overlays        - input: OPTIONAL. struct array (name/wave/color/style),
%                   same one plot_waveform_comparison_n.m's build_overlays_n
%                   produces for this group. Omit or pass [] for no overlays.
%
% ALSO WRITES, alongside the 3 PNGs, the actual data behind them so you can
% verify the plots independently of eyeballing them:
%   view<N>_<label>_groups.mat       - the exact `groups` struct (name, spike
%                                       times, and full quarantine_properties
%                                       table) for that view, reloadable for spot-checks
%   view<N>_<label>_properties.csv   - every group's properties table for
%                                       that view, stacked with a 'category'
%                                       column -- open directly in a
%                                       spreadsheet to eyeball the actual
%                                       width/amplitude/prominence_ratio/etc.
%                                       numbers behind each flag and judge
%                                       for yourself whether it's a real
%                                       violation or a sign of a bug
%   provenance_summary.csv           - one row per (view, category): n_spikes,
%                                       across all 3 views in this subfolder
    if nargin < 8, overlays = []; end
    if nargin < 9, fixed_ylim = []; end
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    summary_view = {}; summary_category = {}; summary_n = [];
    for vi = 1:numel(view_sets)
        groups = compute_spike_waveform_groups(times_struct, spikes_struct, view_sets{vi}, tol);
        view_fname = matlab.lang.makeValidName(view_labels{vi});
        save(fullfile(out_dir, sprintf('view%d_%s_groups.mat', vi, view_fname)), 'groups');

        props_all = table();
        for gi = 1:numel(groups)
            summary_view{end+1,1} = view_labels{vi}; %#ok<AGROW>
            summary_category{end+1,1} = groups(gi).name; %#ok<AGROW>
            summary_n(end+1,1) = numel(groups(gi).times); %#ok<AGROW>
            pt = groups(gi).properties;
            if ~isempty(pt)
                pt.category = repmat({groups(gi).name}, height(pt), 1);
                pt = movevars(pt, 'category', 'Before', 1);
                props_all = [props_all; pt]; %#ok<AGROW>
            end
        end
        if ~isempty(props_all)
            writetable(props_all, fullfile(out_dir, sprintf('view%d_%s_properties.csv', vi, view_fname)));
        end

        n_panels = numel(groups);
        if n_panels == 0
            % This view has no spikes at all (e.g. a class with no response-
            % window spikes) -- nothing to draw, and a 0-row layout errors.
            fprintf('[breakdown] %s -- %s: no spikes in this view, page skipped\n', title_prefix, view_labels{vi});
            continue;
        end
        ncols = max(1, min(3, n_panels));
        nrows = ceil(n_panels / ncols);

        % Pull every panel's waveforms up front so ALL panels on this page
        % share ONE y-axis range (set from whichever category/overlay has
        % the highest peak-to-peak amplitude) -- same "compare apples to
        % apples" fix as plot_waveform_comparison_n.m's main figures,
        % applied here across the whole page instead of auto-scaling each
        % category's panel independently.
        Ws = cell(1, n_panels);
        for gi = 1:n_panels
            Ws{gi} = get_waveforms_for_times(groups(gi).times, times_struct.cluster_class(:,2), times_struct.spikes, tol);
        end
        if isempty(fixed_ylim)
            shared_ylim = compute_shared_ylim(Ws, overlays);
        else
            shared_ylim = fixed_ylim;
        end

        f = figure('Visible','off','Position',[100 100 380*ncols 320*nrows],'Color','w');
        tl = tiledlayout(f, nrows, ncols, 'Padding','loose','TileSpacing','loose');
        title(tl, sprintf('%s -- %s', title_prefix, view_labels{vi}), 'Interpreter','none');

        axs = gobjects(1, n_panels);
        for gi = 1:n_panels
            ax = nexttile(tl); hold(ax,'on');
            axs(gi) = ax;
            g = groups(gi);
            W = Ws{gi};
            if isempty(W)
                text(ax,0.5,0.5,'no spikes','HorizontalAlignment','center','Units','normalized');
            else
                mu = mean(W,1); sd = std(W,0,1); xx = 1:numel(mu);
                fill(ax,[xx fliplr(xx)],[mu+sd fliplr(mu-sd)],[0.2 0.4 0.8],'FaceAlpha',0.25,'EdgeColor','none', ...
                    'DisplayName','mean +/- std (this group)');
                plot(ax, xx, mu, 'Color',[0.1 0.2 0.6], 'LineWidth',1.6, 'DisplayName','mean (this group)');
            end
            % Thickest first so thinner templates stay visible on top.
            if ~isempty(overlays)
                [~, ov_order] = sort([overlays.width], 'descend');
                for oi = ov_order
                    ov = overlays(oi);
                    plot(ax, 1:numel(ov.wave), ov.wave, ov.style, 'Color', ov.color, 'LineWidth', ov.width, 'DisplayName', ov.name);
                end
            end
            title(ax, sprintf('%s (n=%d)', g.name, numel(g.times)), 'Interpreter','none','FontSize',9);
            xlabel(ax,'sample'); ylabel(ax,'amplitude');
            grid(ax,'on');
            if gi == 1 && (~isempty(W) || ~isempty(overlays))
                % A corner INSIDE the axes ('southwest', 'best', 'southeast'
                % all tried) kept colliding with the trough or the shaded
                % band on one panel or another -- these panels are narrower
                % (380px) than plot_waveform_comparison_n.m's, so the same
                % legend text nearly spans the full panel width regardless
                % of which corner it's anchored to, guaranteeing overlap
                % with SOME curve. Placing it in the tiledlayout's own
                % dedicated 'south' margin tile instead means it physically
                % cannot overlap any panel's data -- verified by rendering
                % against real data and inspecting every panel.
                lgd = legend(ax, 'show', 'FontSize',7, 'Interpreter','none', 'Orientation','horizontal');
                lgd.Layout.Tile = 'south';
            end
        end
        if ~isempty(shared_ylim)
            for gi = 1:n_panels, ylim(axs(gi), shared_ylim); end
        end
        for gi = 1:n_panels, xlim(axs(gi), [1 size(times_struct.spikes,2)]); end

        fname = sprintf('view%d_%s.png', vi, matlab.lang.makeValidName(view_labels{vi}));
        save_fig_local(f, fullfile(out_dir, fname));
    end

    summary = table(summary_view, summary_category, summary_n, ...
        'VariableNames', {'view','category','n_spikes'});
    writetable(summary, fullfile(out_dir, 'provenance_summary.csv'));
end

function yl = compute_shared_ylim(Ws, overlays)
% Same convention as plot_waveform_comparison_n.m's compute_shared_ylim:
% union of every panel's mean+/-std band and every overlay's own curve --
% whichever is most extreme sets the y-range every panel on this page
% shares, so a genuinely bigger-amplitude category isn't visually flattened
% to look the same size as a smaller/noisier one.
    lo = Inf; hi = -Inf;
    for gi = 1:numel(Ws)
        W = Ws{gi};
        if isempty(W), continue; end
        mu = mean(W,1); sd = std(W,0,1);
        lo = min(lo, min(mu-sd)); hi = max(hi, max(mu+sd));
    end
    for oi = 1:numel(overlays)
        lo = min(lo, min(overlays(oi).wave)); hi = max(hi, max(overlays(oi).wave));
    end
    if ~isfinite(lo) || ~isfinite(hi) || lo == hi
        yl = [];
    else
        pad = 0.05*(hi-lo);
        yl = [lo-pad, hi+pad];
    end
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
