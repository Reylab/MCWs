function plot_stimlist_summary(summary, pct_in_sd1_vs_sd2, out_dir, ch, tag, window_label)
% PLOT_STIMLIST_SUMMARY  One 2x2 figure characterizing INSIDE-window
% response spikes for a given stim set (outside-window is not plotted here
% -- see build_arm_summary_table's caller for why n_out isn't a trustworthy
% baseline at the stim_list scope). tag/window_label let the caller
% distinguish different window definitions.
%
% Standalone copy of the local function already used in
% test_rescue_grape_metrics.m -- pulled out (unchanged) so other run scripts
% can call it too. Despite the param name, pct_in_sd1_vs_sd2 is just
% whatever one-off pairwise %% diff the caller wants shown in panel 2 (e.g.
% build_arm_summary_table's pct_last_vs_prev) -- it isn't sd1/sd2-specific.
    % VARIABLES
    %   summary     - the summary table for THIS window definition (arm,
    %                 class_id, n_in, pct_in_vs_gt, mean_zscore, mu_base, std_base)
    %   pct_in_sd1_vs_sd2 - the one-off pairwise %% diff for this window definition
    %   out_dir/ch  - where to save, channel number (for title/filename)
    %   tag         - filename fragment distinguishing fixed vs dynamic (e.g. 'fixed_pm100ms')
    %   window_label- human-readable label for the figure title (e.g. 'fixed +/-100ms window')
    %   arms        - arm labels as categorical, x-axis for panels 1/3/4
    %   f/tl        - figure and tiledlayout handles
    %   ax          - current panel's axes handle
    %   pct_arms    - x-axis categories for panel 2
    %   pct_data    - panel 2's bar data: inside %% diff per category
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    arms = categorical(summary.arm, summary.arm);

    f = figure('Visible','off','Position',[100 100 950 760],'Color','w');
    tl = tiledlayout(f,2,2,'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d class%d -- %s (%s)', ch, summary.class_id(1), strjoin(cellstr(summary.arm),' vs '), window_label), 'Interpreter','none');

    % panel 1: inside-window counts
    ax = nexttile(tl);
    bar(ax, arms, summary.n_in);
    ax.YAxis.Exponent = 0;   % show full numbers, no lopsided "x10^4" offset label
    ylabel(ax, '# spikes'); title(ax, 'spike counts (inside window)');
    grid(ax,'on');

    % panel 2: % diff vs the reference (first) arm, + the one-off last-pair comparison.
    % The last-pair bar is skipped with exactly 2 arms total: last vs
    % second-to-last IS arm(2) vs arm(1), so it would just duplicate the
    % "vs GT" bar already shown -- only adds information with 3+ arms.
    ax = nexttile(tl);
    pct_labels = cellstr(summary.arm(2:end));
    pct_data = summary.pct_in_vs_gt(2:end);
    if height(summary) > 2
        pct_labels = [pct_labels; {sprintf('%s->%s', summary.arm{end-1}, summary.arm{end})}];
        pct_data = [pct_data; pct_in_sd1_vs_sd2];
    end
    pct_arms = categorical(pct_labels, pct_labels);
    bar(ax, pct_arms, pct_data);
    yline(ax, 0, 'k-');
    ylabel(ax, '% difference'); title(ax, sprintf('%% diff, inside window (vs %s)', summary.arm{1}));
    grid(ax,'on');

    % panel 3: mean response z-score per arm
    ax = nexttile(tl);
    bar(ax, arms, summary.mean_zscore);
    yline(ax, 0, 'k-');
    ylabel(ax, 'mean z-score'); title(ax, 'response z-score (all scored stimuli)');
    grid(ax,'on');

    % panel 4: baseline mu +/- std per arm
    ax = nexttile(tl);
    bar(ax, arms, summary.mu_base); hold(ax,'on');
    errorbar(ax, arms, summary.mu_base, summary.std_base, 'k.', 'LineWidth', 1.2, 'CapSize', 12);
    ylabel(ax, 'baseline spike count'); title(ax, 'baseline mu +/- std');
    grid(ax,'on');

    save_fig_local(f, fullfile(out_dir, sprintf('stimlist_summary_%s_ch%d_class%d.png', tag, ch, summary.class_id(1))));
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
