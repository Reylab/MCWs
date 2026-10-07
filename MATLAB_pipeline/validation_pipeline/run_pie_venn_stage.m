function run_pie_venn_stage(cfg, arm_classes)
% RUN_PIE_VENN_STAGE  Companion figure linking the composition pie to the
% cluster venn, one page per arm: the pie (draw_composition_pie, same as the
% composition page) and, under it, one bar chart per pie group --
%   original SPC (I) | sd1 first pass, not original (II-I) |
%   forced, sd3 (III-II) | rescued (IV)
% -- showing which arm-side venn regions that group's spikes fall in:
%   matched (GT & arm, not rescued) | rescued, in GT | arm only (not in GT,
%   not rescued) | rescued, not in GT
% Regions and colors are the cluster venn's (three_set_membership, GT matched
% to the arm first), so each group's bars add up to its pie count.
%
%   run_pie_venn_stage(cfg)               % matches arm classes itself
%   run_pie_venn_stage(cfg, arm_classes)  % reuse run_validation's matching
%
% OUTPUTS (under cfg.out_dir/pie_venn)
%   pie_venn_ch<ch>_<arm>.png
%   pie_venn_counts.csv - one row per (arm, pie group)

if nargin < 2, arm_classes = []; end
[Gs, ~, ~, arm_classes] = prepare_arm_groups(cfg, arm_classes);

out_dir = fullfile(cfg.out_dir, 'pie_venn');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% arm-side venn regions, in the venn's own fill colors
reg_keys   = {'gt_arm', 'gt_arm_q', 'arm_only', 'arm_q'};
% (two-line tick labels via the tex interpreter -- a newline in a plain
% XTickLabel string splits it into separate ticks)
reg_labels = {'matched\newline(GT & arm)', 'rescued,\newlinein GT', ...
              'arm only\newline(not in GT)', 'rescued,\newlinenot in GT'};
reg_colors = [0.2 0.7 0.3; 0.2 0.7 0.3; 0.3 0.4 0.85; 0.1 0.6 0.7];

grp_keys   = {'is_I', 'is_core', 'is_III', 'is_IV'};
grp_names  = {'original SPC (I)', 'sd1 first pass, not original (II-I)', ...
              'forced, sd3 (III-II)', 'rescued (IV)'};
grp_colors = [0.05 0.35 0.12; 0.15 0.55 0.25; 0.45 0.6 0.9; 0.95 0.6 0.1];

rows = {};
for k = 1:numel(cfg.arms)
    G = Gs{k};
    cls = arm_classes(k);
    n_spk = numel(G.masks.in_cls);
    R = false(n_spk, numel(reg_keys));
    for r = 1:numel(reg_keys)
        R(G.venn_rows.(reg_keys{r}), r) = true;
    end

    f = figure('Visible','off','Position',[100 100 1700 1400],'Color','w');
    annotation(f, 'textbox', [0 0.95 1 0.04], 'EdgeColor', 'none', 'FontSize', 14, 'FontWeight', 'bold', ...
        'Interpreter', 'none', 'HorizontalAlignment', 'center', ...
        'String', sprintf('ch%d  %s class %d  (n=%d) -- pie groups by %s-vs-%s venn region', ...
                          cfg.ch, cfg.arms(k).name, cls, G.counts.n_total, cfg.gt.name, cfg.arms(k).name));
    draw_composition_pie(axes(f, 'Position', [0.30 0.50 0.40 0.42]), G.counts);

    w = 0.205; gap = 0.035; x0 = 0.045;
    for g = 1:numel(grp_keys)
        m = G.masks.(grp_keys{g});
        cnt = sum(R(m, :), 1);
        n_g = nnz(m);
        assert(sum(cnt) == n_g, 'pie group %s does not split cleanly into venn regions', grp_names{g});
        ax = axes(f, 'Position', [x0 + (g-1)*(w+gap), 0.09, w, 0.33]); hold(ax, 'on');
        for r = 1:numel(reg_keys)
            bar(ax, r, cnt(r), 0.7, 'FaceColor', reg_colors(r,:), 'EdgeColor', 'none');
        end
        y_max = max(max(cnt) * 1.25, 1);
        for r = 1:numel(reg_keys)
            text(ax, r, cnt(r) + 0.02*y_max, sprintf('%d\n(%.1f%%)', cnt(r), 100*cnt(r)/max(n_g,1)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
        end
        set(ax, 'XTick', 1:numel(reg_keys), 'XTickLabel', reg_labels, 'FontSize', 10, ...
            'XLim', [0.4 numel(reg_keys)+0.6], 'TickLabelInterpreter', 'tex');
        ylim(ax, [0 y_max]); grid(ax, 'on'); box(ax, 'off');
        if g == 1, ylabel(ax, 'spikes'); end
        title(ax, sprintf('%s\nn=%d', grp_names{g}, n_g), 'FontSize', 12, 'Color', grp_colors(g,:), 'Interpreter', 'none');
        rows(end+1, :) = [{cfg.arms(k).name, cls, grp_names{g}, n_g}, num2cell(cnt)]; %#ok<AGROW>
    end

    print(f, fullfile(out_dir, sprintf('pie_venn_ch%d_%s.png', cfg.ch, cfg.arms(k).name)), '-dpng', '-r130');
    close(f);
end

writetable(cell2table(rows, 'VariableNames', [{'arm','class','pie_group','n'}, ...
    {'matched_not_rescued','rescued_in_gt','arm_only','rescued_not_in_gt'}]), ...
    fullfile(out_dir, 'pie_venn_counts.csv'));
fprintf('[pie-venn] -> %s\n', out_dir);
end
