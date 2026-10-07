function draw_composition_pie(ax, C)
% DRAW_COMPOSITION_PIE  The matched-cluster composition pie on AX, shared by
% run_composition_stage and run_pie_venn_stage so both draw the same thing.
%   II      sd1 first pass (slice), with the original SPC spikes (I) drawn as a
%           darker wedge inside it, area proportional to n_I/n_II
%   III-II  forced (sd3) spikes not in the sd1 core
%   IV      rescued spikes
% C - arm_wave_groups counts (n_I, n_core, n_II, n_III_II, n_IV, n_total)
    vals = [C.n_II, C.n_III_II, C.n_IV];
    pct = 100*vals/max(C.n_total,1);
    lbls = {sprintf('sd1 first pass,\nnot original (II-I)\nn=%d (%.1f%%)', C.n_core, 100*C.n_core/max(C.n_total,1)), ...
            sprintf('forced, sd3 (III-II)\nn=%d (%.1f%%)', C.n_III_II, pct(2)), ...
            sprintf('rescued (IV)\nn=%d (%.1f%%)', C.n_IV, pct(3))};
    colors = [0.15 0.55 0.25; 0.45 0.6 0.9; 0.95 0.6 0.1];

    keep = vals > 0;
    hp = pie(ax, vals(keep), lbls(keep));
    % pie returns [patch text patch text ...] in slice order
    ci = find(keep);
    for j = 1:numel(ci)
        hp(2*j-1).FaceColor = colors(ci(j), :);
        hp(2*j-1).EdgeColor = 'w'; hp(2*j-1).LineWidth = 1.5;
        set(hp(2*j), 'FontSize', 11, 'FontWeight', 'bold');
    end
    % I (original SPC) drawn INSIDE slice II: same angle, radius scaled so its
    % area is n_I/n_II of the slice (scale the slice's own vertices toward
    % the center, which keeps the exact arc)
    if keep(1) && C.n_I > 0
        hold(ax, 'on');
        sc = sqrt(C.n_I / C.n_II);
        xv = hp(1).XData * sc; yv = hp(1).YData * sc;
        patch(ax, xv, yv, [0.05 0.35 0.12], 'EdgeColor', 'w', 'LineWidth', 1.2);
        text(ax, mean(xv), mean(yv), sprintf('original\nSPC (I)\n%d\n(%.1f%%)', C.n_I, 100*C.n_I/max(C.n_total,1)), 'Color', 'w', ...
            'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end
