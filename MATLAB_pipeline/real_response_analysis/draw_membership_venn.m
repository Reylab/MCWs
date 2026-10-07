function draw_membership_venn(ax, gtc, trial_name, arm_class, n_gt, n_arm, n_m, n_gt_only, n_arm_only, left_name, ch)
% DRAW_MEMBERSHIP_VENN  Draws the actual two-circle GT-vs-arm venn (circles,
% a green-filled overlap lens for the matched region, region labels with
% counts + %-of-union, title) onto AX. Shared by plot_membership_set.m (its
% own standalone figure/PNG) and plot_venn_waveform_group.m (embedded as
% one panel of the composite waveform page) so both draw an IDENTICAL venn
% -- fixing this function fixes both places at once.
%
% VARIABLES
%   gtc/arm_class     - left/right class numbers (labeling only)
%   trial_name/left_name - right/left side labels (e.g. 'rescue_sd1'/'gt')
%   n_gt/n_arm/n_m     - total left-side spikes, total right-side spikes, matched-pair count
%   n_gt_only/n_arm_only - unpaired counts on each side
%   ch                 - channel number (labeling only)
    n_union = n_m + n_gt_only + n_arm_only;
    pct = @(x) 100*x/max(n_union,1);

    hold(ax,'on'); axis(ax,'equal'); axis(ax,'off');
    r1 = sqrt(n_gt); r2 = sqrt(n_arm); rr = max([r1 r2 1]);
    r1n = 0.9*r1/rr; r2n = 0.9*r2/rr;
    ov = n_m/max(min(n_gt,n_arm),1);
    d = (r1n+r2n)*(1-0.85*ov);
    % linspace(0,2*pi,200) would include BOTH 0 and 2*pi -- since
    % cos(0)==cos(2*pi) and sin(0)==sin(2*pi), that's a literal duplicate
    % point (start==end), which is exactly what polyshape's "duplicate
    % vertices" warning below is about. Drop the redundant closing point;
    % fill() still closes the loop back to the first point on its own.
    th = linspace(0, 2*pi, 201);
    th(end) = [];
    fill(ax, -d/2+r1n*cos(th), r1n*sin(th), [0.85 0.3 0.3], 'FaceAlpha',0.35,'EdgeColor',[0.6 0.1 0.1]);
    fill(ax,  d/2+r2n*cos(th), r2n*sin(th), [0.3 0.4 0.85], 'FaceAlpha',0.35,'EdgeColor',[0.1 0.1 0.6]);

    % Matched region = the actual circle-circle overlap lens, filled green
    % (the color used for "matched" everywhere else in this pipeline's
    % plots -- see plot_venn_waveform_group.m) instead of leaving it as
    % just an implicit red+blue color blend.
    if d < r1n+r2n
        p1 = polyshape(-d/2+r1n*cos(th), r1n*sin(th));
        p2 = polyshape( d/2+r2n*cos(th), r2n*sin(th));
        p_ov = intersect(p1, p2);
        plot(ax, p_ov, 'FaceColor', [0.2 0.7 0.3], 'FaceAlpha', 0.65, 'EdgeColor', 'none');
    end

    % Pushed further out from the circles (was *0.6) -- too close made the
    % gt-only/arm-only labels crowd the circle edges and each other.
    text(ax, -d/2-r1n*1.2, 0, sprintf('%s-only\n%d (%.1f%%)',left_name,n_gt_only,pct(n_gt_only)), 'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
    text(ax,  d/2+r2n*1.2, 0, sprintf('%s-only\n%d (%.1f%%)',trial_name,n_arm_only,pct(n_arm_only)), 'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
    text(ax, 0, 0, sprintf('matched\n%d (%.1f%%)',n_m,pct(n_m)), 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold','Color',[0.05 0.4 0.15]);
    title(ax, sprintf('ch%d %s-class%d vs %s(class%d) -- matched %.1f%% of union (n=%d)\n%s n=%d, arm n=%d, %.0f%% of %s recovered', ...
          ch,left_name,gtc,trial_name,arm_class,pct(n_m),n_union,left_name,n_gt,n_arm,100*n_m/max(n_gt,1),left_name), 'Interpreter','none','FontSize',9);
end
