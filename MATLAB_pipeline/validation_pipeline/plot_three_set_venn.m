function plot_three_set_venn(S, names, ttl, out_png, wf)
% PLOT_THREE_SET_VENN  Three-circle GT / arm / quarantined venn, in the style
% of real_response_analysis/draw_membership_venn.m (GT red, arm blue, GT-arm
% overlap green; quarantined added in orange).
%
%   S       - three_set_membership output
%   names   - {gt_name, arm_name, q_name} labels
%   ttl     - title text
%   out_png - output file
%   wf      - OPTIONAL waveform panels drawn to the right of the venn on the
%             same page (see run_venn_stage):
%               wf.panels(i).title, wf.panels(i).sets = struct array with
%                 .W (n x samples), .color, .name -- one band per set
%               wf.templates = struct array .name/.wave/.color/.style/.width,
%                 overlaid on every panel
%               wf.region_colors (optional) - struct, venn region key ->
%                 color; only these regions are filled (circles become
%                 outlines). Keys: gt_only arm_only q_only gt_arm gt_q arm_q gt_arm_q
%
% Fixed layout: GT left, arm right, quarantined bottom, so every one of the
% 7 regions is always drawn and the picture reads the same in every venn.
% Circle sizes only loosely follow set size (radius 0.85-1.0) -- strictly
% area-proportional circles nest inside each other for the full spike set
% and hide regions. The numbers in each region are the exact counts; a
% region too small for its label gets it outside, with an arrow.

    c = S.counts;
    n = [S.n_gt, S.n_arm, S.n_q];
    n_union = c.gt_only + c.arm_only + c.q_only + c.gt_arm + c.gt_q + c.arm_q + c.gt_arm_q;
    pct = @(x) 100*x/max(n_union,1);
    % GT u arm: every region except quarantined-only. Regions inside it also
    % show their share of it -- the recovery picture without the quarantine
    % spikes neither sorting kept.
    n_ga = n_union - c.q_only;
    pct_ga = @(key, x) ifelse_ga(key, 100*x/max(n_ga,1));

    r = 0.85 + 0.15*sqrt(n/max(max(n),1));
    ctr = [-0.62 0.36; 0.62 0.36; 0 -0.70];

    th = linspace(0, 2*pi, 241); th(end) = [];
    P = cell(1,3);
    for k = 1:3
        P{k} = polyshape(ctr(k,1)+r(k)*cos(th), ctr(k,2)+r(k)*sin(th));
    end
    col  = {[0.85 0.3 0.3], [0.3 0.4 0.85], [0.95 0.6 0.1]};
    edge = {[0.6 0.1 0.1], [0.1 0.1 0.6], [0.7 0.4 0.0]};

    have_wf = nargin >= 5 && ~isempty(wf);
    if have_wf
        f = figure('Visible','off','Position',[100 100 2000 900],'Color','w');
        ax = axes(f, 'Position', [0.005 0.02 0.36 0.86]);
    else
        f = figure('Visible','off','Position',[100 100 820 820],'Color','w');
        ax = axes(f, 'Position', [0.02 0.02 0.96 0.88]);
    end
    hold(ax,'on'); axis(ax,'equal'); axis(ax,'off');
    lens = subtract(intersect(P{1}, P{2}), P{3});   % GT & arm, not quarantined = "matched"

    % set names outside each circle: GT upper-left, arm upper-right, quarantined below
    name_pos = [ctr(1,1)-0.35, ctr(1,2)+r(1)+0.12; ctr(2,1)+0.35, ctr(2,2)+r(2)+0.12; ctr(3,1), ctr(3,2)-r(3)-0.14];
    for k = 1:3
        text(ax, name_pos(k,1), name_pos(k,2), sprintf('%s  n=%d', names{k}, n(k)), 'Color', edge{k}, ...
            'HorizontalAlignment','center', 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter','none');
    end

    % region labels: count, % of union, and what the overlap means. In a
    % single-cluster venn some quarantined-only spikes were rescued into
    % OTHER classes, so "not rescued" would overstate it there.
    q_tag = 'not rescued';
    if S.q_rescued_elsewhere > 0, q_tag = 'not in this cluster'; end
    regions = { ...
        'gt_only',  subtract(subtract(P{1},P{2}),P{3}), ''; ...
        'arm_only', subtract(subtract(P{2},P{1}),P{3}), ''; ...
        'q_only',   subtract(subtract(P{3},P{1}),P{2}), q_tag; ...
        'gt_arm',   lens,                                'matched'; ...
        'gt_q',     subtract(intersect(P{1},P{3}),P{2}), 'lost'; ...
        'arm_q',    subtract(intersect(P{2},P{3}),P{1}), 'rescued, extra'; ...
        'gt_arm_q', intersect(intersect(P{1},P{2}),P{3}), 'rescued, in GT'};

    % Fills. With waveform panels, ONLY the regions that have a waveform
    % group are filled, each in that group's own color (wf.region_colors),
    % and the circles are outlines -- so a colored region always means "its
    % waveforms are on this page, in this color". Without panels, the
    % original translucent circles + green matched lens.
    if have_wf && isfield(wf, 'region_colors')
        for i = 1:size(regions,1)
            key = regions{i,1};
            if isfield(wf.region_colors, key) && regions{i,2}.NumRegions > 0
                plot(ax, regions{i,2}, 'FaceColor', wf.region_colors.(key), 'FaceAlpha', 0.45, 'EdgeColor', 'none');
            end
        end
        for k = 1:3
            plot(ax, P{k}, 'FaceColor', 'none', 'EdgeColor', edge{k}, 'LineWidth', 1.6);
        end
    else
        for k = 1:3
            plot(ax, P{k}, 'FaceColor', col{k}, 'FaceAlpha', 0.30, 'EdgeColor', edge{k}, 'LineWidth', 1.4);
        end
        plot(ax, lens, 'FaceColor', [0.2 0.7 0.3], 'FaceAlpha', 0.55, 'EdgeColor', 'none');
    end
    fit_r = 0.16;                        % inner radius a label needs
    out_r = max(sqrt(sum(ctr.^2,2)) + r(:)) + 0.20;
    inside_xy = zeros(0,2);
    outside = [];                        % [region row, anchor x, anchor y, angle]
    for i = 1:size(regions,1)
        key = regions{i,1}; poly = regions{i,2};
        [px, py, rin] = widest_point(poly);
        lbl = region_label(c.(key), pct(c.(key)), regions{i,3}, pct_ga(key, c.(key)));
        if rin >= fit_r
            text(ax, px, py, lbl, 'HorizontalAlignment','center', 'FontSize', 11, 'FontWeight', 'bold');
            inside_xy(end+1,:) = [px py]; %#ok<AGROW>
        else
            outside(end+1,:) = [i, px, py, atan2(py, px)]; %#ok<AGROW>
        end
    end
    % fallback arrows: nearest angle whose line clears the in-region labels
    cand = deg2rad(0:5:355);
    ang = nan(size(outside,1),1);
    for j = 1:size(outside,1)
        [~, o] = sort(abs(angle(exp(1i*(cand - outside(j,4))))));
        ang(j) = cand(o(1));
        for a = cand(o)
            e = out_r*[cos(a) sin(a)];
            clear_lbl = isempty(inside_xy) || all(seg_dist(inside_xy, outside(j,2:3), e) > 0.2);
            clear_ang = all(abs(angle(exp(1i*(a - ang(1:j-1))))) >= deg2rad(25));
            if clear_lbl && clear_ang, ang(j) = a; break; end
        end
        i = outside(j,1); key = regions{i,1};
        lx = out_r*cos(ang(j)); ly = out_r*sin(ang(j));
        annotate_arrow(ax, [lx ly], outside(j,2:3));
        ha = 'left'; if cos(ang(j)) < 0, ha = 'right'; end
        text(ax, lx, ly, region_label(c.(key), pct(c.(key)), regions{i,3}, pct_ga(key, c.(key))), ...
            'HorizontalAlignment', ha, 'FontSize', 11, 'FontWeight', 'bold');
    end

    lim_x = [min(ctr(:,1)-r(:)) max(ctr(:,1)+r(:))] + [-0.55 0.55];
    lim_y = [min(ctr(:,2)-r(:)) max(ctr(:,2)+r(:))] + [-0.35 0.35];
    xlim(ax, lim_x); ylim(ax, lim_y);

    sub = sprintf(['matched %d  |  %s not in %s %d  |  %s not in %s %d     ' ...
                   '(%% of all three, n=%d  |  %% of %s u %s, n=%d)'], ...
        c.gt_arm + c.gt_arm_q, names{1}, names{2}, c.gt_only + c.gt_q, ...
        names{2}, names{1}, c.arm_only + c.arm_q, n_union, names{1}, names{2}, n_ga);
    if S.q_rescued_elsewhere > 0
        sub = sprintf('%s   |   %s rescued into other classes: %d', sub, names{3}, S.q_rescued_elsewhere);
    end
    annotation(f, 'textbox', [0 0.91 1 0.08], 'String', {ttl, sub}, 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'EdgeColor', 'none', 'Interpreter', 'none');

    if have_wf
        draw_waveform_panels(f, wf);
    end

    out_dir = fileparts(out_png);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    print(f, out_png, '-dpng', '-r130');
    close(f);
end

function draw_waveform_panels(f, wf)
% 3x2 grid right of the venn: one panel per wf.panels entry, the last cell
% holds the template legend. Each set: up to 100 faint single traces, then its
% mean +/- std band and mean line; the templates go on top of every panel.
% One y range for every panel so amplitudes compare across them.
    x0 = 0.385; x1 = 0.99; y0 = 0.07; y1 = 0.88;
    ncol = 3; nrow = 2; gx = 0.035; gy = 0.10;
    w = (x1 - x0 - (ncol-1)*gx)/ncol; h = (y1 - y0 - (nrow-1)*gy)/nrow;
    pos = @(i) [x0 + mod(i-1,ncol)*(w+gx), y1 - ceil(i/ncol)*h - (ceil(i/ncol)-1)*gy, w, h];

    lo = inf; hi = -inf;
    for i = 1:numel(wf.panels)
        for s = wf.panels(i).sets(:)'
            if isempty(s.W), continue; end
            mu = mean(s.W,1); sd = std(s.W,0,1);
            lo = min(lo, min(mu-sd)); hi = max(hi, max(mu+sd));
        end
    end
    for t = wf.templates(:)'
        lo = min(lo, min(t.wave)); hi = max(hi, max(t.wave));
    end
    yl = [lo hi] + 0.08*(hi-lo)*[-1 1];

    rs = RandStream('mt19937ar', 'Seed', 0);
    for i = 1:numel(wf.panels)
        ax = axes(f, 'Position', pos(i)); hold(ax, 'on');
        p = wf.panels(i);
        parts = {};
        for s = p.sets(:)'
            n_s = size(s.W,1);
            parts{end+1} = sprintf('%s n=%d', s.name, n_s); %#ok<AGROW>
            if n_s == 0, continue; end
            pick = randperm(rs, n_s, min(n_s, 100));
            plot(ax, s.W(pick,:)', 'Color', [s.color 0.06], 'LineWidth', 0.4);
        end
        for s = p.sets(:)'
            if isempty(s.W), continue; end
            mu = mean(s.W,1); sd = std(s.W,0,1); xx = 1:numel(mu);
            fill(ax, [xx fliplr(xx)], [mu+sd fliplr(mu-sd)], s.color, 'FaceAlpha', 0.22, 'EdgeColor', 'none');
            plot(ax, xx, mu, 'Color', s.color, 'LineWidth', 2.0);
        end
        for t = wf.templates(:)'
            plot(ax, 1:numel(t.wave), t.wave, t.style, 'Color', t.color, 'LineWidth', t.width);
        end
        if numel(p.sets) > 1
            % which color is which spike group (templates are in the shared legend)
            hs = gobjects(0); ls = {};
            for s = p.sets(:)'
                hs(end+1) = plot(ax, NaN, NaN, '-', 'Color', s.color, 'LineWidth', 4); %#ok<AGROW>
                ls{end+1} = s.name; %#ok<AGROW>
            end
            legend(ax, hs, ls, 'Location', 'southeast', 'FontSize', 8, 'Interpreter', 'none', ...
                'Box', 'on', 'AutoUpdate', 'off');
        end
        ylim(ax, yl); xlim(ax, [1 numel(wf.templates(1).wave)]);
        grid(ax, 'on'); box(ax, 'off'); set(ax, 'FontSize', 9);
        title(ax, {p.title, strjoin(parts, ',  ')}, 'FontSize', 10, 'Interpreter', 'none');
        if ceil(i/ncol) == nrow || i + ncol > numel(wf.panels), xlabel(ax, 'sample'); end
        if mod(i-1, ncol) == 0, ylabel(ax, 'amplitude'); end
    end

    % legend cell: template lines only (each panel's title names its sets)
    lax = axes(f, 'Position', pos(numel(wf.panels)+1)); hold(lax, 'on'); axis(lax, 'off');
    h = gobjects(0); lbl = {};
    for t = wf.templates(:)'
        h(end+1) = plot(lax, NaN, NaN, t.style, 'Color', t.color, 'LineWidth', t.width); %#ok<AGROW>
        lbl{end+1} = t.name; %#ok<AGROW>
    end
    legend(lax, h, lbl, 'Location', 'west', 'FontSize', 10, 'Interpreter', 'none', 'Box', 'off');
end

function s = region_label(v, p, tag, p_ga)
% count, then % of all three (and % of GT u arm when the region is inside it)
    if isnan(p_ga)
        s = sprintf('%d\n(%.1f%%)', v, p);
    else
        s = sprintf('%d\n(%.1f%% | %.1f%%)', v, p, p_ga);
    end
    if ~isempty(tag), s = sprintf('%s\n%s', s, tag); end
end

function v = ifelse_ga(key, val)
% quarantined-only is outside GT u arm, so it gets no GT u arm share
    if strcmp(key, 'q_only'), v = NaN; else, v = val; end
end


function annotate_arrow(ax, from, to)
% data-space arrow from the label to the region
    d = to - from;
    quiver(ax, from(1), from(2), d(1), d(2), 0, 'Color', [0.25 0.25 0.25], ...
        'LineWidth', 0.8, 'MaxHeadSize', 0.4/max(norm(d), eps)*0.3);
end

function d = seg_dist(P, a, b)
% distance from each row of P to segment a-b
    ab = b - a;
    t = max(0, min(1, ((P(:,1)-a(1))*ab(1) + (P(:,2)-a(2))*ab(2)) / max(ab*ab', eps)));
    d = hypot(P(:,1) - (a(1) + t*ab(1)), P(:,2) - (a(2) + t*ab(2)));
end

function [px, py, rin] = widest_point(poly)
% interior point farthest from the region's edge (grid search) and that distance
    [xl, yl] = boundingbox(poly);
    [gx, gy] = meshgrid(linspace(xl(1), xl(2), 80), linspace(yl(1), yl(2), 80));
    in = isinterior(poly, gx(:), gy(:));
    gx = gx(in); gy = gy(in);
    V = poly.Vertices; V = V(all(isfinite(V),2), :);
    if isempty(gx)
        [px, py] = centroid(poly); rin = 0; return
    end
    dmin = inf(numel(gx),1);
    for v = 1:size(V,1)
        dmin = min(dmin, hypot(gx - V(v,1), gy - V(v,2)));
    end
    [rin, b] = max(dmin);
    px = gx(b); py = gy(b);
end
