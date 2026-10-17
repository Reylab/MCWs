function class_match_figure(detail_tbl, comp_tbl, channels, arm_names, cmp_dir, ref_name)
% CLASS_MATCH_FIGURE  Per (channel, arm): two stacked-bar panels -
%   left  = where each ground-truth class's spikes go (arm classK / class0 /
%           unmatched), as a fraction of the GT class
%   right = what each matched arm cluster is made of (this gt class / other gt /
%           gt class0 / new spikes with no gt counterpart), as a fraction of
%           the arm cluster
% detail_tbl/comp_tbl are match_cc's DET/COMP output, stacked across arms
% with an added 'arm' column. Shared by validate_recluster_vs_handselect.m
% and analyze_cluster_membership.m.
% ref_name (optional) names the reference in every label instead of 'GT'
% (e.g. 'wave_clus1' when a wave_clus run is the reference); when given,
% the file is class_match_ch<ch>_<ref>_vs_<arm>.png.
    if isempty(detail_tbl), return; end
    named = nargin >= 6 && ~isempty(ref_name);
    if ~named, ref_name = 'GT'; end
    ref_name = char(ref_name);
    arm_names = cellstr(arm_names);
    for ai = 1:numel(arm_names)
        an = arm_names{ai};
        for ch = channels
            D = detail_tbl(detail_tbl.arm == string(an) & detail_tbl.channel_id == ch, :);
            C = comp_tbl(comp_tbl.arm == string(an) & comp_tbl.channel_id == ch, :);
            if isempty(D), continue; end
            gcs = unique(D.gt_class)';
            ng = numel(gcs);

            % ---- panel 1: destinations ----
            dests = unique(D.dest, 'stable');
            cls_n = nan(size(dests));
            for i = 1:numel(dests)
                tk = regexp(dests(i), '^class(\d+)$', 'tokens', 'once');
                if ~isempty(tk), cls_n(i) = str2double(tk{1}); end
            end
            [~, ordn] = sort(cls_n + 1e6*isnan(cls_n));   % numeric classes, then class0/unmatched
            % keep 'unmatched' truly last
            um = dests(ordn) == "unmatched"; ordn = [ordn(~um); ordn(um)];
            dests = dests(ordn);
            M1 = zeros(ng, numel(dests));
            for r = 1:height(D)
                gi = find(gcs == D.gt_class(r));
                di = find(dests == D.dest(r));
                if ~isempty(gi) && ~isempty(di), M1(gi, di) = D.frac_of_gt_class(r); end
            end

            % ---- panel 2: composition ----
            M2 = zeros(ng, 4); rlab = strings(ng, 1);
            for gi = 1:ng
                sub = C(C.gt_class == gcs(gi), :);
                if isempty(sub), rlab(gi) = sprintf('%s %d', ref_name, gcs(gi)); continue; end
                rlab(gi) = sprintf('%s %d -> cluster %d', ref_name, gcs(gi), sub.arm_class(1));
                keys = ["gt_class" + string(gcs(gi)), "gt_other", "gt_class0", "new_no_gt_spike"];
                for si = 1:4
                    m = sub.src == keys(si);
                    if any(m), M2(gi, si) = sub.frac_of_arm_cluster(find(m, 1)); end
                end
            end

            f = figure('Visible', 'off', 'Position', [80 80 1200 max(340, 70*ng + 200)], 'Color', 'w');
            tl = tiledlayout(f, 1, 2, 'Padding', 'loose', 'TileSpacing', 'loose');
            title(tl, {sprintf('ch%d class matching', ch), sprintf('reference: %s     compared: %s', ref_name, an)}, ...
                  'Interpreter', 'none');

            ax = nexttile(tl);
            barh(ax, 1:ng, M1, 'stacked');
            yticks(ax, 1:ng); yticklabels(ax, string(ref_name) + " " + string(gcs)); ylim(ax, [0.4 ng+0.6]);
            set(ax, 'TickLabelInterpreter', 'none');
            xlim(ax, [0 1]); xlabel(ax, sprintf('fraction of %s class', ref_name), 'Interpreter', 'none');
            title(ax, sprintf('where each %s class''s spikes go', ref_name), 'Interpreter', 'none');
            legend(ax, cellstr(dests), 'Interpreter', 'none', 'Location', 'southoutside', ...
                   'Orientation', 'horizontal', 'NumColumns', 4);

            ax = nexttile(tl);
            barh(ax, 1:ng, M2, 'stacked');
            yticks(ax, 1:ng); yticklabels(ax, rlab); ylim(ax, [0.4 ng+0.6]);
            set(ax, 'TickLabelInterpreter', 'none');
            xlim(ax, [0 1]); xlabel(ax, 'fraction of the matched cluster');
            title(ax, 'what the matched cluster is made of');
            legend(ax, {sprintf('this %s class', ref_name), sprintf('other %s', ref_name), ...
                        sprintf('%s class0', ref_name), sprintf('new (no %s spike)', ref_name)}, ...
                   'Interpreter', 'none', 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2);

            if named
                fn = sprintf('class_match_ch%d_%s_vs_%s.png', ch, matlab.lang.makeValidName(ref_name), ...
                    matlab.lang.makeValidName(an));
            else
                fn = sprintf('class_match_ch%d_%s.png', ch, matlab.lang.makeValidName(an));
            end
            save_fig_local(f, fullfile(cmp_dir, fn));
        end
    end
end

function save_fig_local(f, out_png)
    set(f, 'Color', 'w');
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        try
            exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
        catch
        end
    end
    close(f);
end
