function plot_waveform_comparison_n(all_names, all_class_ids, all_times, ...
        inside_stimlist_list, inside_other_list, outside_list, tol, ch, out_dir, varargin)
% PLOT_WAVEFORM_COMPARISON_N  Generalizes plot_waveform_comparison.m (fixed
% at exactly GT+sd1+sd2, 5 hardcoded groups) to an arbitrary number of arms.
% Produces waveform panels (mean +/- std), one composite PNG per group, each
% split into the same 3 views as the original (in-response/stim-list,
% in-response/other-stimuli, outside-response) x reference-template overlays.
%
% GROUPS (data-driven from numel(all_names)-1 = number of rescue arms, N):
%   entire_gt         - GT's WHOLE class population, not restricted to any
%                       exclusive subset (GT-sourced: single row of 3 views).
%   gt_missed_by_all  - GT spikes with no match in ANY arm's full population
%                       (the strongest "rescue can't recover this" case).
%                       Only gets the GT-template overlay (no single arm to
%                       pair it with). GT-sourced: single row.
%   For each arm k (1..N):
%     entire_armK     - armK's WHOLE class population (arm-sourced: gets the
%                       provenance/mask breakdown rows, see below).
%     armK_not_gt     - armK's spikes not in GT's full population (gained).
%                       Arm-sourced.
%     gt_not_armK     - GT's spikes not in armK's full population (missed).
%                       GT-sourced: single row.
%     entire_armK and armK_not_gt both get {GT template, armK's stored
%     template, matched(GT,armK) template} overlays, same 3-template scheme
%     as the original; gt_not_armK gets the same overlay set too.
%   For each consecutive pair (k,k+1) when N>=2:
%     arm(k+1)_not_armK - what the next arm in the given order adds beyond
%     the previous one (generalizes the original's SD2-not-SD1 group),
%     overlaid with {armK template, arm(k+1) template, matched(armK,arm(k+1))}.
%     Arm-sourced (uses arm(k+1)'s own spikes/provenance).
%
% ARM-SOURCED figures (entire_armK, armK_not_gt, arm(k+1)_not_armK) get TWO
% EXTRA ROWS below the waveform row, one column per view, computed from
% that view's own spike subset via compute_spike_provenance_breakdown:
%   row 2: original / forced / rescued, as a %% of that view's spike count
%   row 3: for the rescued portion only, which quarantine mask(s) they
%          originally failed (non-disjoint counts -- a spike failing 2
%          masks counts under both, same convention as
%          batch_files/rescue_mask_breakdown.m)
% GT-sourced figures (entire_gt, gt_missed_by_all, gt_not_armK) have no such
% metadata (hand-selected, not wave_clus+rescue_spikes output) and stay a
% single row of 3, exactly as before.
%
% entire_armK and armK_not_gt ALSO each get their own subfolder
% ('<armK>_entire_breakdown/', '<armK>_not_<gt>_breakdown/') containing 3
% pages, one per view -- 'original', 'forced', and one panel per quarantine
% mask actually present, each showing that category's own mean +/- std
% waveform (compute_spike_waveform_groups.m / plot_provenance_waveform_pages.m)
% -- for visually checking whether e.g. spikes rescued from a specific mask
% actually look like real class waveforms. Not added for the consecutive-pair
% groups (arm(k+1)_not_armK) since only "entire" and "not-in-GT" were asked for.
%
% VARIABLES
%   all_names       - input: cellstr, all_names{1}='gt', all_names{2:end}=rescue arms
%   all_class_ids   - input: numeric vector, matched class id per entry of all_names
%   all_times       - input: cell array, loaded times_*.mat struct per entry of all_names
%                     (needs .cluster_class and .spikes, i.e. the FULL times file)
%   inside_stimlist_list/inside_other_list/outside_list - input: cell arrays, one
%                     per entry of all_names -- e.g. compute_response_window_sets's
%                     .inside_stimlist/.inside_other/.outside per arm
%   tol             - input: spike-time match tolerance (ms)
%   ch/out_dir      - input: channel (labels only), output folder
%
% OPTIONAL NAME/VALUE ARGS:
%   'arm_template_files'  cellstr, one templates_wc_offline.mat path per
%                          RESCUE arm (length N, aligned with all_names(2:end)).
%                          Default {} (or any entry left '' ) skips that
%                          arm's stored-template overlay.
%   'arm_spikes_files'     cellstr, one *_spikes.mat path per RESCUE arm
%                          (length N). Default {} (or any entry left '')
%                          skips the provenance/mask breakdown rows for that
%                          arm's figures (they fall back to a single row).
%
%   n/gt_*         - N = number of rescue arms; gt_name/gt_times/gt_class/
%                    gt_full_t/gt_template - GT's own slice of the inputs
%   arm_full_t     - cell array, one arm's full-population times per rescue arm
%   arm_spikes     - cell array, one arm's loaded *_spikes.mat struct (or [] if
%                    unavailable) per rescue arm
%   view_labels    - the 3 view names, shared by every group
    p = inputParser;
    addParameter(p, 'arm_template_files', {});
    addParameter(p, 'arm_spikes_files', {});
    parse(p, varargin{:});
    arm_template_files = p.Results.arm_template_files;
    arm_spikes_files = p.Results.arm_spikes_files;

    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    n = numel(all_names) - 1;
    if isempty(arm_template_files), arm_template_files = repmat({''}, 1, n); end
    if isempty(arm_spikes_files), arm_spikes_files = repmat({''}, 1, n); end

    arm_spikes = cell(1,n);
    for k = 1:n
        if ~isempty(arm_spikes_files{k}) && exist(arm_spikes_files{k}, 'file')
            arm_spikes{k} = load(arm_spikes_files{k});
        else
            arm_spikes{k} = [];
        end
    end

    gt_name  = all_names{1};
    gt_times = all_times{1};
    gt_class = all_class_ids(1);
    gt_full_t = gt_times.cluster_class(gt_times.cluster_class(:,1)==gt_class, 2);
    gt_template = mean(gt_times.spikes(gt_times.cluster_class(:,1)==gt_class, :), 1);

    arm_full_t = cell(1,n);
    for k = 1:n
        idx = k+1;
        arm_full_t{k} = all_times{idx}.cluster_class(all_times{idx}.cluster_class(:,1)==all_class_ids(idx), 2);
    end

    view_labels = {'in-response (stim list)', 'in-response (other stimuli)', 'outside-response (baseline)'};

    %% Group: GT entire population (GT-sourced, no breakdown)
    views = bucket_views(gt_full_t, inside_stimlist_list{1}, inside_other_list{1}, outside_list{1}, tol);
    overlays_gt_entire = build_overlays_n(gt_name, gt_template, '', [], '', []);
    plot_waveform_group(views, view_labels, gt_times.cluster_class(:,2), gt_times.spikes, tol, ...
        sprintf('ch%d -- %s entire population (n=%d total)', ch, gt_name, numel(gt_full_t)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_%s_entire.png', ch, gt_name)), overlays_gt_entire, [], []);

    %% Group: GT missed by ALL arms
    in_any = false(size(gt_full_t));
    for k = 1:n
        in_any = in_any | (nearest_match(gt_full_t, arm_full_t{k}, tol) > 0);
    end
    group_missed_all = gt_full_t(~in_any);
    overlays_missed_all = build_overlays_n(gt_name, gt_template, '', [], '', []);
    views = bucket_views(group_missed_all, inside_stimlist_list{1}, inside_other_list{1}, outside_list{1}, tol);
    plot_waveform_group(views, view_labels, gt_times.cluster_class(:,2), gt_times.spikes, tol, ...
        sprintf('ch%d -- %s missed by ALL %d arm(s) (n=%d total)', ch, gt_name, n, numel(group_missed_all)), ...
        fullfile(out_dir, sprintf('waveforms_ch%d_%s_missed_by_all.png', ch, gt_name)), overlays_missed_all, [], []);

    n_gained = zeros(1,n); n_missed = zeros(1,n);
    %% Pre-pass: per-arm sets/overlays/views, plus ONE fixed y-limit shared by
    % every rescue-mask breakdown page (every arm, entire + not-GT, all 3 views)
    arm_ov = cell(1,n); views_entire = cell(1,n); views_gained = cell(1,n);
    gt_missed_all = cell(1,n); arm_gained_all = cell(1,n);
    bd_lo = Inf; bd_hi = -Inf;
    for k = 1:n
        idx = k+1;
        arm_times = all_times{idx};
        [matched_gt_k, matched_arm_k, gt_missed_all{k}, arm_gained_all{k}] = membership_sets(gt_full_t, arm_full_t{k}, tol);
        n_gained(k) = numel(arm_gained_all{k}); n_missed(k) = numel(gt_missed_all{k});
        arm_template = load_stored_template(arm_template_files{k}, ch, all_class_ids(idx));
        matched_template = mean([ ...
            get_waveforms_for_times(matched_gt_k, gt_times.cluster_class(:,2), gt_times.spikes, tol); ...
            get_waveforms_for_times(matched_arm_k, arm_times.cluster_class(:,2), arm_times.spikes, tol)], 1);
        arm_ov{k} = build_overlays_n(gt_name, gt_template, all_names{idx}, arm_template, all_names{idx}, matched_template);
        views_entire{k} = bucket_views(arm_full_t{k}, inside_stimlist_list{idx}, inside_other_list{idx}, outside_list{idx}, tol);
        views_gained{k} = bucket_views(arm_gained_all{k}, inside_stimlist_list{idx}, inside_other_list{idx}, outside_list{idx}, tol);
        if ~isempty(arm_spikes{k})
            [lo1, hi1] = provenance_pages_extent(views_entire{k}, arm_times, arm_spikes{k}, tol, arm_ov{k});
            [lo2, hi2] = provenance_pages_extent(views_gained{k}, arm_times, arm_spikes{k}, tol, arm_ov{k});
            bd_lo = min([bd_lo lo1 lo2]); bd_hi = max([bd_hi hi1 hi2]);
        end
    end
    if isfinite(bd_lo) && isfinite(bd_hi) && bd_lo < bd_hi
        breakdown_ylim = [bd_lo bd_hi] + [-1 1]*0.05*(bd_hi-bd_lo);
        fprintf('[waveforms] rescue-mask breakdown y-limit (all arms/subsets/views): [%.1f %.1f]\n', breakdown_ylim);
    else
        breakdown_ylim = [];
    end

    %% Per-arm groups: entire armK, armK-not-GT (gained), GT-not-armK (missed)
    for k = 1:n
        idx = k+1;
        arm_name = all_names{idx};
        arm_times = all_times{idx};
        spikes_k = arm_spikes{k};
        overlays_k = arm_ov{k};
        gt_missed_k = gt_missed_all{k};

        views = views_entire{k};
        plot_waveform_group(views, view_labels, arm_times.cluster_class(:,2), arm_times.spikes, tol, ...
            sprintf('ch%d -- %s entire population (n=%d total)', ch, arm_name, numel(arm_full_t{k})), ...
            fullfile(out_dir, sprintf('waveforms_ch%d_%s_entire.png', ch, arm_name)), overlays_k, arm_times, spikes_k);
        if ~isempty(spikes_k)
            plot_provenance_waveform_pages(views, view_labels, arm_times, spikes_k, tol, ...
                sprintf('ch%d -- %s entire', ch, arm_name), fullfile(out_dir, sprintf('%s_entire_breakdown', arm_name)), ...
                overlays_k, breakdown_ylim);
        end

        views = views_gained{k};
        plot_waveform_group(views, view_labels, arm_times.cluster_class(:,2), arm_times.spikes, tol, ...
            sprintf('ch%d -- %s - not - %s (n=%d total)', ch, arm_name, gt_name, numel(arm_gained_all{k})), ...
            fullfile(out_dir, sprintf('waveforms_ch%d_%s_not_%s.png', ch, arm_name, gt_name)), overlays_k, arm_times, spikes_k);
        if ~isempty(spikes_k)
            plot_provenance_waveform_pages(views, view_labels, arm_times, spikes_k, tol, ...
                sprintf('ch%d -- %s - not - %s', ch, arm_name, gt_name), fullfile(out_dir, sprintf('%s_not_%s_breakdown', arm_name, gt_name)), ...
                overlays_k, breakdown_ylim);
        end

        views = bucket_views(gt_missed_k, inside_stimlist_list{1}, inside_other_list{1}, outside_list{1}, tol);
        plot_waveform_group(views, view_labels, gt_times.cluster_class(:,2), gt_times.spikes, tol, ...
            sprintf('ch%d -- %s - not - %s (n=%d total)', ch, gt_name, arm_name, numel(gt_missed_k)), ...
            fullfile(out_dir, sprintf('waveforms_ch%d_%s_not_%s.png', ch, gt_name, arm_name)), overlays_k, [], []);
    end
    fprintf('[waveforms] group sizes: %s-missed-by-all=%d  gained=%s  missed=%s\n', ...
            gt_name, numel(group_missed_all), mat2str(n_gained), mat2str(n_missed));

    %% Consecutive-pair groups (only meaningful with 2+ arms): arm(k+1)-not-armK
    if n >= 2
        for k = 1:n-1
            idxA = k+1; idxB = k+2;
            nameA = all_names{idxA}; nameB = all_names{idxB};
            timesA = all_times{idxA}; timesB = all_times{idxB};
            spikes_b = arm_spikes{k+1};

            [matched_a, matched_b, ~, gained_b] = membership_sets(arm_full_t{k}, arm_full_t{k+1}, tol);
            templateA = load_stored_template(arm_template_files{k}, ch, all_class_ids(idxA));
            templateB = load_stored_template(arm_template_files{k+1}, ch, all_class_ids(idxB));
            matched_template = mean([ ...
                get_waveforms_for_times(matched_a, timesA.cluster_class(:,2), timesA.spikes, tol); ...
                get_waveforms_for_times(matched_b, timesB.cluster_class(:,2), timesB.spikes, tol)], 1);
            overlays_ab = build_overlays_n(nameA, templateA, nameB, templateB, nameB, matched_template);

            views = bucket_views(gained_b, inside_stimlist_list{idxB}, inside_other_list{idxB}, outside_list{idxB}, tol);
            plot_waveform_group(views, view_labels, timesB.cluster_class(:,2), timesB.spikes, tol, ...
                sprintf('ch%d -- %s - not - %s (n=%d total)', ch, nameB, nameA, numel(gained_b)), ...
                fullfile(out_dir, sprintf('waveforms_ch%d_%s_not_%s.png', ch, nameB, nameA)), overlays_ab, timesB, spikes_b);
        end
    end
end

% ==========================================================================
function views = bucket_views(group_t, inside_stimlist, inside_other, outside, tol)
    v1 = group_t(nearest_match(group_t, inside_stimlist, tol) > 0);
    v2 = group_t(nearest_match(group_t, inside_other, tol) > 0);
    v3 = group_t(nearest_match(group_t, outside, tol) > 0);
    views = {v1, v2, v3};
end

% build_overlays_n is now its own file (build_overlays_n.m) so
% plot_venn_waveform_group.m can share it too, instead of a duplicated copy.

function plot_waveform_group(view_sets, view_labels, times_cc_col2, spikes_matrix, tol, group_title, out_png, overlays, times_struct, spikes_struct)
% Single row of 3 waveform panels (GT-sourced groups), or 3 rows of 3 when
% times_struct is non-empty (arm-sourced groups): row 2 = original/forced/
% rescued %% of that view's own spikes, row 3 = mask breakdown of the
% rescued portion. See compute_spike_provenance_breakdown.m.
    has_breakdown = ~isempty(times_struct);
    if has_breakdown
        nrows = 3; fig_h = 1050;
    else
        nrows = 1; fig_h = 420;
    end
    f = figure('Visible','off','Position',[100 100 1300 fig_h],'Color','w');
    tl = tiledlayout(f,nrows,3,'Padding','loose','TileSpacing','loose');
    title(tl, group_title, 'Interpreter','none');

    % Pull every view's waveforms up front so all 3 panels can share ONE
    % y-axis range (set from whichever view/overlay has the highest
    % peak-to-peak amplitude) -- otherwise each panel auto-scales on its
    % own and a genuinely bigger/smaller waveform can look deceptively
    % similar-sized next to a sparse or noisy one.
    Ws = cell(1,3);
    for vi = 1:3
        Ws{vi} = get_waveforms_for_times(view_sets{vi}, times_cc_col2, spikes_matrix, tol);
    end
    shared_ylim = compute_shared_ylim(Ws, overlays);

    % Precompute all 3 views' provenance/mask breakdown up front (not just
    % the current view's, inside the loop below) so the mask-breakdown
    % panel can show each mask's % of THIS view's own rescued spikes AND
    % its % of the whole group's rescued spikes (all 3 views combined) --
    % the second number needs every view's n_resc summed before any panel
    % is drawn.
    pv_all = cell(1,3);
    total_resc_group = 0;
    if has_breakdown
        for vi = 1:3
            pv_all{vi} = compute_spike_provenance_breakdown(times_struct, spikes_struct, view_sets{vi}, tol);
            total_resc_group = total_resc_group + pv_all{vi}.n_resc;
        end
    end

    ax1 = gobjects(1,3);
    for vi = 1:3
        ax = nexttile(tl, vi); hold(ax,'on');
        ax1(vi) = ax;
        W = Ws{vi};
        if isempty(W)
            text(ax,0.5,0.5,'no spikes','HorizontalAlignment','center','Units','normalized');
            n_str = '0';
        else
            mu = mean(W,1); sd = std(W,0,1); xx = 1:numel(mu);
            fill(ax,[xx fliplr(xx)],[mu+sd fliplr(mu-sd)],[0.2 0.4 0.8],'FaceAlpha',0.25,'EdgeColor','none', ...
                'DisplayName','mean +/- std (this group)');
            plot(ax, xx, mu, 'Color',[0.1 0.2 0.6], 'LineWidth',1.6, 'DisplayName','mean (this group)');
            n_str = num2str(size(W,1));
        end
        % Thickest overlay drawn first (back), thinnest last (front) -- see
        % build_overlays_n.m -- so the thin lines stay visible wherever
        % templates cross instead of whichever was drawn last winning by
        % accident of array order.
        if ~isempty(overlays)
            [~, ov_order] = sort([overlays.width], 'descend');
            for oi = ov_order
                ov = overlays(oi);
                plot(ax, 1:numel(ov.wave), ov.wave, ov.style, 'Color', ov.color, 'LineWidth', ov.width, 'DisplayName', ov.name);
            end
        end
        title(ax, sprintf('%s (n=%s)', view_labels{vi}, n_str), 'Interpreter','none','FontSize',9);
        xlabel(ax,'sample'); ylabel(ax,'amplitude');
        grid(ax,'on');
        if vi == 1 && (~isempty(W) || ~isempty(overlays))
            % A corner INSIDE the axes ('southwest', 'best', 'southeast' all
            % tried) kept colliding with the trough or the shaded band on
            % some panel width/group -- the legend text is wide enough that
            % it can span nearly the full panel regardless of which corner
            % it's anchored to, guaranteeing an overlap with SOME curve for
            % at least one group. Placing it in the tiledlayout's own
            % dedicated 'south' margin tile instead means it physically
            % cannot overlap any panel's data -- verified by rendering
            % against real data and inspecting every panel/group.
            lgd = legend(ax, 'show', 'FontSize',7, 'Interpreter','none', 'Orientation','horizontal');
            lgd.Layout.Tile = 'south';
        end

        if has_breakdown
            pv = pv_all{vi};
            n_tot = numel(view_sets{vi});

            ax2 = nexttile(tl, 3+vi);
            pct = 100*[pv.n_orig, pv.n_forced, pv.n_resc] / max(n_tot,1);
            prov_labels = categorical({'Orig','Forced','Rescued'}, {'Orig','Forced','Rescued'});
            bar(ax2, prov_labels, pct, 'FaceColor', [0.3 0.5 0.8]);
            ylabel(ax2, '% of spikes'); ylim(ax2, [0 100]); grid(ax2,'on');
            title(ax2, sprintf('provenance (n=%d)', n_tot), 'FontSize',8, 'Interpreter','none');

            ax3 = nexttile(tl, 6+vi);
            if pv.n_resc == 0
                text(ax3,0.5,0.5,'no rescued spikes','HorizontalAlignment','center','Units','normalized','FontSize',8);
                axis(ax3,'off');
            elseif isempty(pv.mask_labels)
                text(ax3,0.5,0.5,'no mask data','HorizontalAlignment','center','Units','normalized','FontSize',8);
                axis(ax3,'off');
            else
                % Two %s per mask: of THIS view's own rescued spikes
                % (pv.n_resc), and of the whole group's rescued spikes,
                % all 3 views combined (total_resc_group) -- a mask that
                % looks dominant within one sparse view can be a small
                % sliver of the group's real rescued total, or vice versa.
                mask_cats = categorical(pv.mask_labels, pv.mask_labels);
                pct_sub = 100*pv.mask_counts/max(pv.n_resc,1);
                pct_all = 100*pv.mask_counts/max(total_resc_group,1);
                bar(ax3, mask_cats, [pct_sub(:) pct_all(:)], 'grouped');
                legend(ax3, {'% of this subgroup','% of all rescued'}, 'FontSize',6, 'Location','best');
                ylabel(ax3, '% of rescued spikes'); ylim(ax3, [0 100]); grid(ax3,'on');
                title(ax3, sprintf('mask breakdown (rescued, n=%d of %d total)', pv.n_resc, total_resc_group), 'FontSize',8, 'Interpreter','none');
            end
        end
    end
    if ~isempty(shared_ylim)
        for vi = 1:3, ylim(ax1(vi), shared_ylim); end
    end
    save_fig_local(f, out_png);
end

function yl = compute_shared_ylim(Ws, overlays)
% Union of every view's mean+/-std band and every overlay's own curve --
% whichever is most extreme sets the y-range every panel in this figure
% shares, so a genuinely bigger-amplitude waveform in one view isn't
% visually flattened to look the same size as a smaller one in another.
    lo = Inf; hi = -Inf;
    for vi = 1:numel(Ws)
        W = Ws{vi};
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
