function plot_stim_comparison_page(stim_num, gt_inside_dyn, sd1_inside_dyn, sd2_inside_dyn, ...
        gt_stim_of, sd1_stim_of, sd2_stim_of, gt_full_t, sd1_full_t, sd2_full_t, omap, m_gt, m_sd1, m_sd2, grapes_gt, ...
        class_num, arm_class_sd1, arm_class_sd2, tol, dyn_buffer, ch, root, out_dir)
% PLOT_STIM_COMPARISON_PAGE  ONE composite page per stimulus: thumbnail
% image, zscore/median_post/spk_post_std line graph, lat/dur table,
% gained/lost table (small panels across the top), and a LARGE colored
% union raster underneath spanning the full width -- all 3 arms' spikes on
% one raster, PER-TRIAL rows (not pooled), colored by 3-way membership,
% with all 3 arms' own window boundaries drawn as separate line pairs and
% included in the same legend as the 7 membership categories.
%
% Two bugs fixed from the first version:
%   1) spikes are now converted to TIME RELATIVE TO EACH TRIAL'S OWN ONSET
%      before plotting -- previously they were plotted at their absolute
%      (session-wide, ms-since-recording-start) time, so the window lines
%      (drawn on the correct relative scale) collapsed to an invisible
%      sliver next to a multi-million-ms-wide spike scatter.
%   2) which stimulus/trial each spike belongs to now comes DIRECTLY from
%      find_response_spikes_dynamic's own window match (gt_stim_of etc.),
%      not a separately re-derived nearest-onset guess (assign_spikes_to_stim)
%      that could disagree with the actual classification whenever windows
%      have different widths per stimulus -- that mismatch was why counts
%      here didn't reconcile against the venn diagrams.
%
% VARIABLES
%   stim_num      - which stimulus this page is for
%   gt_inside_dyn/sd1_inside_dyn/sd2_inside_dyn - each arm's FULL (pooled
%                   across stim_list) dynamic in-window spike list (absolute time)
%   gt_stim_of/sd1_stim_of/sd2_stim_of - SAME length/order as the _inside_dyn
%                   lists: which stim ID each spike was actually matched to
%                   (find_response_spikes_dynamic's own output, not re-derived)
%   gt_full_t/sd1_full_t/sd2_full_t - each arm's FULL session spike times
%                   (class-filtered, absolute, NOT window- or stim-restricted).
%                   Passed straight through to classify_three_way so the
%                   raster's category ("only" vs shared) is judged against
%                   whether a spike exists ANYWHERE in another arm's data,
%                   not just within that arm's own window for this stimulus
%                   -- see classify_three_way.m for why that distinction matters.
%   omap          - onset map (stim ID -> real trial onset times), for
%                   converting bucketed spikes to relative-to-trial-onset time
%   m_gt/m_sd1/m_sd2 - get_class_metrics output per arm (per_stim table)
%   grapes_gt     - loaded GT grapes struct (ImageNames source -- same
%                   images regardless of arm, so only one grapes struct needed here)
%   class_num/arm_class_sd1/arm_class_sd2 - matched class number per arm (labels only)
%   tol           - spike-time match tolerance (ms), same convention as elsewhere
%   dyn_buffer    - ms padding for the dynamic window (same as main script)
%   ch/root/out_dir - channel, session root (for image path), output folder
%
%   gt_t/sd1_t/sd2_t - this stimulus's own spike times (absolute), bucketed
%                   using gt_stim_of/sd1_stim_of/sd2_stim_of (exact provenance)
%   trial_onsets  - this stimulus's own trial onset times (omap(stim_num)), sorted
%   row_gt/row_sd1/row_sd2 - this stimulus's single row from each arm's per_stim table
%   gt_win/sd1_win/sd2_win - [lo hi] window bounds per arm (relative ms), or
%                   [NaN NaN] if no confident onset
%   Tcls          - classify_three_way's per-arm-per-spike category table (absolute time)
%   n_gt/n_gained_*/n_lost_*/pct_*/net_pct_* - gained/lost table's numbers
%   f/tl          - figure and tiledlayout handles

    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    % ---- bucket each arm's pooled spike list down to this stim, using the
    % EXACT provenance find_response_spikes_dynamic already computed ----
    gt_t  = gt_inside_dyn(gt_stim_of   == stim_num);
    sd1_t = sd1_inside_dyn(sd1_stim_of == stim_num);
    sd2_t = sd2_inside_dyn(sd2_stim_of == stim_num);

    trial_onsets = [];
    if isKey(omap, stim_num), trial_onsets = sort(omap(stim_num)); end

    row_gt  = m_gt.per_stim(m_gt.per_stim.stim_number   == stim_num, :);
    row_sd1 = m_sd1.per_stim(m_sd1.per_stim.stim_number == stim_num, :);
    row_sd2 = m_sd2.per_stim(m_sd2.per_stim.stim_number == stim_num, :);

    gt_win  = win_bounds(row_gt,  dyn_buffer);
    sd1_win = win_bounds(row_sd1, dyn_buffer);
    sd2_win = win_bounds(row_sd2, dyn_buffer);

    % classification stays in ABSOLUTE time (matching is unaffected by units);
    % gt_t/sd1_t/sd2_t decide WHICH spikes are shown (window-restricted, this
    % stimulus only), gt_full_t/sd1_full_t/sd2_full_t decide the CATEGORY
    % (full session population, so a spike outside another arm's window but
    % still present in its data isn't mislabeled "only")
    Tcls = classify_three_way(gt_t, sd1_t, sd2_t, gt_full_t, sd1_full_t, sd2_full_t, tol);

    [~,~,gt_only_sd1,sd1_only] = membership_sets(gt_t, sd1_t, tol);
    [~,~,gt_only_sd2,sd2_only] = membership_sets(gt_t, sd2_t, tol);
    n_gt = numel(gt_t);
    n_gained_sd1 = numel(sd1_only); n_lost_sd1 = numel(gt_only_sd1);
    n_gained_sd2 = numel(sd2_only); n_lost_sd2 = numel(gt_only_sd2);
    pct_gained_sd1 = 100*n_gained_sd1/max(n_gt,1); pct_lost_sd1 = 100*n_lost_sd1/max(n_gt,1);
    pct_gained_sd2 = 100*n_gained_sd2/max(n_gt,1); pct_lost_sd2 = 100*n_lost_sd2/max(n_gt,1);
    net_pct_sd1 = 100*(numel(sd1_t)-n_gt)/max(n_gt,1);
    net_pct_sd2 = 100*(numel(sd2_t)-n_gt)/max(n_gt,1);

    % ---- one figure, everything on it ----
    f = figure('Visible','off','Position',[80 80 1500 1000],'Color','w');
    tl = tiledlayout(f,4,4,'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d stim%d -- gt vs sd1 vs sd2', ch, stim_num), 'Interpreter','none');

    ax = nexttile(tl,1,[1 1]);
    plot_stim_thumbnail(ax, grapes_gt, stim_num, root);

    ax = nexttile(tl,2,[1 1]);
    plot_line_graph(ax, row_gt, row_sd1, row_sd2);

    ax = nexttile(tl,3,[1 1]); axis(ax,'off');
    txt = sprintf('%-6s %8s %8s %8s\n', 'arm','onset','dura','good_lat');
    txt = [txt sprintf('%-6s %8s %8s %8s\n', '----','-----','----','--------')];
    txt = [txt row_txt('gt',  row_gt)];
    txt = [txt row_txt('sd1', row_sd1)];
    txt = [txt row_txt('sd2', row_sd2)];
    text(ax, 0, 1, txt, 'FontName','FixedWidth', 'FontSize',10, 'VerticalAlignment','top', 'Interpreter','none');
    title(ax,'latency / duration', 'Interpreter','none');

    ax = nexttile(tl,4,[1 1]); axis(ax,'off');
    txt2 = sprintf('n gt = %d\n\n', n_gt);
    txt2 = [txt2 sprintf('%-6s %8s %8s %8s %8s %8s\n','arm','gained','lost','%gain','%lost','net %')];
    txt2 = [txt2 sprintf('%-6s %8d %8d %7.1f%% %7.1f%% %7.1f%%\n', 'sd1', n_gained_sd1, n_lost_sd1, pct_gained_sd1, pct_lost_sd1, net_pct_sd1)];
    txt2 = [txt2 sprintf('%-6s %8d %8d %7.1f%% %7.1f%% %7.1f%%\n', 'sd2', n_gained_sd2, n_lost_sd2, pct_gained_sd2, pct_lost_sd2, net_pct_sd2)];
    text(ax, 0, 1, txt2, 'FontName','FixedWidth', 'FontSize',10, 'VerticalAlignment','top', 'Interpreter','none');
    title(ax,'gained/lost vs GT', 'Interpreter','none');

    ax = nexttile(tl,5,[3 4]);
    plot_big_union_raster(ax, Tcls, trial_onsets, gt_win, sd1_win, sd2_win, stim_num, ch);

    save_fig_local(f, fullfile(out_dir, sprintf('stim_page_ch%d_stim%d.png', ch, stim_num)));
end

% ==========================================================================
function w = win_bounds(row, buffer)
    if isempty(row) || isnan(row.onset(1)) || isnan(row.dura(1))
        w = [NaN NaN];
    else
        w = [row.onset(1)-buffer, row.onset(1)+row.dura(1)+buffer];
    end
end

function plot_stim_thumbnail(ax, grapes, stim_num, root)
    if istable(grapes.ImageNames)
        imgnames = grapes.ImageNames;
    else
        imgnames = table(repmat({fullfile(root, 'pics_used')}, numel(grapes.ImageNames), 1), ...
                         grapes.ImageNames(:), 'VariableNames', {'folder', 'name'});
    end
    imagename = fullfile(imgnames.folder{stim_num}, imgnames.name{stim_num});
    axes(ax);
    if exist(imagename,'file') && contains(lower(imagename),'.jp')
        try
            A = imread(imagename);
            image(ax, A); set(ax,'PlotBoxAspectRatio',[1 1 1]);
        catch
            text(ax,0.5,0.5,'(image load failed)','HorizontalAlignment','center');
        end
    else
        text(ax,0.5,0.5,sprintf('stim %d',stim_num),'HorizontalAlignment','center');
    end
    set(ax,'xtick',[],'ytick',[]); box(ax,'off');
    title(ax, sprintf('stim %d', stim_num), 'Interpreter','none');
end

function plot_line_graph(ax, row_gt, row_sd1, row_sd2)
% mu_base/std_base are the whole-stimset baseline REFERENCE (same value on
% every per_stim row, computed once per class -- see create_responses_data.m:
% zscore = (median_post - mu_base)/std_base). What's actually stim-specific
% -- the values this stimulus's own z-score was built from -- are
% median_post (the exact numerator term) and spk_post_std (this stimulus's
% own post-window response variability across trials), so plot those instead.
    arms = categorical({'gt','sd1','sd2'}, {'gt','sd1','sd2'});
    zsc = [pick(row_gt,'zscore');      pick(row_sd1,'zscore');      pick(row_sd2,'zscore')];
    mpost = [pick(row_gt,'median_post'); pick(row_sd1,'median_post'); pick(row_sd2,'median_post')];
    spstd = [pick(row_gt,'spk_post_std'); pick(row_sd1,'spk_post_std'); pick(row_sd2,'spk_post_std')];
    hold(ax,'on');
    plot(ax, arms, zsc,   '-o', 'DisplayName','zscore');
    plot(ax, arms, mpost, '-s', 'DisplayName','median post');
    plot(ax, arms, spstd, '-^', 'DisplayName','spk post std');
    legend(ax,'Location','northoutside','FontSize',7,'Interpreter','none','Orientation','horizontal'); grid(ax,'on');
    title(ax,'zscore / median post / spk post std', 'Interpreter','none');
end

function v = pick(row, col)
    if isempty(row), v = NaN; else, v = row.(col)(1); end
end

function s = row_txt(name, row)
    if isempty(row)
        s = sprintf('%-6s %8s %8s %8s\n', name, 'n/a','n/a','n/a');
    else
        s = sprintf('%-6s %8.1f %8.1f %8s\n', name, row.onset(1), row.dura(1), mat2str(row.good_lat(1)));
    end
end

% ==========================================================================
function plot_big_union_raster(ax, Tcls, trial_onsets, gt_win, sd1_win, sd2_win, stim_num, ch)
% Per-trial rows, 3 stacked bands (bottom-to-top: SD2, SD1, GT), sharing the
% same trial index across bands so the SAME trial's row lines up vertically
% across arms -- makes per-trial agreement/disagreement directly readable.
    n_trials = numel(trial_onsets);
    if n_trials == 0
        text(ax,0.5,0.5,'no trials found for this stimulus','HorizontalAlignment','center');
        return;
    end
    gap = 1;   % blank rows between bands
    band_base = struct('sd2', 0, 'sd1', n_trials+gap, 'gt', 2*(n_trials+gap));
    arm_of = struct('gt','gt','sd1','sd1','sd2','sd2');

    cats   = {'gt_only','sd1_only','sd2_only','all3','gt_sd1','gt_sd2','sd1_sd2'};
    labels = {'GT only','SD1 only','SD2 only','all 3','GT+SD1 (not SD2)','GT+SD2 (not SD1)','SD1+SD2 (not GT)'};

    hold(ax,'on');
    leg_h   = gobjects(1, numel(cats)+3);
    leg_lbl = cell(1, numel(cats)+3);
    for ci = 1:numel(cats)
        [col,mk,filled] = category_style(cats{ci});
        sel = strcmp(Tcls.category, cats{ci});
        if any(sel)
            idx = find(sel);
            rel = nan(numel(idx),1); y = nan(numel(idx),1);
            for k = 1:numel(idx)
                [tr_idx, tr_rel] = nearest_trial(Tcls.time(idx(k)), trial_onsets);
                rel(k) = tr_rel;
                y(k) = band_base.(arm_of.(Tcls.arm{idx(k)})) + tr_idx;
            end
            if filled
                scatter(ax, rel, y, 55, col, mk, 'filled');
            else
                scatter(ax, rel, y, 55, col, mk, 'LineWidth',1.6);
            end
        end
        if filled
            leg_h(ci) = scatter(ax, NaN, NaN, 55, col, mk, 'filled');
        else
            leg_h(ci) = scatter(ax, NaN, NaN, 55, col, mk, 'LineWidth',1.6);
        end
        leg_lbl{ci} = labels{ci};
    end

    ytop = band_base.gt + n_trials + 0.5;
    ylim(ax, [0.5 ytop]);
    yticks(ax, [band_base.sd2 + (n_trials+1)/2, band_base.sd1 + (n_trials+1)/2, band_base.gt + (n_trials+1)/2]);
    yticklabels(ax, {'SD2','SD1','GT'});
    set(ax,'FontSize',11);

    win_colors = struct('gt',[0.85 0.10 0.10], 'sd1',[0.10 0.60 0.10], 'sd2',[0.10 0.30 0.85]);
    wins  = struct('gt',gt_win, 'sd1',sd1_win, 'sd2',sd2_win);
    names = {'gt','sd1','sd2'};
    for wi = 1:3
        w = wins.(names{wi}); col = win_colors.(names{wi});
        if ~any(isnan(w))
            line(ax,[w(1) w(1)], ylim(ax), 'Color',col,'LineStyle','--','LineWidth',2);
            h = line(ax,[w(2) w(2)], ylim(ax), 'Color',col,'LineStyle','--','LineWidth',2);
        else
            h = line(ax, [NaN NaN], [NaN NaN], 'Color',col,'LineStyle','--','LineWidth',2);
        end
        leg_h(numel(cats)+wi)   = h;
        leg_lbl{numel(cats)+wi} = sprintf('%s window', upper(names{wi}));
    end

    xlabel(ax,'time (ms, relative to this trial''s onset)','FontSize',11,'Interpreter','none');
    ylabel(ax,'trial (grouped by arm)','FontSize',11,'Interpreter','none');
    grid(ax,'on');
    legend(ax, leg_h, leg_lbl, 'Location','eastoutside','FontSize',9,'Interpreter','none');
    title(ax, sprintf('ch%d stim%d -- union raster, per trial (all arms + each arm''s own window)', ch, stim_num), 'Interpreter','none');
end

function [trial_idx, rel] = nearest_trial(t, trial_onsets)
    [~, trial_idx] = min(abs(trial_onsets - t));
    rel = t - trial_onsets(trial_idx);
end

function [col, mk, filled] = category_style(c)
    switch c
        case 'gt_only',  col = [0.85 0.10 0.10]; mk = 'x'; filled = false;
        case 'sd1_only', col = [0.10 0.60 0.10]; mk = 'x'; filled = false;
        case 'sd2_only', col = [0.10 0.30 0.85]; mk = 'o'; filled = true;
        case 'all3',     col = [0.10 0.60 0.10]; mk = 'o'; filled = true;
        case 'gt_sd1',   col = [0.85 0.10 0.10]; mk = 'o'; filled = true;
        case 'gt_sd2',   col = [0.55 0.55 0.55]; mk = 'o'; filled = false;
        case 'sd1_sd2',  col = [0.10 0.60 0.10]; mk = 'o'; filled = false;
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
