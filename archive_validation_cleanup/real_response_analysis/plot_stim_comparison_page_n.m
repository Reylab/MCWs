function plot_stim_comparison_page_n(stim_num, all_names, all_class_ids, all_cc, ...
        omap, per_stim_list, grapes_gt, tol, dyn_buffer, display_buffer, ch, root, out_dir)
% PLOT_STIM_COMPARISON_PAGE_N  Generalizes plot_stim_comparison_page.m
% (fixed at exactly GT+sd1+sd2) to an arbitrary number of arms. ONE
% composite page per stimulus: thumbnail image, zscore/median_post/
% spk_post_std line graph, lat/dur table, gained/lost-vs-GT table (small
% panels across the top), and a LARGE colored union raster underneath.
%
% The raster shows MORE than the response window: for display purposes only
% (this does not change dyn_buffer or any counting/summary logic elsewhere
% in the pipeline), each arm's own response window is widened by
% display_buffer on each side so baseline/background spikes around the
% response are visible too.
%
% RASTER IS FULLY POOLED: every arm's spikes (response-window AND baseline)
% share ONE row per trial, not one row per arm. The same physical spike
% independently detected by more than one arm (within tol) is collapsed
% into a SINGLE dot (pool_all_events, same greedy tol-matching convention
% as nearest_match/membership_sets elsewhere) -- that's the whole point of
% the marker scheme below: since position no longer tells you which arm(s)
% a dot came from, shape/color/fill do.
%
% MARKER RULES (fixed, not data-driven/random -- same meaning on every page):
%   shape:  triangle if GT is one of the contributing arms, circle otherwise
%   fill:   filled if this pooled event falls inside at least one
%           contributing arm's own real response window, hollow if it's
%           baseline (outside every contributing arm's window) -- true for
%           BOTH triangles and circles
%   color, when GT is involved (triangle) -- this is the whole point of the
%   scheme: green means the rescue arm INDEPENDENTLY calls this a response
%   within ITS OWN window, not merely that it has a matching spike somewhere:
%     red X    - GT alone, no rescue arm has this spike at all (special
%                shape override, not really a "colored triangle")
%     black    - GT + a rescue arm both have it, but the rescue arm's own
%                copy is NOT inside that rescue arm's own window
%     green    - GT + a rescue arm both have it, AND the rescue arm's own
%                copy IS inside its own window too
%   Baseline (outside every window) GT-involved events are a single uniform
%   black hollow triangle regardless of rescue-arm match -- baseline means
%   nobody is calling it a response, so there's nothing to color-code.
%   color, when GT is NOT involved (circle), based on plain arm presence:
%     green if ALL configured rescue arms agree on this spike; green
%           (hollow) if only SOME do; if there are 2+ rescue arms, the ONE
%           arm added most recently (all_names{end}) having this spike
%           EXCLUSIVELY (no earlier rescue arm does) gets its own color
%           (blue) instead of green
%
% VARIABLES
%   stim_num        - input: which stimulus this page is for
%   all_names       - input: cellstr, all_names{1}='gt', all_names{2:end}=rescue arms
%   all_class_ids   - input: numeric vector, matched class id per entry of all_names
%   all_cc          - input: cell array, cluster_class per entry of all_names (for full-population times)
%   omap            - input: onset map (stim ID -> real trial onset times)
%   per_stim_list   - input: cell array, one arm's per_stim table (get_class_metrics output)
%   grapes_gt       - input: loaded GT grapes struct (ImageNames source)
%   tol             - input: spike-time match tolerance (ms)
%   dyn_buffer      - input: ms padding defining the REAL response window (same value used
%                     everywhere else in the pipeline for counting -- unaffected by this plot)
%   display_buffer  - input: EXTRA ms shown on each side of the response window, for
%                     visual context only (e.g. 500) -- does not change any window used
%                     for counting/classification elsewhere
%   ch/root/out_dir - input: same meaning as the original
%
%   n              - numel(all_names) (GT + every arm)
%   n_rescue       - n-1 (number of rescue arms)
%   full_t_list    - each arm's FULL session population (source for gathering raw spikes)
%   row_list/win_list - this stimulus's per_stim row / [lo hi] REAL response window bounds, per arm
%   t_abs/t_rel/t_trial/t_inwin - per arm, this stimulus's raw spikes within the display
%                    range: absolute time, time relative to their own trial's onset,
%                    which trial index, and whether they fall inside that arm's REAL window
%   window_t_list  - per arm, just the in-window subset of t_abs (for the gained/lost table,
%                    same definition the original script used)
%   events         - pool_all_events' output: one row per pooled cross-arm event
%                    (trial_idx, rel_time, arms involved, is_in_window)
%   gained/lost/pct_*/net_pct - gained/lost table's numbers, one entry per rescue arm (vs GT)
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    n = numel(all_names);
    n_rescue = n - 1;

    full_t_list = cell(1,n);
    row_list = cell(1,n);
    win_list = cell(1,n);
    for a = 1:n
        full_t_list{a} = all_cc{a}(all_cc{a}(:,1)==all_class_ids(a), 2);
        row_list{a} = per_stim_list{a}(per_stim_list{a}.stim_number == stim_num, :);
        win_list{a} = win_bounds(row_list{a}, dyn_buffer);
    end

    trial_onsets = [];
    if isKey(omap, stim_num), trial_onsets = sort(omap(stim_num)); end

    % ---- display range: union of every arm's own window, +/- display_buffer,
    % falling back to a plain +/-display_buffer around 0 if no arm has a
    % confident window this stimulus ----
    valid = ~cellfun(@(w) any(isnan(w)), win_list);
    if any(valid)
        lo_all = min(cellfun(@(w) w(1), win_list(valid))) - display_buffer;
        hi_all = max(cellfun(@(w) w(2), win_list(valid))) + display_buffer;
    else
        lo_all = -display_buffer; hi_all = display_buffer;
    end

    % ---- gather raw spikes per arm, per trial, within the display range ----
    t_abs = cell(1,n); t_rel = cell(1,n); t_trial = cell(1,n); t_inwin = cell(1,n);
    for a = 1:n
        ta = []; tr = []; ti = [];
        for k = 1:numel(trial_onsets)
            onset = trial_onsets(k);
            cand = full_t_list{a}(full_t_list{a} >= onset+lo_all & full_t_list{a} <= onset+hi_all);
            ta = [ta; cand(:)]; tr = [tr; cand(:)-onset]; ti = [ti; repmat(k, numel(cand),1)];
        end
        t_abs{a} = ta; t_rel{a} = tr; t_trial{a} = ti;
        w = win_list{a};
        if any(isnan(w))
            t_inwin{a} = false(size(tr));
        else
            t_inwin{a} = tr >= w(1) & tr <= w(2);
        end
    end
    window_t_list = cellfun(@(ta,iw) ta(iw), t_abs, t_inwin, 'UniformOutput', false);

    events = pool_all_events(t_rel, t_inwin, t_trial, numel(trial_onsets), all_names, tol);

    n_gt = numel(window_t_list{1});
    gained = zeros(1,n_rescue); lost = zeros(1,n_rescue);
    pct_gained = zeros(1,n_rescue); pct_lost = zeros(1,n_rescue); net_pct = zeros(1,n_rescue);
    for k = 1:n_rescue
        [~,~,gt_only_k,arm_only_k] = membership_sets(window_t_list{1}, window_t_list{k+1}, tol);
        gained(k) = numel(arm_only_k); lost(k) = numel(gt_only_k);
        pct_gained(k) = 100*gained(k)/max(n_gt,1); pct_lost(k) = 100*lost(k)/max(n_gt,1);
        net_pct(k) = 100*(numel(window_t_list{k+1})-n_gt)/max(n_gt,1);
    end

    % ---- one figure, everything on it ----
    f = figure('Visible','off','Position',[80 80 1700 1000],'Color','w');
    tl = tiledlayout(f,4,4,'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d stim%d -- %s', ch, stim_num, strjoin(all_names,' vs ')), 'Interpreter','none');

    ax = nexttile(tl,1,[1 1]);
    plot_stim_thumbnail(ax, grapes_gt, stim_num, root);

    ax = nexttile(tl,2,[1 1]);
    plot_line_graph_n(ax, row_list, all_names);

    ax = nexttile(tl,3,[1 1]); axis(ax,'off');
    txt = sprintf('%-10s %8s %8s %8s\n', 'arm','onset','dura','good_lat');
    txt = [txt sprintf('%-10s %8s %8s %8s\n', '----','-----','----','--------')];
    for a = 1:n
        txt = [txt row_txt(all_names{a}, row_list{a})];
    end
    text(ax, 0, 1, txt, 'FontName','FixedWidth', 'FontSize',10, 'VerticalAlignment','top', 'Interpreter','none');
    title(ax,'latency / duration', 'Interpreter','none');

    ax = nexttile(tl,4,[1 1]); axis(ax,'off');
    txt2 = sprintf('n %s = %d\n\n', all_names{1}, n_gt);
    txt2 = [txt2 sprintf('%-10s %8s %8s %8s %8s %8s\n','arm','gained','lost','%gain','%lost','net %')];
    for k = 1:n_rescue
        txt2 = [txt2 sprintf('%-10s %8d %8d %7.1f%% %7.1f%% %7.1f%%\n', all_names{k+1}, gained(k), lost(k), pct_gained(k), pct_lost(k), net_pct(k))];
    end
    text(ax, 0, 1, txt2, 'FontName','FixedWidth', 'FontSize',10, 'VerticalAlignment','top', 'Interpreter','none');
    title(ax, sprintf('gained/lost vs %s', all_names{1}), 'Interpreter','none');

    ax = nexttile(tl,5,[3 4]);
    plot_pooled_raster_n(ax, events, win_list, trial_onsets, all_names, stim_num, ch);

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
    % NOT axes(ax) here -- calling axes() on an EXISTING axes handle brings
    % its parent figure to the foreground (documented MATLAB behavior),
    % which forces the figure visible even though plot_stim_comparison_
    % page_n.m created it with 'Visible','off' -- exactly why the per-stim
    % pages were popping up on screen during a batch run. Not needed anyway:
    % image(ax, A) below already targets ax explicitly, it doesn't rely on
    % axes() having set the current-axes context first.
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

function plot_line_graph_n(ax, row_list, arm_names)
% mu_base/std_base are the whole-SESSION baseline REFERENCE for that arm's
% class (same value on every per_stim row -- see create_responses_data_
% parallel.m: sp_count_base is a FIXED pre-onset window
% [-resp_conf.tmax_median, -resp_conf.tmin_median], identical whether or not
% the dynamic onset+dura latency path is used for the POST-window side --
% mu_base/std_base themselves are never anchored to that per-stimulus
% latency/duration). Plotted here as an errorbar alongside the stim-specific
% numbers (median_post is the exact z-score numerator, on the same raw-count
% scale as mu_base, so they're directly comparable at a glance).
    n = numel(arm_names);
    arms = categorical(arm_names, arm_names);
    zsc = nan(n,1); mpost = nan(n,1); spstd = nan(n,1); mubase = nan(n,1); stdbase = nan(n,1);
    for a = 1:n
        zsc(a) = pick(row_list{a}, 'zscore');
        mpost(a) = pick(row_list{a}, 'median_post');
        spstd(a) = pick(row_list{a}, 'spk_post_std');
        mubase(a) = pick(row_list{a}, 'mu_base');
        stdbase(a) = pick(row_list{a}, 'std_base');
    end
    hold(ax,'on');
    plot(ax, arms, zsc,   '-o', 'DisplayName','zscore');
    plot(ax, arms, mpost, '-s', 'DisplayName','median post');
    plot(ax, arms, spstd, '-^', 'DisplayName','spk post std');
    errorbar(ax, arms, mubase, stdbase, '-d', 'DisplayName','baseline mu +/- std');
    legend(ax,'Location','northoutside','FontSize',7,'Interpreter','none','Orientation','horizontal'); grid(ax,'on');
    title(ax,'zscore / median post / spk post std / baseline', 'Interpreter','none');
end

function v = pick(row, col)
    if isempty(row), v = NaN; else, v = row.(col)(1); end
end

function s = row_txt(name, row)
    if isempty(row)
        s = sprintf('%-10s %8s %8s %8s\n', name, 'n/a','n/a','n/a');
    else
        s = sprintf('%-10s %8.1f %8.1f %8s\n', name, row.onset(1), row.dura(1), mat2str(row.good_lat(1)));
    end
end

% ==========================================================================
function events = pool_all_events(t_rel, t_inwin, t_trial, n_trials, arm_names, tol)
% POOL_ALL_EVENTS  Merges every arm's spikes (response-window AND baseline
% alike) into ONE set of events per trial: points from different arms
% within tol of each other are the same underlying physical spike
% (detected independently by each arm's own sorting) and collapse into a
% single event tagged with every arm that has it, instead of one dot per
% arm -- that's what lets the raster show ONE pooled row instead of one row
% per arm, with the marker style alone telling you the origin.
%
% Greedy clustering (same tol-matching convention as nearest_match/
% membership_sets elsewhere): process points earliest-time-first, and for
% each unclaimed seed, claim the nearest still-unclaimed point (if any,
% within tol) from every OTHER arm. Not globally optimal for pathological
% overlapping cases, but exact for the sparse, mostly-agree-or-disagree
% spike counts this deals with in practice.
%
% VARIABLES
%   t_rel/t_inwin/t_trial - input: cell arrays (one per arm): time relative to
%                  trial onset, whether in that arm's own REAL response window,
%                  which trial index
%   n_trials      - input: number of trials for this stimulus
%   arm_names     - input: cellstr, arm_names{1}='gt'
%   tol           - input: match tolerance (ms)
%   events        - output: struct array, one row per pooled event, with
%                  .trial_idx, .rel_time (GT's own time if GT contributed,
%                  else the earliest contributing arm's time), .arms (cellstr
%                  of every arm that has this event), .is_in_window (true if
%                  ANY contributing arm's own copy fell inside ITS OWN window),
%                  .inwin_arms (cellstr, SUBSET of .arms whose own copy was
%                  specifically inside THAT arm's own window -- an arm can be
%                  in .arms without being in .inwin_arms, e.g. GT has this
%                  spike in its window but the matched rescue-arm copy is
%                  sitting in that arm's own baseline)
    n = numel(arm_names);
    events = struct('trial_idx',{},'rel_time',{},'arms',{},'is_in_window',{},'inwin_arms',{});
    for tt = 1:n_trials
        pts = zeros(0,3);   % [arm_idx, rel_time, inwin(0/1)]
        for a = 1:n
            sel = (t_trial{a} == tt);
            rt = t_rel{a}(sel);
            iw = t_inwin{a}(sel);
            pts = [pts; [repmat(a, numel(rt), 1), rt(:), double(iw(:))]]; %#ok<AGROW>
        end
        if isempty(pts), continue; end
        [~, ord] = sort(pts(:,2));
        pts = pts(ord,:);
        claimed = false(size(pts,1),1);
        for i = 1:size(pts,1)
            if claimed(i), continue; end
            claimed(i) = true;
            cluster_rows = i;
            for a = 1:n
                if any(pts(cluster_rows,1) == a), continue; end
                cand = find(~claimed & pts(:,1)==a & abs(pts(:,2)-pts(i,2)) <= tol);
                if ~isempty(cand)
                    [~, mi] = min(abs(pts(cand,2) - pts(i,2)));
                    j = cand(mi);
                    claimed(j) = true;
                    cluster_rows(end+1) = j; %#ok<AGROW>
                end
            end
            arms_in_cluster = pts(cluster_rows,1);
            gi = find(arms_in_cluster == 1, 1);
            if ~isempty(gi)
                rep_rel = pts(cluster_rows(gi), 2);
            else
                rep_rel = min(pts(cluster_rows, 2));
            end
            is_inwin = any(pts(cluster_rows,3) == 1);
            inwin_arm_idx = arms_in_cluster(pts(cluster_rows,3) == 1);
            events(end+1) = struct('trial_idx', tt, 'rel_time', rep_rel, ...
                'arms', {arm_names(arms_in_cluster)}, 'is_in_window', is_inwin, ...
                'inwin_arms', {arm_names(inwin_arm_idx)}); %#ok<AGROW>
        end
    end
end

% ==========================================================================
function plot_pooled_raster_n(ax, events, win_list, trial_onsets, arm_names, stim_num, ch)
% ONE row per trial -- every arm's spikes share it (see pool_all_events).
% Every point's marker style is looked up from a FIXED catalog built from
% arm_names (spike_style_catalog + baseline_style_catalog), never a
% per-page data-driven color index, so a given category always looks the
% same across every stimulus page.
    n = numel(arm_names);
    n_rescue = n - 1;
    n_trials = numel(trial_onsets);
    if n_trials == 0
        text(ax,0.5,0.5,'no trials found for this stimulus','HorizontalAlignment','center');
        return;
    end

    catalog = [spike_style_catalog(arm_names), baseline_style_catalog(arm_names)];
    y_by_cat = cell(1, numel(catalog));
    rel_by_cat = cell(1, numel(catalog));
    for k = 1:numel(catalog), y_by_cat{k} = []; rel_by_cat{k} = []; end

    for e = 1:numel(events)
        ev = events(e);
        has_gt = any(strcmp(ev.arms, arm_names{1}));
        % Two membership vectors on purpose: ANY (does the arm have this
        % event at all, regardless of its own window) drives circle color
        % and the alone-vs-matched split; INWIN (does that arm's OWN copy
        % sit inside ITS OWN window) is what actually earns a rescue arm
        % the green color for a GT-triangle -- an arm simply having a
        % matching spike somewhere isn't enough if it's not calling it a
        % response itself. See spike_key.
        rescue_any = false(1,n_rescue);
        rescue_inwin = false(1,n_rescue);
        for r = 1:n_rescue
            rescue_any(r) = any(strcmp(ev.arms, arm_names{r+1}));
            rescue_inwin(r) = any(strcmp(ev.inwin_arms, arm_names{r+1}));
        end
        if ev.is_in_window
            if has_gt
                key = spike_key(true, rescue_any, rescue_inwin, n_rescue);
            else
                key = spike_key(false, rescue_any, rescue_any, n_rescue);
            end
        else
            multi = numel(ev.arms) >= 2;
            key = pick_key(has_gt, 'gt_baseline', pick_key(multi,'arm_multi','arm_solo'));
        end
        ki = find(strcmp({catalog.key}, key), 1);
        y_by_cat{ki}(end+1,1)   = ev.trial_idx;
        rel_by_cat{ki}(end+1,1) = ev.rel_time;
    end

    hold(ax,'on');
    leg_h = gobjects(1, numel(catalog)+n);
    leg_lbl = cell(1, numel(catalog)+n);
    n_leg = 0;
    for k = 1:numel(catalog)
        c = catalog(k);
        [leg_h, leg_lbl, n_leg] = draw_and_legend(ax, rel_by_cat{k}, y_by_cat{k}, c, leg_h, leg_lbl, n_leg, strcmp(c.key,'gt_only'));
    end

    ylim(ax, [0.5, n_trials+0.5]);

    win_colors = lines(n);
    for a = 1:n
        w = win_list{a}; col = win_colors(a,:);
        if ~any(isnan(w))
            line(ax,[w(1) w(1)], ylim(ax), 'Color',col,'LineStyle','--','LineWidth',2);
            h = line(ax,[w(2) w(2)], ylim(ax), 'Color',col,'LineStyle','--','LineWidth',2);
        else
            h = line(ax, [NaN NaN], [NaN NaN], 'Color',col,'LineStyle','--','LineWidth',2);
        end
        n_leg = n_leg + 1;
        leg_h(n_leg)   = h;
        leg_lbl{n_leg} = sprintf('%s window', upper(arm_names{a}));
    end
    leg_h = leg_h(1:n_leg); leg_lbl = leg_lbl(1:n_leg);

    xlabel(ax,'time (ms, relative to this trial''s onset)','FontSize',11,'Interpreter','none');
    ylabel(ax,'trial','FontSize',11,'Interpreter','none');
    grid(ax,'on');
    legend(ax, leg_h, leg_lbl, 'Location','eastoutside','FontSize',9,'Interpreter','none');
    title(ax, sprintf('ch%d stim%d -- pooled union raster (response window + baseline)', ch, stim_num), 'Interpreter','none');
end

function catalog = spike_style_catalog(arm_names)
% Fixed, ordered catalog of every possible IN-WINDOW marker style -- built
% once from arm_names (not from what happens to appear on any given page),
% so the same category always maps to the same color/shape/legend position
% on every stimulus page.
%
% Triangles (GT involved), 3 states:
%   'gt_only'         RED X   - GT alone, no rescue arm has this spike at all
%   'gt_matched_black' BLACK  - GT + a rescue arm both have it, but that
%                      rescue arm's OWN copy is NOT inside the rescue arm's
%                      OWN window (it doesn't call this a response itself)
%   'gt_matched'      GREEN   - GT + a rescue arm both have it, AND the
%                      rescue arm's own copy IS inside its own window too
% (This is the entire point of the color scheme: green means the rescue arm
% independently confirms the response within ITS OWN window, not merely
% that it has a matching spike somewhere.) Circles (no GT) are unaffected
% and keep the green/blue agreement-level colors, based on plain presence.
    gt_name = arm_names{1};
    n_rescue = numel(arm_names) - 1;
    green = [0.1 0.7 0.1]; blue = [0.1 0.3 0.85]; red = [0.85 0.1 0.1]; black = [0 0 0];

    catalog = struct('key',{},'label',{},'color',{},'marker',{},'filled',{},'size',{});
    catalog(end+1) = mk_style('gt_only', sprintf('%s only (missed by every rescue arm)', gt_name), red, 'x', false, 55);
    catalog(end+1) = mk_style('gt_matched_black', sprintf('%s + rescue arm (not in rescue''s own window)', gt_name), black, '^', true, 55);
    catalog(end+1) = mk_style('gt_matched', sprintf('%s + rescue arm (confirmed in rescue''s own window)', gt_name), green, '^', true, 55);
    catalog(end+1) = mk_style('consensus_nogt', 'all rescue arms agree (no GT match)', green, 'o', true, 55);
    if n_rescue >= 2
        newest = arm_names{end};
        catalog(end+1) = mk_style('newest_nogt', sprintf('%s only (newest arm, no GT match)', newest), blue, 'o', true, 55);
        catalog(end+1) = mk_style('partial_nogt', 'some rescue arms (no GT match)', green, 'o', false, 55);
    end
end

function catalog = baseline_style_catalog(arm_names)
% Fixed catalog for BASELINE (outside-EVERY-contributing-arm's-own-window)
% events. Triangles (GT involved) are a SINGLE uniform black hollow style
% here regardless of whether a rescue arm also has it -- baseline means no
% arm (GT included) is calling this a response, so there's nothing to
% color-code by rescue-arm agreement. Circles (no GT) keep the solo/multi
% hollow-vs-filled distinction.
    black = [0 0 0];
    catalog = struct('key',{},'label',{},'color',{},'marker',{},'filled',{},'size',{});
    catalog(end+1) = mk_style('gt_baseline', sprintf('%s baseline (outside every window)', arm_names{1}), black, '^', false, 45);
    catalog(end+1) = mk_style('arm_solo', 'rescue arm baseline only', black, 'o', false, 45);
    catalog(end+1) = mk_style('arm_multi','rescue arms baseline (no GT)', black, 'o', true, 45);
end

function s = mk_style(key, label, color, marker, filled, sz)
    s = struct('key',key,'label',label,'color',color,'marker',marker,'filled',filled,'size',sz);
end

function key = spike_key(has_gt, rescue_any, rescue_inwin, n_rescue)
    if has_gt
        if ~any(rescue_any)
            key = 'gt_only';
        elseif any(rescue_inwin)
            key = 'gt_matched';
        else
            key = 'gt_matched_black';
        end
        return;
    end
    n_present = sum(rescue_any);
    if n_present == n_rescue
        key = 'consensus_nogt'; return;
    end
    if n_rescue >= 2 && n_present == 1 && rescue_any(end)
        key = 'newest_nogt'; return;
    end
    key = 'partial_nogt';
end

function k = pick_key(cond, if_true, if_false)
    if cond, k = if_true; else, k = if_false; end
end

function [leg_h, leg_lbl, n_leg] = draw_and_legend(ax, rel, y, c, leg_h, leg_lbl, n_leg, always_show)
    if ~isempty(rel)
        if c.filled
            scatter(ax, rel, y, c.size, c.color, c.marker, 'filled');
        else
            scatter(ax, rel, y, c.size, c.color, c.marker, 'LineWidth',1.6);
        end
    end
    if ~isempty(rel) || always_show
        n_leg = n_leg + 1;
        if c.filled
            leg_h(n_leg) = scatter(ax, NaN, NaN, c.size, c.color, c.marker, 'filled');
        else
            leg_h(n_leg) = scatter(ax, NaN, NaN, c.size, c.color, c.marker, 'LineWidth',1.6);
        end
        leg_lbl{n_leg} = c.label;
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
