function out = compare_gt_vs_arm_classes(root, ch, gtc, ref_grapes_file, ref_times_file, ...
        arm_grapes_file, arm_times_file, arm_classes, varargin)
% COMPARE_GT_VS_ARM_CLASSES  Venn + raster + waveform-overlay comparison of
% ONE ground-truth class against an ARBITRARY, user-chosen list of arm
% classes -- not just whichever single class analyze_cluster_membership.m's
% match_arm_class() auto-picks as "best".
%
% WHY THIS EXISTS: when a GT class's spikes are genuinely split across
% several arm clusters (see class_match_where_gt_spikes_go_ch<ch>.csv), the
% one auto-matched class (highest recall) can still be the wrong story --
% e.g. it may hold the most spikes but not the ones that actually carry the
% stimulus response. This function makes it cheap to look at several
% candidate classes side by side (typically: the auto-matched class, plus
% whatever else class_match_where_gt_spikes_go_ch<ch>.csv shows a chunk of
% GT's spikes landing in) and see which one actually looks like the real
% unit -- by venn overlap, by raster response, AND by waveform shape.
%
% Reuses the EXACT SAME full-vs-windowed scope rule, one-to-one spike
% matching, and synthetic-raster mechanism already verified in
% analyze_cluster_membership.m (see check_class_count_agreement,
% membership_sets, synth_grapes_for_subset there) -- duplicated here as
% local functions (MATLAB can't call another file's local functions
% directly) rather than refactored into a shared file, so this new,
% unverified entry point carries zero risk to the already-verified main
% pipeline. If one changes, check the other.
%
% USAGE (typical: ground truth vs one rescue_sd trial's real clustering,
% checking classes 1,3,4,5 -- e.g. because gt1's spikes were seen split
% across exactly those four in class_match_where_gt_spikes_go_ch300.csv):
%
%   out = compare_gt_vs_arm_classes(root, 300, 1, ...
%       fullfile(root,'hand selected','grapes_blanks.mat'), ...
%       fullfile(root,'hand selected','times_mRFUS03 raw_300.mat'), ...
%       fullfile(sd1dir,'grapes_blanks_rescue.mat'), ...
%       fullfile(sd1dir,'times_mRFUS03 raw_300.mat'), ...
%       [1 3 4 5], 'stim_list', stim_list, 'arm_name','rescue_sd1', ...
%       'ref_spikes_file', ref_spk, 'arm_spikes_file', arm_spk, ...
%       'out_dir', fullfile(out_root,'class_compare','rescue_sd1'));
%
% To sweep all 3 rescue_sd trials in one go, call this once per trial (see
% the loop at the bottom of this file's help, or just call 3x from your own
% script -- deliberately kept single-arm-per-call so a failure/crash on one
% trial can't take out the other two, matching the rest of this tool's
% per-trial isolation).
%
% REQUIRED
%   root              session root (has stimulus.mat, finalevents.mat, ...)
%   ch                channel number
%   gtc               ONE ground-truth class number
%   ref_grapes_file, ref_times_file    hand-select GT files
%   arm_grapes_file, arm_times_file    the ONE arm's files (e.g. one
%                     rescue_sd trial's grapes_blanks_rescue.mat + times)
%   arm_classes       vector of arm class numbers to compare against gtc,
%                     e.g. [1 3 4 5] -- NOT restricted to the auto-matched one
%
% NAME/VALUE (all optional)
%   'stim_list'         []      ordered stim numbers for rasters (strongly
%                                recommended -- same as analyze_cluster_
%                                membership.m's own stim_list; falls back to
%                                whatever stims are present, with a warning)
%   'ref_spikes_file'   ''      raw spikes file (spikes_all/index_all) for
%                                GT waveforms; omit to skip GT in the overlay
%   'arm_spikes_file'   ''      same, for the arm; omit to skip the overlay
%   'ref_name'          'handselect'
%   'arm_name'          'arm'
%   'match_tol_ms'      0.5     spike-time match tolerance
%   'circshiftblanks'   false
%   'out_dir'           fullfile(fileparts(arm_grapes_file), sprintf('class_compare_ch%d_gt%d', ch, gtc))
%
% OUTPUT (struct) + files under out_dir:
%   out.table                     one row per arm_class: same columns as
%                                  analyze_cluster_membership's own
%                                  membership_ch<ch>.csv (scope, n_gt, n_arm,
%                                  n_matched, n_gt_only, n_arm_only, ...,
%                                  n_gt_full/windowed, n_arm_full/windowed)
%   class_compare_ch<ch>_gt<gtc>_<arm_name>.csv     out.table, on disk
%   venn_ch<ch>_gt<gtc>_<arm_name>_class<K>.png      per arm_class
%   raster_ch<ch>_gt<gtc>_<arm_name>_class<K>.png    matched/gt-only/arm-only
%                                                     tick plot, per arm_class
%   rasters/class<K>/{gt,arm,matched,gt_only,arm_only}/   real PNG raster
%                                                          panels, per class
%   waveform_overlay_ch<ch>_gt<gtc>_<arm_name>_multiclass.png   GT (black) +
%       one line per requested arm_class, mean +/- std, all on one plot --
%       so you can compare waveform SHAPE across candidates, not just count
%   pairwise_ch<ch>_gt<gtc>_class<A>_vs_class<B>.png   real characterize_
%       grapes_responses metrics (zscore, auroc, mean_fr_post_hz, ...),
%       every pair of {GT, requested arm classes}, GT plotted as a red '+'
%       crosshair on every panel -- same plot analyze_cluster_membership.m
%       produces for arm-vs-arm, here for class-vs-class within one arm
%   trends_ch<ch>_gt<gtc>.png   one line per {GT, requested arm class} per
%       metric, x = stim_list (your order) -- "version C" trend view
%   response_metrics_ch<ch>_gt<gtc>.csv   the raw metrics table behind both

    % within_channel/refract_viol/Do_clustering/plot_grapes_as_online (via
    % resolve_session_root) all auto-detect off pwd -- same requirement
    % run_membership_analysis.m satisfies with its own cd(root) up front.
    % Do it here too so this function is self-contained and callable from
    % anywhere, restoring the caller's directory on exit (even on error).
    prev_dir = pwd;
    restore_dir = onCleanup(@() cd(prev_dir));
    cd(root);

    p = inputParser;
    addParameter(p, 'stim_list', []);
    addParameter(p, 'ref_spikes_file', '');
    addParameter(p, 'arm_spikes_file', '');
    addParameter(p, 'ref_name', 'handselect');
    addParameter(p, 'arm_name', 'arm');
    addParameter(p, 'match_tol_ms', 0.5);
    addParameter(p, 'circshiftblanks', false);
    addParameter(p, 'out_dir', '');
    parse(p, varargin{:});
    o = p.Results;
    if isempty(o.out_dir)
        o.out_dir = fullfile(fileparts(arm_grapes_file), sprintf('class_compare_ch%d_gt%d', ch, gtc));
    end
    if ~exist(o.out_dir, 'dir'), mkdir(o.out_dir); end
    arm_classes = unique(arm_classes(:)', 'stable');

    fprintf('[compare_gt_vs_arm_classes] ch%d gt-class%d vs %s, arm classes %s\n', ...
        ch, gtc, o.arm_name, mat2str(arm_classes));

    omap = build_onset_map(root);
    ref_cc = reconstruct_class_cc(ref_grapes_file, ch, omap);
    arm_cc = reconstruct_class_cc(arm_grapes_file, ch, omap);
    gt_t_win = ref_cc(ref_cc(:,1)==gtc, 2);
    [gt_match_ok, gt_n_times, ~] = check_class_count_agreement(ref_times_file, ref_grapes_file, ch, gtc);

    nm = nm_for_ch(root, ch);
    is_rescue_arm = contains(arm_grapes_file, '_rescue');

    gt_detail = reconstruct_class_detail(ref_grapes_file, ch, gtc, omap);
    if ~isempty(o.stim_list)
        stim_order = o.stim_list(:)';
    else
        stim_order = unique(gt_detail.snum, 'stable');
        warning('cgva:noStimList', ['no ''stim_list'' given -- rasters falling back to whatever stims ' ...
            'are present in first-seen order. Pass an explicit ordered stim_list.']);
    end

    % GT's own raster, built once, reused for every arm_class below.
    gt_dest = fullfile(o.out_dir, 'rasters', 'gt');
    run_plot_grapes_as_online(ref_grapes_file, false, ch, stim_order, gt_dest);

    rows = struct('arm_class',{},'scope',{},'n_gt',{},'n_arm',{},'n_matched',{},'n_gt_only',{}, ...
        'n_arm_only',{},'pct_gt_lost',{},'pct_arm_new',{},'n_gt_full',{},'n_gt_windowed',{}, ...
        'n_arm_full',{},'n_arm_windowed',{});

    % ---- waveform overlay across ALL requested classes, one plot ----
    wf = figure('Visible','off','Position',[100 100 900 520],'Color','w');
    wax = axes(wf); hold(wax,'on');
    cmap = lines(numel(arm_classes)+1);
    leg = {}; hh = []; plotted = false;

    if ~isempty(o.ref_spikes_file) && exist(o.ref_spikes_file, 'file')
        [gt_w, gt_scope_wf] = class_waveforms(ref_times_file, gt_t_win, gtc, gt_match_ok, o.ref_spikes_file, o.match_tol_ms);
        if ~isempty(gt_w)
            [hh, leg, plotted] = add_overlay_line(wax, gt_w, [0 0 0], 2.5, ...
                sprintf('%s (gt c%d, %s, n=%d)', o.ref_name, gtc, gt_scope_wf, size(gt_w,1)), hh, leg, plotted);
        end
    end

    for ci = 1:numel(arm_classes)
        acl = arm_classes(ci);
        arm_t_win = arm_cc(arm_cc(:,1)==acl, 2);
        if isempty(arm_t_win)
            fprintf('  [compare_gt_vs_arm_classes] arm class %d: not present in this arm''s reconstruction, skipped.\n', acl);
            continue;
        end
        [arm_match_ok, arm_n_times, ~] = check_class_count_agreement(arm_times_file, arm_grapes_file, ch, acl);
        use_full = gt_match_ok && arm_match_ok;
        if use_full
            gt_t  = full_class_times_from_times_file(ref_times_file, gtc);
            arm_t = full_class_times_from_times_file(arm_times_file, acl);
            scope = 'full';
        else
            gt_t  = gt_t_win;
            arm_t = arm_t_win;
            scope = 'windowed';
        end
        [matched_gt, matched_arm, gt_only, arm_only] = membership_sets(gt_t, arm_t, o.match_tol_ms);
        n_gt = numel(gt_t); n_arm = numel(arm_t);
        rows(end+1) = struct('arm_class',acl, 'scope',string(scope), 'n_gt',n_gt, 'n_arm',n_arm, ...
            'n_matched',numel(matched_gt), 'n_gt_only',numel(gt_only), 'n_arm_only',numel(arm_only), ...
            'pct_gt_lost', 100*numel(gt_only)/max(n_gt,1), 'pct_arm_new', 100*numel(arm_only)/max(n_arm,1), ...
            'n_gt_full', gt_n_times, 'n_gt_windowed', numel(gt_t_win), ...
            'n_arm_full', arm_n_times, 'n_arm_windowed', numel(arm_t_win));

        class_venn_and_raster(gtc, o.arm_name, acl, gt_t, arm_t, matched_gt, gt_only, arm_only, ...
            o.out_dir, ch, scope, gt_n_times, arm_n_times);

        class_dir = fullfile(o.out_dir, 'rasters', sprintf('class%d', acl));
        arm_dest = fullfile(class_dir, 'arm');
        run_plot_grapes_as_online(arm_grapes_file, is_rescue_arm, ch, stim_order, arm_dest);
        derived = {'matched', matched_arm; 'gt_only', gt_only; 'arm_only', arm_only};
        for si = 1:size(derived,1)
            synth_dir = fullfile(class_dir, [derived{si,1} '_synth']);
            gf = synth_grapes_for_subset(derived{si,2}, ch, root, ref_grapes_file, synth_dir, nm, o.circshiftblanks);
            if isempty(gf)
                fprintf('  [compare_gt_vs_arm_classes] gt%d class%d %s: 0 spikes, no raster.\n', gtc, acl, derived{si,1});
                continue;
            end
            run_plot_grapes_as_online(gf, false, ch, stim_order, fullfile(class_dir, derived{si,1}));
        end

        if ~isempty(o.arm_spikes_file) && exist(o.arm_spikes_file, 'file')
            [arm_w, arm_scope_wf] = class_waveforms(arm_times_file, arm_t_win, acl, arm_match_ok, o.arm_spikes_file, o.match_tol_ms);
            if ~isempty(arm_w)
                lbl = sprintf('%s (c%d, %s, n=%d)', o.arm_name, acl, arm_scope_wf, size(arm_w,1));
                [hh, leg, plotted] = add_overlay_line(wax, arm_w, cmap(ci,:), 1.2, lbl, hh, leg, plotted);
            end
        end
    end

    if plotted
        hold(wax,'off'); grid(wax,'on');
        legend(wax, hh, leg, 'Interpreter','none','Location','best');
        title(wax, sprintf('ch%d gt-class%d vs %s: candidate classes %s -- mean waveform +/- std', ...
            ch, gtc, o.arm_name, mat2str(arm_classes)), 'Interpreter','none');
        xlabel(wax,'sample'); ylabel(wax,'amplitude');
        save_fig_local2(wf, fullfile(o.out_dir, sprintf('waveform_overlay_ch%d_gt%d_%s_multiclass.png', ch, gtc, o.arm_name)));
    else
        close(wf);
    end

    % ---- response metrics: pairwise (incl. GT-crosshair) + trend-across-
    % stim-list plots, one "trial" series per requested arm class (plus GT)
    % -- same real characterize_grapes_responses metrics and same plotting
    % code as analyze_cluster_membership.m's section 2, just re-labeled so
    % the "trials" being compared are classes within ONE arm instead of
    % separate arms.
    class_labels = arrayfun(@(c) sprintf('class%d', c), arm_classes, 'UniformOutput', false);
    trial_names_wf = [{o.ref_name}, class_labels];
    metrics = struct('trial',{},'gt_class',{},'arm_class',{},'stim',{},'m',{});
    Tref = run_characterize_grapes(ref_grapes_file, ch, stim_order, root);
    Tarm = run_characterize_grapes(arm_grapes_file, ch, stim_order, root);
    metrics = append_class_metrics(metrics, Tref, sprintf('class%d', gtc), o.ref_name, gtc, gtc, stim_order);
    for ci = 1:numel(arm_classes)
        acl = arm_classes(ci);
        metrics = append_class_metrics(metrics, Tarm, sprintf('class%d', acl), class_labels{ci}, gtc, acl, stim_order);
    end
    fprintf('[compare_gt_vs_arm_classes] response metrics: %d (label,stim) rows\n', numel(metrics));
    pairwise_metric_plots(metrics, trial_names_wf, o.ref_name, gtc, o.out_dir, ch);
    plot_metric_trends(metrics, gtc, stim_order, trial_names_wf, o.ref_name, o.out_dir, ch);

    T = struct2table(rows);
    csv_path = fullfile(o.out_dir, sprintf('class_compare_ch%d_gt%d_%s.csv', ch, gtc, o.arm_name));
    if ~isempty(T), writetable(T, csv_path); end
    fprintf('[compare_gt_vs_arm_classes] %d arm class(es) -> %s\n', height(T), o.out_dir);

    out = struct('table', T, 'out_dir', o.out_dir, 'gt_class', gtc, 'arm_name', o.arm_name, ...
        'arm_classes', arm_classes);
end

function [hh, leg, plotted] = add_overlay_line(ax, w, color, lw, label, hh, leg, plotted)
    mu = mean(w,1); sd = std(w,0,1); xx = 1:numel(mu);
    fill(ax, [xx fliplr(xx)], [mu+sd fliplr(mu-sd)], color, 'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
    h = plot(ax, xx, mu, '-', 'Color', color, 'LineWidth', lw);
    hh(end+1) = h; leg{end+1} = label; plotted = true;
end

% ======================================================================
%  RESPONSE METRICS (real, via characterize_grapes_responses) -- same
%  fields/plots as analyze_cluster_membership.m section 2, "trial" here
%  means "class label" (GT, or one requested arm class).
% ======================================================================
function metrics = append_class_metrics(metrics, T, class_field, trial_label, gtc, acl, stim_order)
    if isempty(T), return; end
    sel_cls = strcmp(T.class, class_field);
    for si = 1:numel(stim_order)
        snum = stim_order(si);
        r = T(sel_cls & T.stim_number==snum, :);
        if isempty(r), continue; end
        r = r(1,:);
        m = struct('zscore', r.zscore, 'mu_base', r.mu_base, 'std_base', r.std_base, ...
            'median_post', r.median_post, 'p_value_sign', r.p_value_sign, 'p_ranksum', r.p_ranksum, ...
            'auroc', r.auroc, 'resp_delta_fr_hz', r.resp_delta_fr_hz, ...
            'mean_fr_base_hz', r.mean_fr_base_hz, 'mean_fr_post_hz', r.mean_fr_post_hz, ...
            'spk_post_mean', r.spk_post_mean, 'spk_post_std', r.spk_post_std, ...
            'frac_active_trials', r.frac_active_trials, 'peak_ifr', r.peak_ifr, ...
            'peak_ifr_lat', r.peak_ifr_lat, 'onset_ms', r.onset, 'duration_ms', r.dura, ...
            'tons_ms', r.tons, 'ntrials', r.ntrials, 'ifr_thr', r.ifr_thr, ...
            'first_spk_lat_med_ms', r.first_spk_lat_med, ...
            'min_spk_test', r.min_spk_test, 'p_test', r.p_test, 'good_lat', r.good_lat);
        metrics(end+1) = struct('trial',trial_label,'gt_class',gtc,'arm_class',acl,'stim',snum,'m',m);
    end
end

function T = run_characterize_grapes(grapes_file, ch, stim_list, root)
    D = fileparts(grapes_file);
    if isempty(D), D = pwd; end
    old = pwd; c = onCleanup(@() cd(old));
    cd(D);
    [~, gbase, gext] = fileparts(grapes_file);
    res = characterize_grapes_responses('channels2plot', ch, 'stim_list', stim_list, ...
        'session_root', root, 'grapes_files', {[gbase gext]}, ...
        'outfile', fullfile(D, 'grapes_response_metrics.mat'));
    gtag = matlab.lang.makeValidName(gbase);
    if isfield(res, gtag) && isfield(res.(gtag), 'metrics')
        T = res.(gtag).metrics;
    else
        T = table();
    end
end

function pairwise_metric_plots(metrics, trial_names, ref_name, gtc, out_dir, ch)
% Every pair of class-labels (incl. arm-class-vs-arm-class), each panel
% also marking where GT sits (crosshair on the y=x line for that stim).
    if isempty(metrics), return; end
    flds = {'zscore','mu_base','std_base','median_post','p_value_sign','p_ranksum','auroc', ...
           'resp_delta_fr_hz','mean_fr_base_hz','mean_fr_post_hz','spk_post_mean','spk_post_std', ...
           'frac_active_trials','peak_ifr','peak_ifr_lat','onset_ms','duration_ms','tons_ms', ...
           'ntrials','ifr_thr','first_spk_lat_med_ms'};
    sel = [metrics.gt_class] == gtc;
    Mgt = metrics(sel);
    present = unique({Mgt.trial});
    present = trial_names(ismember(trial_names, present));
    Ref = Mgt(strcmp({Mgt.trial}, ref_name));
    [stimRef, iRef] = sort([Ref.stim]);
    for a = 1:numel(present)
        for b = a+1:numel(present)
            tA = present{a}; tB = present{b};
            A = Mgt(strcmp({Mgt.trial}, tA)); B = Mgt(strcmp({Mgt.trial}, tB));
            [stimA, iA] = sort([A.stim]); [stimB, iB] = sort([B.stim]);
            common = intersect(stimA, stimB, 'stable');
            if isempty(common), continue; end
            f = figure('Visible','off','Position',[100 100 1700 1400],'Color','w');
            tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
            title(tl, sprintf('ch%d gt-class%d: %s vs %s  (n=%d common stim, + = %s)', ...
                 ch,gtc,tA,tB,numel(common),ref_name), 'Interpreter','none');
            for fi = 1:numel(flds)
                ax = nexttile(tl); hold(ax,'on');
                xv = arrayfun(@(s) A(iA(stimA==s)).m.(flds{fi}), common);
                yv = arrayfun(@(s) B(iB(stimB==s)).m.(flds{fi}), common);
                if strcmp(flds{fi}, 'zscore')
                    % zscore = (median_post-mu_base)/(std_base+1e-10) --
                    % when std_base is ~0 (a real, observed case: a sparse
                    % class whose baseline-window spike count is 0 on every
                    % stimulus) this divides by the epsilon and explodes to
                    % ~1e10, a number with no real meaning -- NOT "extreme",
                    % undefined. Left in, it also wrecks the shared axis
                    % scale and makes every OTHER series' real zscore look
                    % like a flat line at 0. Gap those points instead of
                    % plotting a fake magnitude.
                    xv(abs(arrayfun(@(s) A(iA(stimA==s)).m.std_base, common)) < 1e-6) = NaN;
                    yv(abs(arrayfun(@(s) B(iB(stimB==s)).m.std_base, common)) < 1e-6) = NaN;
                end
                good = isfinite(xv) & isfinite(yv);
                scatter(ax, xv(good), yv(good), 22, 'filled', 'MarkerFaceAlpha',0.6, 'HandleVisibility','off');
                if any(good)
                    lo = min([xv(good) yv(good)]); hi = max([xv(good) yv(good)]);
                    if isfinite(lo) && isfinite(hi) && hi>lo
                        plot(ax,[lo hi],[lo hi],'k--', 'HandleVisibility','off');
                    end
                end
                if ~isempty(Ref)
                    gv = arrayfun(@(s) metric_lookup_row(Ref, iRef, stimRef, s, flds{fi}), common);
                    if strcmp(flds{fi}, 'zscore')
                        gv(abs(arrayfun(@(s) metric_lookup_row(Ref, iRef, stimRef, s, 'std_base'), common)) < 1e-6) = NaN;
                    end
                    gg = isfinite(gv);
                    if any(gg)
                        plot(ax, gv(gg), gv(gg), '+', 'Color',[0.85 0.1 0.1], 'MarkerSize',7, ...
                            'LineWidth',1.3, 'DisplayName', ref_name);
                    end
                end
                grid(ax,'on'); axis(ax,'square');
                xlabel(ax,tA,'Interpreter','none'); ylabel(ax,tB,'Interpreter','none');
                title(ax, flds{fi}, 'Interpreter','none','FontSize',9);
                if fi==1, legend(ax,'Location','best','FontSize',7); end
            end
            save_fig_local2(f, fullfile(out_dir, sprintf('pairwise_ch%d_gt%d_%s_vs_%s.png', ch,gtc,tA,tB)));
        end
    end
    rows = struct2table(metrics, 'AsArray', true);
    if ismember('m', rows.Properties.VariableNames)
        mm = struct2table([metrics.m]);
        rows = [removevars(rows,'m'), mm];
    end
    writetable(rows, fullfile(out_dir, sprintf('response_metrics_ch%d_gt%d.csv', ch, gtc)));
end

function val = metric_lookup(metrics, gtc, trial, stim, fname)
    val = NaN;
    idx = find([metrics.gt_class]==gtc & strcmp({metrics.trial}, trial) & [metrics.stim]==stim, 1);
    if ~isempty(idx), val = metrics(idx).m.(fname); end
end

function val = metric_lookup_row(rows, sorted_idx, sorted_stims, stim, fname)
    val = NaN;
    if isempty(rows), return; end
    k = find(sorted_stims == stim, 1);
    if ~isempty(k), val = rows(sorted_idx(k)).m.(fname); end
end

function plot_metric_trends(metrics, gtc, all_stims, trial_names, ref_name, out_dir, ch)
% one figure for this gt_class -- a line-trend panel per metric, x = stim
% (your order, whole list), one line per class label.
    flds = {'zscore','mu_base','std_base','median_post','p_value_sign','p_ranksum','auroc', ...
           'resp_delta_fr_hz','mean_fr_base_hz','mean_fr_post_hz','spk_post_mean','spk_post_std', ...
           'frac_active_trials','peak_ifr','peak_ifr_lat','onset_ms','duration_ms','tons_ms', ...
           'ntrials','ifr_thr','first_spk_lat_med_ms'};
    other_trials = setdiff(trial_names, ref_name, 'stable');
    trial_order = [ref_name, other_trials];
    cmap = lines(numel(trial_order));
    f = figure('Visible','off','Position',[80 80 1900 1400],'Color','w');
    tl = tiledlayout(f, 'flow', 'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d gt-class%d  -  trend across the whole stim list (x = stim, your order)', ...
         ch, gtc), 'Interpreter','none');
    hh = [];
    for fi = 1:numel(flds)
        ax = nexttile(tl); hold(ax,'on');
        for ti = 1:numel(trial_order)
            v = arrayfun(@(s) metric_lookup(metrics, gtc, trial_order{ti}, s, flds{fi}), all_stims);
            if strcmp(flds{fi}, 'zscore')
                % same std_base~0 guard as pairwise_metric_plots -- see its
                % comment. Without this, one class's undefined-blowup point
                % rescales the whole shared axis and flattens every other
                % class's real zscore into an indistinguishable line at 0.
                sb = arrayfun(@(s) metric_lookup(metrics, gtc, trial_order{ti}, s, 'std_base'), all_stims);
                v(abs(sb) < 1e-6) = NaN;
            end
            h = plot(ax, 1:numel(all_stims), v, '-o', 'Color', cmap(ti,:), 'MarkerSize', 3, 'LineWidth', 1.2);
            if fi==1, hh(end+1) = h; end
        end
        xticks(ax, 1:numel(all_stims)); xticklabels(ax, string(all_stims));
        set(ax,'FontSize',6); xtickangle(ax,90); grid(ax,'on');
        title(ax, flds{fi}, 'Interpreter','none', 'FontSize', 9);
    end
    lg = legend(hh, trial_order, 'Interpreter','none', 'Orientation','horizontal');
    lg.Layout.Tile = 'north';
    save_fig_local2(f, fullfile(out_dir, sprintf('trends_ch%d_gt%d.png', ch, gtc)));
end

function [w, scope] = class_waveforms(times_file, t_win, cls, match_ok, spikes_file, tol)
    w = []; scope = 'win';
    if match_ok
        t = full_class_times_from_times_file(times_file, cls); scope = 'full';
    else
        t = t_win;
    end
    if isempty(t), return; end
    S = load(spikes_file);
    if isfield(S,'spikes_all') && isfield(S,'index_all')
        SA = S.spikes_all; IA = double(S.index_all(:));
    elseif isfield(S,'spikes') && isfield(S,'index')
        SA = S.spikes; IA = double(S.index(:));
    else
        return;
    end
    idx = nearest_match(t, IA, tol);
    idx = idx(idx>0);
    if isempty(idx), return; end
    w = SA(idx,:);
end

function class_venn_and_raster(gtc, arm_name, arm_class, gt_t, arm_t, matched_gt, gt_only, arm_only, ...
        out_dir, ch, scope, n_gt_full, n_arm_full)
    n_gt = numel(gt_t); n_arm = numel(arm_t); n_m = numel(matched_gt);
    tag = sprintf('ch%d_gt%d_%s_class%d', ch, gtc, arm_name, arm_class);
    full_note = '';
    if strcmp(scope, 'windowed')
        full_note = sprintf('  [times/grapes mismatch -> windowed shown; full session: GT=%s, %s=%s]', ...
            n2s(n_gt_full), arm_name, n2s(n_arm_full));
    end

    f = figure('Visible','off','Position',[100 100 1100 380],'Color','w');
    ax = axes(f); hold(ax,'on');
    plot(ax, matched_gt/1000/60, ones(size(matched_gt)), '|', 'Color',[0.2 0.6 0.2]);
    plot(ax, gt_only/1000/60, 2*ones(size(gt_only)), '|', 'Color',[0.8 0.2 0.2]);
    plot(ax, arm_only/1000/60, 3*ones(size(arm_only)), '|', 'Color',[0.2 0.3 0.8]);
    ylim(ax,[0.5 3.5]); yticks(ax,1:3);
    yticklabels(ax, {sprintf('matched (%d)',n_m), sprintf('GT-only (%d)',numel(gt_only)), ...
                     sprintf('%s class%d-only (%d)',arm_name,arm_class,numel(arm_only))});
    set(ax,'TickLabelInterpreter','none'); xlabel(ax,'time (min)'); grid(ax,'on');
    title(ax, sprintf('ch%d gt-class%d vs %s(class%d)  |  spikes over the session  (%s)%s', ...
          ch,gtc,arm_name,arm_class,scope,full_note), 'Interpreter','none','FontSize',9);
    save_fig_local2(f, fullfile(out_dir, ['raster_' tag '.png']));

    f = figure('Visible','off','Position',[100 100 520 460],'Color','w');
    ax = axes(f); hold(ax,'on'); axis(ax,'equal'); axis(ax,'off');
    r1 = sqrt(n_gt); r2 = sqrt(n_arm); rr = max([r1 r2 1]);
    r1n = 0.9*r1/rr; r2n = 0.9*r2/rr;
    ov = n_m/max(min(n_gt,n_arm),1);
    d = (r1n+r2n)*(1-0.85*ov);
    th = linspace(0,2*pi,200);
    fill(ax, -d/2+r1n*cos(th), r1n*sin(th), [0.85 0.3 0.3], 'FaceAlpha',0.45,'EdgeColor',[0.6 0.1 0.1]);
    fill(ax,  d/2+r2n*cos(th), r2n*sin(th), [0.3 0.4 0.85], 'FaceAlpha',0.45,'EdgeColor',[0.1 0.1 0.6]);
    text(ax, -d/2-r1n*0.6, 0, sprintf('GT-only\n%d',numel(gt_only)), 'HorizontalAlignment','center','FontSize',10);
    text(ax,  d/2+r2n*0.6, 0, sprintf('%s class%d-only\n%d',arm_name,arm_class,numel(arm_only)), ...
        'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
    text(ax, 0, 0, sprintf('matched\n%d',n_m), 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
    title(ax, sprintf('ch%d gt-class%d vs %s(class%d)  (%s)\nGT n=%d, arm n=%d, %.0f%% of GT recovered%s', ...
          ch,gtc,arm_name,arm_class,scope,n_gt,n_arm,100*n_m/max(n_gt,1),full_note), ...
          'Interpreter','none','FontSize',9);
    save_fig_local2(f, fullfile(out_dir, ['venn_' tag '.png']));
end

function s = n2s(v)
    if isnan(v), s = 'n/a'; else, s = sprintf('%d', round(v)); end
end

% ======================================================================
%  duplicated from analyze_cluster_membership.m -- kept in sync manually,
%  see the module header comment. Behaviour must stay byte-identical.
% ======================================================================
function omap = build_onset_map(root)
    omap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    S = load(fullfile(root, 'stimulus.mat'));       stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat'));    pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_end_cell', 'scr_config_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));
    have_cfg = isfield(E, 'scr_config_cell');
    for scri = 1:Nscr
        po = pics_onset{scri}; st = stimulus{scri};
        p2u = [];
        if have_cfg && numel(E.scr_config_cell) >= scri && isfield(E.scr_config_cell{scri}, 'pics2use')
            p2u = E.scr_config_cell{scri}.pics2use;
        end
        for j = 1:numel(st)
            snum = resolve_stim_id(st(j).ID, p2u); terna = st(j).terna_onset;
            if ndims(po) == 3
                ts = po(sub2ind(size(po), terna(:,1), terna(:,2), terna(:,3)));
            else
                M = po(terna(:,1), terna(:,3)); ts = M(1:length(M));
            end
            ts = ts(:).';
            if ~isKey(omap, snum), omap(snum) = []; end
            omap(snum) = [omap(snum), ts];
        end
    end
end

function snum = resolve_stim_id(raw_id, pics2use)
    if raw_id == round(raw_id)
        snum = raw_id;
        return;
    end
    ipic = floor(raw_id);
    if ~isempty(pics2use) && ipic >= 1 && ipic <= numel(pics2use)
        snum = pics2use(ipic);
    else
        warning('cgva:oldFormatStimIdNoPics2use', ['st(j).ID=%.4g looks like the pre-2024-11-18 ' ...
            'ipic+iISI/10 encoding but scr_config_cell.pics2use is unavailable -- falling back to ' ...
            'round(), which is only exact when pics2use==1:N for this session.'], raw_id);
        snum = round(raw_id);
    end
end

function [match_ok, n_times, n_details] = check_class_count_agreement(times_file, grapes_file, ch, class_num)
    match_ok = false; n_times = NaN; n_details = NaN;
    if isempty(times_file) || ~exist(times_file, 'file'), return; end
    T = load(times_file, 'cluster_class');
    if ~isfield(T, 'cluster_class'), return; end
    n_times = nnz(T.cluster_class(:,1) == class_num);
    if isempty(grapes_file) || ~exist(grapes_file, 'file'), return; end
    G = load(grapes_file, 'rasters');
    cf = sprintf('chan%d', ch); fld = sprintf('class%d', class_num);
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf) || ~isfield(G.rasters.(cf),'details') || ~isfield(G.rasters.(cf).details, fld)
        return;
    end
    n_details = G.rasters.(cf).details.(fld);
    match_ok = (n_times == n_details);
end

function t = full_class_times_from_times_file(times_file, class_num)
    t = zeros(0,1);
    if isempty(times_file) || ~exist(times_file, 'file'), return; end
    T = load(times_file, 'cluster_class');
    if ~isfield(T, 'cluster_class'), return; end
    t = T.cluster_class(T.cluster_class(:,1)==class_num, 2);
end

function cc = reconstruct_class_cc(grapes_file, ch, omap)
    cc = zeros(0,2);
    G = load(grapes_file); cf = ['chan' num2str(ch)];
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf), return; end
    R = G.rasters.(cf); fn = fieldnames(R);
    cls = sort(cellfun(@(x) sscanf(x,'class%d'), fn(~cellfun('isempty', regexp(fn,'^class\d+$','once')))))';
    allt=[]; allc=[];
    for ci=1:numel(cls)
        C = sprintf('class%d', cls(ci)); stimc = R.(C).stim;
        for snum=1:numel(stimc)
            tr = stimc{snum}; if isempty(tr), continue; end
            if ~iscell(tr), tr = num2cell(tr,2); end
            if ~isKey(omap,snum), continue; end
            O = omap(snum); if numel(O)~=numel(tr), continue; end
            for t=1:numel(tr)
                rel = tr{t}(:).'; rel = rel(rel>=0 & rel<9000);
                if isempty(rel), continue; end
                allt=[allt, rel+O(t)]; allc=[allc, repmat(cls(ci),1,numel(rel))];
            end
        end
    end
    if isempty(allt), return; end
    [allt,ord]=sort(allt); allc=allc(ord);
    keep=[true, ~(diff(allt)<=0.05 & diff(allc)==0)];
    cc=[allc(keep)', allt(keep)'];
end

function rows = reconstruct_class_detail(grapes_file, ch, class_id, omap)
    rows = struct('snum',{}, 'trial_idx',{}, 'rel',{}, 'abs',{});
    G = load(grapes_file); cf = ['chan' num2str(ch)];
    if ~isfield(G,'rasters') || ~isfield(G.rasters,cf), return; end
    R = G.rasters.(cf); C = sprintf('class%d', class_id);
    if ~isfield(R, C), return; end
    stimc = R.(C).stim;
    a_snum=[]; a_tidx=[]; a_rel=[]; a_abs=[];
    for snum = 1:numel(stimc)
        tr = stimc{snum}; if isempty(tr), continue; end
        if ~iscell(tr), tr = num2cell(tr,2); end
        if ~isKey(omap, snum), continue; end
        O = omap(snum); if numel(O) ~= numel(tr), continue; end
        for t = 1:numel(tr)
            rel = tr{t}(:).'; rel = rel(rel>=0 & rel<9000);
            if isempty(rel), continue; end
            a_snum = [a_snum, repmat(snum,1,numel(rel))];
            a_tidx = [a_tidx, repmat(t,1,numel(rel))];
            a_rel  = [a_rel, rel];
            a_abs  = [a_abs, rel + O(t)];
        end
    end
    if isempty(a_abs), return; end
    [a_abs, ord] = sort(a_abs);
    a_snum = a_snum(ord); a_tidx = a_tidx(ord); a_rel = a_rel(ord);
    keep = [true, diff(a_abs) > 0.05];
    rows = struct('snum', num2cell(a_snum(keep)), 'trial_idx', num2cell(a_tidx(keep)), ...
                  'rel', num2cell(a_rel(keep)), 'abs', num2cell(a_abs(keep)));
end

function run_plot_grapes_as_online(grapes_file, is_rescue, ch, stim_order, dest_dir)
    target_dir = fileparts(grapes_file);
    if isempty(target_dir), target_dir = pwd; end
    try
        plot_grapes_as_online('target_dir', target_dir, 'channels2plot', ch, ...
            'use_blanks', true, 'rescue', is_rescue, 'stim_list', stim_order, ...
            'order_by_rank', false, 'copy2miniscrfolder', false, 'grapes_offline', true);
    catch ME
        warning('%s', ['run_plot_grapes_as_online failed for ' grapes_file ': ' ME.message]);
        return
    end
    suffix = 'offline_grapes_new_order_results';
    if is_rescue, suffix = [suffix '_rescue_post']; end
    src = fullfile(target_dir, suffix);
    if ~exist(src, 'dir')
        d = dir(fullfile(target_dir, 'offline_grapes_new_order_results*'));
        d = d([d.isdir]);
        if isempty(d), return; end
        [~, k] = max([d.datenum]); src = fullfile(target_dir, d(k).name);
    end
    if ~exist(dest_dir, 'dir'), mkdir(dest_dir); end
    movefile(fullfile(src, '*'), dest_dir);
    rmdir(src);
end

function nm = nm_for_ch(root, ch)
    NSx = getfield(load(fullfile(root, 'NSx.mat')), 'NSx');
    pos = find(arrayfun(@(x) x.chan_ID == ch, NSx), 1);
    if isempty(pos), nm = sprintf('ch%d', ch); else, nm = NSx(pos).output_name; end
end

function gf = synth_grapes_for_subset(abs_times, ch, root, template_grapes_file, out_dir, nm, circshiftblanks)
    gf = '';
    abs_times = sort(abs_times(:)');
    if isempty(abs_times), return; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    T = load(template_grapes_file, 'ImageNames', 'exp_type');
    S = load(fullfile(root, 'stimulus.mat'));    stimulus = S.stimulus;
    F = load(fullfile(root, 'finalevents.mat')); pics_onset = F.pics_onset;
    E = load(fullfile(root, 'experiment_properties_online3.mat'), 'scr_config_cell', 'scr_end_cell');
    Nscr = min(numel(stimulus), numel(E.scr_end_cell));

    grapes = struct('time_pre', 1000, 'time_pos', 2000, 'exp_type', T.exp_type, 'ImageNames', {T.ImageNames});
    for scri = 1:Nscr
        grapes = update_grapes_blanks(grapes, pics_onset{scri}, F.seq_beg_blanks_cell{scri}, ...
            stimulus{scri}, {abs_times}, ch, {nm}, 1, {[]}, ...
            E.scr_config_cell{scri}.pics2use, scri, circshiftblanks, false);
    end
    gf = fullfile(out_dir, 'grapes_blanks.mat');
    save(gf, '-struct', 'grapes');
end

function [matched_gt, matched_arm, gt_only, arm_only] = membership_sets(gt_t, arm_t, tol)
    gm = nearest_match(gt_t, arm_t, tol);
    [gm_u, gt_order] = enforce_one_to_one(gt_t, arm_t, gm, tol);
    matched_gt  = gt_t(gt_order(gm_u>0));
    matched_arm = arm_t(gm_u(gm_u>0));
    gt_only  = gt_t(gt_order(gm_u==0));
    used = false(numel(arm_t),1); used(gm_u(gm_u>0)) = true;
    arm_only = arm_t(~used);
end

function [gm_u, order] = enforce_one_to_one(gt_t, arm_t, gm, tol)
    order = (1:numel(gt_t))';
    dist = nan(size(gm));
    ok = gm>0; dist(ok) = abs(gt_t(ok) - arm_t(gm(ok)));
    [~, seq] = sort(dist);
    gm_u = zeros(size(gm));
    claimed = false(numel(arm_t),1);
    for ii = seq(:)'
        if gm(ii)==0, continue; end
        if ~claimed(gm(ii))
            gm_u(ii) = gm(ii); claimed(gm(ii)) = true;
        end
    end
end

function gm = nearest_match(gt_t, ar_t, tol)
    gm = zeros(numel(gt_t), 1);
    if isempty(ar_t), return; end
    [as, order] = sort(ar_t(:));
    for i = 1:numel(gt_t)
        k = find(as >= gt_t(i) - tol, 1, 'first');
        best = 0; bestd = tol + 1;
        while ~isempty(k) && k <= numel(as) && as(k) <= gt_t(i) + tol
            dd = abs(as(k) - gt_t(i));
            if dd < bestd, bestd = dd; best = k; end
            k = k + 1;
        end
        if best > 0, gm(i) = order(best); end
    end
end

function save_fig_local2(f, out_png)
    set(f, 'Color', 'w');
    d = fileparts(out_png);
    if ~isempty(d) && ~exist(d, 'dir'), mkdir(d); end
    try
        exportgraphics(f, out_png, 'Resolution', 120);
    catch ME
        warning('%s', ['save_fig_local2: exportgraphics failed for ' out_png ': ' ME.message]);
    end
    close(f);
end
