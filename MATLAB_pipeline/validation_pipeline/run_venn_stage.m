function run_venn_stage(cfg, arm_classes)
% RUN_VENN_STAGE  GT / arm / quarantined three-circle venns, per arm:
%   cluster - GT's class vs the arm's matched class
%   full    - every GT spike in a class > 0 vs every arm spike in a class > 0
% The quarantined set (quarantine.all, every rejector) is the same in both;
% only the GT and arm sets change.
%
%   run_venn_stage(cfg)               % matches arm classes itself
%   run_venn_stage(cfg, arm_classes)  % reuse run_validation's matching
%
% Needs the arm times file in the Python layout (full length, every spike,
% with a quarantine struct) -- cluster_class rows line up 1:1 with
% quarantine.all.
%
% Each page is the venn plus waveform panels for its regions (arm-file
% waveforms, except GT-only which only exists in the GT file):
% GT vs arm is the comparison (three_set_membership matches GT to the arm
% alone); quarantine only splits those sets. Waveform panels:
%   GT & arm matched -- every matched spike, rescued or not (counts of each in the title) |
%   quarantined, not in arm (GT part = lost, in its own color) |
%   rescued into arm -- every rescued arm spike, GT or not |
%   arm only -- arm not in GT and not rescued |
%   GT only -- every GT spike not in the arm (GT-file waveforms)
% with two templates overlaid on every panel: the matched template (mean of
% every GT & arm spike) and the sd1 first-pass template of the matched class
% (sd1_pass_templates, clustering pass -- the rescue pass's sd1 template comes
% out effectively the same, so only one is drawn; the max difference is printed).
%
% OUTPUTS (under cfg.out_dir/venn_three_set)
%   venn3_ch<ch>_<arm>_cluster.png / _full.png
%   venn3_ch<ch>_<arm>_detections.png - only with cfg.venn_detections = true: two-set
%       venn of every GT spike vs every arm spike on the channel (any class, incl. 0,
%       quarantined or not), waveforms of both / GT only / arm only
%   venn3_counts.csv - every region count, one row per (arm, scope)

here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'real_response_analysis'));
addpath(fullfile(fileparts(here), 'responses'));

ch = cfg.ch;
out_dir = fullfile(cfg.out_dir, 'venn_three_set');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

times_gt = load(fullfile(cfg.gt.dir, cfg.gt.times_file));
gt_cc = times_gt.cluster_class;
arms = cfg.arms;
n_arms = numel(arms);
times_arms = cell(1, n_arms);
for k = 1:n_arms
    times_arms{k} = load(fullfile(arms(k).dir, arms(k).times_file));
end

if nargin < 2 || isempty(arm_classes)
    arm_classes = resolve_arm_classes(cfg, times_gt, times_arms);
end

rows = {};
for k = 1:n_arms
    T = times_arms{k};
    if ~isfield(T, 'quarantine') || ~isfield(T.quarantine, 'all')
        warning('run_venn_stage: %s has no quarantine.all, skipping.', arms(k).name);
        continue
    end
    lab = T.cluster_class(:,1); t = T.cluster_class(:,2);
    q = logical(T.quarantine.all(:));
    if numel(q) ~= numel(lab)
        warning('run_venn_stage: %s quarantine.all (%d) != cluster_class (%d) rows, skipping.', ...
            arms(k).name, numel(q), numel(lab));
        continue
    end
    names = {cfg.gt.name, arms(k).name, 'quarantined'};
    W_arm = double(T.spikes);
    [tmpl_clu, tmpl_res, tinfo] = sd1_pass_templates(T, arm_classes(k), 1, [1 99]);
    fprintf('[venn3] %s class %d sd1 templates: clustering %d seeds + %d core, rescue %d seeds + %d core, max |diff| = %.2f\n', ...
        arms(k).name, arm_classes(k), tinfo.clu_seed, tinfo.clu_core, tinfo.res_seed, tinfo.res_core, ...
        max(abs(tmpl_clu - tmpl_res)));
    if size(times_gt.spikes, 2) ~= size(W_arm, 2)
        warning('run_venn_stage: GT waveforms have %d samples, %s has %d.', ...
            size(times_gt.spikes,2), arms(k).name, size(W_arm,2));
    end

    scopes = struct( ...
        'name',   {'cluster', 'full'}, ...
        'gt_rows',{find(gt_cc(:,1) == cfg.gt.class), find(gt_cc(:,1) > 0)}, ...
        'in_arm', {lab == arm_classes(k), lab > 0}, ...
        'desc',   {sprintf('%s class %d vs %s class %d', cfg.gt.name, cfg.gt.class, arms(k).name, arm_classes(k)), ...
                   sprintf('%s all classes vs %s all classes', cfg.gt.name, arms(k).name)});

    % all detections (cfg.venn_detections): every GT spike vs every arm spike
    % on the channel, any class incl. 0 -- two sets, no quarantine split:
    % how the two detectors differ, not the sorting
    if isfield(cfg, 'venn_detections') && cfg.venn_detections
        D = plot_detection_venn(cfg, arms(k).name, gt_cc, times_gt, t, W_arm, tmpl_clu, arm_classes(k), out_dir);
        rows(end+1, :) = {arms(k).name, 'detections', arm_classes(k), D.n_gt, D.n_arm, 0, ...
            D.gt_only, D.arm_only, 0, D.both, 0, 0, 0, 0}; %#ok<AGROW>
        fprintf('[venn3] %s detections: gt=%d arm=%d | both %d | gt only %d | arm only %d\n', ...
            arms(k).name, D.n_gt, D.n_arm, D.both, D.gt_only, D.arm_only);
    end

    for si = 1:numel(scopes)
        sc = scopes(si);
        S = three_set_membership(gt_cc(sc.gt_rows, 2), lab, t, q, sc.in_arm, cfg.tol);
        ttl = sprintf('ch%d  %s  (%s)  -- GT match tol %.1f ms', ch, sc.desc, sc.name, cfg.tol);

        R = S.rows;
        G = @(rows) W_arm(rows, :);
        wf.panels = struct( ...
            'title', {sprintf('%s & %s matched', names{1}, names{2}), ...
                      sprintf('quarantined, not in %s', names{2}), ...
                      sprintf('rescued into %s', names{2}), ...
                      sprintf('%s only (not in %s, not rescued)', names{2}, names{1}), ...
                      sprintf('%s only (not in %s)', names{1}, names{2})}, ...
            'sets', {wset(G([R.gt_arm; R.gt_arm_q]), [0.2 0.7 0.3], ...
                          sprintf('not rescued %d + rescued %d, total', numel(R.gt_arm), numel(R.gt_arm_q))), ...
                     [wset(G(R.q_only), [0.95 0.6 0.1], sprintf('not in %s', names{1})), ...
                      wset(G(R.gt_q), [0.6 0.05 0.05], sprintf('in %s (lost)', names{1}))], ...
                     wset(G([R.arm_q; R.gt_arm_q]), [0.1 0.6 0.7], 'rescued'), ...
                     wset(G(R.arm_only), [0.3 0.4 0.85], sprintf('%s only', names{2})), ...
                     wset(double(times_gt.spikes(sc.gt_rows([S.gt_only_pos; S.gt_lost_pos]), :)), [0.85 0.3 0.3], ...
                          sprintf('%s only', names{1}))});
        % venn regions filled in the color of the panel that shows them; the
        % GT & arm overlap is one region color (matched, regardless of rescue)
        wf.region_colors = struct('gt_arm', [0.2 0.7 0.3], 'gt_arm_q', [0.2 0.7 0.3], ...
            'q_only', [0.95 0.6 0.1], 'gt_q', [0.6 0.05 0.05], 'arm_q', [0.1 0.6 0.7], ...
            'arm_only', [0.3 0.4 0.85], 'gt_only', [0.85 0.3 0.3]);
        wf.templates = struct( ...
            'name',  {'matched template (GT & arm)', ...
                      sprintf('sd1 template (class %d)', arm_classes(k))}, ...
            'wave',  {mean(G([R.gt_arm; R.gt_arm_q]), 1), tmpl_clu}, ...
            'color', {[0 0 0], [0.85 0.1 0.6]}, ...
            'style', {'-', '--'}, ...
            'width', {1.8, 1.8});

        plot_three_set_venn(S, names, ttl, ...
            fullfile(out_dir, sprintf('venn3_ch%d_%s_%s.png', ch, arms(k).name, sc.name)), wf);

        c = S.counts;
        rows(end+1, :) = {arms(k).name, sc.name, arm_classes(k), S.n_gt, S.n_arm, S.n_q, ...
            c.gt_only, c.arm_only, c.q_only, c.gt_arm, c.gt_q, c.arm_q, c.gt_arm_q, S.q_rescued_elsewhere}; %#ok<AGROW>
        fprintf('[venn3] %s %s: gt=%d arm=%d q=%d | gt&arm=%d gt&q(lost)=%d arm&q(rescued, not gt)=%d all3=%d\n', ...
            arms(k).name, sc.name, S.n_gt, S.n_arm, S.n_q, c.gt_arm, c.gt_q, c.arm_q, c.gt_arm_q);
    end
end

if ~isempty(rows)
    writetable(cell2table(rows, 'VariableNames', {'arm','scope','arm_class','n_gt','n_arm','n_quarantined', ...
        'gt_only','arm_only','q_only','gt_arm','gt_q','arm_q','gt_arm_q','q_rescued_other_class'}), ...
        fullfile(out_dir, 'venn3_counts.csv'));
end
end

function D = plot_detection_venn(cfg, arm_name, gt_cc, times_gt, arm_t, W_arm, tmpl_clu, cls, out_dir)
% Two-set venn of every GT spike vs every arm spike on the channel (any
% class, incl. 0; quarantined or not -- all detections), one-to-one within
% cfg.tol, with waveform panels: detected by both | GT only | arm only.
    [gs, go] = sort(gt_cc(:, 2));
    gm = nearest_match(gs, arm_t, cfg.tol);
    gm_u = enforce_one_to_one(gs, arm_t, gm, cfg.tol);
    gt_hit = false(size(gs)); gt_hit(gm_u > 0) = true;
    arm_hit = false(size(arm_t)); arm_hit(gm_u(gm_u > 0)) = true;
    D = struct('n_gt', numel(gs), 'n_arm', numel(arm_t), 'both', nnz(gt_hit), ...
               'gt_only', nnz(~gt_hit), 'arm_only', nnz(~arm_hit));
    f = figure('Visible', 'off', 'Position', [100 100 2000 760], 'Color', 'w');
    ax = axes(f, 'Position', [0.03 0.10 0.29 0.70]);
    draw_membership_venn(ax, 0, arm_name, 0, D.n_gt, D.n_arm, D.both, D.gt_only, D.arm_only, cfg.gt.name, cfg.ch);
    xlim(ax, [-2.0 2.0]); ylim(ax, [-1.1 1.1]);
    title(ax, sprintf('%s n=%d  |  %s n=%d\nboth %d = %.1f%% of %s, %.1f%% of %s', cfg.gt.name, D.n_gt, ...
        arm_name, D.n_arm, D.both, 100*D.both/max(D.n_gt,1), cfg.gt.name, 100*D.both/max(D.n_arm,1), arm_name), ...
        'Interpreter', 'none', 'FontSize', 11);
    grp = @(name, W, color) struct('name', name, 'W', W, 'color', color);
    G = [grp(sprintf('detected by both (n=%d)', D.both), W_arm(gm_u(gm_u > 0), :), [0.2 0.7 0.3]), ...
         grp(sprintf('%s only (n=%d)', cfg.gt.name, D.gt_only), double(times_gt.spikes(go(~gt_hit), :)), [0.85 0.3 0.3]), ...
         grp(sprintf('%s only (n=%d)', arm_name, D.arm_only), W_arm(~arm_hit, :), [0.3 0.4 0.85])];
    T = struct('name', sprintf('sd1 template (class %d)', cls), 'wave', tmpl_clu, 'color', [0.85 0.1 0.6], 'style', '--');
    yl = [inf -inf];
    for g = G
        if size(g.W, 1) >= 5
            mu = mean(g.W, 1); sd = std(g.W, 0, 1);
            yl = [min(yl(1), min(mu - sd)), max(yl(2), max(mu + sd))];
        end
    end
    yl = yl + [-0.08 0.08] * diff(yl);
    for i = 1:numel(G)
        a = axes(f, 'Position', [0.36 + (i-1)*0.215, 0.12, 0.19, 0.70]);
        draw_wave_group(a, G(i), T, yl, i == 1);
        title(a, G(i).name, 'Interpreter', 'none', 'FontSize', 11);
        xlabel(a, 'sample'); if i == 1, ylabel(a, 'amplitude'); end
    end
    annotation(f, 'textbox', [0 0.92 1 0.07], 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 13, ...
        'FontWeight', 'bold', 'Interpreter', 'none', 'String', sprintf(['ch%d  all %s spikes vs all %s spikes on the channel ' ...
        '(every class incl. 0, quarantined or not) -- detectors, not sorting; spike-time match %.1f ms'], cfg.ch, cfg.gt.name, arm_name, cfg.tol));
    print(f, fullfile(out_dir, sprintf('venn3_ch%d_%s_detections.png', cfg.ch, arm_name)), '-dpng', '-r130');
    close(f);
end

function s = wset(W, color, name)
    s = struct('W', W, 'color', color, 'name', name);
end
