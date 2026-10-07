function R = plot_response_spikes(arm1_ref)
% PLOT_RESPONSE_SPIKES  Waveforms of the spikes inside the stim-list response
% windows, GT vs arm 1, with the arm's matched (GT & arm) and sd1 (II)
% templates.
%
%   arm1_ref = load(fullfile(cfg.out_dir, 'arm1_ref.mat'));   % saved by run_validation
%   R = plot_response_spikes(arm1_ref);
%
% or, inside run_validation right after the raster loop,
%   plot_response_spikes(arm1_ref)   % arm1_ref = the arm-1 table there; the
%                                    % other inputs come from run_validation's workspace
%
% arm1_ref is the run_validation workspace saved after the raster loop; used
% from it, with nothing recomputed:
%   .arm1_ref / .gt_ref - arm-1 / GT data_table from plot_response_window_grapes
%                         (stim_number, win_lo, win_hi = the raster windows)
%   .onset_map          - stim number -> trial onset times (ms)
%   .times_gt / .times_arms{1} - GT / arm-1 times files (cluster_class, spikes)
%   .cfg, .arm_classes(1), .arm_names{1}
%
% A spike of the class is "in" when it falls in [trial onset + win_lo,
% trial onset + win_hi] for any trial of any stimulus in the table (rows with
% NaN windows -- no latency -- contribute none).
%
% Page: GT in window | arm in window / GT only | GT & arm matched / arm only,
% all in window, matching one-to-one within cfg.tol on the whole class
% populations; one fixed y range (and the same x range) on every panel.
%
% R - in-window spike rows and per-stimulus counts, for GT and the arm, plus
%     R.gt_only / R.matched (GT-file rows) and R.arm_only (arm-file rows)
%
% OUTPUT: cfg.out_dir/response_waveforms/response_waveforms_ch<ch>_<arm>.png

if istable(arm1_ref)
    % called inside run_validation as plot_response_spikes(arm1_ref): there
    % arm1_ref is the arm-1 table, and the rest is in run_validation's
    % workspace at that line -- take it from there, nothing is reloaded
    S = struct('arm1_ref', arm1_ref);
    for v = {'gt_ref', 'onset_map', 'times_gt', 'times_arms', 'cfg', 'arm_classes', 'arm_names'}
        S.(v{1}) = evalin('caller', v{1});
    end
    arm1_ref = S;
end
dt_k      = arm1_ref.arm1_ref;
gt_ref    = arm1_ref.gt_ref;
onset_map = arm1_ref.onset_map;
times_gt  = arm1_ref.times_gt;
times_arm = arm1_ref.times_arms{1};
cfg       = arm1_ref.cfg;
arm_class = arm1_ref.arm_classes(1);
arm_name  = arm1_ref.arm_names{1};

[R.gt.rows, R.gt.per_stim]   = in_window_rows(times_gt.cluster_class, cfg.gt.class, gt_ref, onset_map);
[R.arm.rows, R.arm.per_stim] = in_window_rows(times_arm.cluster_class, arm_class, dt_k, onset_map);

% templates and the shared y range of the other waveform pages
gt_t = times_gt.cluster_class(times_gt.cluster_class(:,1) == cfg.gt.class, 2);
G = arm_wave_groups(times_arm, arm_class, gt_t, cfg.tol);
yl = wave_ylim({G});

% GT vs arm, one-to-one within cfg.tol on the whole class populations (as in
% the venn), then restricted to the in-window spikes
gt_cls_rows  = find(times_gt.cluster_class(:,1) == cfg.gt.class);
arm_cls_rows = find(times_arm.cluster_class(:,1) == arm_class);
[gt_sorted_t, o_gt] = sort(times_gt.cluster_class(gt_cls_rows, 2));
gm = nearest_match(gt_sorted_t, times_arm.cluster_class(arm_cls_rows, 2), cfg.tol);
gm_u = enforce_one_to_one(gt_sorted_t, times_arm.cluster_class(arm_cls_rows, 2), gm, cfg.tol);
gt_matched = false(size(times_gt.cluster_class, 1), 1);
gt_matched(gt_cls_rows(o_gt(gm_u > 0))) = true;
arm_matched = false(size(times_arm.cluster_class, 1), 1);
arm_matched(arm_cls_rows(gm_u(gm_u > 0))) = true;

R.gt_only  = R.gt.rows(~gt_matched(R.gt.rows));
R.matched  = R.gt.rows(gt_matched(R.gt.rows));
R.arm_only = R.arm.rows(~arm_matched(R.arm.rows));

Wg = @(rows) double(times_gt.spikes(rows, :));
Wa = @(rows) double(times_arm.spikes(rows, :));
panels = { ...
    struct('name', sprintf('%s class %d, in response window', cfg.gt.name, cfg.gt.class), 'W', Wg(R.gt.rows), 'color', [0.85 0.3 0.3]), ...
    struct('name', sprintf('%s class %d, in response window', arm_name, arm_class), 'W', Wa(R.arm.rows), 'color', [0.3 0.4 0.85]), ...
    struct('name', sprintf('%s only (not in %s), in window', cfg.gt.name, arm_name), 'W', Wg(R.gt_only), 'color', [0.85 0.3 0.3]), ...
    struct('name', sprintf('%s & %s matched, in window', cfg.gt.name, arm_name), 'W', Wg(R.matched), 'color', [0.2 0.7 0.3]), ...
    struct('name', sprintf('%s only (not in %s), in window', arm_name, cfg.gt.name), 'W', Wa(R.arm_only), 'color', [0.3 0.4 0.85])};

% one y range for every panel: the shared waveform-page range, widened if
% any panel's mean +/- std goes past it
for pi_ = 1:numel(panels)
    Wp = panels{pi_}.W;
    if size(Wp, 1) >= 5
        mu = mean(Wp, 1); sdv = std(Wp, 0, 1);
        yl = [min(yl(1), min(mu - sdv)), max(yl(2), max(mu + sdv))];
    end
end

f = figure('Visible', 'off', 'Position', [100 100 1500 1450], 'Color', 'w');
annotation(f, 'textbox', [0 0.955 1 0.04], 'EdgeColor', 'none', 'FontSize', 13, 'FontWeight', 'bold', ...
    'Interpreter', 'none', 'HorizontalAlignment', 'center', 'String', ...
    sprintf('ch%d  spikes inside the stim-list response windows (%d stimuli)', cfg.ch, height(dt_k)));
pos = {[0.06 0.67 0.42 0.25], [0.55 0.67 0.42 0.25], ...
       [0.06 0.36 0.42 0.25], [0.55 0.36 0.42 0.25], ...
       [0.06 0.05 0.42 0.25]};
for pi_ = 1:numel(panels)
    ax = axes(f, 'Position', pos{pi_});
    draw_wave_group(ax, panels{pi_}, G.templates, yl, pi_ == 1);
    xlabel(ax, 'sample');
    if mod(pi_, 2) == 1, ylabel(ax, 'amplitude'); end
end

out_dir = fullfile(cfg.out_dir, 'response_waveforms');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
print(f, fullfile(out_dir, sprintf('response_waveforms_ch%d_%s.png', cfg.ch, arm_name)), '-dpng', '-r130');
close(f);
fprintf('[response] %s: GT %d in-window (only %d, matched %d), %s %d (only %d)\n', arm_name, ...
    numel(R.gt.rows), numel(R.gt_only), numel(R.matched), arm_name, numel(R.arm.rows), numel(R.arm_only));
end

function [rows, per] = in_window_rows(cc, cls, dt, onset_map)
% rows of cc (class CLS) inside any stimulus's trial windows from data_table DT
    t = cc(:,2);
    in_cls = cc(:,1) == cls;
    hit = false(size(t));
    n_in = zeros(height(dt), 1);
    for i = 1:height(dt)
        s = dt.stim_number(i);
        if ~isKey(onset_map, s) || ~isfinite(dt.win_lo(i)) || ~isfinite(dt.win_hi(i)), continue; end
        t0 = onset_map(s);
        h = false(size(t));
        for tr = 1:numel(t0)
            h = h | (t >= t0(tr) + dt.win_lo(i) & t <= t0(tr) + dt.win_hi(i));
        end
        h = h & in_cls;
        n_in(i) = nnz(h);
        hit = hit | h;
    end
    rows = find(hit);
    per = table(dt.stim_number(:), n_in, 'VariableNames', {'stim_number', 'n_in_window'});
end
