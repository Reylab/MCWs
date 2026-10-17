function data_table = plot_response_window_grapes(grapes, per_stim, ch, class_num, n_response_spikes, ...
                                     win_lo_vec, win_hi_vec, rank_config, ejex, ss_num, ...
                                     extra_label_base, output_dir, root, unit_field, clip_to_window, ...
                                     cluster_total, gt_ref, varargin)
% PLOT_RESPONSE_WINDOW_GRAPES  Single-channel/class raster+PSTH grid (one
% arm, one window definition), built the same way plot_channel_grapes.m
% builds its per-(channel,class) jobs, but scoped down to just the one
% (channel,class) we already know we want -- no multi-channel looping, no
% parallel batching. Two differences from the real production plot:
%   1) the figure title shows n_response_spikes (this arm+window's IN-WINDOW
%      count) instead of grapes.rasters.chanN.details.classN (the full,
%      whole-session class population)
%   2) every panel gets two extra green lines at win_lo(stim)/win_hi(stim) --
%      the actual in/out classification boundary for THIS window definition,
%      which is not the same as the existing onset/dura or win_cent markers
%
% VARIABLES
%   grapes        - loaded grapes struct for this arm (e.g. grapes_gt/grapes_sd1/grapes_sd2)
%   per_stim      - this class's per-stim metrics table (get_class_metrics' m.per_stim),
%                   MUST already be restricted+ordered to the stimuli you want plotted
%   ch/class_num  - which channel/class to plot
%   n_response_spikes - the IN-WINDOW spike count for this arm+window (from
%                   find_response_spikes/find_response_spikes_dynamic's `inside` length)
%   win_lo_vec/win_hi_vec - per-stimulus window bounds (ms, relative to onset),
%                   SAME length/order as per_stim's rows -- fixed window: constant
%                   [-def_pr_pre, +def_pr_pos] repeated; dynamic window: per-stim
%                   [onset-buffer, onset+dura+buffer]
%   rank_config/ejex - from get_class_metrics (m.rank_config/m.ejex)
%   ss_num        - sub-screening count, cosmetic (figure title text only)
%   extra_label_base - short tag identifying this arm+window (e.g. 'gt_fixed_pm100ms')
%   output_dir    - where the PNG(s) land
%
%   cf/cls        - 'chan<N>'/'class<N>' field names
%   data_table    - per_stim + win_lo/win_hi columns, what the plotting engine reads
%   channel_grapes- minimal grapes struct: just this channel's class raster + details
%   ifr_thr       - this class's IFR threshold (same value on every per_stim row)
%   lbl           - figure title string, built with n_response_spikes (not the full-class count)
%   unit_field    - OPTIONAL. Omit or [] to use ['class' num2str(class_num)] (the normal case).
%                   Pass a literal field name (e.g. 'mu') to plot a raster field that isn't a
%                   numbered class -- used for synth_grapes_for_subset outputs (an exclusive
%                   membership-set raster has no real class, only a .mu field). When set, also
%                   overrides data_table's own 'class' column so it matches the raster field
%                   actually being read (per_stim is typically borrowed from a different arm's
%                   real per-stim table in that case, so its 'class' column would say the wrong
%                   thing otherwise).
%   clip_to_window - OPTIONAL, default false. When true, each panel's raster ticks are
%                   restricted to [win_lo,win_hi] instead of showing the full raw window --
%                   use for a sparse/exclusive spike set where showing the full window makes
%                   it look like spikes belong to trials they weren't actually classified to.
%   cluster_total - OPTIONAL, default []. This class's WHOLE session-wide spike count
%                   (a single number, e.g. numel(full_t)). When given, the engine's 4th
%                   sub-row shows a small bar chart of this stimulus's own response-window
%                   count vs. cluster_total, and the image panel adds a mu_base/std_base
%                   text line (per_stim already carries those columns) -- omit to skip
%                   both and keep the original 3-sub-row layout (e.g. test_rescue_grape_metrics.m).
%   gt_ref        - OPTIONAL, default []. GT's own returned data_table (this function's
%                   output, from the GT call in run_rescue_grape_analysis.m's stimulus_rasters
%                   loop) -- when given, this stimulus's arm values are joined against GT's
%                   SAME stim_number row to add gt_n_response/gt_median_post/gt_mean_post/
%                   gt_mu_base/gt_std_base/gt_cluster_total/gt_zscore columns to data_table,
%                   which the engine reads for its GT-vs-ARM paired-line summary figure
%                   (plot_gt_arm_summary_figure). Omit (or pass [] / call for GT
%                   itself) to skip -- the engine just leaves that figure out.
%
%   varargin      - OPTIONAL name-value pairs forwarded to loop_plot_responses_BCM_response_window
%                   (e.g. 'x_jitter', 0.12 to jitter the GT-vs-ARM paired-line panels).
%
% RETURNS
%   data_table    - the per_stim table actually plotted (win_lo/win_hi, and when applicable
%                   n_response/cluster_total/gt_* columns), so a GT call's output can be
%                   captured and reused as the gt_ref argument for later arm calls.

    if nargin < 14, unit_field = []; end
    if nargin < 15, clip_to_window = false; end
    if nargin < 16, cluster_total = []; end
    if nargin < 17, gt_ref = []; end
    cf  = ['chan' num2str(ch)];
    if isempty(unit_field)
        cls = ['class' num2str(class_num)];
    else
        cls = unit_field;
    end

    if ~exist(output_dir,'dir'), mkdir(output_dir); end

    data_table = per_stim;
    data_table.win_lo = win_lo_vec(:);
    data_table.win_hi = win_hi_vec(:);
    if ~isempty(unit_field)
        data_table.class = repmat({cls}, height(data_table), 1);
    end

    channel_grapes            = struct;
    channel_grapes.time_pre   = grapes.time_pre;
    channel_grapes.time_pos   = grapes.time_pos;
    channel_grapes.exp_type   = grapes.exp_type;
    % raw grapes.mat stores ImageNames as a plain name list -- the raster
    % plotting code (plot_img_zscore_class) expects a folder/name TABLE, the
    % same conversion plot_grapes_as_online.m does before it ever reaches
    % this plotting code (codes_emu/online_v3/plot_grapes_as_online.m:210).
    if istable(grapes.ImageNames)
        channel_grapes.ImageNames = grapes.ImageNames;
    else
        channel_grapes.ImageNames = table(repmat({fullfile(root, 'pics_used')}, ...
                                          numel(grapes.ImageNames), 1), grapes.ImageNames(:), ...
                                          'VariableNames', {'folder', 'name'});
    end
    channel_grapes.rasters    = struct;
    channel_grapes.rasters.(cf).(cls)     = grapes.rasters.(cf).(cls);
    channel_grapes.rasters.(cf).details   = grapes.rasters.(cf).details;
    if isfield(grapes,'ISI_min')
        channel_grapes.ISI_min = grapes.ISI_min;
    end

    ifr_thr = data_table.ifr_thr(1);
    % cluster_total is a single session-wide constant (same number on every
    % stimulus), so it belongs ONCE in the page title, not repeated per
    % stimulus -- fold it in here when given. When cluster_total equals
    % n_response_spikes (e.g. an exclusive/derived subset -- matched/gt_only/
    % arm_only -- passed here purely to turn on the per-stimulus n=NNN count,
    % see below, not because there's a real distinct "whole cluster" to
    % compare against), showing "X/X resp/cluster spks" is a meaningless
    % ratio -- collapse to the plain single-number form instead.
    if ~isempty(cluster_total) && cluster_total ~= n_response_spikes
        lbl = sprintf('%s_ch_%s_%s (%d/%d resp/cluster spks)(ifr_t=%.1fHz)', extra_label_base, ...
                      grapes.rasters.(cf).details.ch_label, cls, n_response_spikes, cluster_total, ifr_thr);
    else
        lbl = sprintf('%s_ch_%s_%s (%d response spks)(ifr_t=%.1fHz)', extra_label_base, ...
                      grapes.rasters.(cf).details.ch_label, cls, n_response_spikes, ifr_thr);
    end
    % inside = spikes in the green windows of the stimuli on this page (the stim list only, every
    % presentation); outside = every other spike of the unit / subset
    if ~isempty(cluster_total) && cluster_total > 0
        n_out = cluster_total - n_response_spikes;
        lbl = sprintf('%s | in %d (%.1f%% of all) | out %d (%.1f%% of all) | in/out %.1f%%', lbl, ...
                      n_response_spikes, 100 * n_response_spikes / cluster_total, n_out, ...
                      100 * n_out / cluster_total, 100 * n_response_spikes / max(n_out, 1));
    end

    % OPTIONAL: per-stimulus response count AND mean_post, folded into the
    % image panel's own mu/std text line (see
    % loop_plot_responses_BCM_response_window.m's plot_img_zscore_class)
    % instead of a dedicated subplot row -- a whole panel for one number
    % that changes per stimulus was still only ONE bar, a poor use of a
    % full subplot. Both computed straight from the SAME raster trial data
    % plot_rasters itself draws (win_lo/win_hi are already in that same
    % trial-relative coordinate space), so the numbers match what's visible
    % in the raster above it:
    %   n_response = TOTAL in-window spikes, pooled across all this
    %                stimulus's trials (matches the raster's raw tick count)
    %   mean_post  = the ARITHMETIC MEAN in-window count PER TRIAL -- a true
    %                mean, distinct from the production pipeline's
    %                median_post (see codes_emu/tasks_tools/
    %                create_responses_data_parallel.m, which only ever
    %                computes a per-trial MEDIAN, never a mean) -- computed
    %                here since nothing upstream provides it
    % Skipped entirely when cluster_total isn't passed -- existing callers
    % (test_rescue_grape_metrics.m) keep working unchanged, just without
    % this text. cluster_total ITSELF is still stored per-row below
    % (data_table.cluster_total) for any caller that wants the whole-class
    % total alongside these.
    if ~isempty(cluster_total)
        n_stims = height(data_table);
        n_response_per_stim = zeros(n_stims,1);
        mean_post_per_stim = nan(n_stims,1);
        for i = 1:n_stims
            trials_i = channel_grapes.rasters.(cf).(cls).stim{data_table.stim_number(i)};
            if ~iscell(trials_i), trials_i = num2cell(trials_i, 2); end
            per_trial_counts = zeros(numel(trials_i),1);
            for tt = 1:numel(trials_i)
                v = trials_i{tt};
                per_trial_counts(tt) = sum(v >= data_table.win_lo(i) & v <= data_table.win_hi(i));
            end
            n_response_per_stim(i) = sum(per_trial_counts);
            if ~isempty(per_trial_counts)
                mean_post_per_stim(i) = mean(per_trial_counts);
            end
        end
        data_table.n_response = n_response_per_stim;
        data_table.mean_post = mean_post_per_stim;
        data_table.cluster_total = repmat(cluster_total, n_stims, 1);
    end

    % OPTIONAL: join GT's reference values onto this arm's data_table by
    % stim_number, for the engine's GT-vs-ARM paired-line summary figure
    % (plot_gt_arm_summary_figure). Only meaningful when gt_ref itself
    % carries n_response/cluster_total (i.e. GT's own call was also given a
    % cluster_total) -- otherwise there's nothing to compare against, so skip
    % quietly and leave data_table without gt_* columns (the engine turns
    % that figure out in that case too). gt_mu_base/gt_std_base/gt_cluster_total are
    % pooled, session-wide CONSTANTS (see codes_emu/tasks_tools/
    % create_responses_data_parallel.m) -- still joined below since
    % plot_gt_arm_summary_figure and test_rescue_grape_metrics.m-style
    % callers use them.
    if ~isempty(gt_ref) && all(ismember({'n_response','cluster_total','median_post','mean_post','mu_base','std_base','zscore'}, ...
                                         gt_ref.Properties.VariableNames))
        n_stims = height(data_table);
        gt_n_response    = nan(n_stims,1);
        gt_median_post   = nan(n_stims,1);
        gt_mean_post     = nan(n_stims,1);
        gt_mu_base       = nan(n_stims,1);
        gt_std_base      = nan(n_stims,1);
        gt_cluster_total = nan(n_stims,1);
        gt_zscore        = nan(n_stims,1);
        for i = 1:n_stims
            gi = find(gt_ref.stim_number == data_table.stim_number(i), 1);
            if isempty(gi), continue; end
            gt_n_response(i)    = gt_ref.n_response(gi);
            gt_median_post(i)   = gt_ref.median_post(gi);
            gt_mean_post(i)     = gt_ref.mean_post(gi);
            gt_mu_base(i)       = gt_ref.mu_base(gi);
            gt_std_base(i)      = gt_ref.std_base(gi);
            gt_cluster_total(i) = gt_ref.cluster_total(gi);
            gt_zscore(i)        = gt_ref.zscore(gi);
        end
        data_table.gt_n_response    = gt_n_response;
        data_table.gt_median_post   = gt_median_post;
        data_table.gt_mean_post     = gt_mean_post;
        data_table.gt_mu_base       = gt_mu_base;
        data_table.gt_std_base      = gt_std_base;
        data_table.gt_cluster_total = gt_cluster_total;
        data_table.gt_zscore        = gt_zscore;
        % GT's own per-stimulus sign-test p (same test the production grapes
        % plots mark with '*'), for the GT-vs-ARM summary and sign-test pages
        if ismember('p_value_sign', gt_ref.Properties.VariableNames)
            gt_p_value_sign = nan(n_stims,1);
            for i = 1:n_stims
                gi = find(gt_ref.stim_number == data_table.stim_number(i), 1);
                if ~isempty(gi), gt_p_value_sign(i) = gt_ref.p_value_sign(gi); end
            end
            data_table.gt_p_value_sign = gt_p_value_sign;
        end
    end
    % every call, with or without a GT join (GT itself, subsets): the numbers
    % behind the page
    save(fullfile(output_dir,'data_table.mat'),'data_table');

    nwins2plot = max(1, ceil(height(data_table)/20));

    loop_plot_responses_BCM_response_window(data_table, channel_grapes, ss_num, nwins2plot, ...
        rank_config, ejex, true, lbl, true, 0, [], true, false, false, false, ...
        'output_dir', output_dir, 'clip_to_window', clip_to_window, varargin{:});
end
