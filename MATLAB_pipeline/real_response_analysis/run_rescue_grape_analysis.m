function run_rescue_grape_analysis(stim_list, class_num)
% RUN_RESCUE_GRAPE_ANALYSIS  Orchestrates the GT-vs-rescue analysis using
% the generalized stage functions in real_response_analysis/
% (match_arms_to_ground_truth, get_class_metrics, compute_dynamic_window_sets,
% compute_response_window_sets, build_arm_summary_table) plus the existing
% plotting functions, instead of test_rescue_grape_metrics.m's single
% hand-duplicated script.
%
% test_rescue_grape_metrics.m is left as-is (untouched, still the original
% reference implementation). Everything here should be arm-count-agnostic
% (loop over ARM_NAMES, not hand-copied per arm), but the dataset itself --
% paths, channel, arm folders -- is still hardcoded to this one session/
% class for now (see CONFIG below). Generalizing CONFIG to other datasets
% is future work.
%
% NOT ported yet (still only in test_rescue_grape_metrics.m): the
% exclusive-set rasters (gt_not_sd1/sd1_not_gt/sd2_only), the response-window
% venn sanity checks, the per-stimulus comparison pages, and the waveform
% comparison. Those are pairwise/sd-sweep-specific comparisons whose shape
% for an arbitrary number of arms hasn't been decided yet -- run
% test_rescue_grape_metrics.m directly if you need them.
%
% VARIABLES (top-level)
%   stim_list/class_num - inputs: same meaning as test_rescue_grape_metrics.m
%   ch          - channel number
%   root        - session root (has NSx.mat, finalevents.mat, ...)
%   arm_names   - rescue-arm labels, in the order they'll be compared (GT is added separately as arm 1)
%   arm_dirs    - one folder per arm_names entry
%   all_names/all_class_ids/all_grapes/all_cc/all_metrics - GT + every arm, same order, index 1 = GT

%% CONFIG -- hardcoded to this one dataset for now
ch = 368;
root = pwd;
tol = 0.5;
min_recall = 0.5;
def_pr_pre = 100; def_pr_pos = 100;
dyn_buffer = 30;
display_buffer = 600;   % ms shown on each side of the response window on the per-stimulus
                         % pages ONLY (visual context/baseline activity) -- does not change
                         % dyn_buffer or any counting/summary logic elsewhere
out_dir = fullfile(root, 'response_plots_sd1_sd2');
gt_dir = fullfile(root, 'hand selected');
% gt_dir = fullfile(root, 'hand selected');
gt_times_file = fullfile(gt_dir, 'times_mRFUS07 raw_368.mat');
gt_grapes_file = fullfile(gt_dir, 'grapes_blanks.mat');


% sweep_dir = fullfile(root, 'MEMBERSHIP_ch328_20260918_003947', 'rescue_sd_sweep');
% sweep_dir = fullfile(root, 'python_rescue_test');
sweep_dir = root;

arm_names = {'rescue_sd1'};
% arm_names = {'rescue_sd1'};
% sweep_dir = fullfile(root, 'times_20260928_0100');

% arm_dirs = {fullfile(sweep_dir)};


% sweep_dir = fullfile(root, 'MEMBERSHIP_ch328_20260918_003947', 'rescue_sd_sweep');
% % arm_names = {'rescue_sd1','rescue_sd2'};
% arm_names = {'rescue_sd1'};
% 
% arm_dirs = {fullfile(sweep_dir,'rescue_sd1')};
arm_dirs = {fullfile(sweep_dir,'python_rescue_no_refract_sd1')};

arm_times_file = 'times_mRFUS07 raw_368.mat';
arm_grapes_file = 'grapes_blanks_rescue.mat';

addpath(genpath(root));

%% load session-wide data + onset map (stim_list scope)
finalevents = load('finalevents.mat');
stimulus = load('stimulus.mat');
exp_prop = load('experiment_properties_online3.mat');
Nscr = min(numel(stimulus.stimulus), numel(exp_prop.scr_end_cell));
onset_map = build_all_stim_onset_map(stimulus, finalevents, Nscr, stim_list);

%% load GT + build its full matched-class spike-time population
times_gt = load(gt_times_file);
grapes_gt = load(gt_grapes_file);
[gt_inside, gt_outside] = find_response_spikes(class_num, grapes_gt, times_gt.cluster_class, onset_map, stim_list, ch, def_pr_pre, def_pr_pos);
gt_abs_times = sort([gt_inside(:); gt_outside(:)]);
ccg = [repmat(class_num, numel(gt_abs_times),1), gt_abs_times(:)];

%% load rescue arms
n_arms = numel(arm_names);
times_arms = cell(1,n_arms); grapes_arms = cell(1,n_arms);
for k = 1:n_arms
    times_arms{k} = load(fullfile(arm_dirs{k}, arm_times_file));
    grapes_arms{k} = load(fullfile(arm_dirs{k}, arm_grapes_file));
end
arm_cc_list = cellfun(@(t) t.cluster_class, times_arms, 'UniformOutput', false);
all_times = [{times_gt}, times_arms];

%% stage: match each arm's class to GT, plot the full-population venns + match figure
[arm_classes, ~, DET_all, COMP_all, matches] = match_arms_to_ground_truth( ...
    ccg, gt_abs_times, arm_names, arm_cc_list, ch, tol, min_recall);

if ~exist(out_dir,'dir'), mkdir(out_dir); end
% for k = 1:n_arms
%     plot_membership_set(class_num, arm_names{k}, arm_classes(k), gt_abs_times, matches(k).arm_t, ...
%         matches(k).matched_gt, matches(k).gt_only, matches(k).arm_only, out_dir, ch);
% end
class_match_figure(DET_all, COMP_all, ch, arm_names, out_dir);

%% stage: per-arm metrics (GT + each rescue arm)
m_gt = get_class_metrics(gt_dir, 'grapes_blanks.mat', ch, class_num, stim_list, root);
m_arms = cell(1,n_arms);
for k = 1:n_arms
    m_arms{k} = get_class_metrics(arm_dirs{k}, arm_grapes_file, ch, arm_classes(k), stim_list, root);
end

all_names = [{'gt'}, arm_names];
all_class_ids = [class_num; arm_classes(:)];
all_grapes = [{grapes_gt}, grapes_arms];
all_cc = [{times_gt.cluster_class}, arm_cc_list];
all_metrics = [{m_gt}, m_arms];
all_per_stim = cellfun(@(m) m.per_stim, all_metrics, 'UniformOutput', false);
all_mean_zscore = cellfun(@(m) m.mean_zscore, all_metrics);
all_mu_base = cellfun(@(m) m.mu_base, all_metrics);
all_std_base = cellfun(@(m) m.std_base, all_metrics);

%% stage: dynamic-window sets + summary, stim_list scope
dyn = compute_dynamic_window_sets(all_class_ids, all_grapes, all_cc, onset_map, all_per_stim, stim_list, ch, dyn_buffer);
n_in_dyn = arrayfun(@(d) numel(d.inside), dyn)';

[stimlist_summary, pct_dyn_last_vs_prev] = build_arm_summary_table( ...
    all_names, all_class_ids, n_in_dyn, all_mean_zscore, all_mu_base, all_std_base, ...
    out_dir, 'stimlist_summary_dynamic.csv');
fprintf('[dynamic onset+dura+%dms] %s -> %s inside %% diff = %.1f%%\n', dyn_buffer, all_names{end-1}, all_names{end}, pct_dyn_last_vs_prev);
plot_stimlist_summary(stimlist_summary, pct_dyn_last_vs_prev, out_dir, ch, 'dynamic_latency_dura', ...
    sprintf('dynamic onset+dura (+/-%dms buffer)', dyn_buffer));

%% response-window raster plots (gt + each arm), dynamic window only
% rw_dir = fullfile(out_dir, 'response_window_plots');
% for k = 1:numel(all_names)
%     per_stim_k = all_per_stim{k};
%     lo = per_stim_k.onset - dyn_buffer;
%     hi = per_stim_k.onset + per_stim_k.dura + dyn_buffer;
%     plot_response_window_grapes(all_grapes{k}, per_stim_k, ch, all_class_ids(k), n_in_dyn(k), lo, hi, ...
%         all_metrics{k}.rank_config, all_metrics{k}.ejex, Nscr, sprintf('%s_dynamic_latdur', all_names{k}), rw_dir, root);
% end

%% stimulus raster pages -- SAME dense production-style grid as the response-
% window plots above (images + rasters + PSTH + window lines, ~20 stimuli
% per page), but with cluster_total passed so the image panel's mu_base/
% std_base/n_response text also shows up. SEPARATE subfolder per arm (GT and
% each rescue arm each get their own). GT (k==1) is plotted first and its
% returned data_table is kept as gt_ref so every rescue arm's page also gets
% the engine's 4th sub-row (this stimulus's actual GT vs ARM values --
% response count/median_post/mean_post/zscore, see plot_arm_gt_value_bar)
% plus a separate GT-vs-ARM paired-line summary figure covering every
% stimulus at once (plot_gt_arm_summary_figure). GT's own page has no gt_ref
% (nothing to compare itself against), so both stay off for it.
stim_raster_dir = fullfile(out_dir, 'stimulus_rasters');
gt_ref = [];
for k = 1:numel(all_names)
    per_stim_k = all_per_stim{k};
    lo = per_stim_k.onset - dyn_buffer;
    hi = per_stim_k.onset + per_stim_k.dura + dyn_buffer;
    cluster_total_k = numel(all_cc{k}(all_cc{k}(:,1)==all_class_ids(k), 2));

    dt_k = plot_response_window_grapes(all_grapes{k}, per_stim_k, ch, all_class_ids(k), n_in_dyn(k), lo, hi, ...
        all_metrics{k}.rank_config, all_metrics{k}.ejex, Nscr, sprintf('%s_stimulus_rasters', all_names{k}), ...
        fullfile(stim_raster_dir, all_names{k}), root, [], false, cluster_total_k, gt_ref);
    if k == 1
        gt_ref = dt_k;
    end
end

%% matched / GT-only / arm-only exclusive-set rasters -- SAME dense grid as
% the stimulus_rasters pages above, restricted to just the spikes in ONE
% membership category: spikes matched between GT and the arm, spikes unique
% to GT, and spikes unique to the arm. Scoped to EACH class's FULL spike
% population (gt_abs_times / matches(k).arm_t -- the SAME full-population
% data the "all_spikes" venn above already uses), NOT just the dynamic
% in-window response spikes -- every spike gets classified to its own trial
% by the real windowing code and shown across the FULL raw trial window
% (clip_to_window=false), same as the regular stimulus_rasters pages, so a
% unique/matched spike sitting outside the green response-window lines is
% still visible instead of being cut off. Each category is built into a
% real synthetic grapes raster via synth_grapes_for_subset (calls the actual
% production update_grapes_blanks windowing code, not a hand-rolled
% reimplementation) so it plots through the identical engine -- same
% pattern test_rescue_grape_metrics.m already established for its
% exclusive_set_plots (gt_not_sd1/sd1_not_gt/sd2_only), generalized over
% arm_names and extended with the "matched" set that script didn't have.
excl_dir = fullfile(out_dir, 'exclusive_set_rasters');
nm = grapes_gt.rasters.(sprintf('chan%d',ch)).details.output_name;
gt_lo = all_per_stim{1}.onset - dyn_buffer; gt_hi = all_per_stim{1}.onset + all_per_stim{1}.dura + dyn_buffer;
% Real, stable whole-class totals (GT's/each arm's TRUE cluster size, same
% numbers the all_spikes venn's title already shows as "GT n=.../arm n=...")
% -- NOT the size of whichever exclusive subset happens to be plotted, which
% changes per page and would make the title's "resp/cluster" ratio compare a
% subset against itself instead of against the real cluster it came from.
gt_cluster_total = numel(gt_abs_times);
for k = 1:n_arms
    idx = k+1;
    arm_template = fullfile(arm_dirs{k}, arm_grapes_file);
    arm_lo = all_per_stim{idx}.onset - dyn_buffer; arm_hi = all_per_stim{idx}.onset + all_per_stim{idx}.dura + dyn_buffer;
    arm_cluster_total = numel(matches(k).arm_t);

    [matched_gt, ~, gt_only, arm_only] = membership_sets(gt_abs_times, matches(k).arm_t, tol);

    plot_excl_raster(matched_gt, gt_grapes_file, all_per_stim{1}, class_num, gt_lo, gt_hi, ...
        all_metrics{1}.rank_config, all_metrics{1}.ejex, Nscr, sprintf('%s_matched_gt', arm_names{k}), excl_dir, root, ch, nm, gt_cluster_total);
    plot_excl_raster(gt_only, gt_grapes_file, all_per_stim{1}, class_num, gt_lo, gt_hi, ...
        all_metrics{1}.rank_config, all_metrics{1}.ejex, Nscr, sprintf('%s_gt_only', arm_names{k}), excl_dir, root, ch, nm, gt_cluster_total);
    plot_excl_raster(arm_only, arm_template, all_per_stim{idx}, all_class_ids(idx), arm_lo, arm_hi, ...
        all_metrics{idx}.rank_config, all_metrics{idx}.ejex, Nscr, sprintf('%s_arm_only', arm_names{k}), excl_dir, root, ch, nm, arm_cluster_total);
end

%% "other" (non-stim_list) response summary -- every OTHER scored stimulus,
% scored against a window BORROWED from stim_list's own median onset/dura
% (most non-stim_list stimuli were never expected to elicit a response, so
% their own latency estimate is meaningless/absent -- we still want a
% window that's comparable to the real responses, not each stim's own
% noisy/missing one). This is disjoint from the stim_list-scope summary
% above by construction (compute_response_window_sets builds ONE unified
% window set over all_stim_list, so a spike lands in exactly one bucket) --
% unlike the old pattern of calling compute_dynamic_window_sets separately
% over all_stim_list, which double-counted stim_list's own responses since
% all_stim_list is a superset of stim_list.
all_stim_list = unique(m_gt.per_stim_all.stim_number)';
all_onset_map = build_all_stim_onset_map(stimulus, finalevents, Nscr, all_stim_list);
all_per_stim_all = cellfun(@(m) m.per_stim_all, all_metrics, 'UniformOutput', false);

resp = compute_response_window_sets(all_class_ids, all_grapes, all_cc, all_onset_map, ...
    all_per_stim, all_per_stim_all, stim_list, all_stim_list, ch, dyn_buffer);
n_in_other = arrayfun(@(d) numel(d.inside_other), resp)';

% Filter per_stim_all down to the OTHER stimuli before averaging -- zscore
% varies per stimulus, so pooling in stim_list's own (real, responsive)
% rows here would silently make this identical to an all-320-stim average,
% not a true "other-only" one. (mu_base/std_base are session-wide constants
% repeated in every row by create_responses_data_parallel.m regardless of
% which stimuli are scored, so filtering can't change those two -- that
% part is expected to match the stim_list-scope summary, not a bug.)
other_stim_list = resp(1).other_stim_list;
other_per_stim_all = cellfun(@(t) t(ismember(t.stim_number, other_stim_list), :), all_per_stim_all, 'UniformOutput', false);
mean_zscore_all = cellfun(@(t) mean(t.zscore,'omitnan'), other_per_stim_all);
mu_base_all = cellfun(@(t) t.mu_base(1), other_per_stim_all);
std_base_all = cellfun(@(t) t.std_base(1), other_per_stim_all);

[other_summary, pct_other_last_vs_prev] = build_arm_summary_table( ...
    all_names, all_class_ids, n_in_other, mean_zscore_all, mu_base_all, std_base_all, ...
    out_dir, 'other_summary.csv');
fprintf('[other, %d non-stim-list stims, borrowed window onset=%.1fms dura=%.1fms] %s -> %s inside %% diff = %.1f%%\n', ...
    numel(resp(1).other_stim_list), resp(1).fixed_onset, resp(1).fixed_dura, all_names{end-1}, all_names{end}, pct_other_last_vs_prev);
plot_stimlist_summary(other_summary, pct_other_last_vs_prev, out_dir, ch, 'other_stimuli', ...
    sprintf('other (non-stim-list) stimuli, borrowed window (median onset=%.0fms dura=%.0fms)', ...
    resp(1).fixed_onset, resp(1).fixed_dura));

%% response / non-response / all-spikes venn diagrams, GT vs each arm -- same
% 2-circle plot_membership_set already used for the full-population venn at
% the top of this script (class_match/venn_ch... above), now ALSO split into
% just the response-window spikes and just the non-response (baseline)
% spikes, so recovery can be judged separately for the window that actually
% matters vs everything else, not just the whole-population number.
%
% IMPORTANT: response and non-response counts must add up to the all-spikes
% count (they didn't in an earlier version of this section -- e.g. GT-only
% response=28 + GT-only non-response=44 = 72, not the true GT-only total of
% 65). That happened because (1) "non-response" was built from
% resp(...).outside alone, which excludes resp(...).inside_other (response-
% window spikes for OTHER, non-stim-list stimuli) -- so response union
% non-response wasn't actually the full population -- and (2) matched/
% gt_only/arm_only were recomputed with a FRESH membership_sets call on each
% restricted subset, so a GT spike could come out "unmatched" in the
% response-only comparison merely because its true arm counterpart happens
% to live in the arm's non-response bucket, not because it's genuinely
% unmatched overall.
%
% Fixed by computing the match ONCE on the full population (matches(k),
% already done above for the all-spikes venn) and then splitting THAT
% single matched_gt/gt_only/arm_only assignment by response-window
% membership, instead of re-matching independently per scope:
%   - gt_resp_all/gt_nonresp_all EXACTLY partition gt_abs_times (nonresp is
%     defined as the plain set-complement of resp, not a separately-scored
%     bucket), so every one of matches(k).gt_only's 65 spikes lands in
%     exactly one of the two venns -- 28 + 37 = 65, not 28 + 44.
%   - same for the arm side against matches(k).arm_t.
% Six scopes now, all just different spike-time SUBSETS run through the same
% two shared functions (plot_membership_set / plot_venn_waveform_group) --
% fixing either function fixes all six at once:
%   response/non_response    - resp(...).inside_stimlist vs its complement.
%                               NOTE this is a DIFFERENT window definition
%                               than inside_response_total/outside_response
%                               below -- non_response here also swallows
%                               resp(...).inside_other (response-window
%                               spikes for OTHER, non-stim-list stimuli),
%                               so it does NOT equal outside_response.
%   all_spikes                - the whole matched population, no window filter.
%   inside_response_total     - resp(...).inside_stimlist UNION inside_other,
%                               i.e. the SAME "inside response" the waveform
%                               pages (plot_waveform_comparison_n.m) use.
%   outside_response           - resp(...).outside, the SAME "outside" bucket
%                               the waveform pages use.
%   pre_rescue                 - GT vs the arm's PRE-rescue population
%                               (cluster_class_pre_rescue, the same field
%                               do_structure_rescue_BCM_online3.m reads),
%                               whole population only, not further split.
%
% IMPORTANT (response/non_response only): counts must add up to the
% all-spikes count (they didn't in an earlier version -- e.g. GT-only
% response=28 + GT-only non-response=44 = 72, not the true GT-only total of
% 65). Fixed by computing the match ONCE on the full population (matches(k))
% and splitting THAT single matched_gt/gt_only/arm_only assignment by window
% membership, instead of re-matching independently per scope -- every scope
% below follows the same pattern, including the two new ones and pre_rescue.
venn_dir = fullfile(out_dir, 'response_venn_plots');
waveform_dir = fullfile(out_dir, 'waveform_plots');
inside_stimlist_list = {resp.inside_stimlist};
inside_other_list    = {resp.inside_other};
outside_list         = {resp.outside};
arm_template_files = cellfun(@(d) fullfile(d, 'templates_wc_offline.mat'), arm_dirs, 'UniformOutput', false);
gt_template = mean(times_gt.spikes(times_gt.cluster_class(:,1)==class_num, :), 1);
% Needed here (not just later, for plot_waveform_comparison_n) so this
% section can also build the quarantined-not-rescued waveform page --
% moved up from where it used to be defined, right before that call.
arm_spikes_filename = regexprep(arm_times_file, '^times_(.*)\.mat$', '$1_spikes.mat');
arm_spikes_files = cellfun(@(d) fullfile(d, arm_spikes_filename), arm_dirs, 'UniformOutput', false);

% ---- pass 1: build every (arm, scope) group's spike-time sets, and collect
% the shared y-axis extents before drawing anything. TWO separate limits,
% each data-driven (min/max of mean+/-std, plus templates, +5% pad) and fixed
% across its own pages: one for every venn waveform page, one for every
% quarantined-not-rescued page. They used to share one limit, but the
% rejected spikes' spread (to ~ -93/+74 on ch328 vs ~ -44/+20 for the venn
% regions) squashed the venn spikes into a third of the axis. ----
groups = struct('arm_k',{},'scope',{},'gt_scope_t',{},'arm_scope_t',{}, ...
    'matched_gt',{},'matched_arm',{},'gt_only',{},'arm_only',{}, ...
    'gt_template',{},'arm_template',{},'matched_template_global',{});
qnr_groups_all = cell(1, n_arms);   % quarantined-not-rescued groups, one cell per arm
qnr_overlays_all = cell(1, n_arms); % {GT, arm, matched} templates for each arm's quarantined page
shared_lo = Inf; shared_hi = -Inf;  % venn waveform pages
qnr_lo = Inf; qnr_hi = -Inf;        % quarantined-not-rescued pages

for k = 1:n_arms
    arm_template_k = load_stored_template(arm_template_files{k}, ch, arm_classes(k));

    % Quarantined-and-never-rescued waveform groups -- arm-level, not
    % scoped to GT or any venn (see classify_quarantined_not_rescued.m).
    if exist(arm_spikes_files{k}, 'file')
        arm_spikes_k = load(arm_spikes_files{k});
    else
        arm_spikes_k = [];
    end
    qnr_groups_all{k} = classify_quarantined_not_rescued(times_arms{k}, arm_spikes_k);
    for gi2 = 1:numel(qnr_groups_all{k})
        W = qnr_groups_all{k}(gi2).waveforms;
        if isempty(W), continue; end
        mu = mean(W,1); sd = std(W,0,1);
        qnr_lo = min(qnr_lo, min(mu-sd));
        qnr_hi = max(qnr_hi, max(mu+sd));
    end
    if ~isempty(arm_template_k) && ~all(isnan(arm_template_k))
        shared_lo = min(shared_lo, min(arm_template_k));
        shared_hi = max(shared_hi, max(arm_template_k));
        qnr_lo = min(qnr_lo, min(arm_template_k));
        qnr_hi = max(qnr_hi, max(arm_template_k));
    end

    % matches(k) only kept the GT-side matched set (matched_arm was
    % discarded) -- recompute once here so the venn/waveform pages below
    % can look up matched waveforms on the ARM side too.
    [matched_gt_all, matched_arm_all, ~, ~] = membership_sets(gt_abs_times, matches(k).arm_t, tol);
    matched_template_global = mean([ ...
        get_waveforms_for_times(matched_gt_all, times_gt.cluster_class(:,2), times_gt.spikes, tol); ...
        get_waveforms_for_times(matched_arm_all, times_arms{k}.cluster_class(:,2), times_arms{k}.spikes, tol)], 1);

    qnr_overlays_all{k} = build_overlays_n('gt', gt_template, arm_names{k}, arm_template_k, ...
        arm_names{k}, matched_template_global);
    for oi = 1:numel(qnr_overlays_all{k})
        qnr_lo = min(qnr_lo, min(qnr_overlays_all{k}(oi).wave));
        qnr_hi = max(qnr_hi, max(qnr_overlays_all{k}(oi).wave));
    end

    gt_resp_all  = resp(1).inside_stimlist;
    gt_nonresp_all = setdiff(gt_abs_times, gt_resp_all);
    arm_resp_all = resp(k+1).inside_stimlist;
    arm_nonresp_all = setdiff(matches(k).arm_t, arm_resp_all);

    gt_total_all  = union(resp(1).inside_stimlist,   resp(1).inside_other);
    arm_total_all = union(resp(k+1).inside_stimlist, resp(k+1).inside_other);
    gt_outside_all  = resp(1).outside;
    arm_outside_all = resp(k+1).outside;

    arm_t_pre = times_arms{k}.cluster_class_pre_rescue(times_arms{k}.cluster_class_pre_rescue(:,1)==arm_classes(k), 2);
    [matched_gt_pre, matched_arm_pre, gt_only_pre, arm_only_pre] = membership_sets(gt_abs_times, arm_t_pre, tol);

    scope_defs = struct( ...
        'name',        {'response',    'non_response',   'all_spikes',     'inside_response_total', 'outside_response',  'pre_rescue'}, ...
        'gt_scope_t',  {gt_resp_all,   gt_nonresp_all,    gt_abs_times,     gt_total_all,             gt_outside_all,      gt_abs_times}, ...
        'arm_scope_t', {arm_resp_all,  arm_nonresp_all,   matches(k).arm_t, arm_total_all,            arm_outside_all,     arm_t_pre}, ...
        'matched_gt',  {intersect(matches(k).matched_gt, gt_resp_all), intersect(matches(k).matched_gt, gt_nonresp_all), matches(k).matched_gt, intersect(matches(k).matched_gt, gt_total_all), intersect(matches(k).matched_gt, gt_outside_all), matched_gt_pre}, ...
        'matched_arm', {intersect(matched_arm_all, arm_resp_all), intersect(matched_arm_all, arm_nonresp_all), matched_arm_all, intersect(matched_arm_all, arm_total_all), intersect(matched_arm_all, arm_outside_all), matched_arm_pre}, ...
        'gt_only',     {intersect(matches(k).gt_only, gt_resp_all), intersect(matches(k).gt_only, gt_nonresp_all), matches(k).gt_only, intersect(matches(k).gt_only, gt_total_all), intersect(matches(k).gt_only, gt_outside_all), gt_only_pre}, ...
        'arm_only',    {intersect(matches(k).arm_only, arm_resp_all), intersect(matches(k).arm_only, arm_nonresp_all), matches(k).arm_only, intersect(matches(k).arm_only, arm_total_all), intersect(matches(k).arm_only, arm_outside_all), arm_only_pre});

    for si = 1:numel(scope_defs)
        sd_ = scope_defs(si);
        groups(end+1) = struct('arm_k',k,'scope',sd_.name,'gt_scope_t',sd_.gt_scope_t,'arm_scope_t',sd_.arm_scope_t, ...
            'matched_gt',sd_.matched_gt,'matched_arm',sd_.matched_arm,'gt_only',sd_.gt_only,'arm_only',sd_.arm_only, ...
            'gt_template',gt_template,'arm_template',arm_template_k,'matched_template_global',matched_template_global); %#ok<AGROW>

        % matched_name = arm_names{k} (plain name, not a compound string) --
        % see plot_venn_waveform_group.m's identical build_overlays_n call
        % for why: its own label format already prepends ref_name.
        ov = build_overlays_n('gt', gt_template, arm_names{k}, arm_template_k, ...
            arm_names{k}, matched_template_global);
        yl = venn_waveform_extent(times_gt, times_arms{k}, sd_.matched_gt, sd_.matched_arm, sd_.gt_only, sd_.arm_only, tol, ov);
        if ~isempty(yl)
            shared_lo = min(shared_lo, yl(1));
            shared_hi = max(shared_hi, yl(2));
        end
    end
end
shared_ylim = padded_ylim(shared_lo, shared_hi);
qnr_ylim    = padded_ylim(qnr_lo, qnr_hi);

% ---- pass 2: draw every venn + its matching 4-panel waveform page ----
for gi = 1:numel(groups)
    g = groups(gi);
    k = g.arm_k;
    plot_membership_set(class_num, arm_names{k}, arm_classes(k), g.gt_scope_t, g.arm_scope_t, ...
        g.matched_gt, g.gt_only, g.arm_only, fullfile(venn_dir, g.scope), ch);

    waveform_venn_dir = fullfile(waveform_dir, 'venn_breakdowns', g.scope);
    if ~exist(waveform_venn_dir, 'dir'), mkdir(waveform_venn_dir); end
    plot_venn_waveform_group('gt', times_gt, arm_names{k}, times_arms{k}, class_num, arm_classes(k), ch, ...
        g.matched_gt, g.matched_arm, g.gt_only, g.arm_only, tol, ...
        g.gt_template, g.arm_template, g.matched_template_global, ...
        sprintf('ch%d gt vs %s -- %s', ch, arm_names{k}, strrep(g.scope,'_',' ')), ...
        fullfile(waveform_venn_dir, sprintf('venn_waveforms_ch%d_gt_%s_%s.png', ch, arm_names{k}, g.scope)), ...
        shared_ylim);
end

% ---- pass 2 (cont'd): each arm's quarantined-not-rescued waveform page --
% arm-level, not per-scope, so one page per arm (not per venn) ----
qnr_dir = fullfile(waveform_dir, 'quarantined_not_rescued');
if ~exist(qnr_dir, 'dir'), mkdir(qnr_dir); end
for k = 1:n_arms
    plot_quarantined_not_rescued_waveforms(qnr_groups_all{k}, qnr_overlays_all{k}, ...
        sprintf('ch%d %s -- quarantined, never rescued', ch, arm_names{k}), ...
        fullfile(qnr_dir, sprintf('quarantined_not_rescued_ch%d_%s.png', ch, arm_names{k})), ...
        qnr_ylim);
end

%% waveform comparison -- N-arm-generic version of plot_waveform_comparison.m
% (that original stays fixed at exactly GT+sd1+sd2 since test_rescue_grape_metrics.m
% still calls it positionally; plot_waveform_comparison_n.m generalizes the
% same group/overlay logic to however many rescue arms ARM_NAMES has).
% Reuses resp's already-computed, non-overlapping inside_stimlist/inside_other/
% outside sets, arm_template_files, and arm_spikes_files -- all already
% built above, in the venn section (needed there too now, for the venn
% waveform pages and the quarantined-not-rescued page).
plot_waveform_comparison_n(all_names, all_class_ids, all_times, ...
    inside_stimlist_list, inside_other_list, outside_list, tol, ch, waveform_dir, ...
    'arm_template_files', arm_template_files, 'arm_spikes_files', arm_spikes_files);

%% per-stimulus comparison pages -- N-arm-generic version of plot_stim_comparison_page.m,
% one page per stimulus in stim_list. Each arm's own full population +
% onset_map + per_stim window are passed straight through; the function
% itself gathers raw spikes (response window +/- display_buffer) and
% classifies them, so no pre-filtered inside/stim_of lists are needed here.
stim_page_dir = fullfile(out_dir, 'stim_pages');
for sp = stim_list(:)'
    plot_stim_comparison_page_n(sp, all_names, all_class_ids, all_cc, ...
        onset_map, all_per_stim, grapes_gt, ...
        tol, dyn_buffer, display_buffer, ch, root, stim_page_dir);
end

%% SPC temperature diagram for GT and every arm -- how each clustering was
% chosen. Drawn from the folder's own data_<name>.dg_01; skipped with a
% note when the folder has none.
temp_dir = fullfile(out_dir, 'temperature_plots');
if ~exist(temp_dir, 'dir'), mkdir(temp_dir); end
all_dirs = [{gt_dir}, arm_dirs];
[~, gt_times_name] = fileparts(gt_times_file);
all_out_names = [{regexprep(gt_times_name, '^times_', '')}, ...
    repmat({regexprep(arm_times_file, '^times_(.*)\.mat$', '$1')}, 1, n_arms)];
for k = 1:numel(all_names)
    make_arm_temperature_plot(all_dirs{k}, all_out_names{k}, all_names{k}, ...
        fullfile(temp_dir, sprintf('temperature_ch%d_%s.png', ch, all_names{k})));
end

%% bundle every PNG output into a handful of organized, paged PDFs -- much
% easier to flip through as a report than dozens of loose PNGs scattered
% across subfolders. One PDF per thematic group, pages in filename order;
% each PNG's own relative path is printed as a caption strip so a page can
% still be traced back to its source file.
pdf_dir = fullfile(out_dir, 'pdf_reports');

pngs = list_pngs_sorted(out_dir, {'class_match_*.png','stimlist_summary_*.png'});
compile_pngs_to_pdf(pngs, fullfile(pdf_dir,'1_membership_overview.pdf'), ...
    cellfun(@(p) erase(p, [out_dir filesep]), pngs, 'UniformOutput', false));

pngs = list_pngs_sorted(venn_dir, {'response/*.png','non_response/*.png','all_spikes/*.png', ...
    'inside_response_total/*.png','outside_response/*.png','pre_rescue/*.png'});
compile_pngs_to_pdf(pngs, fullfile(pdf_dir,'2_response_venns.pdf'), ...
    cellfun(@(p) erase(p, [venn_dir filesep]), pngs, 'UniformOutput', false));

% '3_response_window_plots.pdf' removed: its source (rw_dir, populated by the
% response-window raster loop above) has been commented out since the
% stim_raster_dir pages below produce the same dense grid, so rw_dir was
% never being defined here -- this bundling step just errored on every run.

% Combined rasters+summary+venns PDF -- GT's raster page(s), then the mu/std/
% zscore summary charts, then for EACH arm: that arm's raster page(s)
% immediately followed by its own 3 venns (response/non_response/all_spikes)
% -- so the raster (which already carries per-stimulus mu/std/n via
% plot_img_zscore_class) and the membership venns for the SAME arm sit right
% next to each other in one document instead of two you have to flip between.
combined_pngs = list_pngs_sorted(stim_raster_dir, {'gt/*.png'});
combined_caps = cellfun(@(p) erase(p, [stim_raster_dir filesep]), combined_pngs, 'UniformOutput', false);

summary_pngs = list_pngs_sorted(out_dir, {'stimlist_summary_*.png'});
combined_pngs = [combined_pngs; summary_pngs];
combined_caps = [combined_caps; cellfun(@(p) erase(p, [out_dir filesep]), summary_pngs, 'UniformOutput', false)];

for k = 1:n_arms
    arm_pngs = list_pngs_sorted(stim_raster_dir, {fullfile(arm_names{k},'*.png')});
    arm_caps = cellfun(@(p) erase(p, [stim_raster_dir filesep]), arm_pngs, 'UniformOutput', false);

    excl_pngs = list_pngs_sorted(excl_dir, {[arm_names{k} '_matched_gt*.png'], ...
        [arm_names{k} '_gt_only*.png'], [arm_names{k} '_arm_only*.png']});
    excl_caps = cellfun(@(p) erase(p, [excl_dir filesep]), excl_pngs, 'UniformOutput', false);

    venn_tag = sprintf('venn_ch%d_GT%d_%s.png', ch, class_num, arm_names{k});
    venn_scopes = {'response','non_response','all_spikes','inside_response_total','outside_response','pre_rescue'};
    venn_pngs = cellfun(@(s) fullfile(venn_dir,s,venn_tag), venn_scopes, 'UniformOutput', false)';
    venn_caps = cellfun(@(s) sprintf('%s -- %s venn', arm_names{k}, strrep(s,'_',' ')), venn_scopes, 'UniformOutput', false)';

    combined_pngs = [combined_pngs; arm_pngs; excl_pngs; venn_pngs];
    combined_caps = [combined_caps; arm_caps; excl_caps; venn_caps];
end
compile_pngs_to_pdf(combined_pngs, fullfile(pdf_dir,'4_stimulus_rasters_and_venns.pdf'), combined_caps);

pngs = list_pngs_sorted(waveform_dir, {'*.png'});
compile_pngs_to_pdf(pngs, fullfile(pdf_dir,'5_waveform_comparisons.pdf'), ...
    cellfun(@(p) erase(p, [waveform_dir filesep]), pngs, 'UniformOutput', false));

% Per-mask provenance/waveform breakdown PDF -- the '<arm>_entire_breakdown'/
% '<arm>_not_gt_breakdown' subfolders (3 view pages each: in-response stim
% list / in-response other stimuli / outside-response baseline) built by
% plot_waveform_comparison_n.m -- NOT swept up by the plain '*.png' glob
% above since they live one level down.
breakdown_pngs = {};
breakdown_caps = {};
for k = 1:n_arms
    pngs_k = list_pngs_sorted(waveform_dir, ...
        {[arm_names{k} '_entire_breakdown/*.png'], [arm_names{k} '_not_gt_breakdown/*.png']});
    breakdown_pngs = [breakdown_pngs; pngs_k];
    breakdown_caps = [breakdown_caps; cellfun(@(p) erase(p, [waveform_dir filesep]), pngs_k, 'UniformOutput', false)];
end
compile_pngs_to_pdf(breakdown_pngs, fullfile(pdf_dir,'6_waveform_mask_breakdowns.pdf'), breakdown_caps);

pngs = list_pngs_sorted(stim_page_dir, {'*.png'});
compile_pngs_to_pdf(pngs, fullfile(pdf_dir,'7_stim_pages.pdf'), ...
    cellfun(@(p) erase(p, [stim_page_dir filesep]), pngs, 'UniformOutput', false));

pngs = list_pngs_sorted(temp_dir, {'*.png'});
compile_pngs_to_pdf(pngs, fullfile(pdf_dir,'8_temperature_plots.pdf'), ...
    cellfun(@(p) erase(p, [temp_dir filesep]), pngs, 'UniformOutput', false));

fprintf('[pdf] bundled reports written to %s\n', pdf_dir);

end


function plot_excl_raster(times_subset, template_file, per_stim, class_id, lo, hi, ...
        rank_config, ejex, Nscr, label, excl_dir, root, ch, nm, real_cluster_total)
% PLOT_EXCL_RASTER  One membership-category exclusive-set raster page: builds
% TIMES_SUBSET (this class's FULL population, not just its response-window
% spikes -- see caller) into a real synthetic grapes raster
% (synth_grapes_for_subset, the actual production update_grapes_blanks
% windowing code) and plots it through the same engine as the regular
% stimulus_rasters pages, restricted to just this subset (unit_field='mu'
% since a derived subset has no real class). clip_to_window is left false
% (full raw trial window shown, same as the regular stimulus_rasters pages)
% so a unique/matched spike sitting outside the green response-window lines
% is still visible instead of being cut off.
%
% real_cluster_total - the REAL whole-class total this subset was carved out
% of (GT's or the arm's true cluster size, from the caller -- NOT
% numel(times_subset) itself, which would make the title's "X/Y resp/cluster"
% compare the subset against its own size instead of the real cluster; it
% would also silently change per exclusive-set page (65/741/5284), which
% isn't a meaningful "cluster total" at all). Passing it (non-empty) also
% turns on the per-stimulus n=NNN count next to mu/sd on each raster panel --
% the page title already gives the subset's own total, but the number that
% matters PER STIMULUS (not another bar chart, just the number) was missing
% until now.
%
% A genuinely empty subset is skipped, not an error -- see
% synth_grapes_for_subset.m.
    gf = synth_grapes_for_subset(times_subset, ch, root, template_file, excl_dir, nm, false);
    if isempty(gf)
        fprintf('[exclusive] %s: 0 spikes -- nothing to plot\n', label);
        return;
    end
    synth = load(gf);
    plot_response_window_grapes(synth, per_stim, ch, class_id, numel(times_subset), lo, hi, ...
        rank_config, ejex, Nscr, label, excl_dir, root, 'mu', false, real_cluster_total);
end

function yl = padded_ylim(lo, hi)
    if isfinite(lo) && isfinite(hi) && lo < hi
        pad = 0.05*(hi-lo);
        yl = [lo-pad, hi+pad];
    else
        yl = [];
    end
end
