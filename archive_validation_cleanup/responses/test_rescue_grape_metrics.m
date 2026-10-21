function test_rescue_grape_metrics(stim_list,class_num)
%
% this scripts analyzes the grapes data with different masks
% assuming all have been redone
% looking at specific stimuli that elicit responses only
% looking with sd1-2 variation
% 
% if looking at specific stimuli // look at latency / duration 
% mean, std, z score
% percentages
% venn diagrama
% waveforms in rescue - what gets rescued from each within channel
%    what doesnt get rescued
% see waveforms inside and outside response window (diff between sd1 and sd2)
% 
% this will be done for hand selected and the automated pipeline
% ultimately comparing sd1-sd2 and ground truth

% we will use a modified plot grapes function
% to visualize the response windows

% VARIABLES (top-level script)
%   finalevents   - loaded finalevents.mat (struct; holds pics_onset etc.)
%   stimulus      - loaded stimulus.mat (struct; holds the stimulus{} cell)
%   NSx           - loaded NSx.mat (channel ID -> output_name lookup)
%   exp_prop      - loaded experiment_properties_online3.mat (scr_end_cell, scr_config_cell)
%   def_pr_pre    - response-window half-width BEFORE onset, ms
%   def_pr_pos    - response-window half-width AFTER onset, ms
%   onset_map     - stim ID -> real onset times (ms), the "onset_map"
%   want          - Map used as arm_c O(1) membership test for stim_list
%   Nscr          - number of completed sub-screenings to loop over
%   times_gt      - loaded ground-truth times_*.mat (cluster_class etc.)
%   grapes_gt     - loaded ground-truth grapes_blanks.mat raster
%   gt_inside     - GT class's spikes classified as inside the response window
%   gt_outside    - GT class's spikes classified as outside the response window
%   tol           - spike-time match tolerance (ms) for class matching
%   min_recall    - match_cc's threshold for calling a class "matched"
%   out_dir       - folder every plot in this run gets saved to
%   ccg           - ground-truth spikes as [class time_ms], one row per GT spike
%   T_all/DET_all/COMP_all - match_cc outputs accumulated across arms
%   arm_names     - labels for each arm being compared (e.g. rescue_sd1/sd2)
%   arm_cc_list   - each arm's full cluster_class, same order as arm_names
%   arm_c         - current arm's name, this loop iteration
%   cca           - current arm's cluster_class, this loop iteration
%   T/DET/COMP    - current arm's match_cc outputs, this loop iteration
%   arm_class     - class number in the current arm matched to class_num
%   arm_t         - current arm's spike times for arm_class
%   matched_gt/gt_only/arm_only - membership_sets output for the Venn
%   gt_inside/gt_outside - GT's fixed-window split; kept ONLY to recover the
%     full class population for matching (their union), not reported as a
%     window result -- see the dynamic-window analysis for that
finalevents = load('finalevents.mat');
stimulus = load('stimulus.mat');
NSx = load('NSx.mat');
exp_prop = load('experiment_properties_online3.mat');

def_pr_pos = 100;
def_pr_pre = 100;
dyn_buffer = 40;   % ms padding on each side of a stimulus's own [onset, onset+dura] window

%% onset times
onset_map = containers.Map('KeyType','double','ValueType','any');

want = containers.Map(num2cell(stim_list),num2cell(true(size(stim_list))));

Nscr = min(numel(stimulus.stimulus),numel(exp_prop.scr_end_cell));

for i = 1:Nscr
    pic_on = finalevents.pics_onset{i};
    stim = stimulus.stimulus{i};

    for j = 1:numel(stim)
        stim_num = stim(j).ID;
        if ~isKey(want,stim_num),continue;end

        terna = stim(j).terna_onset;

        if ndims(pic_on) ==3
            ts = pic_on(sub2ind(size(pic_on), terna(:,1),terna(:,2),terna(:,3)))
        else
            M = po(terna(:,1),terna(:,3));
            ts = M(1:length(M));
        end
        ts = ts(:).';

        if ~isKey(onset_map,stim_num),onset_map(stim_num) = []; end
        onset_map(stim_num) = [onset_map(stim_num),ts];
    end
end
missing = setdiff(stim_list,cell2mat(keys(onset_map)));
if ~isempty(missing)
    warning('build onset map: stim ID(s) %s never appeared in any subscreening', mat2str(missing));
end

%% inside outside spikes per trial
% ground truth spikes
times_gt = load('./hand selected/times_mLFUS08 raw_328.mat');
grapes_gt = load('./hand selected/grapes_blanks.mat');

times_gt_class = times_gt.cluster_class(:,1) == class_num;
% find_response_spikes here is ONLY used to recover the full class
% population (inside+outside always reconstructs everything regardless of
% window width, since they're a complete partition of class_abs_times) --
% needed for class matching below. The fixed +/-100ms window itself is not
% reported anywhere anymore (confirmed not capturing real responses; see
% the dynamic-window analysis instead).
[gt_inside,gt_outside] = find_response_spikes(class_num,grapes_gt,times_gt.cluster_class,onset_map,stim_list,328,def_pr_pre,def_pr_pos);

%% find matching class to GT
tol = 0.5;
min_recall = 0.5;
out_dir = fullfile(pwd,'membership_plots');
gt_abs_times = sort([gt_inside(:); gt_outside(:)]);
ccg = [repmat(class_num,numel(gt_abs_times),1),gt_abs_times(:)];

DET_all = table();
COMP_all = table();
T_all = table();
times_sd1 = load('./MEMBERSHIP_ch328_20260918_003947/rescue_sd_sweep/rescue_sd1/times_mLFUS08 raw_328.mat');
grapes_sd1 = load('./MEMBERSHIP_ch328_20260918_003947/rescue_sd_sweep/rescue_sd1/grapes_blanks_rescue.mat');
times_sd2 = load('./MEMBERSHIP_ch328_20260918_003947/rescue_sd_sweep/rescue_sd2/times_mLFUS08 raw_328.mat');
grapes_sd2 = load('./MEMBERSHIP_ch328_20260918_003947/rescue_sd_sweep/rescue_sd2/grapes_blanks_rescue.mat');

arm_names = {'rescue_sd1','rescue_sd2'};

arm_cc_list = {times_sd1.cluster_class, times_sd2.cluster_class};

for arm = 1:numel(arm_names)
    arm_c = arm_names{arm};
    cca = arm_cc_list{arm};
    [T, DET, COMP] = match_cc(ccg,cca,328,tol,min_recall);
    T.arm = repmat(string(arm_c),height(T),1);
    DET.arm = repmat(string(arm_c),height(DET),1);
    COMP.arm = repmat(string(arm_c),height(COMP),1);
    T_all = [T_all;T];
    DET_all = [DET_all;DET];
    COMP_all = [COMP_all;COMP];
    arm_class = T.arm_class(1);
    fprintf('[%s] matched class%d -> arm class%d (recall=%.2f, precision=%.2f, status=%s)\n', ...
            arm_c, class_num, arm_class, T.recall(1), T.precision(1), T.status(1));

    arm_t = cca(cca(:,1)==arm_class,2);
    [matched_gt,~,gt_only,arm_only] = membership_sets(gt_abs_times,arm_t,tol);
    plot_membership_set(class_num,arm_c,arm_class,gt_abs_times,arm_t,matched_gt,gt_only,arm_only,out_dir,328);
    if strcmp(arm_c,'rescue_sd1')
        arm_class_sd1 = arm_class;
    else
        arm_class_sd2 = arm_class;
    end
end
class_match_figure(DET_all, COMP_all, 328, arm_names, out_dir);

% rescue with sd1



%% sd 1 spikes

% find matching class venn diagram

% analyze responses

% rasters 

% waveforms


%% sd 2 spikes
% find matching class venn diagram

% analyze responses

% rasters 

% waveforms

%% stimlist summary page
% Characterizes response spikes for THIS stim set only (not an overall/
% session-wide summary -- that comes later once the true-baseline "outside"
% definition is sorted out for every stimulus in the session, not just this
% list). Outside-window counts are tabled for now: n_out is contaminated by
% real responses to stimuli that aren't in stim_list, so it's not a clean
% "no response" baseline yet -- only inside-window numbers are trustworthy
% here. The venn diagrams above are unaffected -- they compare full class
% populations and never touch stim_list scoping.
%
% Dynamic window only: each stimulus's OWN measured latency+duration
% ([onset-dyn_buffer, onset+dura+dyn_buffer]), computed independently per
% arm since rescue can shift the measured response timing. The fixed
% +/-100ms window was tried and dropped -- confirmed not capturing real
% responses (see find_response_spikes_dynamic for the onset-anchored
% fixed-width option if you want a middle ground between the two).
root = pwd;
m_gt  = get_class_metrics(fullfile(root,'hand selected'), 'grapes_blanks.mat', ...
                          328, class_num, stim_list, root);
m_sd1 = get_class_metrics(fullfile(root,'MEMBERSHIP_ch328_20260918_003947','rescue_sd_sweep','rescue_sd1'), ...
                          'grapes_blanks_rescue.mat', 328, arm_class_sd1, stim_list, root);
m_sd2 = get_class_metrics(fullfile(root,'MEMBERSHIP_ch328_20260918_003947','rescue_sd_sweep','rescue_sd2'), ...
                          'grapes_blanks_rescue.mat', 328, arm_class_sd2, stim_list, root);

% per-stim latency/duration, all 3 arms, for manual verification against
% your own data -- onset/dura are NaN-free but good_lat may still be false
% (see find_response_spikes_dynamic's skip logic for why that matters)
lat_cols = {'stim_number','onset','dura','good_lat','ntrials','ifr_thr'};
fprintf('\n--- per-stim latency/duration: GT (class%d) ---\n', class_num);
disp(m_gt.per_stim(:, lat_cols));
fprintf('--- per-stim latency/duration: sd1 (class%d) ---\n', arm_class_sd1);
disp(m_sd1.per_stim(:, lat_cols));
fprintf('--- per-stim latency/duration: sd2 (class%d) ---\n', arm_class_sd2);
disp(m_sd2.per_stim(:, lat_cols));

arm      = {'gt'; 'sd1'; 'sd2'};
class_id = [class_num; arm_class_sd1; arm_class_sd2];
mean_zscore = [m_gt.mean_zscore; m_sd1.mean_zscore; m_sd2.mean_zscore];
mu_base     = [m_gt.mu_base;     m_sd1.mu_base;     m_sd2.mu_base];
std_base    = [m_gt.std_base;    m_sd1.std_base;    m_sd2.std_base];

% ---- dynamic window: each stim's own onset+dura, +/- dyn_buffer ----
% (the fixed +/-100ms window was dropped from reporting -- confirmed not
% capturing real responses; gt_inside/gt_outside above are kept only as
% class-matching plumbing, not a window definition worth reporting on)
[gt_inside_dyn,  gt_outside_dyn,  gt_stim_of_inside]  = find_response_spikes_dynamic(class_num,    grapes_gt,  times_gt.cluster_class,  onset_map, m_gt.per_stim,  stim_list, 328, dyn_buffer);
[sd1_inside_dyn, sd1_outside_dyn, sd1_stim_of_inside] = find_response_spikes_dynamic(arm_class_sd1, grapes_sd1, times_sd1.cluster_class, onset_map, m_sd1.per_stim, stim_list, 328, dyn_buffer);
[sd2_inside_dyn, sd2_outside_dyn, sd2_stim_of_inside] = find_response_spikes_dynamic(arm_class_sd2, grapes_sd2, times_sd2.cluster_class, onset_map, m_sd2.per_stim, stim_list, 328, dyn_buffer);

n_in_dyn = [numel(gt_inside_dyn); numel(sd1_inside_dyn); numel(sd2_inside_dyn)];
pct_in_dyn_vs_gt = 100*(n_in_dyn - n_in_dyn(1)) ./ n_in_dyn(1);
pct_in_dyn_vs_gt(1) = NaN;
pct_in_dyn_sd1_vs_sd2 = 100*(n_in_dyn(3) - n_in_dyn(2)) / n_in_dyn(2);

stimlist_summary_dynamic = table(arm, class_id, n_in_dyn, pct_in_dyn_vs_gt, mean_zscore, mu_base, std_base, ...
                                 'VariableNames', {'arm','class_id','n_in','pct_in_vs_gt','mean_zscore','mu_base','std_base'});
disp(stimlist_summary_dynamic);
writetable(stimlist_summary_dynamic, fullfile(out_dir, 'stimlist_summary_dynamic.csv'));
fprintf('[dynamic onset+dura+%dms] sd1 -> sd2 inside %% diff = %.1f%%\n', dyn_buffer, pct_in_dyn_sd1_vs_sd2);

plot_stimlist_summary(stimlist_summary_dynamic, pct_in_dyn_sd1_vs_sd2, out_dir, 328, ...
                      'dynamic_latency_dura', sprintf('dynamic onset+dura (+/-%dms buffer)', dyn_buffer));

%% response-window raster plots -- 3 total: {gt,sd1,sd2}, dynamic window only
% Same raster+PSTH grid as the real pipeline's plot_grapes_as_online, but
% (1) the title shows the IN-WINDOW response spike count for THIS arm,
% not the full-session class population, and (2) two green lines mark
% exactly where our in/out boundary sits for each stimulus.
rw_dir = fullfile(out_dir, 'response_window_plots');

% ---- dynamic: each stim's own onset+dura+buffer (straight from per_stim) ----
gt_lo  = m_gt.per_stim.onset  - dyn_buffer; gt_hi  = m_gt.per_stim.onset  + m_gt.per_stim.dura  + dyn_buffer;
sd1_lo = m_sd1.per_stim.onset - dyn_buffer; sd1_hi = m_sd1.per_stim.onset + m_sd1.per_stim.dura + dyn_buffer;
sd2_lo = m_sd2.per_stim.onset - dyn_buffer; sd2_hi = m_sd2.per_stim.onset + m_sd2.per_stim.dura + dyn_buffer;
plot_response_window_grapes(grapes_gt,  m_gt.per_stim,  328, class_num,    numel(gt_inside_dyn),  gt_lo,  gt_hi,  m_gt.rank_config,  m_gt.ejex,  Nscr, 'gt_dynamic_latdur',  rw_dir, root);
plot_response_window_grapes(grapes_sd1, m_sd1.per_stim, 328, arm_class_sd1, numel(sd1_inside_dyn), sd1_lo, sd1_hi, m_sd1.rank_config, m_sd1.ejex, Nscr, 'sd1_dynamic_latdur', rw_dir, root);
plot_response_window_grapes(grapes_sd2, m_sd2.per_stim, 328, arm_class_sd2, numel(sd2_inside_dyn), sd2_lo, sd2_hi, m_sd2.rank_config, m_sd2.ejex, Nscr, 'sd2_dynamic_latdur', rw_dir, root);

%% response-window exclusive-set rasters: GT\SD1, SD1\GT, SD2\(GT u SD1)
% All three restricted to the DYNAMIC in-window response spikes, not full
% class populations. Each set is built into a real (synthetic) grapes
% raster via synth_grapes_for_subset -- the actual production
% update_grapes_blanks, not a hand-rolled reimplementation -- so it plots
% through the same engine as everything else. The window lines/onset/dura
% context are borrowed from the SOURCE arm's own per_stim (GT's for
% GT\SD1, etc.) since that's genuinely where those spikes' window applies;
% note the red IFR trace/z-score in these plots still reflect the source
% class's FULL response, not just this exclusive subset -- only the raster
% ticks and green window lines are specific to the subset itself.
gt_template  = './hand selected/grapes_blanks.mat';
sd1_template = './MEMBERSHIP_ch328_20260918_003947/rescue_sd_sweep/rescue_sd1/grapes_blanks_rescue.mat';
sd2_template = './MEMBERSHIP_ch328_20260918_003947/rescue_sd_sweep/rescue_sd2/grapes_blanks_rescue.mat';
nm = grapes_gt.rasters.chan328.details.output_name;

[~, ~, gt_not_sd1, sd1_not_gt] = membership_sets(gt_inside_dyn, sd1_inside_dyn, tol);
gm_gt2  = nearest_match(sd2_inside_dyn, gt_inside_dyn,  tol);
gm_sd1b = nearest_match(sd2_inside_dyn, sd1_inside_dyn, tol);
sd2_only = sd2_inside_dyn(gm_gt2==0 & gm_sd1b==0);

excl_dir = fullfile(out_dir, 'exclusive_set_plots');
gf = synth_grapes_for_subset(gt_not_sd1, 328, root, gt_template, excl_dir, nm, false);
if isempty(gf)
    fprintf('[exclusive] gt_not_sd1: 0 spikes -- nothing to plot\n');
else
    synth = load(gf);
    plot_response_window_grapes(synth, m_gt.per_stim, 328, class_num, numel(gt_not_sd1), gt_lo, gt_hi, ...
        m_gt.rank_config, m_gt.ejex, Nscr, 'gt_not_sd1', excl_dir, root, 'mu', true);
end

gf = synth_grapes_for_subset(sd1_not_gt, 328, root, sd1_template, excl_dir, nm, false);
if isempty(gf)
    fprintf('[exclusive] sd1_not_gt: 0 spikes -- nothing to plot\n');
else
    synth = load(gf);
    plot_response_window_grapes(synth, m_sd1.per_stim, 328, arm_class_sd1, numel(sd1_not_gt), sd1_lo, sd1_hi, ...
        m_sd1.rank_config, m_sd1.ejex, Nscr, 'sd1_not_gt', excl_dir, root, 'mu', true);
end

gf = synth_grapes_for_subset(sd2_only, 328, root, sd2_template, excl_dir, nm, false);
if isempty(gf)
    fprintf('[exclusive] sd2_only (not gt, not sd1): 0 spikes -- nothing to plot\n');
else
    synth = load(gf);
    plot_response_window_grapes(synth, m_sd2.per_stim, 328, arm_class_sd2, numel(sd2_only), sd2_lo, sd2_hi, ...
        m_sd2.rank_config, m_sd2.ejex, Nscr, 'sd2_only', excl_dir, root, 'mu', true);
end

%% response-window venn diagrams: gt v sd1, gt v sd2, sd1 v sd2 (sanity check)
% Restricted to the dynamic in-window response spikes (NOT the full-class
% venns already built during class matching earlier).
resp_venn_dir = fullfile(out_dir, 'response_venn_plots');
[m_gs1,~,g1_only,s1_only] = membership_sets(gt_inside_dyn, sd1_inside_dyn, tol);
plot_membership_set(class_num, 'sd1_response', arm_class_sd1, gt_inside_dyn, sd1_inside_dyn, m_gs1, g1_only, s1_only, resp_venn_dir, 328);

[m_gs2,~,g2_only,s2_only] = membership_sets(gt_inside_dyn, sd2_inside_dyn, tol);
plot_membership_set(class_num, 'sd2_response', arm_class_sd2, gt_inside_dyn, sd2_inside_dyn, m_gs2, g2_only, s2_only, resp_venn_dir, 328);

[m_s1s2,~,s1v2_only,s2v1_only] = membership_sets(sd1_inside_dyn, sd2_inside_dyn, tol);
plot_membership_set(arm_class_sd1, 'sd2_response_sanity_check', arm_class_sd2, sd1_inside_dyn, sd2_inside_dyn, m_s1s2, s1v2_only, s2v1_only, resp_venn_dir, 328, 'sd1');

%% per-stimulus comparison pages -- one call per stimulus in stim_list
% gt_full_t/sd1_full_t/sd2_full_t: each arm's FULL session spike times for
% the matched class (NOT window- or stim-restricted). Passed into the union
% raster so a spike's category is judged against whether it exists ANYWHERE
% in another arm's data, not just within that arm's own response window --
% confirmed directly on stim513 that "X only" markers based on window-
% restricted matching were misleading (all 20 turned out present in SD2's
% full population, just outside SD2's narrower window for that stimulus).
gt_full_t  = times_gt.cluster_class(times_gt.cluster_class(:,1)   == class_num,     2);
sd1_full_t = times_sd1.cluster_class(times_sd1.cluster_class(:,1) == arm_class_sd1, 2);
sd2_full_t = times_sd2.cluster_class(times_sd2.cluster_class(:,1) == arm_class_sd2, 2);

stim_page_dir = fullfile(out_dir, 'stim_pages');
for sp = stim_list(:)'
    plot_stim_comparison_page(sp, gt_inside_dyn, sd1_inside_dyn, sd2_inside_dyn, ...
        gt_stim_of_inside, sd1_stim_of_inside, sd2_stim_of_inside, ...
        gt_full_t, sd1_full_t, sd2_full_t, ...
        onset_map, m_gt, m_sd1, m_sd2, grapes_gt, ...
        class_num, arm_class_sd1, arm_class_sd2, tol, dyn_buffer, 328, root, stim_page_dir);
end

%% overall (session-wide) summary -- ONE aggregate number set, not per-stimulus.
% Same shape as the stimlist summary, but scoped to EVERY stimulus that was
% ever scored (m_gt/m_sd1/m_sd2's per_stim_all, already computed for free
% back in get_class_metrics), not just stim_list -- so "outside" is finally
% a clean, uncontaminated baseline (no more real responses to un-scored
% stimuli hiding in it, the issue that made us drop "outside" from the
% stimlist summary earlier).
all_stim_list = unique(m_gt.per_stim_all.stim_number)';
all_onset_map = build_all_stim_onset_map(stimulus, finalevents, Nscr, all_stim_list);

[gt_inside_all,  gt_outside_all,  gt_stim_of_all]  = find_response_spikes_dynamic(class_num,     grapes_gt,  times_gt.cluster_class,  all_onset_map, m_gt.per_stim_all,  all_stim_list, 328, dyn_buffer);
[sd1_inside_all, sd1_outside_all, sd1_stim_of_all] = find_response_spikes_dynamic(arm_class_sd1,  grapes_sd1, times_sd1.cluster_class, all_onset_map, m_sd1.per_stim_all, all_stim_list, 328, dyn_buffer);
[sd2_inside_all, sd2_outside_all, sd2_stim_of_all] = find_response_spikes_dynamic(arm_class_sd2,  grapes_sd2, times_sd2.cluster_class, all_onset_map, m_sd2.per_stim_all, all_stim_list, 328, dyn_buffer);

n_in_all  = [numel(gt_inside_all);  numel(sd1_inside_all);  numel(sd2_inside_all)];
n_out_all = [numel(gt_outside_all); numel(sd1_outside_all); numel(sd2_outside_all)];
pct_in_all_vs_gt = 100*(n_in_all - n_in_all(1)) ./ n_in_all(1);
pct_in_all_vs_gt(1) = NaN;
pct_in_all_sd1_vs_sd2 = 100*(n_in_all(3) - n_in_all(2)) / n_in_all(2);

mean_zscore_all = [mean(m_gt.per_stim_all.zscore,'omitnan'); mean(m_sd1.per_stim_all.zscore,'omitnan'); mean(m_sd2.per_stim_all.zscore,'omitnan')];
mu_base_all     = [m_gt.per_stim_all.mu_base(1);  m_sd1.per_stim_all.mu_base(1);  m_sd2.per_stim_all.mu_base(1)];
std_base_all    = [m_gt.per_stim_all.std_base(1); m_sd1.per_stim_all.std_base(1); m_sd2.per_stim_all.std_base(1)];

overall_summary = table(arm, class_id, n_in_all, pct_in_all_vs_gt, mean_zscore_all, mu_base_all, std_base_all, ...
                        'VariableNames', {'arm','class_id','n_in','pct_in_vs_gt','mean_zscore','mu_base','std_base'});
disp(overall_summary);
writetable(overall_summary, fullfile(out_dir, 'overall_summary.csv'));
fprintf('[overall, all %d stims] sd1 -> sd2 inside %% diff = %.1f%%\n', numel(all_stim_list), pct_in_all_sd1_vs_sd2);
fprintf('[overall] outside-window counts (now a clean baseline): gt=%d sd1=%d sd2=%d\n', n_out_all(1), n_out_all(2), n_out_all(3));

plot_stimlist_summary(overall_summary, pct_in_all_sd1_vs_sd2, out_dir, 328, ...
                      'overall_all_stims', sprintf('overall, all %d scored stimuli', numel(all_stim_list)));

%% waveform comparison -- 3 groups (GT-missed-by-both, SD1-not-GT, SD2-not-SD1)
% x 3 views (in-response/stim_list, in-response/all-stim, outside-response),
% 9 panels total across 3 composite PNGs. Uses each arm's FULL matched-class
% population (times_gt/times_sd1/times_sd2's own cluster_class+spikes),
% not response-window-restricted, then splits by where those spikes land.
waveform_dir = fullfile(out_dir, 'waveform_plots');
plot_waveform_comparison(times_gt, times_sd1, times_sd2, class_num, arm_class_sd1, arm_class_sd2, ...
    gt_inside_dyn, sd1_inside_dyn, sd2_inside_dyn, ...
    gt_inside_all, sd1_inside_all, sd2_inside_all, ...
    gt_outside_all, sd1_outside_all, sd2_outside_all, ...
    tol, 328, waveform_dir);

%% helper functions
function [matched_gt, matched_arm,gt_only,arm_only] = membership_sets(gt_t,arm_t,tol)
    % VARIABLES
    %   gt_t        - input: GT spike times
    %   arm_t       - input: arm spike times
    %   tol         - input: match tolerance (ms)
    %   gm          - raw greedy nearest-match (index into arm_t per GT spike, 0=none)
    %   gm_u        - gm after enforce_one_to_one (each arm spike claimed once)
    %   gt_order    - GT spike index order returned by enforce_one_to_one
    %   used        - boolean per arm spike: was it claimed by some GT spike?
    %   matched_gt/matched_arm - output: paired-up GT/arm spike times
    %   gt_only/arm_only       - output: unpaired spikes on each side
    gm =  nearest_match(gt_t,arm_t,tol);
    [gm_u,gt_order] = enforce_one_to_one(gt_t,arm_t,gm,tol);
    matched_gt = gt_t(gt_order(gm_u>0));
    matched_arm = arm_t(gm_u(gm_u>0));
    gt_only = gt_t(gt_order(gm_u==0));
    used = false(numel(arm_t),1); 
    used(gm_u(gm_u>0)) = true;
    arm_only = arm_t(~used);
end
function [gm_u,order] = enforce_one_to_one(gt_t,arm_t,gm,tol)
    % VARIABLES
    %   gt_t/arm_t/gm/tol - inputs (gm = nearest_match's raw result to clean up)
    %   order   - output: fixed reference list of GT spike positions (1:numel(gt_t))
    %   dist    - match distance (ms) for each GT spike's tentative match, NaN if unmatched
    %   ok      - boolean: which GT spikes had any tentative match
    %   seq     - GT spike indices sorted by match distance, tightest first
    %   gm_u    - output: one-to-one-cleaned match array
    %   claimed - boolean per arm spike: already given to a GT spike this pass?
    %   z       - loop variable: current GT spike index, in seq order
    order = (1:numel(gt_t))';
    dist = nan(size(gm));
    ok = gm>0;
    dist(ok) = abs(gt_t(ok)-arm_t(gm(ok)));
    [~,seq] = sort(dist);
    gm_u = zeros(size(gm));
    claimed = false(numel(arm_t),1);

    for z = seq(:)'
        if gm(z) == 0
            continue
        end
        if ~claimed(gm(z))
            gm_u(z) = gm(z);
            claimed(gm(z)) = true;
        end
    end
end
function gm = nearest_match(gt_t,ar_t,tol)
    % VARIABLES
    %   gt_t/ar_t/tol - inputs: GT spike times, arm spike times to search, tolerance (ms)
    %   gm      - output: index into ar_t of each GT spike's nearest match (0=none within tol)
    %   as      - ar_t sorted ascending
    %   order   - permutation mapping sorted position back to original ar_t index
    %   i       - loop variable: which GT spike (gt_t(i)) is being matched
    %   k       - search pointer into the sorted array as
    %   best/bestd - running best candidate index / its distance so far
    %   dd      - distance being tested against bestd for the current candidate
    gm = zeros(numel(gt_t),1);
    if isempty(ar_t)
        return
    end
    [as, order] = sort(ar_t(:));
    for i = 1:numel(gt_t)
        k = find(as >= gt_t(i)-tol,1,'first');
        best = 0;
        bestd = tol + 1;
        while ~isempty(k) && k <= numel(as) && as(k) <= gt_t(i) + tol
            dd = abs(as(k) - gt_t(i));
            if dd < bestd
                bestd = dd;
                best = k;
            end
            k = k+1;
        end
        if best > 0 
            gm(i) = order(best);
        end
    end
end
function plot_membership_set(gtc,trial_name,arm_class,gt_t,arm_t,matched_gt,gt_only,arm_only,out_dir,ch,left_name)
    % VARIABLES
    %   gtc         - left-side class number (for labeling the plot)
    %   trial_name  - right-side arm name (for labeling)
    %   arm_class   - right-side matched arm class number (for labeling)
    %   gt_t/arm_t  - full left-side / right-side spike time lists (drive circle sizes)
    %   matched_gt  - paired spikes (drives the "matched" count in the middle)
    %   gt_only/arm_only - unpaired spikes on each side
    %   our_dir     - output folder (NOTE: typo'd param name, see out_dir usage below)
    %   ch          - channel number (for labeling/filename)
    %   left_name   - OPTIONAL. Label for the LEFT side (default 'GT'). Pass something
    %                 else (e.g. 'sd1') when the comparison isn't GT-vs-arm at all --
    %                 e.g. an sd1-vs-sd2 sanity check -- so the plot doesn't say "gt"
    %                 when GT isn't part of the comparison.
    %   n_gt/n_arm/n_m - total left-side spikes, total right-side spikes, matched-pair count
    %   tag         - filename fragment (channel/left-class/arm-name)
    %   r1/r2       - raw circle radii, proportional to sqrt(n_gt)/sqrt(n_arm)
    %   rr          - normalizing factor so the bigger circle fits the figure
    %   r1n/r2n     - final normalized radii used for drawing
    %   ov          - overlap fraction (matched / smaller set size)
    %   d           - center-to-center distance between the two circles
    %   th          - angle values 0:2pi used to trace each circle's outline
    if nargin < 11 || isempty(left_name), left_name = 'GT'; end
    n_gt = numel(gt_t); n_arm = numel(arm_t); n_m = numel(matched_gt);
    tag = sprintf('ch%d_%s%d_%s', ch, left_name, gtc, trial_name);
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    f = figure('Visible','off','Position',[100 100 520 460],'Color','w');
    ax = axes(f); hold(ax,'on'); axis(ax,'equal'); axis(ax,'off');
    r1 = sqrt(n_gt); r2 = sqrt(n_arm); rr = max([r1 r2 1]);
    r1n = 0.9*r1/rr; r2n = 0.9*r2/rr;
    ov = n_m/max(min(n_gt,n_arm),1);
    d = (r1n+r2n)*(1-0.85*ov);
    th = linspace(0,2*pi,200);
    fill(ax, -d/2+r1n*cos(th), r1n*sin(th), [0.85 0.3 0.3], 'FaceAlpha',0.45,'EdgeColor',[0.6 0.1 0.1]);
    fill(ax,  d/2+r2n*cos(th), r2n*sin(th), [0.3 0.4 0.85], 'FaceAlpha',0.45,'EdgeColor',[0.1 0.1 0.6]);
    text(ax, -d/2-r1n*0.6, 0, sprintf('%s-only\n%d',left_name,numel(gt_only)), 'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
    text(ax,  d/2+r2n*0.6, 0, sprintf('%s-only\n%d',trial_name,numel(arm_only)), 'HorizontalAlignment','center','FontSize',10,'Interpreter','none');
    text(ax, 0, 0, sprintf('matched\n%d',n_m), 'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
    title(ax, sprintf('ch%d %s-class%d vs %s(class%d)\n%s n=%d, arm n=%d, %.0f%% of %s recovered', ...
          ch,left_name,gtc,trial_name,arm_class,left_name,n_gt,n_arm,100*n_m/max(n_gt,1),left_name), 'Interpreter','none','FontSize',9);
    save_fig_local(f, fullfile(out_dir, ['venn_' tag '.png']));
end

function m = get_class_metrics(dir_path, grapes_filename, ch, class_num, stim_list, root)
% Runs characterize_grapes_responses on ONE grapes file (in its own folder),
% scored against the FULL stimulus set (not just stim_list) -- mu_base/
% std_base/IFR_thr are pooled across every stimulus in the scoring call
% (see create_responses_data.m), so restricting to a small stim_list would
% compute a different, noisier baseline/threshold than what plot_grapes_
% as_online.m's raster plots use (which score everything by default) --
% that mismatch is exactly what showed up as onset/dura not matching the
% raster images. Score 'all', THEN filter the returned rows down to
% stim_list, so onset/dura/ifr_thr match the reference plots exactly while
% still only reporting the stimuli you asked about.
    % VARIABLES
    %   dir_path/grapes_filename - where the grapes file lives, and its name
    %   ch/class_num/stim_list   - which channel/class/stimuli to report (NOT to score)
    %   root        - session root (has NSx.mat, experiment_properties, ...)
    %   old         - pwd before cd'ing in, restored on exit via onCleanup
    %   out         - characterize_grapes_responses' full return struct
    %   gbase/tag   - grapes filename without extension / the out.(tag) field name
    %   T           - out.(tag).metrics, one row per (channel,class,stim), EVERY stim scored
    %   clname      - 'class<N>' string used to filter T down to this class
    %   sel         - logical row mask: this channel AND this class AND in stim_list
    %   Tc          - T filtered down to just this class's rows, restricted to stim_list
    %   m           - output struct: mean_zscore, mu_base, std_base, n_stims, n_trials, per_stim
    old = pwd;
    cleanup = onCleanup(@() cd(old));
    cd(dir_path);
    out = characterize_grapes_responses('channels2plot', ch, 'stim_list', 'all', ...
                                        'session_root', root, 'grapes_files', {grapes_filename}, ...
                                        'save_ifr', true);   % need the actual IFR trace for the response-window plots
    [~, gbase] = fileparts(grapes_filename);
    tag = matlab.lang.makeValidName(gbase);
    T = out.(tag).metrics;
    m.rank_config = out.(tag).rank_config;
    m.ejex        = T.Properties.UserData.ifr_ejex;
    clname = sprintf('class%d', class_num);
    sel_class = T.channel_num == ch & strcmp(cellstr(T.class), clname);
    Tc = T(sel_class & ismember(T.stim_number, stim_list), :);
    m.mean_zscore = mean(Tc.zscore, 'omitnan');
    m.mu_base     = Tc.mu_base(1);
    m.std_base    = Tc.std_base(1);
    m.n_stims     = height(Tc);
    m.n_trials    = sum(Tc.ntrials);   % total trials scored, summed across stim_list -- for rate calc
    m.per_stim    = Tc;                % per-(class,stim) rows, stim_list only -- for the dynamic window
    m.per_stim_all = T(sel_class, :);  % SAME class, but EVERY stimulus scored this call -- reuse this for the
                                        % overall/session-wide summary later instead of re-running
                                        % characterize_grapes_responses; it's already sitting here for free.
end

function plot_stimlist_summary(summary, pct_in_sd1_vs_sd2, out_dir, ch, tag, window_label)
% One 2x2 figure characterizing INSIDE-window response spikes for a given
% stim set only (outside-window is tabled -- see the caller's comment on why
% n_out isn't trustworthy yet). tag/window_label let the caller distinguish
% different window definitions if more than one is ever run again (fixed
% +/-100ms was tried and dropped -- see the caller's comment).
    % VARIABLES
    %   summary     - the summary table for THIS window definition (arm,
    %                 class_id, n_in, pct_in_vs_gt, mean_zscore, mu_base, std_base)
    %   pct_in_sd1_vs_sd2 - the one-off sd1-vs-sd2 %% diff for this window definition
    %   out_dir/ch  - where to save, channel number (for title/filename)
    %   tag         - filename fragment distinguishing fixed vs dynamic (e.g. 'fixed_pm100ms')
    %   window_label- human-readable label for the figure title (e.g. 'fixed +/-100ms window')
    %   arms        - arm labels (gt/sd1/sd2) as categorical, x-axis for panels 1/3/4
    %   f/tl        - figure and tiledlayout handles
    %   ax          - current panel's axes handle
    %   pct_arms    - x-axis categories for panel 2 (sd1, sd2, sd1->sd2)
    %   pct_data    - panel 2's bar data: inside %% diff per category
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    arms = categorical(summary.arm, summary.arm);

    f = figure('Visible','off','Position',[100 100 950 760],'Color','w');
    tl = tiledlayout(f,2,2,'Padding','loose','TileSpacing','loose');
    title(tl, sprintf('ch%d class%d -- gt vs sd1 vs sd2 (%s)', ch, summary.class_id(1), window_label), 'Interpreter','none');

    % panel 1: inside-window counts
    ax = nexttile(tl);
    bar(ax, arms, summary.n_in);
    ax.YAxis.Exponent = 0;   % show full numbers, no lopsided "x10^4" offset label
    ylabel(ax, '# spikes'); title(ax, 'spike counts (inside window)');
    grid(ax,'on');

    % panel 2: % diff vs GT (sd1, sd2) + the one-off sd1->sd2 comparison
    ax = nexttile(tl);
    pct_arms = categorical({'sd1','sd2','sd1->sd2'}, {'sd1','sd2','sd1->sd2'});
    pct_data = [summary.pct_in_vs_gt(2); summary.pct_in_vs_gt(3); pct_in_sd1_vs_sd2];
    bar(ax, pct_arms, pct_data);
    yline(ax, 0, 'k-');
    ylabel(ax, '% difference'); title(ax, '% diff, inside window (vs GT; sd1->sd2 vs sd1)');
    grid(ax,'on');

    % panel 3: mean response z-score per arm
    ax = nexttile(tl);
    bar(ax, arms, summary.mean_zscore);
    yline(ax, 0, 'k-');
    ylabel(ax, 'mean z-score'); title(ax, 'response z-score (all scored stimuli)');
    grid(ax,'on');

    % panel 4: baseline mu +/- std per arm
    ax = nexttile(tl);
    bar(ax, arms, summary.mu_base); hold(ax,'on');
    errorbar(ax, arms, summary.mu_base, summary.std_base, 'k.', 'LineWidth', 1.2, 'CapSize', 12);
    ylabel(ax, 'baseline spike count'); title(ax, 'baseline mu +/- std');
    grid(ax,'on');

    save_fig_local(f, fullfile(out_dir, sprintf('stimlist_summary_%s_ch%d_class%d.png', tag, ch, summary.class_id(1))));
end

function save_fig_local(f, out_png)
    % VARIABLES
    %   f       - figure handle to save
    %   out_png - full output file path
    set(f, 'Color', 'w');
    try
        print(f, out_png, '-dpng', '-r130');
    catch
        exportgraphics(f, out_png, 'Resolution', 130, 'BackgroundColor', 'white');
    end
    close(f);
end

end