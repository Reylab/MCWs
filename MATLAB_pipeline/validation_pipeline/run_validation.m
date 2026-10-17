function run_validation(cfg)
% run_validation('pt11_ch328_second_pass') or run_validation(cfg): a session id
% from validation_sessions.m, or a cfg struct.
% RUN_VALIDATION  GT-vs-arm response validation for one channel/class.
%
%   run_validation('pt10_ch368_recluster_sd1')     % id from validation_sessions.m
%
% Stage 1 (this file, so far): the stimulus raster pages and the GT-vs-ARM
% paired-line comparison, same as run_rescue_grape_analysis.m's
% stimulus_rasters section, with x-axis jitter on the paired-line points so
% stimuli with equal values no longer stack on top of each other.
%
% Reuses the real_response_analysis/ stage functions unchanged
% (match_arms_to_ground_truth, get_class_metrics,
% plot_response_window_grapes), so numbers are identical to
% run_rescue_grape_analysis.m for the same inputs.
%
% OUTPUTS (under cfg.out_dir)
%   class_match_ch<ch>_<ref>_vs_<arm>.png  - which arm class was matched to the reference (GT or e.g. wave_clus)
%   class_match_ch<ch>_<arm>_vs_<ref>.png  - the other way round: the matched arm cluster over the reference's classes
%   stimulus_rasters/<name>/*.png          - raster pages (GT first, then each arm)
%   stimulus_rasters/<arm>/*_gt_vs_arm_summary.png - jittered paired-line summary
%   stimulus_rasters/<arm>/data_table.mat  - per-stimulus GT/arm values behind the plots
%   arm_classes.csv                        - resolved class per arm
%   pdf_reports/stimulus_rasters.pdf       - all of the above pages in one PDF
%   venn_three_set/                        - GT / arm / quarantined venns (run_venn_stage)
%   cluster_composition/                   - matched-cluster composition pages (run_composition_stage)
%   pie_venn/                              - pie groups by venn region (run_pie_venn_stage)
%   waveform_breakdowns/                   - rescued / not-rescued waveform pages (run_waveform_breakdown_stage)
%   subsets/                               - saved spike subsets per arm (save_validation_subsets)
%   subset_rasters/<arm>/<subset>/         - (only when 'subset_rasters' is asked for) raster pages of gt_only, arm_only, matched, quarantined
%                                            never rescued, and each of its quarantine properties,
%                                            on the GT / arm latency and windows (plot_subset_rasters)
%   results/                               - every raster page's numbers, GT / arms / subsets, long
%                                            format for comparing runs (export_validation_results)
%   validation_state.mat                   - cfg, arm classes, per-stimulus tables, onset map:
%                                            what rerun_validation / plot_subset_rasters reload
%
% cfg.stages (optional, default 'all' = every stage but subset_rasters): cell of stage names to run, any of
%   'class_match' 'rasters' 'venn' 'composition' 'pie_venn' 'waveforms' 'subsets'
%   'subset_rasters' 'results' 'check' 'report'
% ('rasters' includes the response-waveform page and the PDF). To redo stages
% on a finished run, or with a different class, use rerun_validation.

here = fileparts(mfilename('fullpath'));
pipeline_root = fileparts(here);
addpath(here);   % the stages live here; run_validation cd's to the session root below
if ischar(cfg) || isstring(cfg), cfg = validation_sessions(char(cfg)); end
addpath(fullfile(pipeline_root, 'real_response_analysis'));
addpath(fullfile(pipeline_root, 'responses'));
addpath(genpath(fullfile(pipeline_root, 'codes_emu')));
addpath(genpath(fullfile(pipeline_root, 'useful_functions')));

old = pwd;
cleanup = onCleanup(@() cd(old));
cd(cfg.root);

ch = cfg.ch;
stim_list = cfg.stim_list;
if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end
if ~isfield(cfg, 'stages') || isempty(cfg.stages), cfg.stages = 'all'; end
if ~isfield(cfg, 'max_dura'), cfg.max_dura = 250; end   % ms; longer responses also get a default-window raster version
if ~isfield(cfg, 'max_lat'), cfg.max_lat = 400; end     % ms; later latencies also get a default-window raster version
% 'all' = every stage except subset_rasters, which runs only when named
do = @(stage) (ischar(cfg.stages) && strcmp(cfg.stages, 'all') && ~strcmp(stage, 'subset_rasters')) || ...
    any(strcmp(cellstr(cfg.stages), stage));

%% session-wide data + onset map
finalevents = load(fullfile(cfg.root, 'finalevents.mat'));
stimulus    = load(fullfile(cfg.root, 'stimulus.mat'));
exp_prop    = load(fullfile(cfg.root, 'experiment_properties_online3.mat'));
Nscr = min(numel(stimulus.stimulus), numel(exp_prop.scr_end_cell));
onset_map = build_all_stim_onset_map(stimulus, finalevents, Nscr, stim_list, exp_prop);

%% GT full class population (matching reference)
times_gt  = load(fullfile(cfg.gt.dir, cfg.gt.times_file));
grapes_gt = load(fullfile(cfg.gt.dir, cfg.gt.grapes_file));
gt_abs_times = sort(times_gt.cluster_class(times_gt.cluster_class(:,1) == cfg.gt.class, 2));
ccg = [repmat(cfg.gt.class, numel(gt_abs_times), 1), gt_abs_times(:)];

%% arms + class matching
arms = cfg.arms;
n_arms = numel(arms);
arm_names = {arms.name};
times_arms = cell(1, n_arms); grapes_arms = cell(1, n_arms);
for k = 1:n_arms
    times_arms{k}  = load(fullfile(arms(k).dir, arms(k).times_file));
    grapes_arms{k} = load(fullfile(arms(k).dir, arms(k).grapes_file));
end
arm_cc_list = cellfun(@(t) t.cluster_class, times_arms, 'UniformOutput', false);

[arm_classes, ~, DET_all, COMP_all] = match_arms_to_ground_truth( ...
    ccg, gt_abs_times, arm_names, arm_cc_list, ch, cfg.tol, cfg.min_recall);
for k = 1:n_arms
    if ~isempty(arms(k).class)
        arm_classes(k) = arms(k).class;
    end
end
ref_name = cfg.gt.name; if strcmpi(ref_name, 'gt'), ref_name = 'GT'; end   % the reference, in labels
% an arm whose class was fixed in cfg: its class-match tables follow that class,
% not the automatic (most-overlap) pick
for k = 1:n_arms
    if isempty(arms(k).class), continue; end
    [~, Dk, Ck] = match_cc(ccg, arm_cc_list{k}, ch, cfg.tol, cfg.min_recall, arms(k).class);
    Dk.arm = repmat(string(arm_names{k}), height(Dk), 1); Ck.arm = repmat(string(arm_names{k}), height(Ck), 1);
    DET_all  = [DET_all(DET_all.arm ~= string(arm_names{k}), :); Dk];
    COMP_all = [COMP_all(COMP_all.arm ~= string(arm_names{k}), :); Ck];
end
if do('class_match')
    class_match_figure(DET_all, COMP_all, ch, arm_names, cfg.out_dir, ref_name);
    % and the other way round: each arm's matched cluster as the reference,
    % broken down over the reference's classes (class_match_ch<ch>_<arm>_vs_<ref>.png)
    for k = 1:n_arms
        at = sort(arm_cc_list{k}(arm_cc_list{k}(:, 1) == arm_classes(k), 2));
        if isnan(arm_classes(k)) || isempty(at), continue; end
        [~, ~, Dr, Cr] = match_arms_to_ground_truth([repmat(arm_classes(k), numel(at), 1) at], at, ...
            {ref_name}, {times_gt.cluster_class}, ch, cfg.tol, cfg.min_recall);
        class_match_figure(Dr, Cr, ch, {ref_name}, cfg.out_dir, arm_names{k});
    end
end
writetable(table(arm_names(:), arm_classes(:), 'VariableNames', {'arm','class'}), ...
    fullfile(cfg.out_dir, 'arm_classes.csv'));
for k = 1:n_arms
    fprintf('[validation] %s -> class %d (%s class %d)\n', arm_names{k}, arm_classes(k), ref_name, cfg.gt.class);
end

%% run state, reloaded by rerun_validation / plot_subset_rasters. The raster
% tables are kept from an earlier run when this call skips the rasters.
state_file = fullfile(cfg.out_dir, 'validation_state.mat');
ref_tables = {}; n_in_window = [];
if ~do('rasters') && exist(state_file, 'file')
    old_state = load(state_file);
    if isfield(old_state, 'arm_classes') && isequal(old_state.arm_classes(:), arm_classes(:))
        ref_tables = old_state.ref_tables;
        if isfield(old_state, 'n_in_window'), n_in_window = old_state.n_in_window; end
    end
end
unit_names = [{cfg.gt.name}, arm_names]; %#ok<NASGU>
save(state_file, 'cfg', 'arm_classes', 'ref_tables', 'n_in_window', 'unit_names', 'onset_map', 'Nscr');

if do('rasters')
%% per-arm metrics (GT first)
all_names     = [{cfg.gt.name}, arm_names];
all_dirs      = [{cfg.gt.dir}, {arms.dir}];
all_gfiles    = [{cfg.gt.grapes_file}, {arms.grapes_file}];
all_class_ids = [cfg.gt.class; arm_classes(:)];
all_grapes    = [{grapes_gt}, grapes_arms];
all_cc        = [{times_gt.cluster_class}, arm_cc_list];

all_metrics = cell(1, numel(all_names));
for k = 1:numel(all_names)
    all_metrics{k} = get_class_metrics(all_dirs{k}, all_gfiles{k}, ch, all_class_ids(k), stim_list, cfg.root);
end
all_per_stim = cellfun(@(m) m.per_stim, all_metrics, 'UniformOutput', false);

%% stimulus raster pages + jittered GT-vs-ARM paired-line summary
% Two versions when any stimulus has a late latency (> cfg.max_lat) or a long response (> cfg.max_dura):
%   defined  (stimulus_rasters/)                 every window = latency / duration (onset - buffer .. onset + dura + buffer)
%   default  (stimulus_rasters_default_window/)  those stimuli get grapes' default post window [tmin_median, tmax_median]
%                                                after stimulus onset; the rest keep latency / duration
% Each page says at the bottom which one it is. ref_tables = defined, ref_tables_default = default (or {}).
flag = cell(1, numel(all_names));
for k = 1:numel(all_names)
    ps = all_per_stim{k}; flag{k} = ps.dura > cfg.max_dura | ps.onset > cfg.max_lat;
end
versions = {'defined'}; if any(cellfun(@any, flag)), versions{end + 1} = 'default'; end
ref_tables_default = {};
for v = 1:numel(versions)
ver = versions{v};
stim_raster_dir = fullfile(cfg.out_dir, 'stimulus_rasters');
if strcmp(ver, 'default'), stim_raster_dir = [stim_raster_dir '_default_window']; end
n_in_window = zeros(1, numel(all_names));
gt_ref = [];
arm1_ref = [];
tabs = cell(1, numel(all_names));
for k = 1:numel(all_names)
    per_stim_k = all_per_stim{k};
    % the scoring windows behind median_post / sign test, kept with the numbers
    rc = all_metrics{k}.resp_conf;
    per_stim_k.post_lo = per_stim_k.tons;
    per_stim_k.post_hi = per_stim_k.tons + all_metrics{k}.rank_config.win_length;
    per_stim_k.base_lo = repmat(-rc.tmax_median, height(per_stim_k), 1);
    per_stim_k.base_hi = repmat(-rc.tmin_median, height(per_stim_k), 1);
    lo = per_stim_k.onset - cfg.dyn_buffer;
    hi = per_stim_k.onset + per_stim_k.dura + cfg.dyn_buffer;
    fl = flag{k}; ids = per_stim_k.stim_number(fl);
    if strcmp(ver, 'default')
        lo(fl) = rc.tmin_median; hi(fl) = rc.tmax_median;
        footer = sprintf(['DEFAULT WINDOW %g-%g ms after onset for stim %s (latency > %d or duration > %d ms); ' ...
            'other stimuli: latency / duration'], rc.tmin_median, rc.tmax_median, mat2str(ids(:)'), cfg.max_lat, cfg.max_dura);
        if isempty(ids), footer = 'WINDOW: latency / duration (no stimulus of this unit has a late latency or a long duration)'; end
    else
        footer = sprintf('WINDOW: latency / duration (onset-%d to onset+dura+%d ms)', cfg.dyn_buffer, cfg.dyn_buffer);
        if ~isempty(ids)
            footer = sprintf('%s | stim %s: latency > %d or duration > %d ms -> also a *_default_window version', ...
                footer, mat2str(ids(:)'), cfg.max_lat, cfg.max_dura);
        end
    end
    cluster_total_k = nnz(all_cc{k}(:,1) == all_class_ids(k));
    % title count: this class's spikes (times file) inside the green windows drawn
    % on the page -- the same windows as each stimulus's n= and the subsets
    n_in_window(k) = count_in_windows(all_cc{k}(all_cc{k}(:,1) == all_class_ids(k), 2), ...
        per_stim_k.stim_number, lo, hi, onset_map); %#ok<AGROW>

    dt_k = plot_response_window_grapes(all_grapes{k}, per_stim_k, ch, all_class_ids(k), n_in_window(k), lo, hi, ...
        all_metrics{k}.rank_config, all_metrics{k}.ejex, Nscr, sprintf('%s_stimulus_rasters', all_names{k}), ...
        fullfile(stim_raster_dir, all_names{k}), cfg.root, [], false, cluster_total_k, gt_ref, ...
        'x_jitter', cfg.x_jitter, 'ref_name', ref_name, 'arm_name', all_names{k}, 'footer', footer);
    [~, ia] = ismember(dt_k.stim_number, per_stim_k.stim_number); dt_k.default_win = false(height(dt_k), 1);
    if strcmp(ver, 'default'), dt_k.default_win(ia > 0) = fl(ia(ia > 0)); end
    dt_k.Properties.UserData.footer = footer;
    if k == 1
        gt_ref = dt_k;
    elseif k==2
        arm1_ref = dt_k;
    end
    tabs{k} = dt_k;
end
if strcmp(ver, 'defined'), ref_tables = tabs; n_in_window_defined = n_in_window; else, ref_tables_default = tabs; end
end
n_in_window = n_in_window_defined;
stim_raster_dir = fullfile(cfg.out_dir, 'stimulus_rasters');
save(state_file, 'ref_tables', 'ref_tables_default', 'n_in_window', '-append');
save(fullfile(cfg.out_dir,'arm1_ref'))

plot_response_spikes(arm1_ref);

%% bundle into one PDF: GT pages, then each arm's pages + its summary
pngs = list_pngs_sorted(stim_raster_dir, [{fullfile(cfg.gt.name, '*.png')}, ...
    cellfun(@(n) fullfile(n, '*.png'), arm_names, 'UniformOutput', false)]);
compile_pngs_to_pdf(pngs, fullfile(cfg.out_dir, 'pdf_reports', 'stimulus_rasters.pdf'), ...
    cellfun(@(p) erase(p, [stim_raster_dir filesep]), pngs, 'UniformOutput', false));
end

%% GT / arm / quarantined three-circle venns (cluster + full spike set)
if do('venn'), run_venn_stage(cfg, arm_classes); end

%% matched-cluster composition pie (sd1 first pass / forced / rescued)
if do('composition'), run_composition_stage(cfg, arm_classes); end

%% pie groups by venn region (companion to the composition pie and the venn)
if do('pie_venn'), run_pie_venn_stage(cfg, arm_classes); end

%% waveform breakdowns: rescued by class / by quarantine property, not rescued by property
if do('waveforms'), run_waveform_breakdown_stage(cfg, arm_classes); end

%% spike subsets (gt only, matched, arm only, pie groups, ...) for later digging
if do('subsets'), save_validation_subsets(cfg, arm_classes); end

%% rasters of those subsets, on the GT / arm latency, duration and windows
% if do('subset_rasters')
%     for k = 1:n_arms
%         plot_subset_rasters('default', 'dir', cfg.out_dir, 'arm', k);
%     end
% end

%% every number behind the raster pages, one long format per run
if do('results'), export_validation_results(cfg.out_dir); end

%% count checks (venn / pie / subsets add up; GT and arm classes times vs grapes)
if do('check'), check_validation_run(cfg); end

%% one PDF of every page, in reading order
if do('report'), make_validation_report(cfg); end

fprintf('[validation] done -> %s\n', cfg.out_dir);
end
