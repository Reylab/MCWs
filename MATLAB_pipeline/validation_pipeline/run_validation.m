function run_validation(cfg)
% RUN_VALIDATION  GT-vs-arm response validation for one channel/class.
%
%   run_validation(validation_cfg_pt10_emu7_ch368())
%
% Stage 1 (this file, so far): the stimulus raster pages and the GT-vs-ARM
% paired-line comparison, same as run_rescue_grape_analysis.m's
% stimulus_rasters section, with x-axis jitter on the paired-line points so
% stimuli with equal values no longer stack on top of each other.
%
% Reuses the real_response_analysis/ stage functions unchanged
% (match_arms_to_ground_truth, get_class_metrics, compute_dynamic_window_sets,
% plot_response_window_grapes), so numbers are identical to
% run_rescue_grape_analysis.m for the same inputs.
%
% OUTPUTS (under cfg.out_dir)
%   class_match_ch<ch>_*.png               - which arm class was matched to GT
%   stimulus_rasters/<name>/*.png          - raster pages (GT first, then each arm)
%   stimulus_rasters/<arm>/*_gt_vs_arm_summary.png - jittered paired-line summary
%   stimulus_rasters/<arm>/data_table.mat  - per-stimulus GT/arm values behind the plots
%   arm_classes.csv                        - resolved class per arm
%   pdf_reports/stimulus_rasters.pdf       - all of the above pages in one PDF
%   venn_three_set/                        - GT / arm / quarantined venns (run_venn_stage)
%   cluster_composition/                   - matched-cluster composition pages (run_composition_stage)
%   pie_venn/                              - pie groups by venn region (run_pie_venn_stage)
%   waveform_breakdowns/                   - rescued / not-rescued waveform pages (run_waveform_breakdown_stage)

here = fileparts(mfilename('fullpath'));
pipeline_root = fileparts(here);
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

%% session-wide data + onset map
finalevents = load(fullfile(cfg.root, 'finalevents.mat'));
stimulus    = load(fullfile(cfg.root, 'stimulus.mat'));
exp_prop    = load(fullfile(cfg.root, 'experiment_properties_online3.mat'));
Nscr = min(numel(stimulus.stimulus), numel(exp_prop.scr_end_cell));
onset_map = build_all_stim_onset_map(stimulus, finalevents, Nscr, stim_list);

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
class_match_figure(DET_all, COMP_all, ch, arm_names, cfg.out_dir);
writetable(table(arm_names(:), arm_classes(:), 'VariableNames', {'arm','class'}), ...
    fullfile(cfg.out_dir, 'arm_classes.csv'));
for k = 1:n_arms
    fprintf('[validation] %s -> class %d (GT class %d)\n', arm_names{k}, arm_classes(k), cfg.gt.class);
end

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

dyn = compute_dynamic_window_sets(all_class_ids, all_grapes, all_cc, onset_map, all_per_stim, stim_list, ch, cfg.dyn_buffer);
n_in_dyn = arrayfun(@(d) numel(d.inside), dyn)';

%% stimulus raster pages + jittered GT-vs-ARM paired-line summary
stim_raster_dir = fullfile(cfg.out_dir, 'stimulus_rasters');
gt_ref = [];
arm1_ref = [];
for k = 1:numel(all_names)
    per_stim_k = all_per_stim{k};
    lo = per_stim_k.onset - cfg.dyn_buffer;
    hi = per_stim_k.onset + per_stim_k.dura + cfg.dyn_buffer;
    cluster_total_k = nnz(all_cc{k}(:,1) == all_class_ids(k));

    dt_k = plot_response_window_grapes(all_grapes{k}, per_stim_k, ch, all_class_ids(k), n_in_dyn(k), lo, hi, ...
        all_metrics{k}.rank_config, all_metrics{k}.ejex, Nscr, sprintf('%s_stimulus_rasters', all_names{k}), ...
        fullfile(stim_raster_dir, all_names{k}), cfg.root, [], false, cluster_total_k, gt_ref, ...
        'x_jitter', cfg.x_jitter);
    if k == 1
        gt_ref = dt_k;
    elseif k==2
        arm1_ref = dt_k;
    end
end
save(fullfile(cfg.out_dir,'arm1_ref'))

plot_response_spikes(arm1_ref);

%% bundle into one PDF: GT pages, then each arm's pages + its summary
pngs = list_pngs_sorted(stim_raster_dir, [{fullfile(cfg.gt.name, '*.png')}, ...
    cellfun(@(n) fullfile(n, '*.png'), arm_names, 'UniformOutput', false)]);
compile_pngs_to_pdf(pngs, fullfile(cfg.out_dir, 'pdf_reports', 'stimulus_rasters.pdf'), ...
    cellfun(@(p) erase(p, [stim_raster_dir filesep]), pngs, 'UniformOutput', false));

%% GT / arm / quarantined three-circle venns (cluster + full spike set)
run_venn_stage(cfg, arm_classes);

%% matched-cluster composition pie (sd1 first pass / forced / rescued)
run_composition_stage(cfg, arm_classes);

%% pie groups by venn region (companion to the composition pie and the venn)
run_pie_venn_stage(cfg, arm_classes);

%% waveform breakdowns: rescued by class / by quarantine property, not rescued by property
run_waveform_breakdown_stage(cfg, arm_classes);

fprintf('[validation] done -> %s\n', cfg.out_dir);
end
