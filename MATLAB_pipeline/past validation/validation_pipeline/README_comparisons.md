# Comparing a wave_clus run with GT and python, and notch vs no notch

Five steps per channel. Every function is in `validation_pipeline/`; add it to the path first:

```matlab
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline')
SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
```

## 0. What a channel folder needs

```
<channel>/                       e.g. 'pt 6 emu 13 ch 262'
  hand selected/                 GT: times_<name>.mat + grapes_blanks.mat
  python_recluster_<date>_second_pass/   python arm: times file + grapes_blanks_rescue.mat + metrics/report_*.pdf
  wave_clus/  wave_clus no notch/        one folder per wave_clus run (name it however you like)
  <name>.NC5                     raw data (only needed for the notch vs no notch "why missing" pages)
  finalevents.mat, stimulus.mat, experiment_properties_online3.mat
```

Each wave_clus run folder should keep everything wave_clus writes: `<name>_spikes.mat`, `times_<name>.mat`,
`data_<name>.dg_01`, `data_<name>.dg_01.lab`, `spc_log.txt`, `fig2print_*.png`.
`spc_log.txt` holds the SPC `RandomSeed` and the times file holds `ipermut` -- those two make the
clustering reproducible. Keep one run per folder (a second run in the same folder overwrites the log).

The channel also needs a row in `validation_sessions.m` (one row per GT unit):
`'<id>', '<channel folder>', ch, [stimulus list], gt('times_<name>.mat', <GT class>), {python arm}, '<out folder>', {}`.
`validation_sessions()` lists every id.

## 1. Run wave_clus (in the run's folder)

```matlab
cd(fullfile(SC, 'pt 6 emu 13 ch 262', 'wave_clus'))   % NC5, NSX_TimeStamps.mat, NSx.mat and (for the notch)
Get_spikes('mRAMY06 raw_262.NC5')                     % pre_processing_info.mat must be reachable here
Do_clustering('mRAMY06 raw_262_spikes.mat')
```

Notch on/off is `par.preprocessing` (true = notch from `pre_processing_info.mat`, saved in the spikes
file as `par.process_info`; the times file does NOT keep `process_info`, so keep the spikes file).

## 2. grapes for the run

```matlab
root = fullfile(SC, 'pt 6 emu 13 ch 262');
build_wave_clus_arm(root, 262, fullfile(root, 'wave_clus'))     % writes grapes_blanks.mat into that folder
```
It needs the GT's `exp_type` (from `hand selected/`) and `finalevents.mat` with the blank onsets.

## 3. GT vs wave_clus and wave_clus vs python  -> `venn_wave_clus.m`

```matlab
venn_wave_clus('pt6emu13_ch262_class3', 'wave_clus')             % session id, wave_clus folder name
venn_wave_clus('pt6emu13_ch262_class3', 'wave_clus no notch')
% options: 'python', '<arm folder>'  'gt_vs_python', true  'tol', 0.7
```

What it does:
1. `validation_sessions(id)` -> channel folder, GT times file, GT class `g`, stimuli, `cfg.tol` (0.5 ms).
2. **GT vs wave_clus**: `run_validation` with stages `class_match` + `venn`, wave_clus as the arm.
   `match_arms_to_ground_truth` pairs spikes one-to-one within `tol` and picks the wave_clus class
   holding most of GT class `g` -> `class_match_figure` (both directions) and `run_venn_stage` ->
   `plot_three_set_venn` (cluster / full / all-detections venns).
3. **python's class X**: the same matching with python as the arm (kept only with `'gt_vs_python', true`).
4. **wave_clus vs python**: `python_cluster_validation(id, wc_folder, X)` -- wave_clus becomes the
   reference (its class = the one holding most of python cluster X), python stays the arm so its
   quarantined / rescued spikes show in the venn.

Output (inside the wave_clus folder):
```
venn_gt_vs_wave_clus_class<g>/        class_match_*_GT_vs_wave_clus.png   which wave_clus class each GT spike went to
                                      class_match_*_wave_clus_vs_GT.png   the matched wave_clus class over GT's classes
                                      venn_three_set/venn3_*_cluster.png  GT class g / wave_clus class / quarantined
                                      venn_three_set/venn3_*_full.png     same, all of the arm's spikes
                                      venn_three_set/venn3_*_detections.png  every detection of both
                                      arm_classes.csv, validation_state.mat (cfg used)
venn_wave_clus_vs_python_c<X>/        same files, wave_clus as reference, python cluster X as the arm
```

## 4. notch vs no notch  -> `compare_detections.m`

```matlab
R = fullfile(SC, 'pt 6 emu 13 ch 262');
compare_detections(fullfile(R, 'wave_clus no notch', 'mRAMY06 raw_262_spikes.mat'), ...
                   fullfile(R, 'wave_clus', 'mRAMY06 raw_262_spikes.mat'), ...
                   'names', {'no notch', 'notch'}, 'tol', 0.7, ...
                   'raw', fullfile(R, 'mRAMY06 raw_262.NC5'), ...
                   'out_dir', fullfile(R, 'compare_no_notch_vs_notch'))
% the two inputs can also be structs from load(file)
```

Use the **spikes** files (they carry `par.process_info`; a times file loses the notch).

How it works:
1. Reads each run's spike times (`index`, ms) and waveforms.
2. Pairs spikes one-to-one, closest first, within `tol` -> both / A only / B only; also at every
   tolerance up to `max_shift` (3 ms) for the tolerance curve.
3. With `'raw'`: rebuilds each run's detection from the NC5 and its own `par` (ellip bandpass,
   `[process_info.SOS; tf2sos]` cascade with gain `G` when preprocessing, `fast_filtfilt`, 5x
   median threshold per 5-min segment, 1.5 ms refractory, peak = min of the sort signal in the
   22 samples after the crossing, stdmax artifact check) and checks it reproduces the run (prints
   thresholds vs the file's and % of spikes reproduced -- both should be 100%).
4. For every spike only one run has, reads the OTHER run's own replay log at that time and records
   why it has no spike there: threshold (never crosses) / refractory period (its own spike 0-1.5 ms
   earlier) / crossed but peak placed > tol away / artifact check / paired with another spike /
   segment edge.

Output:
```
detections_overview.png       venn | matched % vs tolerance | shift of every pair (1 bar = 1 sample)
detections_timeline.png       firing diagram: A only / both / B only over time, rate per 10 s, shifts over time
detections_why_missing.png    unique spikes: stored waveforms, the same times in both detection signals, cause counts
detections_why_timeline.png   unique spikes by cause over time + their peak vs the other run's threshold
detections_why_examples.png   +-5 ms around examples of each cause (v = A spikes, ^ = B spikes, star = the spike)
detections_why_missing.csv    one row per unique spike: time, cause, detail (ms), peak in the other signal
detections_summary.csv        matched count at each tolerance
detections_match.mat          indices of pairs / A only / B only, shifts, causes
```

## 5. Collect the figures for review  -> `collect_comparison_figures.m`

```matlab
T = fullfile(SC, 'temp_notch_vs_no_notch');
c = fullfile(SC, 'pt 6 emu 13 ch 262');
collect_comparison_figures(c, 'wave_clus', T)            % the notch run + the figures shared by the channel
```
Copies only. Layout per channel (this is how `temp_notch_vs_no_notch/` is laid out, notch runs only):
```
ch262/wave_clus_spikes/           the notch run's fig2print
ch262/gt_vs_wave_clus_class3/     GT vs the notch run: both class matches + cluster venn
ch262/wave_clus_vs_python_c1/     the notch run vs python cluster X: both class matches + cluster venn
ch262/gt_vs_python_class3/        GT vs python: both class matches + cluster venn
ch262/python_spikes/              python report cluster pages (png)
ch262/no_notch_vs_notch/          compare_detections pages + csv
```
To put two runs of one channel side by side, use `'label'` (sub-folder for the run's figures) and
`'shared', false` on the second call so the shared figures are not copied twice.
