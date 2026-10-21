# MCWs (python) vs wave_clus vs GT: every step, from the python run to the for ppt report

This file covers the python side only. The wave_clus run (`<channel>/wave_clus notch/`, GT's notch) is done
once per channel and is not re-run (see `README_comparisons.md` for how it was made).

Paths used below:

```
MWS = /home/ajb/Documents/GitHub/MicroWireSorter                 python sorter (venv: $MWS/.venv)
VP  = /home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline
RR  = /mnt/data0/recluster_runs                                   python run output
SC  = /mnt/data0/MCWs Sessions/matlab validation/single channel  one folder per channel
```

Rules: never write into sEEG_DATA (only read it); never use BAD / zneed / need_times / mismatch folders or
circ-shift files; GT (`hand selected/`) is the reference; batches run at most 2 at a time, low priority
(`nice -n 10`, MATLAB `-singleCompThread`), and one channel is checked before the rest are batched.

---

## 0. What the channel folder must already have

```
<SC>/<channel>/                        e.g. 'pt 6 emu 13 ch 264'
  hand selected/                       GT: times_<name>.mat + grapes_blanks.mat (grapes = times class counts)
  wave_clus notch/                     wave_clus with GT's notch (already done; *_spikes.mat holds GT's notch in par.process_info)
  <name>.NC5, NSx.mat                  the channel's data / channel table
  finalevents.mat                      stimulus onsets, with seq_beg_blanks_cell (blank onsets)
  stimulus.mat, experiment_properties_online3.mat
```

## 1. Pick the recording folder python reads (it must give GT's notch)

python takes its notch from the recording folder, first match wins
(`mcws/utils/preprocessing_cache.py`, `load_notches`):

1. `<recording>/preprocessing_info/<file stem>.notches.pkl`   (python's own cache -- if present it beats GT's file)
2. `<recording>/preprocessing_info/pre_processing_info.mat`
3. `<recording>/pre_processing_info.mat`
4. none: python fits its own adaptive notch.

So:
* if the session's own `pre_processing_info.mat` in sEEG_DATA has the same notch for this channel as GT's
  (`wave_clus notch/*_spikes.mat` -> `par.process_info`) and there is no `.notches.pkl`, use the sEEG_DATA
  recording folder directly;
* otherwise use a copy of the recording folder in `$RR/_gt_notch_recordings/<pNN_EMU-EEE...>/` holding GT's
  `pre_processing_info.mat` (and nothing in `preprocessing_info/`).

Always pass the channel number: with a channel subset python never writes a notch cache, so nothing is
written into the recording folder.

## 2. Run the python sorter (clustering -> rescue -> second pass)

One run folder per channel, named `pNN_EMU-EEE_chCCC` (the converter matches folders by this name):

```bash
cd $MWS
nice -n 10 .venv/bin/python test_full_pipeline.py "<recording folder>" "$RR/p06_EMU-013_ch264" 264
```

A fresh SPC seed is drawn for every run (printed as `run seed` in `batch.log`, saved in the times files as
`clustering_metadata.spc_seed` / `permut_seed`, with `ipermut`). No `spc_log.txt` is written.

Output (`$RR/p06_EMU-013_ch264/`):

```
batch.log, logs/pipeline_<ts>.log     everything printed; the notch line, the run seed, the exit code
plots/                                bundle traces, Spectrum/ (notch check)
spikes_<ts>/                          detected spikes
times_<ts1>/                          clustering only (no quarantine.rescued)     -> arm *_clustered
times_<ts2>/                          + artifact rescue                           -> arm *_rescued
   times_<name>.mat, data_<name>.dg_01(.lab) (SPC tree), metrics/ (report_*.pdf, metrics_*.csv)
second_pass_<ts>/summary.csv
second_pass_<ts>/<label>_<ch>/        second_pass_report.pdf, before_after.pdf, decisions.json,
   times_<ts>/                        the second-pass sort (times file, metrics/, temperatures.png)  -> arm *_second_pass
```

Check in `batch.log` before going on:
* `Loaded adaptive notch profile from MATLAB-compatible <the file you meant>` (step 1),
* `Applying filter chain: elliptic BP -> adaptive notch`, `detection='neg'`,
* the last line: `Pipeline session ended (exit_code=0, ...)`.

Second pass alone, on an existing sort: `.venv/bin/python recluster_test/run.py "$RR/<run>/times_<ts2>" ["<out dir>"] [<ch>]`.

## 3. Convert the three sorts to MATLAB arms (and their grapes)

```matlab
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline')
RR = '/mnt/data0/recluster_runs';  SC = '/mnt/data0/MCWs Sessions/matlab validation/single channel';
run = {'p06_EMU-013_ch264'};  sp = 'second_pass_*/*';      % the second-pass sort under the run folder
T = convert_recluster_runs(RR, SC, 'runs', run, 'second_pass', sp, 'dry_run', true)   % match + checks only
T = convert_recluster_runs(RR, SC, 'runs', run, 'second_pass', sp)                    % convert
```

* Matching: `pNN_EMU-EEE_chCCC` -> the one channel folder whose name contains `pt NN emu EEE ch CCC`. Checked
  first: the NC5 name equals the times file's, and the onsets (`finalevents.pics_onset`) and
  `experiment_properties_online3.mat` equal the recording's.
* Each sort -> `<channel>/python_recluster_<date>_{clustered,rescued,second_pass}/`, built by
  `build_python_arm_from_times` (adapts the times / spikes files, then `do_structure_mu_BCM_online3` ->
  `do_structure_sorted_BCM_online3` -> `do_structure_rescue_BCM_online3`): `times_<name>.mat`,
  `<name>_spikes.mat`, `grapes_blanks.mat`, `grapes_blanks_rescue.mat`.
* Every conversion is logged in `<SC>/recluster_conversion_log.csv` (run, folder, channel, arm, source, status).

## 4. Check grapes = times for every arm

For each arm and each class k > 0: `nnz(cluster_class(:,1) == k)` in the arm's `times_<name>.mat` must equal
`rasters.chan<ch>.details.class<k>` in its `grapes_blanks_rescue.mat` (and `grapes_blanks.mat`):

```matlab
a = fullfile(SC, 'pt 6 emu 13 ch 264', 'python_recluster_20261019_second_pass');  ch = 264;
t = dir(fullfile(a, 'times_*.mat'));  c = load(fullfile(a, t(1).name), 'cluster_class');  c = c.cluster_class(:, 1);
G = load(fullfile(a, 'grapes_blanks_rescue.mat'), 'rasters');  dt = G.rasters.(sprintf('chan%d', ch)).details;
k = unique(c(c > 0))';  [arrayfun(@(x) nnz(c == x), k); arrayfun(@(x) dt.(sprintf('class%d', x)), k)]   % rows must match
```

## 5. Copy the python reports into the arms (conversion does not)

```bash
# metrics: each sort's times_<ts>/metrics/ -> <channel>/<arm>/metrics/
cp -rn "$RR/p06_EMU-013_ch264/times_<ts1>/metrics" "$SC/pt 6 emu 13 ch 264/python_recluster_<date>_clustered/"
cp -rn "$RR/p06_EMU-013_ch264/times_<ts2>/metrics" "$SC/pt 6 emu 13 ch 264/python_recluster_<date>_rescued/"
cp -rn "$RR/p06_EMU-013_ch264/second_pass_<ts>/<label>_264/times_<ts>/metrics" "$SC/pt 6 emu 13 ch 264/python_recluster_<date>_second_pass/"

# second-pass report: pdfs + decisions + summary -> <arm>/second_pass_report/ (with SOURCE.txt = where it came from)
S="$RR/p06_EMU-013_ch264/second_pass_<ts>/<label>_264";  D="$SC/pt 6 emu 13 ch 264/python_recluster_<date>_second_pass/second_pass_report"
mkdir -p "$D" && cp -n "$S"/{second_pass_report.pdf,before_after.pdf,decisions.json} "$S/../summary.csv" "$D/" && echo "$S" > "$D/SOURCE.txt"
```

The page images for the report go in in step 7.

## 6. One row per GT unit in `validation_sessions.m`

```
'<id>', '<channel folder>', ch, [stimulus list], gt('times_<name>.mat', <GT class>), {python arm}, '<output folder>', {}
```
e.g. `'pt6emu13_ch264_class3', 'pt 6 emu 13 ch 264', 264, [615 44 842], gt('times_mRAMY08 raw_264.mat', 3), {rec14('times_mRAMY08 raw_264.mat')}, 'validation_gt_vs_python_recluster_20261014_class3', {}`.
`validation_sessions()` lists every id. A channel with several GT units has several rows (one per GT class)
and its report goes in `for ppt/GT c<g>/` (`multi = true` below).

The report code takes the python arm = the newest `python_recluster_*_second_pass` folder in the channel.

## 7. Validation + the for ppt report (one GT unit)

```matlab
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline')
id = 'pt6emu13_ch264_class3';  multi = false;

ppt_session(id, multi)              % a-e below
ppt_wc_py_destinations(id, multi)   % by_destination for wave_clus notch 03, 05, 03b, 03c, 03d
ppt_mahal_forced_rescued(id, multi) % 10_mahal_forced_rescued
ppt_not_rescued_by_mask(id, multi)  % never-rescued quarantined spikes by mask + the two quarantine pages
ppt_check_unit(id, multi)           % accounting.csv: PASS / FAIL
ppt_flags(id, multi)                % flags.csv
```

`ppt_session` does, in order:
a. `build_wave_clus_arm` -- grapes for `wave_clus notch` (once per channel);
b. moves venns made for another python arm to `wave_clus notch/old_venns/`;
c. `venn_wave_clus(id, 'wave_clus notch', 'gt_vs_python', true)` -- GT vs wave_clus, GT vs python and
   wave_clus vs python class matches + venns (`run_validation`, `python_cluster_validation`);
d. `build_ppt_report` -- 01 metrics, 02 venns, 03-06 (06 in each matched class's windows), 08 full raster runs
   (`validation_gt_vs_<python>_class<g>`, `wave_clus notch/validation_wave_clus_vs_python_c<X>`: the windows
   every subset uses);
e. `ppt_python_subsets` (03b / 03c / 03d) and `ppt_gt_subsets` (09 GT vs python, 09 GT vs wave_clus, with
   by_destination).

Then the second-pass report pages (python second pass, into the report's 01_metrics):

```bash
D="$SC/pt 6 emu 13 ch 264/python_recluster_<date>_second_pass/second_pass_report"; O="$SC/pt 6 emu 13 ch 264/for ppt/01_metrics/python_second_pass"
mkdir -p "$O" && pdftoppm -r 110 -png "$D/second_pass_report.pdf" "$O/second_pass_report" && pdftoppm -r 110 -png "$D/before_after.pdf" "$O/before_after"
```

Optional, per question:
* `ppt_not_rescued_by_mask(id, multi, true)` -- only the two quarantine pages;
* `ppt_compare_classes(id, multi, k)` -- the whole comparison with python class k instead of the matched one
  -> `for ppt/alt_python_c<k>/`;
* the Euclidean vs Mahalanobis figure (`10_mahal_forced_rescued/c<k>_euclid_vs_mahal.png`): re-runs the second
  pass's re-match and rescue with python's own `force_membership` from the saved cores and checks it
  reproduces the saved labels -- script `mahal_exact.py` (not in this folder yet).

The deck: `~/.venvs/pptdeck/bin/python $VP/build_channel_deck.py "<channel>/for ppt"` (or `.../for ppt/GT c<g>`).

## 8. Many units

Same calls, 2 MATLABs at a time, each `matlab -singleCompThread -batch "<calls>; disp('UNIT DONE')" < /dev/null`
started with `setsid nohup ... &`; one log per unit, `UNIT DONE` = finished. Test one unit and look at it
before starting the rest. Report folders being replaced go to `<channel>/archive_<date>/` (moved, never deleted).

## 9. Where things are

| step | code |
|---|---|
| python run | `$MWS/test_full_pipeline.py` (second pass: `$MWS/recluster_test/second_pass.py`, `run.py`) |
| conversion + grapes | `convert_recluster_runs.m` -> `real_response_analysis/build_python_arm_from_times.m` |
| sessions | `validation_sessions.m` |
| validation + report | `ppt_session.m`, `build_ppt_report.m`, `venn_wave_clus.m`, `run_validation.m`, `python_cluster_validation.m`, `ppt_python_subsets.m`, `ppt_gt_subsets.m` |
| added sections | `ppt_wc_py_destinations.m`, `ppt_mahal_forced_rescued.m`, `ppt_not_rescued_by_mask.m` (+ `ppt_quarantine_pages.m`), `ppt_compare_classes.m` |
| checks | `ppt_check_unit.m`, `ppt_flags.m` |
| report layout, windows, the check | `README_for_ppt.md` sections 4-6 |
