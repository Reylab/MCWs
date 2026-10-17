# for ppt reports: wave_clus (GT notch) vs MCWs (python) vs GT

All code is in `MATLAB_pipeline/validation_pipeline/` (plus raster plotting in `real_response_analysis/`).
Step-through script for one unit, with editable subsets: **`ppt_interactive.m`**.

## 1. What each channel needs first

| step | what | how |
|---|---|---|
| a | GT notch = the hand-selected spikes file's `par.process_info` | `<channel>/pre_processing_info.mat` entry for the channel must equal it (old ones archived in `archive_2026-10-16/`) |
| b | python with GT's notch | newest `python_recluster_*_second_pass` (the GT-notch run); converted, grapes = times, metrics copied |
| c | wave_clus with GT's notch | `run_wave_clus_gt_notch('pt 4 emu 7 ch 259')` -> `<channel>/wave_clus notch/` (writes `gt_notch_check.txt`: detection identical to GT) |
| d | session row | `validation_sessions.m`: id, channel folder, ch, stim list, GT times file + class |

wave_clus-master reads `<cwd>/preprocessing/pre_processing_info.mat` **before** `<cwd>/pre_processing_info.mat`;
`run_wave_clus_gt_notch` runs inside the run folder with GT's notch copied in, so no other file is read.

## 2. One unit (everything)

```matlab
addpath('/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline')
ppt_session('pt4_ch259', false)        % true = channel with several GT units -> for ppt/GT c<g>/
ppt_check_unit('pt4_ch259', false)     % PASS / FAIL + for ppt/accounting.csv
```

`ppt_session` runs, in order:
1. `build_wave_clus_arm(root, ch, 'wave_clus notch')` - grapes_blanks.mat for the run (once)
2. `venn_wave_clus(session, 'wave_clus notch', 'gt_vs_python', true)` - GT vs wave_clus, wave_clus vs python, GT vs python venns + class matches
3. `build_ppt_report(o)` - sections 01-08 below
4. `ppt_python_subsets(session, multi)` - 03b / 03c / 03d
5. `ppt_gt_subsets(session, multi)` - 09

## 3. All units (batch)

```
"/mnt/data0/MCWs Sessions/matlab validation/run_all_ppt.sh"
```
2 workers, single-threaded MATLAB. Each unit: old report -> `<channel>/archive_2026-10-17/`, `ppt_session`, `ppt_check_unit`.
Progress / result: `/mnt/data0/MCWs Sessions/matlab validation/ppt_status.txt` (one PASS/FAIL per unit, `ALL DONE` at the end).
Only the notch run is compared. No-notch material is kept out of `for ppt`: `move_no_notch_out.sh` puts the no-notch runs (and any `07_no_notch_vs_notch`) in `<channel>/_no_notch/`.

## 4. Report layout (`<channel>/for ppt/`, or `for ppt/GT c<g>/`)

```
01_metrics/{gt, python}          GT metric pages, python report cluster page
02_venns/gt_vs_python            class matches + venns
06_all_classes/                  every GT and python class, each with ITS OWN class windows
09_gt_vs_python/                 01_gt_only 02_python_only 03_matched 04_gt_x_quarantine(+by_property)
                                 05_rescued_only 06_quarantine_only(+by_property)  counts.csv  summary/
accounting.csv                   the check (section 6)
membership_ch<ch>.mat            every spike of GT / python / wave_clus notch and where it is in the others
wave_clus notch/
  01_metrics                     fig2print
  02_venns/{wave_clus_vs_python, gt_vs_wave_clus}
  03_wave_clus_only              wave_clus class spikes python has neither in its class nor in quarantine
  03b_python_only                python class spikes wave_clus lacks, not quarantined
  03c_rescued_only               python class spikes rescued into the class that wave_clus lacks
  03d_quarantine_only            python quarantined spikes in neither class (+ by quarantine property)
  04_subsets                     python class / wave_clus class / matched
  05_wave_clus_x_quarantine      wave_clus class spikes python quarantined, not in python's class (+ by property)
  06_all_classes                 every wave_clus class, own class windows
  08_rasters_wave_clus_vs_python full run: both raster pages + per-stimulus summary (z-score, sign-test p ...) + sign test
  09_gt_vs_wave_clus             01_gt_only 02_wave_clus_only 03_matched counts.csv summary/
```
Every raster folder `<name>` may have a `<name>_default_window` twin (section 5). Waveform pages: up to 5000
single spikes per panel + mean, with the GT / wave_clus / python class templates drawn on top.

## 5. Response windows (what the green lines are)

* Every subset uses the windows of the **full run of the sort its spikes come from**:
  wave_clus spikes -> the wave_clus class's own windows, python spikes -> python's, GT spikes -> GT's.
  These are the raster tables of the full comparison runs:
  * `wave_clus notch/validation_wave_clus_vs_python_c<X>` (ref = wave_clus class, arm = python class X)
  * `validation_gt_vs_<python>_class<g>` (ref = GT, arm = python)
  * `wave_clus notch/validation_gt_vs_wave_clus_notch_class<g>` (ref = GT, arm = wave_clus)
* Window = latency - 30 ms .. latency + duration + 30 ms (`run_validation`, `cfg.dyn_buffer`).
* Stimuli with latency > 400 ms (`cfg.max_lat`) or duration > 250 ms (`cfg.max_dura`) get a second version with
  grapes' default post window (`[tmin_median, tmax_median]`, 50-550 ms after onset): `*_default_window` folders.
* Every page's footer says which windows it uses; the title gives in / out of window counts and %.

## 6. The check (`ppt_check_unit`)

For wave_clus vs python, GT vs python and GT vs wave_clus: each venn region is rebuilt from the membership and
1. its spike count = the venn's number,
2. the regions of each class add up to the class,
3. each sort's regions' in-window counts (own windows) add up to the in-window count on that sort's full raster page.

`accounting.csv` lists every region (n, venn, in-window) and the two class sums per comparison.
Note: matched spikes are counted in both sorts' windows, so the wave_clus and python in-window totals differ by
exactly (wave_clus only + wave_clus x quarantine) - (python only + rescued only), each in its own windows.

## 7. Your own subsets

`ppt_interactive.m`, sections 4-7. The building blocks:

```matlab
M = load('<for ppt>/membership_ch259.mat');           % M.gt / M.python / M.wc_notch tables, M.waves, M.templates
m = M.python.class == 1 & M.python.q_low_amplitude;    % any logical mask on a table
plot_subset_waveforms(struct('name', 'mine', 'W', M.waves.python(m, :), 'color', [1 0 0]), templates, 'title', 'out.png')
plot_subset_rasters([], 'dir', <full run folder>, 'times', M.python.t(m), 'name', 'mine', 'side', 'arm', 'out_dir', out)
plot_subset_rasters([], ..., 'window', 'default')       % the default-window version
plot_class_rasters(<run folder>, <unit dir>, <grapes file>, <times file>, k, 'name', out)   % a whole class, own windows
```

## 8. Flags and decks

```matlab
ppt_flags('pt4_ch259', false)     % -> for ppt/flags.csv: severity | comparison | finding | likely explanation | figures
```
```bash
~/.venvs/pptdeck/bin/python validation_pipeline/build_channel_deck.py "<channel>/for ppt"        # or ".../for ppt/GT c3"
```
The deck (`<channel> wc_mcws_auto.pptx` + `.odp`, in the report folder) follows the ch259 blueprint
(`pt 4 emu 7 ch 259/wave_clus_mcws_compare.odp`), same positions:
1 title | 2 wave_clus fig2print | 3 python cluster page | 4 wave_clus vs python venn + class matches + region counts |
5 full rasters wave_clus / python (08, own windows) | 6 per-stimulus summary | 7 quarantine set (05) |
8 GT metrics | 9 GT vs python + GT vs wave_clus venns + class matches | 10 python only / wave_clus only vs GT (09) |
11 bookkeeping (accounting.csv) | 12 flags (flags.csv).
Flag rules (`ppt_flags.m` header): region > 2% of its class; region in-window % 1.5x off its class's; one quarantine
property > 50% of wave_clus x quarantine; paired sign test on per-stimulus z / sign-p between the full runs;
late-latency / long-duration stimuli; any accounting FAIL. Extras (other classes, per-quarantine-reason responses,
Mahalanobis shift) are added per channel only where a flag needs them.

All units: `"/mnt/data0/MCWs Sessions/matlab validation/run_all_decks.sh"` (after run_all_ppt.sh; result `deck_status.txt`).
python-pptx environment: `python3 -m venv ~/.venvs/pptdeck && ~/.venvs/pptdeck/bin/pip install python-pptx`.

## 9. Files

| file | role |
|---|---|
| `run_wave_clus_gt_notch.m` | wave_clus with GT's notch in `<channel>/wave_clus notch` |
| `build_wave_clus_arm.m` | grapes_blanks.mat for a wave_clus run |
| `venn_wave_clus.m`, `python_cluster_validation.m`, `run_validation.m` | venns, class matches, full raster runs (windows) |
| `ppt_session.m` | one unit, everything |
| `build_ppt_report.m` | sections 01-08 |
| `ppt_python_subsets.m` | 03b / 03c / 03d |
| `ppt_gt_subsets.m` | 09 |
| `ppt_check_unit.m` | the check |
| `ppt_flags.m` | flags.csv |
| `build_channel_deck.py` | the deck |
| `run_all_decks.sh` | flags + decks for every passed unit |
| `build_sort_membership.m` | the membership tables |
| `plot_subset_waveforms.m`, `draw_wave_group.m`, `plot_traces.m` | waveform pages |
| `plot_subset_rasters.m`, `plot_class_rasters.m` | raster pages (`real_response_analysis/plot_response_window_grapes.m`, `loop_plot_responses_BCM_response_window.m` draw them) |
| `compare_detections.m` | no notch vs notch detections (07) |
| `ppt_interactive.m` | step-through for one unit |
| `run_all_ppt.sh` | the batch (copy in `/mnt/data0/MCWs Sessions/matlab validation/`) |
