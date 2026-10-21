# MCWs vs wave_clus vs GT validation

Compares a MicroWireSorter (MCWs, python) sort of one channel with a wave_clus sort of the same channel and with
the hand-selected ground truth (GT), and builds a report folder (`for ppt`) and a slide deck from it.

* One channel at a time, everything in one call: `mcws_validation.m` (see `README_mcws_validation.md`).
* Many channels: `batch/run_batch.sh` (see `README_batch.md`).
* This file: the functions behind it, what they need and what they write, in the order they run.

Add the code to the MATLAB path first:

```matlab
addpath('<repo>/MATLAB_pipeline/validation_pipeline', '<repo>/MATLAB_pipeline/real_response_analysis', '<repo>/MATLAB_pipeline/responses')
```

## 0. Paths (set these once on a new machine)

Every folder the code uses comes from `mcws_paths.m`. Each has a lab-server default and can be changed with an
environment variable, set in the shell before starting MATLAB / python, or in MATLAB with `setenv` (MATLAB's
`system` calls pass it on to the python and shell scripts):

| variable | what | default |
|---|---|---|
| `MCWS_DATA_ROOT` | data root | `/mnt/data0` |
| `MCWS_SINGLE_CHANNEL` | one folder per channel (the session table in `validation_sessions.m` is relative to it) | `<data>/MCWs Sessions/matlab validation/single channel` |
| `MCWS_SEEG_DATA` | sEEG recordings, read only (GT spikes files for `run_wave_clus_gt_notch`) | `<data>/sEEG_DATA` |
| `MCWS_ROOT` | the MicroWireSorter repository (MATLAB, python and shell scripts) | `/home/ajb/Documents/GitHub/MicroWireSorter` |
| `WAVE_CLUS_ROOT` | wave_clus (wave_clus-master), for `run_wave_clus_gt_notch` | `/home/ajb/Documents/wave_clus-master` |

```bash
export MCWS_DATA_ROOT=/data MCWS_ROOT=$HOME/MicroWireSorter WAVE_CLUS_ROOT=$HOME/wave_clus-master
```
```matlab
setenv('MCWS_SINGLE_CHANNEL', '/data/my channels');  P = mcws_paths()   % check what is used
```
`mcws_validation` and the batch scripts work on any channel folder you give them; the defaults above only matter for
the session table, `run_wave_clus_gt_notch`, and where MicroWireSorter / wave_clus are installed. The deck needs a
python with python-pptx (`mcws_validation`'s `'deck_python'`, default `~/.venvs/pptdeck`).

---

## 1. Files a channel folder needs

```
<channel folder>/
  <name>.NC5                           the channel's raw data (only its name is read: <name> = e.g. 'mRAMY08 raw_264')
  NSx.mat  finalevents.mat  stimulus.mat  experiment_properties_online3.mat     session files
  hand selected/  times_<name>.mat     GT: the hand-selected sort (cluster_class)
                  grapes_blanks.mat    GT's grapes (its exp_type is reused for every other sort)
  wave_clus notch/  times_<name>.mat   the wave_clus run compared against (run with GT's notch)
                    <name>_spikes.mat  (its par.process_info = the notch used)
                    fig2print_*.png    (copied into the report)
  python arm(s)                        see 2: one folder per python sort, python_recluster_<date>_<kind>
```

`finalevents.mat` must have the blank onsets (`seq_beg_blanks_cell`), or no grapes can be built.

## 2. Python sort -> MATLAB arm

| function | input | output |
|---|---|---|
| `build_python_arm_from_times('root', C, 'channel', ch, 'exp_type', e, 'arm_name', A, 'python_times_file', f)` (`real_response_analysis/`) | a python `times_<name>.mat` | `C/A/`: `times_<name>.mat`, `<name>_spikes.mat` adapted to MATLAB, `grapes_blanks.mat`, `grapes_blanks_rescue.mat` |
| `convert_recluster_runs(runs_root, sc_root, ...)` | python run folders named `pNN_EMU-EEE_chCCC` under `runs_root` | the three arms of each run in the matching channel folder (`'arm_subdir'` = a subfolder), a row each in `sc_root/recluster_conversion_log.csv` |
| `check_arm_grapes(C, arm_subdir, ch)` | the arms | `[grapes]` lines; true when every class's spike count in times = grapes |

A python run (`test_full_pipeline.py`) gives three sorts, converted to three arms:
`times_<ts1>/` (clustering) -> `python_recluster_<date>_clustered`,
`times_<ts2>/` (+ rescue) -> `..._rescued`,
`second_pass_<ts>/<label>_<ch>/times_<ts>/` -> `..._second_pass` (the one compared). Copy each sort's `metrics/`
into its arm, and the second pass's `second_pass_report.pdf`, `before_after.pdf`, `decisions.json` into
`<arm>/second_pass_report/` with a `SOURCE.txt` holding the python folder it came from.

## 3. The session

Every step below takes a session id; `validation_sessions(id)` turns it into the cfg (channel folder, ch, stimuli,
GT times file + class, tolerances).

| call | use |
|---|---|
| `validation_sessions()` | list the ids in the table |
| `validation_sessions('add', id, C, ch, stims, 'times_<name>.mat', g)` | add a session for this MATLAB session (absolute folder C) |
| id `'<id>@<tag>'` | the same session with a tagged python run (folders: `ppt_paths`) |
| `ppt_paths(cfg, multi)` | where the python arm, the validation runs and the report are |

| tag | python arm | validation runs | report |
|---|---|---|---|
| none | newest `python_recluster_*_second_pass` in C | `C` (GT side), `C/wave_clus notch` (wave_clus side) | `C/for ppt` |
| starts with a letter, e.g. `isi3` | `C/run isi3/` | `C/validation isi3/` (+ `wave_clus notch/`) | `C/for ppt isi3` |
| starts with a digit, e.g. `10_19` | `C/python second pass 10_19/` | `C/10_19 validation/` | `C/10_19 for ppt` |

`multi` = true when the channel has more than one GT unit: the report then goes in `<report>/GT c<g>/`.
The stimulus list (`stims`) chooses the raster pages; GT class `g` is the unit compared.

## 4. Comparisons + validation + report (per GT unit, in this order)

| step | function | what it does | writes |
|---|---|---|---|
| a | `ppt_session(id, multi)` | wave_clus grapes if missing (`build_wave_clus_arm`); class matches + venns of GT vs wave_clus, GT vs python, wave_clus vs python (`venn_wave_clus` -> `run_validation`, `python_cluster_validation`); the report (`build_ppt_report`, `ppt_python_subsets`, `ppt_gt_subsets`) | venn / validation folders (3); report sections 01-09 + `membership_ch<ch>.mat` |
| b | `ppt_wc_py_destinations(id, multi)` | where every wave_clus-vs-python region's spikes are in the other sort | `wave_clus notch/<region>/by_destination/` |
| c | `ppt_mahal_forced_rescued(id, multi)` | the matched python class: clustered / forced / rescued, spikes moved by the Mahalanobis step | `10_mahal_forced_rescued/` |
| d | `ppt_not_rescued_by_mask(id, multi)` | quarantined spikes never rescued, per quarantine mask, + distance to templates and the value that quarantined each spike (`ppt_quarantine_pages`) | `<region>/not_rescued_by_mask/` (4 regions) |
| e | `ppt_check_unit(id, multi)` | every venn region rebuilt and counted; regions add up to each class; in-window counts add up to the raster pages | `accounting.csv`; returns true = PASS |
| f | `ppt_flags(id, multi)` | findings worth a look (large regions, response-rich / -poor regions, ...) | `flags.csv` |
| opt | `ppt_compare_classes(id, multi, k)` | the whole comparison with python class k instead of the matched one | `alt_python_c<k>/` |

Response windows: every raster uses the full-run windows of the sort its spikes come from (stimulus latency - 30 ms
to latency + duration + 30 ms); stimuli with latency > 400 ms or duration > 250 ms also get a default-window page
(50-550 ms), in `*_default_window` folders.

## 5. Figures (python) and deck

| script | input | output |
|---|---|---|
| `mahal_exact.py "<C>" <python class> "<report>" ["<arm subfolder>"]` | the second-pass arm (its `second_pass_report/SOURCE.txt`) | `<report>/10_mahal_forced_rescued/c<k>_euclid_vs_mahal.png`: Euclidean vs Mahalanobis choice of every moved spike (re-runs python's matching; prints the % of labels it reproduces) |
| `reassign_scores.py "<C>" "<arm subfolder>" "<out dir>"` | same | distances, score, bias and cluster covariances of the re-match and rescue steps + `clusters.csv` |
| `opposite_peak_check.py "<python times_ folder>" "<out dir>"` | a python sort | per cluster: side ratios of the opposite-peak rule, second troughs |
| `build_channel_deck.py "<report>"` (python-pptx) | a report folder | `<report>/<channel>[ <tag>] wc_mcws_auto.pptx` + `.odp` |

Python scripts use MicroWireSorter's environment (`MCWS_ROOT`, default `/home/ajb/Documents/GitHub/MicroWireSorter`);
`MCWS_RESCUE_SD` = the run's second-pass rescue radius (default 1). The second-pass report pages in the report:
`pdftoppm -r 110 -png <arm>/second_pass_report/second_pass_report.pdf <report>/01_metrics/python_second_pass/second_pass_report`.

## 6. Report layout

```
<report>/
  01_metrics/{gt, python, python_second_pass}   02_venns/gt_vs_python   06_all_classes/
  09_gt_vs_python/{01_gt_only .. 06_quarantine_only}/   10_mahal_forced_rescued/
  wave_clus notch/{01_metrics, 02_venns, 03_wave_clus_only, 03b_python_only, 03c_rescued_only, 03d_quarantine_only,
                   04_subsets, 05_wave_clus_x_quarantine, 06_all_classes, 08_rasters_wave_clus_vs_python, 09_gt_vs_wave_clus}
  accounting.csv  flags.csv  membership_ch<ch>.mat  <deck>.pptx/.odp  reassignment_scores/ (channel level)
```

## 7. Running the python sorter

```bash
cd /home/ajb/Documents/GitHub/MicroWireSorter
.venv/bin/python test_full_pipeline.py "<recording folder>" "<output folder>" <ch>
# with settings changed for this run only (nothing in MicroWireSorter is edited):
.venv/bin/python <repo>/MATLAB_pipeline/validation_pipeline/run_mcws_with.py \
    artifact_rejection.opposite_peak_ratio=0.4 second_pass:ISI_THR=3 -- "<recording folder>" "<output folder>" <ch>
```
The recording folder must give GT's notch: python reads `<recording>/preprocessing_info/<stem>.notches.pkl` first,
then `pre_processing_info.mat` (in `preprocessing_info/`, then the folder itself); give a channel number so nothing
is written into the recording folder.
