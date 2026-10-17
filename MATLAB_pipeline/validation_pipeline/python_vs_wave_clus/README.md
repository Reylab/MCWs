# python vs wave_clus analysis

Why do the wave_clus runs look better than the python (MicroWireSorter) sorts, and
what in python should change? One flow for every channel. Read-only on every sort:
nothing in either sorter is changed.

## Start here

`<out_dir>/overview.pdf` (out_dir = `single channel/analysis_python_vs_wave_clus/`),
organised by question, every channel side by side:

| Section | Question |
|---|---|
| 1 | Headline per GT unit: python before / after the second pass, + log-det replay, best fair wave_clus |
| 2 | Where does python lose GT spikes? |
| 3 | Is the within-channel rejection too tight? (counts + rejected waveforms) |
| 4 | Mahalanobis and the log-determinant (replay, net moves, covariance reliability) |
| 5 | The second pass (decisions, GT spikes before -> after) |
| 6 | wave_clus vs python: which comparisons are fair, rasters of the fair pairs |
| 7 | Rasters (GT, python, fair wave_clus runs) |
| 8 | Spike times and parameters |

Then `<out_dir>/appendix_ch<ch>.pdf`: every number and figure for one channel.

## Rules used everywhere

- Each side is the cluster **closest to the GT unit** (0.5 ms spike matching): python's
  GT cluster, each wave_clus run's GT cluster. Never a cluster matched through another sort.
- In wave_clus vs python, **python sd1 is always the arm** (its class fixed to python's GT
  cluster) so its quarantine / rescue / Mahalanobis breakdown is analysed; wave_clus is the
  reference.
- One wave_clus run per GT unit (channels.json): the run most similar to both GT and
  python's GT cluster.
- A wave_clus run is a **fair** comparison only when its GT cluster and python's GT
  cluster overlap >= `fair_min` (0.8) both ways; others are listed as low match.
- Z-scores are not compared between sorts (a near-zero median baseline inflates them).

## Add / change channels

Edit `channels.json` (one entry per GT unit: session id from `validation_sessions.m`,
folder, channel, GT class, times file, recluster run folder, wave_clus run folders),
then run `./run_analysis.sh`. Finished validations are skipped.

## Steps (`run_analysis.sh`)

| Step | Code | Writes |
|---|---|---|
| 1 GT + wave_clus export | `pvw_matlab('gt_export')`, `pvw_matlab('wc_export')` | `<out_dir>/gt/gt_<ch>.mat`, `<out_dir>/wc/<run>_ch<ch>.mat` |
| 2 validations | `pvw_matlab('validations')` | see "Where the files are" |
| 3 diagnostics | `pvw_matlab('diagnostics')` -> `python_loss_figures.m` | `.../diagnostics/` |
| 4 log-det replay, wave_clus forcing | `logdet_replay.py`, `logdet_fig.py`, `covariance_check.py`, `wc_forced_fig.py` | `<out_dir>/ld/ch<ch>/`, `covariance_check.csv`, `<out_dir>/wc/wc_forced_*.png` |
| 5 tables + report | `pvw_matlab('tables')`, `build_pack.py`, `build_overview.py` | `<out_dir>/*.csv`, `appendix_ch<ch>.pdf`, `overview.pdf` |

`./run_analysis.sh report` redoes only step 5.

## Where the files are

`<out_dir>/where_everything_is.txt` (from `where.py`, also run at the end of
`run_analysis.sh`) lists every run per GT unit with the time its output was written.

Channel folder = `single channel/<folder>/`; `[_classK]` only for channels with two GT units.

| What | Where |
|---|---|
| GT | `<channel>/hand selected/` |
| newest python sort (converted) | `<channel>/python_recluster_20261012_second_pass/` |
| python before the second pass | `<channel>/python_recluster_20261007_rescued/` |
| wave_clus runs (+ grapes_blanks.mat) | `<channel>/wave_clusN/` |
| GT vs newest python validation | `<channel>/validation_gt_vs_python_recluster_20261012_second_pass[_classK]/` |
| diagnostic figures + CSV | ...that folder`/diagnostics/` |
| GT vs wave_clusN validation | `<channel>/wave_clusN/validation_gt_vs_wave_clusN[_classK]/` |
| wave_clusN vs python validation (python = arm) | `<channel>/wave_clusN/validation_wave_clusN_vs_python_sd1[_classK]/` |
| python second-pass outputs | `/mnt/data0/recluster_runs/<run>/10_12_reclust_test/second_pass_<ts>/<label>_<ch>/` (before_after.pdf, second_pass_report.pdf, decisions.json, times_<ts>/) |
| log-det replay | `<out_dir>/ld/ch<ch>/` (figure, json, labels per variant) |
| tables | `<out_dir>/wc_python_match.csv`, `raster_metrics.csv`, `covariance_check.csv`, `par_compare.csv` |
| logs | `<out_dir>/logs/` |

## Archived (2026-10-14)

`archive_2026-10-14/` holds the analysis code from the tuning / feature / compact-report work
(no longer part of the flow; the tuning outputs were deleted, the compact-report outputs are still on disk: `report_<unit>.pdf` in channel folders, `<out_dir>/compact/`, `subset_rasters/` in the validation folders, `<out_dir>/trials/ladder.*`): `channels_tuned.json`, `trial_eval.py`,
`patch_tuned_qc_params.m`, `ladder_eval.py`, `ladder_fig.py`, `raster_export.m`,
`raster_compare.py`, `compact_report.py`, `subset_sets.py/.m`, and `scratch/` (one-off scripts).
