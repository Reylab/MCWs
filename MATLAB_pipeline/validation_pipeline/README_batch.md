# Running many channels

`batch/run_batch.sh` runs one job per channel, N at a time. A job = the python sorter on the channel's recording
(optional) and then `mcws_validation` on the channel folder (see `README_mcws_validation.md`).

## Job file

One channel per line, fields separated by `|`, `#` lines ignored (example: `batch/jobs_example.txt`):

```
<channel folder>|<recording folder, or - >|<ch>|<gt_class>|<stims>
/data/single channel/pt 6 emu 13 ch 264|/mnt/data0/sEEG_DATA/MCW-FH_006/EMU/EMU-013_..._run-03|264|3|[615 44 842]
/data/single channel/pt 6 emu 13 ch 334|-|334|[3 4]|{[133 418 404], [736 839 744]}
```

* recording folder: the python sorter runs on it into `$RUNS_ROOT/<channel folder name>`; `-` = no python run, use
  the arms already in the channel folder (or in `run <TAG>/`).
* `gt_class` and `stims` are MATLAB expressions, as in `mcws_validation`.

## Run

```bash
cd <where the logs should go>
TAG=isi3 RUNS_ROOT=/mnt/data0/recluster_runs/runs_isi3 \
OVERRIDES="artifact_rejection.opposite_peak_ratio=0.4 second_pass:ISI_THR=3 second_pass:AUTO_MERGE=false" \
  setsid nohup <repo>/MATLAB_pipeline/validation_pipeline/batch/run_batch.sh jobs.txt 4 > batch.out 2>&1 &
```

| variable | |
|---|---|
| 2nd argument | jobs at a time (default 2) |
| `TAG` | run tag: arms in `run <TAG>/`, report in `for ppt <TAG>/`; leave empty only for arms already in the channel folder |
| `RUNS_ROOT` | where the python outputs go (one folder per channel); an existing one is not re-run |
| `OVERRIDES` | python settings changed for these runs (`run_mcws_with.py` syntax: `section.key=v`, `module:NAME=v`, `module.function:arg=v`) |
| `RESCUE_SD` | the runs' second-pass rescue radius, for the figures (default 1) |
| `MCWS_ROOT` | the MicroWireSorter repository (default and other paths: `README.md`, section 0) |
| `LOGS` | log folder (default `./batch_logs`): `<channel>[_<TAG>].python.txt` and `.matlab.txt` per job |

`batch.out` gets one line per finished step: python OK / FAILED, then `[mcws_validation] done: GT c3 PASS, ...` or
MATLAB FAILED with the error. `BATCH DONE` at the end.

## Notes

* Every job runs at low priority (`nice`), MATLAB with `-singleCompThread`; each python run briefly uses many cores.
* Settings that differ per run (e.g. three rescue radii) = one batch per setting, each with its own `TAG` and
  `RUNS_ROOT`.
* To stop: kill `run_batch.sh` and its `run_one.sh` / `matlab` processes; finished channels stay, and an existing
  python output folder is not re-run.
