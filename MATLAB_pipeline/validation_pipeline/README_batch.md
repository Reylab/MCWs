# Running many channels

`batch/run_batch.sh` calls `mcws_validation` (see `README_mcws_validation.md`) once per channel, N at a time.
It does nothing else: run the python sorter beforehand for the channels that need a new run.

## Job file

One channel per line, fields separated by `|`, `#` lines ignored (example: `batch/jobs_example.txt`):

```
<channel folder>|<python run folder to convert, or - >|<gt_class>|<stims>
/data/single channel/pt 6 emu 13 ch 264|/mnt/data0/recluster_runs/runs_isi3/p06_EMU-013_ch264|3|[615 44 842]
/data/single channel/pt 6 emu 13 ch 334|-|[3 4]|{[133 418 404], [736 839 744]}
```

`gt_class` and `stims` are MATLAB expressions, as in `mcws_validation`; `-` = use the arms already there.

## Run

```bash
cd <where the logs should go>
TAG=isi3 setsid nohup <repo>/MATLAB_pipeline/validation_pipeline/batch/run_batch.sh jobs.txt 4 > batch.out 2>&1 &
```

| setting | |
|---|---|
| 2nd argument | channels at a time (default 2) |
| `TAG` | the run tag passed to `mcws_validation` (arms in `run <TAG>/`, report in `for ppt <TAG>/`) |
| `PYTHON` | set to a python with MicroWireSorter's packages to also run `mcws_validation`'s section 5 (figures, deck) |
| `DECK_PYTHON` | a python with python-pptx for the deck (default: `PYTHON`) |
| `LOGS` | log folder (default `./batch_logs`), one MATLAB log per channel |

`batch.out`: one line per GT unit (`<channel>: [mcws_validation] GT c3: PASS`) or `FAILED, see <log>`;
`BATCH DONE` at the end. MATLAB runs with `-singleCompThread` at low priority (`nice`).
