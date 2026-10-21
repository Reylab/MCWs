# mcws_validation.m -- the whole validation of one channel, in one file

For one channel folder, in order (the sections of the file):

1. check the folder has what the validation needs;
2. (optional) convert a python sort into MATLAB arms, with their grapes;
3. check every arm's grapes match its times file;
4. for each GT unit: comparisons with wave_clus and GT, validation, the `for ppt` report, its checks;
5. (optional, only when a python is given) the python figures, the second-pass pages and the slide deck.

Sections 1-4 need MATLAB only. The python sorter itself (`test_full_pipeline.py`) is run before, separately.
Paths (MicroWireSorter, data roots, wave_clus): `mcws_paths.m`, `README.md` section 0.

## Folder layout

```
<channel folder>/
  <name>.NC5                              e.g. 'mRAMY08 raw_264.NC5' (the channel number is read from its name)
  NSx.mat  finalevents.mat  stimulus.mat  experiment_properties_online3.mat
  hand selected/   times_<name>.mat  grapes_blanks.mat              GT
  wave_clus notch/ times_<name>.mat  <name>_spikes.mat  fig2print_*.png     the wave_clus run (GT's notch)
```

plus the python arms (`python_recluster_<date>_clustered / _rescued / _second_pass`, in the channel folder or in
`run <tag>/`), or a python output folder to convert (`'python_run'`).

## Run

```matlab
addpath('<repo>/MATLAB_pipeline/validation_pipeline')
cd('<channel folder>')

mcws_validation(3, [615 44 842])                            % one GT unit, arms already in the folder

mcws_validation([3 4], {[133 418 404], [736 839 744]}, ...    % two GT units, convert a python run first
    'python_run', '/mnt/data0/recluster_runs/runs_isi3/p06_EMU-013_ch334', 'tag', 'isi3')

mcws_validation(3, [615 44 842], 'tag', 'isi3', ...          % + figures, second-pass pages, deck
    'python', '/path/to/MicroWireSorter/.venv/bin/python', 'deck_python', '~/.venvs/pptdeck/bin/python')
```

| argument | default | |
|---|---|---|
| `gt_class` | required | GT class(es) in `hand selected/times_<name>.mat` |
| `stims` | required | stimuli for the raster pages; one vector, or one per GT class in a cell |
| `'folder'` | current folder | the channel folder |
| `'python_run'` | none | python output folder (`test_full_pipeline.py`) to convert first |
| `'tag'` | none | none: arms in the channel folder, report in `for ppt`; `'isi3'`: arms in `run isi3/`, validation in `validation isi3/`, report in `for ppt isi3/` |
| `'python_class'` | none (matched) | compare this python class instead of the one matched to GT, one per GT class; report in `<report>/MCWs c<k>/` |
| `'python'` | none | a python with MicroWireSorter's packages; given = run section 5 |
| `'deck_python'` | `'python'` | a python with python-pptx, for the deck |
| `'rescue_sd'` | 1 | the run's second-pass rescue radius (section 5 figures) |

Returns one row per GT class: report folder, check (true = PASS), matched python class. Several GT classes: each
report is in `<report>/GT c<g>/`. Use a tag for every new python run so earlier arms and reports are kept.

## What each section calls

| section | calls | writes |
|---|---|---|
| 2 | `build_python_arm_from_times` per sort | arm folders: adapted times / spikes, `grapes_blanks.mat`, `grapes_blanks_rescue.mat`, `metrics/`, `second_pass_report/` |
| 3 | `check_arm_grapes` | `[grapes]` lines; stops if any arm does not match |
| 4 | `ppt_session`, `ppt_wc_py_destinations`, `ppt_mahal_forced_rescued`, `ppt_not_rescued_by_mask`, `ppt_check_unit`, `ppt_flags` | the validation folders and the report (`accounting.csv`, `flags.csv`, sections 01-10) |
| 5 | `pdftoppm`, `mahal_exact.py`, `build_channel_deck.py`, `reassign_scores.py` | `01_metrics/python_second_pass/`, `10_mahal_forced_rescued/c<k>_euclid_vs_mahal.png`, the deck, `reassignment_scores/` |

Section 5 runs outside MATLAB (it needs `pdftoppm` and the given pythons on that computer); a failed command there
is a warning, the report from sections 1-4 is kept.

## Comparing a different python class

By default each GT unit is compared with the python class that holds most of its spikes (and wave_clus with the
class matching that python class). To use a python class of your choice instead, give `'python_class'`: the whole
validation and report then use that class, and go in their own `MCWs c<k>/` subfolders (report, validation runs),
so the matched report is kept.

```matlab
mcws_validation(3, [615 44 842], 'tag', 'isi3', 'python_class', 2)        % GT c3 vs python c2 -> for ppt isi3/MCWs c2/
mcws_validation([3 4], {s3, s4}, 'python_class', [2 5])                     % one python class per GT class
```

`ppt_compare_classes(session, multi, k)` does a similar comparison after a matched report, into
`<report>/alt_python_c<k>/` (see `README.md`).
