# mcws_validation.m -- one channel, every step

Runs, for one channel folder: (optionally) the conversion of a python run into MATLAB arms, the grapes check, the
comparisons with wave_clus and GT, the validation, the `for ppt` report, the python figures and the slide deck.

Paths (MicroWireSorter, data roots, wave_clus) come from `mcws_paths.m` / environment variables: `README.md`, section 0.

## Folder layout

Put the files in one channel folder (subfolders as named):

```
<channel folder>/
  <name>.NC5                              e.g. 'mRAMY08 raw_264.NC5' (the channel number is read from its name)
  NSx.mat  finalevents.mat  stimulus.mat  experiment_properties_online3.mat
  hand selected/   times_<name>.mat  grapes_blanks.mat          GT
  wave_clus notch/ times_<name>.mat  <name>_spikes.mat  fig2print_*.png     the wave_clus run (GT's notch)
```

and either the python arms already converted (`python_recluster_<date>_clustered/_rescued/_second_pass`, in the
folder or in `run <tag>/`), or a python output folder (`test_full_pipeline.py`) to convert: `'python_run'`.

## Run

```matlab
addpath('<repo>/MATLAB_pipeline/validation_pipeline')
cd('<channel folder>')

% one GT unit, arms already in the folder
mcws_validation('gt_class', 3, 'stims', [615 44 842])

% two GT units, convert a python run first, keep it apart from earlier runs with a tag
mcws_validation('gt_class', [3 4], 'stims', {[133 418 404], [736 839 744]}, ...
                'python_run', '/mnt/data0/recluster_runs/runs_isi3/p06_EMU-013_ch334', 'tag', 'isi3')
```

| option | default | |
|---|---|---|
| `gt_class` | required | GT class(es) to report on |
| `stims` | required | stimuli for the raster pages; one vector, or one per GT class in a cell |
| `folder` | current folder | the channel folder |
| `ch` | from the .NC5 name | channel number |
| `python_run` | `''` | python output folder to convert first |
| `tag` | `''` | `''`: arms in the channel folder, report in `for ppt`; `isi3`: `run isi3/`, `validation isi3/`, `for ppt isi3/` |
| `steps` | all | any of `convert`, `report`, `figures`, `deck` |
| `python`, `deck_python` | MicroWireSorter `.venv`, `~/.venvs/pptdeck` | pythons for the figures / the deck |
| `rescue_sd` | 1 | the run's second-pass rescue radius (for the figures) |

## What it does, in order

1. checks the files above are there;
2. `python_run` given: converts its three sorts into arms (`build_python_arm_from_times`), copies their metrics and
   the second-pass report;
3. checks grapes = times in every arm (`check_arm_grapes`; stops if not);
4. for each GT class: `ppt_session` -> `ppt_wc_py_destinations` -> `ppt_mahal_forced_rescued` ->
   `ppt_not_rescued_by_mask` -> `ppt_check_unit` -> `ppt_flags`, the second-pass report pages, the Euclidean vs
   Mahalanobis figure, the deck;
5. the channel's reassignment figures (`reassignment_scores/`).

Returns one row per GT class: the report folder and its check (true = PASS). With more than one GT class each report
is in `<report>/GT c<g>/`.

## Notes

* Without a tag the report goes in `for ppt/` and the arms are read from the channel folder; use a tag for every new
  python run so earlier reports and arms are kept.
* Nothing in `hand selected/` or `wave_clus notch/` is changed when a tag is used (venns and validation runs go in
  the tag's validation folder); without a tag the wave_clus-side runs are written in `wave_clus notch/`.
* Needs `pdftoppm` for the second-pass pages; the figures and the deck are skipped with a warning if their python
  fails. Function details: `README.md`.
