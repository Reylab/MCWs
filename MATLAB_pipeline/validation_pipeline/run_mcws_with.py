"""Run MicroWireSorter's test_full_pipeline.py with settings changed for this run only (nothing in MicroWireSorter
is edited). Overrides go before '--', the usual test_full_pipeline.py arguments after it. Three kinds:

    <section>.<key>=<value>            a mcws/config.yaml value       artifact_rejection.opposite_peak_ratio=0.4
    <module>:<NAME>=<value>            a module constant              second_pass:ISI_THR=3   second_pass:AUTO_MERGE=false
                                                                      second_pass:RESCUE_SD=[1.0,1.5]
    <module>.<function>:<arg>=<value>  forced keyword of a function   mcws.clustering.rescue.apply_spike_rescue:template_sdnum=1.5
                                       (every call of it in the run gets arg=value)
Values are read as YAML (numbers, true/false, lists).

    cd $MCWS_ROOT                         # the MicroWireSorter repository (default: see mcws_paths.m)
    .venv/bin/python <this repo>/MATLAB_pipeline/validation_pipeline/run_mcws_with.py \
        artifact_rejection.opposite_peak_ratio=0.4 second_pass:ISI_THR=3 -- "<recording dir>" "<output dir>" <ch>

The run records what it used as usual (times file par / qc_params, decisions.json, batch.log); the overrides are
printed first.
"""
import functools, importlib, os, runpy, sys
import yaml

MWS = os.environ.get('MCWS_ROOT', '/home/ajb/Documents/GitHub/MicroWireSorter')   # MicroWireSorter
if '--' not in sys.argv:
    sys.exit(__doc__)
cut = sys.argv.index('--')
overrides, args = sys.argv[1:cut], sys.argv[cut + 1:]
sys.path.insert(0, MWS); sys.path.insert(0, os.path.join(MWS, 'recluster_test'))   # as test_full_pipeline does
import mcws.config as C

cfg = C.load_config()
forced = {}                                         # function -> {arg: value}
for ov in overrides:
    target, value = ov.split('=', 1); value = yaml.safe_load(value)
    if ':' in target:
        left, name = target.split(':', 1)
        try:
            mod = importlib.import_module(left); is_fn = False           # <module>:<NAME>
        except ImportError:
            m, f = left.rsplit('.', 1); is_fn = callable(getattr(importlib.import_module(m), f, None))
            if not is_fn:
                sys.exit(f'run_mcws_with: {left} is neither a module nor a function')
        if is_fn:
            forced.setdefault(left, {})[name] = value       # forced keyword of a function
            print(f'override {left}(..., {name}={value})', flush=True)
        else:
            if not hasattr(mod, name):
                sys.exit(f'run_mcws_with: {left} has no {name}')
            old = getattr(mod, name); setattr(mod, name, value)
            print(f'override {left}.{name}: {old} -> {value}', flush=True)
    else:
        *sections, key = target.split('.')
        node = cfg
        for s in sections:
            node = node[s]
        if key not in node:
            sys.exit(f'run_mcws_with: no key {target} in mcws/config.yaml')
        old = node[key]; node[key] = value
        print(f'override {target}: {old} -> {value}', flush=True)
C._cached_config = cfg

for path, kw in forced.items():                     # wrap the function in its module (callers import it from there)
    modname, fname = path.rsplit('.', 1); mod = importlib.import_module(modname); fn = getattr(mod, fname)
    @functools.wraps(fn)
    def wrapped(*a, _fn=fn, _kw=kw, **k):
        k.update(_kw); return _fn(*a, **k)
    setattr(mod, fname, wrapped)

os.chdir(MWS)
sys.argv = ['test_full_pipeline.py'] + args
runpy.run_path(os.path.join(MWS, 'test_full_pipeline.py'), run_name='__main__')
