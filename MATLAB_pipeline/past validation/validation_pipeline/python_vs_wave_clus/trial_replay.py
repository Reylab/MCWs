"""Parameter trials, estimated offline and identical on every channel.

For one channel: load the second pass's input sort, recompute the within-channel
waveform rule with trial thresholds (only ever un-flagging spikes the original
rule flagged), then replay the second pass (assess + act, same seed) with or
without the log-determinant in the Mahalanobis step.  Nothing in the sorter is
changed (the rule is re-implemented here; log-det is patched in memory).

Trials (TRIALS below): name -> within-channel thresholds, log-det on/off.
-> <out_dir>/trials/ch<ch>/labels_<trial>.npy, trial_replay.json

usage: trial_replay.py <ch> [trial,trial,...] [trial to save]
"""
import copy, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, MWS, channels
sys.path[:0] = [str(MWS), str(MWS / "recluster_test")]
import matplotlib
matplotlib.use("Agg")
import second_pass as SP
import mcws.utils.template_matching as TM
from mcws.io.spike_io import load_spikes

# current sorter values (config.yaml artifact_rejection): prom_ratio 2.0, width 3-15, low_amp_percentile 5
CURRENT = dict(prom_ratio=2.0, width_lower=3.0, width_upper=15.0, low_amp_percentile=5.0, other_prom_min_ratio=0.01)
LOOSE = dict(CURRENT, prom_ratio=1.5, width_upper=25.0, low_amp_percentile=1.0)
WIDTH = dict(CURRENT, width_upper=25.0)                       # only the width limit
MILD = dict(CURRENT, prom_ratio=1.7, width_upper=25.0)        # width + a milder second-peak cut
TRIALS = {
    "as_run":       dict(within=CURRENT, logdet=False),
    "width":        dict(within=WIDTH,   logdet=False),       # width limit 15 -> 25 only
    "mild":         dict(within=MILD,    logdet=False),       # + second-peak prominence 2.0 -> 1.7
    "logdet":       dict(within=CURRENT, logdet=True),
    "within":       dict(within=LOOSE,   logdet=False),
    "within_logdet": dict(within=LOOSE,  logdet=True),
    "width_logdet": dict(within=WIDTH,   logdet=True),
    "mild_logdet":  dict(within=MILD,    logdet=True),
}
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else list(TRIALS)   # optional: trials to (re)run
SAVE = sys.argv[3] if len(sys.argv) > 3 else None                      # optional: trial to save as a sort


def props(d, n):
    """per-spike waveform measurements stored by the sorter (quarantine_properties)"""
    P = d["quarantine_properties"]
    out = {k: np.full(n, np.nan) for k in ("prominence_ratio", "num_peaks", "other_prominence", "width", "peak_sample_20")}
    for i in range(n):
        s = P[i]
        for k in out:
            v = getattr(s, k, None) if not isinstance(s, dict) else s.get(k)
            if v is not None and np.size(v):
                out[k][i] = float(np.asarray(v).ravel()[0])
    return out


def waveform_rule(p, prm):
    """True = rejected by the sorter's within-channel waveform rule (mcws/pipeline.py, same logic)."""
    pr, npk, op, wd, pk = p["prominence_ratio"], p["num_peaks"], p["other_prominence"], p["width"], p["peak_sample_20"]
    a = np.abs(pk); thr = np.percentile(a[np.isfinite(a)], prm["low_amp_percentile"]) if prm["low_amp_percentile"] > 0 else -np.inf
    low = np.nan_to_num(a < thr, nan=False).astype(bool)
    with np.errstate(invalid="ignore"):
        one = np.nan_to_num(npk == 1, nan=False).astype(bool)
        failed = np.nan_to_num(op < prm["other_prom_min_ratio"] * a, nan=False).astype(bool)
        width_ok = (wd < prm["width_upper"]) & (wd > prm["width_lower"])
        auto = np.isnan(wd) | (np.isnan(pr) & ~one)
        exc = one | failed
        valid_ok = ~(auto | exc) & (pr > prm["prom_ratio"]) & width_ok
        exc_ok = exc & ~np.isnan(wd) & width_ok
    keep = (np.nan_to_num(valid_ok, nan=False).astype(bool) | np.nan_to_num(exc_ok, nan=False).astype(bool)) & ~auto & ~low
    return ~keep


ch = int(sys.argv[1]); C = channels()[ch]; u0 = C["units"][0]
out = OUT / "trials" / f"ch{ch}"; out.mkdir(parents=True, exist_ok=True)
src = load_spikes(u0["second_pass_input"]); chid = next(iter(src)); base = src[chid]
n = len(base["cluster_class"]); P = props(base, n)
q = base["quarantine"]
wch = np.asarray(q["within_channel"]).astype(bool).ravel()
other = np.zeros(n, bool)
for k in ("bundle_collision", "refractory_period"):
    if k in q:
        other |= np.asarray(q[k]).astype(bool).ravel()
old_rule = waveform_rule(P, CURRENT)
orig_feat = TM._feat_mahal


def feat_logdet(features_out, means, inv_covs):
    d = orig_feat(features_out, means, inv_covs)
    for k in range(means.shape[0]):
        d[:, k] -= np.linalg.slogdet(inv_covs[k])[1]
    return d


jf = out / "trial_replay.json"
res = json.loads(jf.read_text()) if jf.exists() else {"channel": ch, "input": u0["second_pass_input"], "trials": {}}
for name, tr in TRIALS.items():
    if name not in ONLY:
        continue
    d = copy.deepcopy(base)
    unflag = wch & old_rule & ~waveform_rule(P, tr["within"])          # only spikes the old rule explained
    qq = d["quarantine"]
    qq["within_channel"] = (wch & ~unflag).astype(np.uint8)
    qq["all"] = (np.asarray(qq["all"]).astype(bool).ravel() & ~(unflag & ~other)).astype(np.uint8)
    TM._feat_mahal = feat_logdet if tr["logdet"] else orig_feat
    chd = SP.assess_channel(f"ch{ch} {name}", {chid: d}, chid)
    SP.apply_plan(chd)
    TM._feat_mahal = orig_feat
    lab = chd["plan"]["labels"]
    prev = out / f"labels_{name}.npy"
    if prev.exists() and not np.array_equal(np.load(prev), lab):
        print(f"[{ch}] {name}: labels differ from the earlier replay", flush=True)
    np.save(prev, lab)
    pl = chd["plan"]   # per-spike provenance: Mahalanobis-moved, template-forced, rescued, SPC core
    np.savez(out / f"plan_{name}.npz", mahal=pl["mahal"], forced=pl["forced"], rescued=pl["rescued"], core=pl["core"])
    if SAVE == name:   # write this trial as a sorter output: times file + metrics report + before_after.pdf
        dest = Path(CFG["recluster_runs"]) / C["run"] / f"tuned_{name}" / f"second_pass_{name}" / SP.channel_folder(chd["d"], chid)
        dest.mkdir(parents=True, exist_ok=True)
        # record the trial's limits in par.qc_params: MATLAB rebuilds the per-check quarantine flags from them
        qc = chd["d"].setdefault("par", {}).setdefault("qc_params", {})
        for k, v in (("min_width_idx", tr["within"]["width_lower"]), ("max_width_idx", tr["within"]["width_upper"]),
                     ("final_prominence_ratio_pass", tr["within"]["prom_ratio"])):
            if isinstance(qc, dict):
                qc[k] = v
            else:
                setattr(qc, k, v)
        TM._feat_mahal = feat_logdet if tr["logdet"] else orig_feat
        SP.write_before_after(chd, dest / "before_after.pdf")
        saved = SP.save_channel(chd, dest)
        TM._feat_mahal = orig_feat
        (dest / "trial_params.json").write_text(json.dumps(dict(trial=name, within=tr["within"], logdet=tr["logdet"],
                                                                unflagged=int(unflag.sum()), input=u0["second_pass_input"]), indent=2))
        print(f"[{ch}] saved {name} -> {saved}", flush=True)
    res["trials"][name] = dict(params=tr["within"], logdet=tr["logdet"], unflagged=int(unflag.sum()),
                               decisions={str(c): chd["rows"][c]["decision"] for c in chd["clusters"]})
    print(f"[{ch}] {name}: un-quarantined {int(unflag.sum()):,}, decisions {res['trials'][name]['decisions']}", flush=True)
np.save(out / "times.npy", base["cluster_class"][:, 1])
(out / "trial_replay.json").write_text(json.dumps(res, indent=2))
