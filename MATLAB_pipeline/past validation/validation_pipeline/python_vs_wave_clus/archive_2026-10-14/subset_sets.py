"""Spike sets for the end of the compact report -- wave_clus vs python only, no GT
(subset_sets.m draws their rasters, in the wave_clus vs python validation):

  python_only     python's matched-cluster spikes not in wave_clus's matched cluster (python's windows)
  gtpy_gt_only / gtpy_python_only   addition: GT vs python exclusive sets (in that validation)
  wave_clus_only  wave_clus's matched-cluster spikes not in python's matched cluster (wave_clus's windows)
  python_c<N>     another python class holding >= OTHER_MIN of wave_clus's matched cluster (whole class)
  wave_clus_c<N>  another wave_clus class holding >= OTHER_MIN of python's matched cluster (whole class)
The matched clusters are the ones the validations compare (each sort's cluster with the most GT
spikes).  Matching: 0.5 ms.
-> <out_dir>/compact/sets/<unit>/<set>.mat (times, ms) + manifest.json
"""
import json, sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, SC
from compact_report import sorts_for          # matched clusters, same rule as the validations
from pvw_scorecard import gt_index

OTHER_MIN = 0.05


def matched(t_from, t_to):
    """bool over t_from: has a spike in t_to within 0.5 ms"""
    hit = np.zeros(len(t_from), bool)
    if len(t_to) and len(t_from):
        hit[gt_index(t_from, np.sort(t_to))] = True
    return hit


def shares(lab, t, ref_t, exclude):
    """{class: share of ref_t spikes that sit in that class} for classes != exclude, != 0"""
    j = gt_index(t, np.sort(ref_t)); L = lab[j]
    return {int(c): float(np.sum(L == c) / max(len(ref_t), 1)) for c in np.unique(L) if c not in (0, exclude)}


for u in CFG["units"]:
    if u.get("python_class") or u.get("wc_class"):
        continue
    S, gt = sorts_for(u)
    root = SC / u["folder"]; tok = u["tok"]; w = u["wave_clus"][0]
    gtpy = root / f"validation_gt_vs_{u['python_arm']}{tok}"
    wcpy = root / w / f"validation_{w}_vs_python_sd1{tok}"
    _, pl, pt, pc = S["python"]; _, wl, wt, wc = S["wave_clus"]
    P, Wt = np.sort(pt[pl == pc]), np.sort(wt[wl == wc])
    out = OUT / "compact" / "sets" / u["id"]; out.mkdir(parents=True, exist_ok=True)
    gt = np.sort(gt)
    sets = [("python_only", P[~matched(P, Wt)], wcpy, "arm", f"python c{pc} spikes not in wave_clus c{wc}"),
            ("wave_clus_only", Wt[~matched(Wt, P)], wcpy, "gt", f"wave_clus c{wc} spikes not in python c{pc}"),
            # addition: GT vs python, drawn in that validation with its own windows
            ("gtpy_gt_only", gt[~matched(gt, P)], gtpy, "gt", f"GT class {u['gt_class']} spikes not in python c{pc}"),
            ("gtpy_python_only", P[~matched(P, gt)], gtpy, "arm", f"python c{pc} spikes not in GT class {u['gt_class']}")]
    other = []
    for c, sh in sorted(shares(pl, pt, Wt, pc).items()):
        if sh >= OTHER_MIN:
            other.append(dict(sort="python", cls=c, share=sh, n=int(np.sum(pl == c))))
            sets.append((f"python_c{c}", np.sort(pt[pl == c]), wcpy, "arm", f"python c{c}: {100*sh:.0f}% of wave_clus c{wc}"))
    for c, sh in sorted(shares(wl, wt, P, wc).items()):
        if sh >= OTHER_MIN:
            other.append(dict(sort="wave_clus", cls=c, share=sh, n=int(np.sum(wl == c))))
            sets.append((f"wave_clus_c{c}", np.sort(wt[wl == c]), wcpy, "gt", f"wave_clus c{c}: {100*sh:.0f}% of python c{pc}"))
    man = []
    for name, times, vdir, side, desc in sets:
        f = out / f"{name}.mat"
        sio.savemat(f, {"times": np.asarray(times, float).reshape(-1, 1)})
        man.append(dict(name=name, file=str(f), dir=str(vdir), side=side, desc=desc, n=int(len(times))))
    (out / "manifest.json").write_text(json.dumps(dict(unit=u["id"], python_class=pc, wc_class=wc, sets=man, other=other), indent=1))
    print(f"{u['id']}: " + ", ".join(f"{m['name']} {m['n']:,}" for m in man), flush=True)
