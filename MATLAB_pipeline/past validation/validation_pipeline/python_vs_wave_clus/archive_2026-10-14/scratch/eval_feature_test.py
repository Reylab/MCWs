"""Score the KS vs GMM test sorts against GT (cluster level, no rasters yet)."""
import glob, sys
from pathlib import Path
import numpy as np, scipy.io as sio
sys.path.insert(0, "/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline/python_vs_wave_clus")
from pvw_common import CFG
from pvw_scorecard import gt_times, gt_index

RUNS = {"p11_EMU-017_ch328": 328, "p12_EMU-018_ch300_faces": 300}


def load(d):
    f = sorted(glob.glob(f"{d}/times_*.mat"))[0]
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False,
                    variable_names=["cluster_class", "forced", "rescue_mask", "ipermut", "features", "clustering_metadata"])
    lab = m["cluster_class"][:, 0].astype(int); t = m["cluster_class"][:, 1]
    fz = np.asarray(m.get("forced", np.zeros(len(lab)))).astype(bool).ravel()
    rs = np.asarray(m.get("rescue_mask", np.zeros(len(lab)))).astype(bool).ravel()
    return lab, t, fz, rs, np.shape(m.get("features", np.zeros((0, 0))))


def score(lab, t, fz, rs, gt, other_gt=None):
    j = gt_index(t, gt); L = lab[j]
    if not np.any(L > 0):
        return None
    c = int(np.bincount(L[L > 0]).argmax()); m = lab == c
    isgt = np.zeros(len(t), bool); isgt[j] = True
    rec, pur = np.mean(L == c), isgt[m].mean()
    isi = np.diff(np.sort(t[m]))
    r = dict(c=c, n=int(m.sum()), rec=rec, pur=pur, f1=2 * rec * pur / (rec + pur), isi=100 * np.mean(isi < 3),
             spc=100 * np.mean(~fz[m] & ~rs[m]), nclu=int(lab.max()))
    if other_gt is not None:
        j2 = gt_index(t, other_gt); L2 = lab[j2]
        r["other_gt_in_same"] = np.mean(L2 == c)
    return r


units = [u for u in CFG["units"] if u["ch"] in RUNS.values() and not u.get("python_class") and not u.get("wc_class")]
for run, ch in RUNS.items():
    base = Path("/mnt/data0/recluster_runs") / run / "gmm_feature_test"
    U = [u for u in units if u["ch"] == ch]
    G = {u["id"]: gt_times(u) for u in U}
    for v in ("ks", "gmm_seed0", "gmm_seed1"):
        d = base / v
        if not d.exists():
            continue
        feat = (d / "features.txt").read_text().splitlines()[0] if (d / "features.txt").exists() else "?"
        stages = sorted(glob.glob(f"{d}/times_*"))
        sp = sorted(glob.glob(f"{d}/second_pass_*/*/times_*"))
        print(f"\nch{ch} {v}: {feat}")
        for name, sd in (("first pass (rescued)", stages[-1] if stages else None), ("second pass", sp[-1] if sp else None)):
            if not sd:
                continue
            lab, t, fz, rs, fs = load(sd)
            for u in U:
                other = [G[k] for k in G if k != u["id"]]
                r = score(lab, t, fz, rs, G[u["id"]], other[0] if other else None)
                if r is None:
                    print(f"   {name:21s} {u['id']}: no GT cluster"); continue
                extra = f"  other GT unit in same cluster {100*r['other_gt_in_same']:.0f}%" if "other_gt_in_same" in r else ""
                print(f"   {name:21s} {u['id']}: c{r['c']} of {r['nclu']} clusters, n={r['n']:6,}  rec {100*r['rec']:5.1f}  pur {100*r['pur']:5.1f}  "
                      f"F1 {r['f1']:.3f}  ISI {r['isi']:.2f}%  built by SPC {r['spc']:3.0f}%{extra}")
