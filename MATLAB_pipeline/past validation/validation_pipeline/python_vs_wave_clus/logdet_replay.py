"""Replay the second pass's act step (apply_plan: template re-match + rescue,
both with the Mahalanobis post-step) three ways, from the same input sort and
seed as the 10_12 run, WITHOUT changing any code (variants are monkeypatched
in memory):

  as_run     Mahalanobis without the log-determinant (current code)
  logdet     Mahalanobis + log|Sigma_k|  (Gaussian log-likelihood comparison)
  euclidean  no Mahalanobis post-step (pure two-pass Euclidean)

usage: logdet_replay.py <ch>      (reads channels.json; writes <out_dir>/ld/ch<ch>/)
"""
import copy, json, sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, MWS, channels, sp_root
ROOT = MWS
sys.path[:0] = [str(ROOT), str(ROOT / "recluster_test")]
import matplotlib
matplotlib.use("Agg")
import second_pass as SP
import mcws.utils.template_matching as TM
from mcws.io.spike_io import load_spikes

ch = int(sys.argv[1]); C = channels()[ch]
run = C["run"]; gt_classes = [u["gt_class"] for u in C["units"]]
gt_file = OUT / "gt" / f"gt_{ch}.mat"
out = OUT / "ld" / f"ch{ch}"
out.mkdir(parents=True, exist_ok=True)
R = Path(CFG["recluster_runs"]) / run
dec_file = sorted(sp_root(C["run"]).glob("second_pass_*/*/decisions.json"))[-1]
dec = json.loads(dec_file.read_text())
source = C["units"][0]["second_pass_input"]   # the pipeline (rescue) sort the second pass ran on
print("input sort:", source, flush=True)
saved = load_spikes(next(dec_file.parent.glob("times_*")))
saved = saved[next(iter(saved))]

orig_feat_mahal = TM._feat_mahal
orig_reassign = TM._mahalanobis_reassign


def feat_mahal_logdet(features_out, means, inv_covs):
    d = orig_feat_mahal(features_out, means, inv_covs)
    for k in range(means.shape[0]):
        sign, logdet_inv = np.linalg.slogdet(inv_covs[k])
        d[:, k] = d[:, k] - logdet_inv          # + log|Sigma_k| = - log|Sigma_k^-1|
    return d


def no_reassign(core_classes, core_features, features_out, labels, allowed):
    return labels


def run_variant(chd, name):
    TM._feat_mahal, TM._mahalanobis_reassign = orig_feat_mahal, orig_reassign
    if name == "logdet":
        TM._feat_mahal = feat_mahal_logdet
    elif name == "euclidean":
        TM._mahalanobis_reassign = no_reassign
    c = copy.deepcopy(chd)
    SP.apply_plan(c)
    TM._feat_mahal, TM._mahalanobis_reassign = orig_feat_mahal, orig_reassign
    return c["plan"]


print(f"[{run}] assessing (same input sort and seed as the 10_12 run) ...", flush=True)
chd = SP.assess_channel(dec["channel"], source, ch)
decisions = {c: chd["rows"][c]["decision"] for c in chd["clusters"]}
t = chd["t"]
G = sio.loadmat(gt_file)["cluster_class"]

res = {"channel": dec["channel"], "decisions": decisions, "variants": {}}
for name in ("as_run", "logdet", "euclidean"):
    p = run_variant(chd, name)
    lab = p["labels"]
    v = {"labels_equal_saved": None, "clusters": {}, "gt": {}, "n_mahal": int(p["mahal"].sum()),
         "n_changed_vs_input": int(np.sum(lab != chd["labels0"]))}
    if name == "as_run":
        v["labels_equal_saved"] = float(np.mean(lab == saved["cluster_class"][:, 0].astype(int)))
    for c in sorted(set(lab.tolist())):
        m = lab == c
        isi = np.diff(np.sort(t[m]))
        v["clusters"][int(c)] = {"n": int(m.sum()), "isi_pct": float(np.mean(isi < 3) * 100) if isi.size else 0.0,
                                 "mahal_in": int(np.sum(p["mahal"] & m))}
    ts = np.sort(t); order = np.argsort(t)
    for g in gt_classes:
        gt = np.sort(G[G[:, 0] == g, 1])
        k = np.clip(np.searchsorted(ts, gt), 1, len(ts) - 1)
        k = np.where(np.abs(ts[k - 1] - gt) < np.abs(ts[k] - gt), k - 1, k)
        det = np.abs(ts[k] - gt) <= 0.5
        j = order[k[det]]
        L = lab[j]
        best = int(np.bincount(L[L > 0]).argmax()) if np.any(L > 0) else 0
        v["gt"][g] = {"best": best, "recall": float(np.sum(L == best) / len(gt)),
                      "purity": float(np.sum(L == best) / max(np.sum(lab == best), 1)),
                      "n_best": int(np.sum(lab == best))}
    res["variants"][name] = v
    print(f"  {name:9s} mahal-moved {v['n_mahal']:6d}  " +
          "  ".join(f"GT{g}: c{x['best']} recall {x['recall']:.3f} purity {x['purity']:.3f}" for g, x in v["gt"].items()) +
          (f"  | as_run == saved sort: {v['labels_equal_saved']:.4f}" if name == "as_run" else ""), flush=True)
    np.save(out / f"labels_{name}.npy", lab)
(out / "logdet_replay.json").write_text(json.dumps(res, indent=2))
np.save(out / "labels_input.npy", chd["labels0"])
np.save(out / "times.npy", t)
