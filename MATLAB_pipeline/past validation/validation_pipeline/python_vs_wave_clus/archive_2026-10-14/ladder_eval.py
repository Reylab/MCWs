"""Loosening ladder: spike metrics per GT unit for python as run -> width -> mild ->
loose (trial_replay.py, no log-det), the unit's wave_clus run and GT itself.
Same measures as trial_eval.py (recovered, purity, F1, ISI < 3 ms, SNR, and d',
L-ratio, isolation distance in 10 PCs of that sort's own waveforms).
-> <out_dir>/trials/ladder_metrics.csv
"""
import csv, sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, MWS
from pvw_scorecard import gt_index, gt_times
sys.path.insert(0, str(MWS))
from mcws.evaluation.spike_metrics import snr, mahalanobis_metrics, d_prime_lda
from mcws.io.spike_io import load_spikes

LADDER = ["as_run", "width", "mild", "within"]          # within = loose (prom 1.5, width 25, low amp 1 %)
rng = np.random.default_rng(0)


def metrics(W, lab, t, c, gt):
    m = lab == c
    L = lab[gt_index(t, gt)]
    isi = np.diff(np.sort(t[m]))
    keep = np.flatnonzero(lab >= 0)
    if keep.size > 40000:
        keep = np.union1d(rng.choice(keep, 40000, replace=False), np.flatnonzero(m)[:20000])
    F = PCA(10, random_state=0).fit_transform(W[keep]); lk = lab[keep]
    try:
        iso, lr = mahalanobis_metrics(F, lk, c)
    except Exception:
        iso, lr = np.nan, np.nan
    try:
        dp = d_prime_lda(F, lk, c)
    except Exception:
        dp = np.nan
    rec, pur = float(np.sum(L == c) / len(gt)), float(np.sum(L == c) / max(m.sum(), 1))
    return dict(n=int(m.sum()), recovered=rec, purity=pur, f1=2 * rec * pur / max(rec + pur, 1e-9),
                isi_pct=float(np.mean(isi < 3) * 100) if isi.size else 0.0, snr=float(snr(W[m])),
                iso_dist=float(iso), l_ratio=float(lr), d_prime=float(dp))


rows, cache = [], {}
for u in CFG["units"]:
    if u.get("python_class") or u.get("wc_class"):
        continue
    ch = u["ch"]; gt = gt_times(u); d = OUT / "trials" / f"ch{ch}"
    if ch not in cache:
        x = load_spikes(u["second_pass_input"]); x = x[next(iter(x))]
        cache[ch] = np.asarray(x["spikes"], float), x["cluster_class"][:, 1]
    W, t = cache[ch]
    for tr in LADDER:
        f = d / f"labels_{tr}.npy"
        if not f.exists():
            continue
        lab = np.load(f); L = lab[gt_index(t, gt)]; c = int(np.bincount(L[L > 0]).argmax())
        rows.append(dict(unit=u["id"], sort=f"python {tr}", cluster=c, **metrics(W, lab, t, c, gt)))
    w = u["wave_clus"][0]; m = sio.loadmat(OUT / "wc" / f"{w}_ch{ch}.mat")
    wl = m["cluster_class"][:, 0].astype(int); wt = m["cluster_class"][:, 1]
    L = wl[gt_index(wt, gt)]; c = int(np.bincount(L[L > 0]).argmax())
    rows.append(dict(unit=u["id"], sort=w, cluster=c, **metrics(m["spikes"].astype(float), wl, wt, c, gt)))
    g = sio.loadmat(OUT / "gt" / f"gt_{ch}.mat")
    rows.append(dict(unit=u["id"], sort="GT", cluster=u["gt_class"],
                     **metrics(g["spikes"].astype(float), g["cluster_class"][:, 0].astype(int), g["cluster_class"][:, 1], u["gt_class"], gt)))
    print(f"{u['id']} done", flush=True)
with open(OUT / "trials" / "ladder_metrics.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("->", OUT / "trials" / "ladder_metrics.csv")
