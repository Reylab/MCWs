"""Score every trial (trial_replay.py) against GT, python as run and wave_clus,
with measures that do not grow with the number of spikes:

  GT recovered / purity        share of the GT unit in the cluster / share of the cluster that is GT
  ISI < 3 ms (%)               refractory violations of the cluster
  SNR                          sorter's snr() on the cluster's waveforms
  isolation distance, L-ratio  sorter's mahalanobis_metrics(), d' = d_prime_lda(), all in the
                               SAME feature space for every sort: 10 principal components of that
                               sort's own waveforms (so sorts are comparable)
-> <out_dir>/trials/trial_metrics.csv
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

TRIAL_NAMES = ["as_run", "logdet", "within", "within_logdet", "width_logdet", "mild_logdet"]
rng = np.random.default_rng(0)


def metrics(W, lab, t, c, gt):
    """cluster c of one sort; W = that sort's waveforms (rows = lab rows)"""
    m = lab == c
    j = gt_index(t, gt); L = lab[j]
    isi = np.diff(np.sort(t[m]))
    # common feature space: 10 PCs of this sort's waveforms (subsample for speed)
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


rows = []
done_inputs = {}
for u in CFG["units"]:
    if u.get("python_class") or u.get("wc_class"):
        continue                                   # trials are scored on each unit's GT-matched clusters
    ch = u["ch"]; gt = gt_times(u)
    if ch not in done_inputs:
        d = load_spikes(u["second_pass_input"]); d = d[next(iter(d))]
        done_inputs[ch] = (np.asarray(d["spikes"], float), d["cluster_class"][:, 1])
    W, t = done_inputs[ch]
    for tr in TRIAL_NAMES:
        lab = np.load(OUT / "trials" / f"ch{ch}" / f"labels_{tr}.npy")
        L = lab[gt_index(t, gt)]; c = int(np.bincount(L[L > 0]).argmax())
        rows.append(dict(unit=u["id"], sort=f"python {tr}", cluster=c, **metrics(W, lab, t, c, gt)))
    w = u["wave_clus"][0]; m = sio.loadmat(OUT / "wc" / f"{w}_ch{ch}.mat")
    wl = m["cluster_class"][:, 0].astype(int); wt = m["cluster_class"][:, 1]; WW = m["spikes"].astype(float)
    L = wl[gt_index(wt, gt)]; c = int(np.bincount(L[L > 0]).argmax())
    rows.append(dict(unit=u["id"], sort=w, cluster=c, **metrics(WW, wl, wt, c, gt)))
    g = sio.loadmat(OUT / "gt" / f"gt_{ch}.mat")
    gl = g["cluster_class"][:, 0].astype(int); gtt = g["cluster_class"][:, 1]
    rows.append(dict(unit=u["id"], sort="GT", cluster=u["gt_class"], **metrics(g["spikes"].astype(float), gl, gtt, u["gt_class"], gt)))
    print(f"{u['id']} done", flush=True)
with open(OUT / "trials" / "trial_metrics.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for r in rows:
    print(f"{r['unit']:10s} {r['sort']:22s} c{r['cluster']:<3d} n={r['n']:6,}  rec {100*r['recovered']:5.1f}  pur {100*r['purity']:5.1f}  F1 {r['f1']:.3f}  "
          f"ISI {r['isi_pct']:5.2f}%  SNR {r['snr']:5.2f}  isoD {r['iso_dist']:7.1f}  Lr {r['l_ratio']:7.3f}  d' {r['d_prime']:5.2f}")
