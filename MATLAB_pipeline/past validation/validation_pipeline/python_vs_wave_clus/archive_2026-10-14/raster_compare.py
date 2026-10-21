"""Stimulus-response comparison, GT vs python as run vs python tuned vs wave_clus,
with measures that do NOT grow with the number of spikes in the cluster
(input: <out_dir>/raster/per_stim.csv from raster_export.m; PVW_CONFIG=channels_tuned.json).

Per unit and sort, over the unit's stimuli (GT-responsive = GT's p_test & z >= 3):
  tuning r          Pearson r of the per-stimulus evoked rate (post - baseline, Hz) with GT's:
                    does the cluster respond to the SAME stimuli in the same proportions as GT?
  MI                modulation index (post - base) / (post + base) on GT-responsive stimuli:
                    contamination by spikes that don't respond pulls it toward 0
  baseline Hz       pre-stimulus rate (non-GT spikes mostly add baseline)
  AUROC             trial-by-trial separability post vs baseline on GT-responsive stimuli
  latency diff      |first-spike latency - GT's| (ms) on GT-responsive stimuli
  resp agreement    responsive stimuli (p_test & z >= 3) shared with GT / union (Jaccard)
  extra responsive  stimuli called responsive that GT does not respond to (false responses)
  in-window share   spikes in response windows per 1,000 cluster spikes
-> <out_dir>/raster/raster_compare.csv, raster_compare.png, raster_tuning.png
"""
import csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import OUT

SORTS = ["GT", "python as run", "python tuned", "wave_clus"]
COL = {"GT": "k", "python as run": "#9aa5b1", "python tuned": "#1f77b4", "wave_clus": "#d62728"}
R = OUT / "raster"
rows = list(csv.DictReader(open(R / "per_stim.csv")))
fl = lambda v: float(v) if v not in ("", "NaN", "nan") else np.nan
tf = lambda v: str(v).strip().lower() in ("1", "true")

D = {}
for r in rows:
    D.setdefault(r["unit"], {}).setdefault(r["sort"], {})[int(float(r["stim_number"]))] = r
units = list(D)

out, tuning = [], {}
for u in units:
    g = D[u]["GT"]; stims = sorted(g)
    resp = lambda s, sort: tf(D[u][sort][s]["p_test"]) and fl(D[u][sort][s]["zscore"]) >= 3
    gt_resp = [s for s in stims if resp(s, "GT")]
    ev = lambda sort: np.array([fl(D[u][sort][s]["mean_fr_post_hz"]) - fl(D[u][sort][s]["mean_fr_base_hz"]) for s in stims])
    tuning[u] = {sort: ev(sort) for sort in SORTS if sort in D[u]}
    for sort in SORTS:
        if sort not in D[u] or set(D[u][sort]) != set(stims):
            continue
        X = D[u][sort]
        post = np.array([fl(X[s]["mean_fr_post_hz"]) for s in gt_resp]); base = np.array([fl(X[s]["mean_fr_base_hz"]) for s in gt_resp])
        mi = np.nanmedian((post - base) / np.maximum(post + base, 1e-9)) if gt_resp else np.nan
        e, eg = tuning[u][sort], tuning[u]["GT"]
        r_ = np.corrcoef(e, eg)[0, 1] if len(stims) > 2 and np.nanstd(e) > 0 else np.nan
        mine = {s for s in stims if resp(s, sort)}
        jac = len(mine & set(gt_resp)) / max(len(mine | set(gt_resp)), 1)
        lat = np.nanmedian([abs(fl(X[s]["first_spk_lat_med"]) - fl(g[s]["first_spk_lat_med"])) for s in gt_resp]) if gt_resp else np.nan
        tot = fl(X[stims[0]]["cluster_total"])
        out.append(dict(unit=u, sort=sort, run=X[stims[0]]["run"], n_stim=len(stims), gt_responsive=len(gt_resp), responsive=len(mine),
                        shared=len(mine & set(gt_resp)), extra=len(mine - set(gt_resp)), missed=len(set(gt_resp) - mine), jaccard=jac,
                        tuning_r=r_, mi=mi, baseline_hz=np.nanmedian([fl(X[s]["mean_fr_base_hz"]) for s in stims]),
                        auroc=np.nanmedian([fl(X[s]["auroc"]) for s in gt_resp]) if gt_resp else np.nan, latency_diff_ms=lat,
                        window_per_1000=1000 * np.nansum([fl(X[s]["n_response"]) for s in stims]) / len(stims) / max(tot, 1),
                        median_log10_sign_p=np.nanmedian(np.log10([fl(X[s]["p_value_sign"]) for s in gt_resp])) if gt_resp else np.nan,
                        cluster_total=int(tot)))

with open(R / "raster_compare.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

# ---- figure 1: per-unit measures, one panel per measure
meas = [("tuning_r", "tuning r with GT (higher = same stimuli)"), ("mi", "modulation index on GT-responsive stim"),
        ("baseline_hz", "baseline rate (Hz, lower = less contamination)"), ("auroc", "AUROC on GT-responsive stim"),
        ("jaccard", "responsive-set agreement with GT"), ("latency_diff_ms", "|latency - GT latency| (ms)")]
fig, axs = plt.subplots(2, 3, figsize=(17, 9))
x = np.arange(len(units)); wdt = 0.2
for ax, (k, t) in zip(axs.flat, meas):
    for i, sort in enumerate(SORTS):
        v = [next((r[k] for r in out if r["unit"] == u and r["sort"] == sort), np.nan) for u in units]
        ax.bar(x + (i - 1.5) * wdt, v, wdt, color=COL[sort], label=sort)
    ax.set_xticks(x); ax.set_xticklabels(units, rotation=30, ha="right", fontsize=9); ax.set_title(t, fontsize=11, fontweight="bold")
axs[0, 0].legend(fontsize=9)
fig.suptitle("Stimulus responses, count-free: GT vs python as run vs python tuned vs wave_clus (each sort's GT-matched cluster)",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(R / "raster_compare.png", dpi=110); plt.close(fig)

# ---- figure 2: tuning curves (evoked rate per stimulus, each sort scaled to its own max so size does not matter)
fig, axs = plt.subplots(2, 4, figsize=(18, 8)); axs = axs.flat
for ax, u in zip(axs, units):
    g = sorted(D[u]["GT"]); order = np.argsort(-tuning[u]["GT"])
    for sort in SORTS:
        e = tuning[u].get(sort)
        if e is None: continue
        ax.plot(np.arange(len(g)), e[order] / max(np.nanmax(np.abs(e)), 1e-9), "-o", ms=3, color=COL[sort], label=sort, lw=2 if sort == "GT" else 1.3)
    rr = {r["sort"]: r["tuning_r"] for r in out if r["unit"] == u}
    ax.set_title(f"{u}\nr: as run {rr.get('python as run', np.nan):.2f}, tuned {rr.get('python tuned', np.nan):.2f}, wc {rr.get('wave_clus', np.nan):.2f}", fontsize=10)
    ax.set_xlabel("stimulus (GT order, strongest first)", fontsize=8); ax.axhline(0, color="0.8", lw=0.8)
axs[0].set_ylabel("evoked rate / own max"); axs[0].legend(fontsize=8)
for ax in list(axs)[len(units):]: ax.axis("off")
fig.suptitle("Tuning curves: evoked rate (post - baseline) per stimulus, each sort normalised to its own maximum", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(R / "raster_tuning.png", dpi=110); plt.close(fig)

for r in out:
    print(f"{r['unit']:10s} {r['sort']:14s} n={r['cluster_total']:6,}  resp {r['responsive']:2d}/{r['n_stim']:2d} (GT {r['gt_responsive']:2d}; shared {r['shared']:2d}, extra {r['extra']}, missed {r['missed']})  "
          f"J {r['jaccard']:.2f}  r {r['tuning_r']:.3f}  MI {r['mi']:.3f}  base {r['baseline_hz']:.2f}Hz  AUROC {r['auroc']:.3f}  "
          f"lat {r['latency_diff_ms']:.1f}ms  win/1k {r['window_per_1000']:.1f}  log10p {r['median_log10_sign_p']:.2f}")
print("->", R)
