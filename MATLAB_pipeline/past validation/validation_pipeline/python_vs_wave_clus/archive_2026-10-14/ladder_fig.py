"""Loosening ladder figure + table: per GT unit, python as run -> width -> mild -> loose
(no log-det), with the unit's wave_clus run and GT as references.
  spike metrics   <out_dir>/trials/ladder_metrics.csv (ladder_eval.py)
  responses       ladder_responses.csv (resp_ladder.m + resp_ladder_export.m): each
                  sort's GT-matched cluster on the unit's configured stimulus list
-> <out_dir>/trials/ladder.png, ladder_summary.csv
usage: ladder_fig.py <ladder_responses.csv>
"""
import csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import OUT

STEPS = ["as_run", "width", "mild", "within"]
LBL = {"as_run": "as run", "width": "width 25", "mild": "+ prom 1.7", "within": "+ prom 1.5,\nlow amp 1%"}
SM = csv.DictReader(open(OUT / "trials" / "ladder_metrics.csv"))
SM = list(SM)
RS = list(csv.DictReader(open(sys.argv[1])))
f = lambda v: float(v) if v not in ("", "NaN", "nan") else np.nan
t = lambda v: str(v).strip().lower() in ("1", "true")
units = list(dict.fromkeys(r["unit"] for r in SM))


def resp(unit, sort, cls):
    rr = [r for r in RS if r["unit"] == unit and r["sort"] == sort and r["class"] == f"class{cls}"]
    if not rr:
        return {}
    z = np.array([f(r["zscore"]) for r in rr]); ok = np.isfinite(z) & (z < 1e6)   # z is unusable when the baseline is 0
    return dict(n_resp=sum(t(r["p_test"]) and f(r["zscore"]) >= 3 for r in rr), n_stim=len(rr),
                mean_z=float(np.mean(z[ok])) if ok.any() else np.nan,
                median_post=float(np.mean([f(r["median_post"]) for r in rr])), baseline=f(rr[0]["mu_base"]),
                sign_p=float(np.median([f(r["p_value_sign"]) for r in rr])), auroc=float(np.nanmedian([f(r["auroc"]) for r in rr])))


rows = []
for u in units:
    for r in [x for x in SM if x["unit"] == u]:
        s = r["sort"]
        rs_sort = (f"python_{s.split()[1]}" if s.startswith("python") else "GT" if s == "GT" else f"wc_{s}")
        rows.append(dict(unit=u, sort=s, cluster=r["cluster"], n=int(r["n"]), recovered=100 * f(r["recovered"]), purity=100 * f(r["purity"]),
                         f1=f(r["f1"]), isi=f(r["isi_pct"]), snr=f(r["snr"]), d_prime=f(r["d_prime"]), l_ratio=f(r["l_ratio"]),
                         iso=f(r["iso_dist"]), **resp(u, rs_sort, r["cluster"])))
with open(OUT / "trials" / "ladder_summary.csv", "w", newline="") as fh:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

M = [("f1", "F1 vs GT", 1), ("purity", "purity %", 1), ("isi", "ISI<3ms %  (lower better)", -1), ("snr", "SNR", 1),
     ("d_prime", "d'", 1), ("l_ratio", "L-ratio (lower better)", -1),
     ("n_resp", "responsive stimuli", 1), ("mean_z", "mean z (finite)", 1), ("median_post", "mean median post", 1),
     ("baseline", "baseline (lower better)", -1), ("sign_p", "median sign p (lower better)", -1), ("auroc", "median AUROC", 1)]
fig, axs = plt.subplots(3, 4, figsize=(22, 14)); axs = axs.ravel()
cols = plt.cm.tab20(np.linspace(0, 1, len(units)))
for ax, (k, title, _) in zip(axs, M):
    for ci, u in enumerate(units):
        R = {r["sort"]: r for r in rows if r["unit"] == u}
        y = [R.get(f"python {s}", {}).get(k, np.nan) for s in STEPS]
        ax.plot(range(len(STEPS)), y, "-o", color=cols[ci], lw=1.6, ms=4, label=u)
        wc = next((r for r in rows if r["unit"] == u and not r["sort"].startswith("python") and r["sort"] != "GT"), None)
        if wc and np.isfinite(wc.get(k, np.nan)):
            ax.plot(len(STEPS) + 0.3, wc[k], "s", color=cols[ci], ms=6, mfc="none")
        g = R.get("GT", {}).get(k, np.nan)
        if np.isfinite(g):
            ax.plot(len(STEPS) + 0.8, g, "*", color=cols[ci], ms=8)
    ax.set_xticks(list(range(len(STEPS))) + [len(STEPS) + 0.3, len(STEPS) + 0.8])
    ax.set_xticklabels([LBL[s] for s in STEPS] + ["wave_clus", "GT"], fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if k in ("sign_p", "l_ratio"):
        ax.set_yscale("log")
axs[0].legend(fontsize=7, ncol=2)
fig.suptitle("Loosening the within-channel rule (same settings on every channel, no log-det): spike metrics (top two rows) "
             "and responses on each unit's stimulus list (bottom).  Lines = python; open square = wave_clus; star = GT.",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "trials" / "ladder.png", dpi=110)
print("->", OUT / "trials" / "ladder.png")
