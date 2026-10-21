"""Hand-selected GT clusters of one channel: every class's waveforms (with its
mean and the share that was template-forced) and ISI histogram; the GT
unit(s) analysed are marked.  -> <out_dir>/gt/gt_clusters_ch<ch>.png

usage: gt_fig.py <ch>
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import OUT, channels

ch = int(sys.argv[1]); C = channels()[ch]
m = sio.loadmat(OUT / "gt" / f"gt_{ch}.mat")
lab = m["cluster_class"][:, 0].astype(int); t = m["cluster_class"][:, 1]
W = m["spikes"].astype(float); fo = m["forced"].ravel().astype(bool) if "forced" in m else np.zeros(len(lab), bool)
used = {u["gt_class"] for u in C["units"]}
cls = [c for c in np.unique(lab) if c > 0]
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 9, "font.weight": "bold", "axes.titleweight": "bold"})
fig, axes = plt.subplots(len(cls), 2, figsize=(11, 2.5 * len(cls) + 0.9), squeeze=False, gridspec_kw={"width_ratios": [1.6, 1]})
for i, c in enumerate(cls):
    mask = lab == c; idx = np.flatnonzero(mask)
    ax = axes[i, 0]
    pick = rng.choice(idx, min(idx.size, 400), replace=False)
    ax.plot(W[pick].T, color="#1f6fb2" if c in used else "#7f8c8d", alpha=0.12, lw=0.5)
    ax.plot(W[idx].mean(0), color="k", lw=2)
    ax.set_ylim(np.percentile(W[idx], [0.5, 99.5])); ax.grid(alpha=.3)
    mark = "   <- GT unit analysed" if c in used else ""
    ax.set_title(f"GT class {c}: n={idx.size:,}  ({100 * fo[mask].mean():.0f}% template-forced){mark}", fontsize=9)
    ax = axes[i, 1]
    isi = np.diff(np.sort(t[mask]))
    ax.hist(isi, bins=np.arange(0, 102, 2), color="#1f6fb2" if c in used else "#7f8c8d", linewidth=0)
    ax.set_xlim(0, 100); ax.grid(alpha=.3); ax.set_xlabel("ISI (ms)")
    ax.set_title(f"{int(np.sum(isi < 3)):,} ISIs < 3 ms ({100 * np.mean(isi < 3):.2f}%)", fontsize=9)
fig.suptitle(f"ch{ch} hand-selected GT: every class ({int(np.sum(lab == 0)):,} spikes in class 0, not shown)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96 if len(cls) > 2 else 0.92))
p = OUT / "gt" / f"gt_clusters_ch{ch}.png"; fig.savefig(p, dpi=120); plt.close(fig)
print(p)
