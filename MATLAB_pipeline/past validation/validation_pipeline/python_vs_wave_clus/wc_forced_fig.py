"""wave_clus run of one channel: every cluster's waveforms clustered by SPC
(forced = 0) vs placed by template forcing (forced = 1), and their means
overlaid.  The cluster matched to GT (and to python's GT cluster) is marked.
-> <out_dir>/wc/wc_forced_<run>_ch<ch>.png

usage: wc_forced_fig.py <ch>
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
from pvw_scorecard import gt_index, gt_times

ch = int(sys.argv[1]); C = channels()[ch]
for w in sorted({w for u in C["units"] for w in u["wave_clus"]}):
    m = sio.loadmat(OUT / "wc" / f"{w}_ch{ch}.mat")
    lab = m["cluster_class"][:, 0].astype(int); t = m["cluster_class"][:, 1]
    fo = m["forced"].ravel().astype(bool); W = m["spikes"].astype(float)
    marks = {}
    for u in C["units"]:
        if w not in u["wave_clus"]:
            continue
        L = lab[gt_index(t, gt_times(u))]
        c = int(np.bincount(L[L > 0]).argmax())
        marks.setdefault(c, []).append(f"GT{u['gt_class']}")
    cls = [c for c in np.unique(lab) if c > 0]
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.size": 9, "font.weight": "bold", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(len(cls), 3, figsize=(13, 2.6 * len(cls) + 0.8), squeeze=False)
    for i, c in enumerate(cls):
        m0, m1 = (lab == c) & ~fo, (lab == c) & fo
        tag = f"  <- matched to {', '.join(marks[c])}" if c in marks else ""
        yl = np.percentile(W[lab == c], [0.5, 99.5])
        for k, (mask, name, col) in enumerate(((m0, "forced = 0 (SPC)", "#1f6fb2"), (m1, "forced = 1 (template)", "#d35400"))):
            ax = axes[i, k]
            idx = np.flatnonzero(mask)
            if idx.size:
                pick = rng.choice(idx, min(idx.size, 300), replace=False)
                ax.plot(W[pick].T, color=col, alpha=0.12, lw=0.5)
                ax.plot(W[idx].mean(0), color="k", lw=2)
            ax.set_ylim(yl); ax.set_title(f"c{c} {name}: n={idx.size:,} ({100 * idx.size / max((lab == c).sum(), 1):.0f}%)", fontsize=9)
            ax.grid(alpha=.3)
        ax = axes[i, 2]
        for mask, col, name in ((m0, "#1f6fb2", "forced = 0"), (m1, "#d35400", "forced = 1")):
            if mask.any():
                mu, sd = W[mask].mean(0), W[mask].std(0)
                ax.plot(mu, color=col, lw=2, label=name); ax.fill_between(np.arange(W.shape[1]), mu - sd, mu + sd, color=col, alpha=0.15)
        ax.set_ylim(yl); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
        ax.set_title(f"c{c} means +/- sd (n={int((lab == c).sum()):,}){tag}", fontsize=9)
    n0 = int(((lab > 0) & ~fo).sum()); n1 = int(((lab > 0) & fo).sum())
    fig.suptitle(f"ch{ch} {w}: clustered by SPC (forced = 0) vs template-forced (forced = 1)\n"
                 f"all clusters: {n0:,} SPC, {n1:,} forced ({100 * n1 / max(n0 + n1, 1):.0f}% forced); cluster 0 not shown ({int((lab == 0).sum()):,})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(cls) > 2 else 0.9))
    p = OUT / "wc" / f"wc_forced_{w}_ch{ch}.png"; fig.savefig(p, dpi=120); plt.close(fig)
    print(p)
