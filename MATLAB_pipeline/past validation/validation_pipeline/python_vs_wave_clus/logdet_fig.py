"""Figure for one channel's three-way replay (logdet_replay.json):
clusters' size and ISI<3ms per variant, GT recall / purity per variant, and how
many spikes each variant puts in a different cluster than the as-run sort."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import OUT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ch = int(sys.argv[1]); d = OUT / "ld" / f"ch{ch}"; out = d / f"logdet_ch{ch}.png"
R = json.loads((d / "logdet_replay.json").read_text())
V = ["as_run", "logdet", "euclidean"]
names = {"as_run": "as run (Mahalanobis)", "logdet": "Mahalanobis + log-det", "euclidean": "Euclidean only"}
cols = {"as_run": "#c0392b", "logdet": "#2471a3", "euclidean": "#7f8c8d"}
lab = {v: np.load(d / f"labels_{v}.npy") for v in V}
clusters = sorted({int(c) for v in V for c in R["variants"][v]["clusters"]} - {0})

plt.rcParams.update({"font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"})
fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
w = 0.27
x = np.arange(len(clusters))
for i, v in enumerate(V):
    C = R["variants"][v]["clusters"]
    ax[0].bar(x + (i - 1) * w, [C.get(str(c), {}).get("n", 0) for c in clusters], w, color=cols[v], label=names[v])
    ax[1].bar(x + (i - 1) * w, [C.get(str(c), {}).get("isi_pct", 0) for c in clusters], w, color=cols[v])
for a, t in ((ax[0], "cluster size"), (ax[1], "% ISI < 3 ms")):
    a.set_xticks(x, [f"c{c}" for c in clusters]); a.set_title(t); a.grid(axis="y", alpha=.3)
ax[1].axhline(1, color="k", ls=":", lw=1)
ax[0].legend(fontsize=8)

gts = list(R["variants"]["as_run"]["gt"])
for i, v in enumerate(V):
    G = R["variants"][v]["gt"]
    xs = np.arange(len(gts))
    ax[2].bar(xs * 2 + (i - 1) * w, [G[g]["recall"] for g in gts], w, color=cols[v])
    ax[2].bar(xs * 2 + 1 + (i - 1) * w, [G[g]["purity"] for g in gts], w, color=cols[v])
ax[2].set_xticks(sum([[2 * k, 2 * k + 1] for k in range(len(gts))], []),
                 sum([[f"GT{g} recall", f"GT{g} purity"] for g in gts], []), rotation=20)
ax[2].set_ylim(0, 1.05); ax[2].set_title("GT unit (matched cluster)"); ax[2].grid(axis="y", alpha=.3)

base = lab["as_run"]
diffs = [int(np.sum(lab[v] != base)) for v in V[1:]]
ax[3].bar([names[v] for v in V[1:]], diffs, color=[cols[v] for v in V[1:]])
for k, n in enumerate(diffs):
    ax[3].text(k, n, f"{n:,}", ha="center", va="bottom")
ax[3].set_title("spikes in a different cluster than as run"); ax[3].tick_params(axis="x", rotation=10)
dec = ", ".join(f"c{c} {s}" for c, s in R["decisions"].items())
fig.suptitle(f"{R['channel']} -- second-pass act step replayed three ways (as run == saved sort: "
             f"{R['variants']['as_run']['labels_equal_saved']:.0%})\nsecond-pass decisions: {dec}", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, dpi=130)
print(out)
