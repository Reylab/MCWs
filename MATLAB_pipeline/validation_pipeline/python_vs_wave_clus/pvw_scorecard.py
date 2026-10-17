"""One-page scorecard per GT unit: every candidate problem in the python sort,
its measured size on that unit, and a plain verdict.  Used as page 1 of each
appendix and in the overview.

Verdict rules (fixed, so they are the same on every channel):
  quarantine rule      COSTS when it loses >= 1 % of the GT unit, minor below
  Mahalanobis          HURTS when it moves more GT spikes out of the GT cluster
                       than in; ADDS NON-GT when non-GT moved in > GT moved in;
                       helps when GT moved in > out; each only at >= 1 % of the
                       unit, else minor
  log-det / Euclidean  HELPS / HURTS when GT recovered changes >= 2 points
                       (purity reported alongside), else neutral
  second pass          HURTS / HELPS when GT recovered changes >= 2 points
  vs wave_clus         from responsive stimuli and spikes in the window
"""
import csv, json
from pathlib import Path
import numpy as np
import scipy.io as sio
from pvw_common import CFG, OUT, SC, second_pass_dir


def _csv(p):
    p = Path(p)
    return list(csv.DictReader(open(p))) if p.exists() else []


def gt_times(u):
    G = sio.loadmat(OUT / "gt" / f"gt_{u['ch']}.mat")["cluster_class"]
    return np.sort(G[G[:, 0] == u["gt_class"], 1])


def gt_index(t, gt):
    """rows of t (spike times) matched to the GT spikes, 0.5 ms"""
    order = np.argsort(t); ts = t[order]
    k = np.clip(np.searchsorted(ts, gt), 1, len(ts) - 1)
    k = np.where(np.abs(ts[k - 1] - gt) < np.abs(ts[k] - gt), k - 1, k)
    return order[k[np.abs(ts[k] - gt) <= 0.5]]


def replay_gt(u):
    """{variant: (cluster, recall, purity)} of the log-det replay for this unit's python
    cluster (the fixed python_class, or the one matched to GT)."""
    d = OUT / "ld" / f"ch{u['ch']}"
    if not (d / "times.npy").exists():
        return {}
    t = np.load(d / "times.npy"); gt = gt_times(u); j = gt_index(t, gt)
    out = {}
    for v in ("as_run", "logdet", "euclidean"):
        lab = np.load(d / f"labels_{v}.npy"); L = lab[j]
        c = u.get("python_class") or int(np.bincount(L[L > 0]).argmax())
        out[v] = (c, np.sum(L == c) / len(gt), np.sum(L == c) / max(np.sum(lab == c), 1))
    return out


def mahal_net(u):
    sp = second_pass_dir(u["run"])
    T = sio.loadmat(sorted(sp.glob("times_*/times_*.mat"))[0], variable_names=["cluster_class", "mahal_reassigned"])
    t = T["cluster_class"][:, 1]; lab = T["cluster_class"][:, 0].astype(int)
    m = T["mahal_reassigned"].ravel().astype(bool)
    j = gt_index(t, gt_times(u))
    isgt = np.zeros(len(t), bool); isgt[j] = True
    L = lab[j]; best = u.get("python_class") or np.bincount(L[L > 0]).argmax()
    return dict(best=int(best), moved=int(m.sum()), into=int(np.sum(m & isgt & (lab == best))),
                away=int(np.sum(m & isgt & (lab != best))), bad_in=int(np.sum(m & ~isgt & (lab == best))))


def scorecard(u):
    """[(problem, measured, verdict)] for one GT unit, plus the unit's GT size."""
    PY = u["python_arm"]
    diag = SC / u["folder"] / f"validation_gt_vs_{PY}{u['tok']}{u.get('py_tag', '')}" / "diagnostics" / f"diagnostics_summary_ch{u['ch']}_gt{u['gt_class']}.csv"
    S = {r["sort"]: r for r in _csv(diag)}
    new, old = S.get(PY), S.get(u["python_before"])
    rm = [r for r in _csv(OUT / "raster_metrics.csv") if r["unit"] == u["id"]]
    rmd = {r["sort"]: r for r in rm}
    fate = [int(float(v)) if v not in ("", "NaN") else 0 for v in list(new.values())[8:]] if new else [0] * 10
    G = sio.loadmat(OUT / "gt" / f"gt_{u['ch']}.mat")["cluster_class"]; gtn = int(np.sum(G[:, 0] == u["gt_class"]))
    pct = lambda n: 100 * n / max(gtn, 1)
    rows = []

    def q(name, n):
        rows.append((name, f"{n:,} GT spikes lost ({pct(n):.1f}%)", "COSTS" if pct(n) >= 1 else "minor"))
    q("within-ch: second-peak rule (prom 2.0)", fate[2])
    q("within-ch: low-amplitude (bottom 5%)", fate[3])
    q("within-ch: width outside 3-15 samples", fate[4])
    q("within-ch: other checks", fate[5])
    q("refractory quarantine (never rescued)", fate[6])

    mn = mahal_net(u)
    if mn["away"] > mn["into"] and pct(mn["away"]) >= 1:
        v = "HURTS: moves GT out"
    elif mn["bad_in"] > mn["into"] and pct(mn["bad_in"]) >= 1:
        v = "ADDS NON-GT"
    elif mn["into"] > mn["away"] and pct(mn["into"]) >= 1:
        v = "helps"
    else:
        v = "minor"
    rows.append(("Mahalanobis (as run)", f"GT in {mn['into']:,} / GT out {mn['away']:,} / non-GT in {mn['bad_in']:,}", v))

    R = replay_gt(u)
    if R:
        a = dict(recall=R["as_run"][1], purity=R["as_run"][2])
        for key, name in (("logdet", "+ log-determinant (replay)"), ("euclidean", "Euclidean only (replay)")):
            b = dict(recall=R[key][1], purity=R[key][2]); d = 100 * (b["recall"] - a["recall"]); dp = 100 * (b["purity"] - a["purity"])
            verdict = "HELPS" if d >= 2 else ("HURTS" if d <= -2 else "neutral")
            rows.append((name, f"recovered {100*a['recall']:.0f}->{100*b['recall']:.0f}, purity {100*a['purity']:.0f}->{100*b['purity']:.0f}",
                         f"{verdict} ({d:+.0f} / {dp:+.0f} pts)"))

    if new and old:
        d = float(new["recall_pct"]) - float(old["recall_pct"])
        verdict = "HURTS" if d <= -2 else ("HELPS" if d >= 2 else "neutral")
        rows.append((CFG.get("before_label", "second pass (its input -> newest)"), f"recovered {float(old['recall_pct']):.0f}->{float(new['recall_pct']):.0f}, "
                     f"purity {float(old['purity_pct']):.0f}->{float(new['purity_pct']):.0f}", verdict))

    wc = [k for k in rmd if k not in ("GT", "python")]
    if "python" in rmd and wc:
        p, w, g0 = rmd["python"], rmd[wc[0]], rmd["GT"]
        np_, nw = int(p["n_responsive"]), int(w["n_responsive"])
        rows.append(("rasters: responsive stimuli", f"GT {g0['n_responsive']}, python {np_}, {wc[0]} {nw}  (of {p['n_stim']})",
                     "python better" if np_ > nw else ("wave_clus better" if nw > np_ else "same")))
        sp_, sw = float(p["mean_n_response"]), float(w["mean_n_response"])
        per = lambda r: 1000 * float(r["mean_n_response"]) / max(float(r["cluster_total"]), 1)
        rows.append(("response share: in-window spikes / 1,000 cluster spikes",
                     f"GT {per(g0):.1f}, python {per(p):.1f}, {wc[0]} {per(w):.1f}",
                     "python more selective" if per(p) > per(w) else "wave_clus more selective"))
        rows.append(("rasters: spikes in window / stim", f"GT {float(g0['mean_n_response']):.0f}, python {sp_:.0f}, {wc[0]} {sw:.0f}",
                     "python more" if sp_ > sw else "wave_clus more"))
    S2 = S.get(wc[0]) if wc else None
    if S2 and new:
        f1 = lambda r: 2 * float(r["recall_pct"]) * float(r["purity_pct"]) / max(float(r["recall_pct"]) + float(r["purity_pct"]), 1e-9) / 100
        rows.append(("F1 (recovery x purity)", f"python {f1(new):.3f}, {wc[0]} {f1(S2):.3f}" + (f", python before {f1(old):.3f}" if old else ""),
                     "python better" if f1(new) > f1(S2) else "wave_clus better"))
    if S2 and new:
        rows.append(("GT recovered / purity", f"python {float(new['recall_pct']):.0f} / {float(new['purity_pct']):.0f}, "
                     f"{wc[0]} {float(S2['recall_pct']):.0f} / {float(S2['purity_pct']):.0f}",
                     "wave_clus merges more" if float(S2["purity_pct"]) < float(new["purity_pct"]) else "python merges more"))
    return rows, gtn
