"""Compact report per GT unit: the wave_clus vs python sd1 validation first, plus the
spike metrics and spikes the validation reports do not carry.

  1  summary       clusters compared; spike metrics (GT / python / wave_clus): recovered,
                   purity, F1, ISI < 3 ms, SNR, d', L-ratio, isolation distance (10 PCs of each
                   sort's own waveforms); responses on the unit's stimulus list
                   (raster_metrics.csv, from the validations); where python loses GT spikes
  2  spikes        the three matched clusters: waveforms, mean +- sd, ISI; means overlaid
  3  GT fate       diagnostics: what happened to each GT spike in python
     + cluster waveforms and metrics (same report style for GT, python, wave_clus)
     + python clustering and second pass: decisions, clusters before -> after, GT spikes before ->
       after, before_after.pdf, second-pass report summary / after acting (as in the appendix)
  4  wave_clus vs python sd1 validation (python = arm), every page, report order
  5  GT vs python validation, every page; GT vs wave_clus: the class match only
  6  wave_clus vs python exclusive rasters (python only, wave_clus only), then any other
     python / wave_clus class holding >= 5% of the other sort's matched cluster: waveforms
     and rasters (subset_sets.py + subset_sets.m make the sets and raster pages)
Only the GT-specified class and each sort's matching cluster, and only the newest python
sort: no whole-sort pages, no all-cluster venn, no channel-wide waveform breakdowns.
Images are fitted to the page (never cropped).

usage: compact_report.py [unit_id ...]     (default: every main unit in channels.json)
-> <out_dir>/compact/report_<unit>.pdf, copied to the unit's channel folder
"""
import csv, glob, shutil, sys, textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, SC, MWS, second_pass_dir
from pvw_scorecard import gt_index, gt_times
sys.path.insert(0, str(MWS))
from mcws.evaluation.spike_metrics import snr, mahalanobis_metrics, d_prime_lda
from mcws.io.spike_io import load_spikes

PAGE = (11, 8.5)
CO = OUT / "compact"; CO.mkdir(exist_ok=True)
COL = {"GT": "k", "python": "#1f77b4", "wave_clus": "#d62728"}
rng = np.random.default_rng(0)


def read_csv(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


# ------------------------------------------------------------------ pages --
def image_page(pdf, png, title):
    png = Path(png)
    if not png.exists():
        return
    img = plt.imread(str(png))
    h, w = img.shape[:2]
    pg = PAGE if w >= h else PAGE[::-1]            # tall images (raster pages) get a portrait page
    fig = plt.figure(figsize=pg)
    fig.text(0.02, 0.975, textwrap.fill(title, 110 if w >= h else 85), fontsize=11, weight="bold", va="top")
    box_w, box_h = 0.96, 0.90                      # page area under the title, as page fractions
    s = min(box_w * pg[0] / w, box_h * pg[1] / h)
    iw, ih = w * s / pg[0], h * s / pg[1]
    ax = fig.add_axes([0.02 + (box_w - iw) / 2, 0.02 + (box_h - ih) / 2, iw, ih])
    ax.imshow(img); ax.axis("off")
    pdf.savefig(fig); plt.close(fig)


def pdf_pages(pdf_file, tag, first=1, last=None):
    """render pages of another PDF to PNGs (cached under compact/_png)"""
    import subprocess
    d = CO / "_png"; d.mkdir(exist_ok=True)
    args = ["pdftoppm", "-r", "110", "-png", "-f", str(first)] + (["-l", str(last)] if last else []) + [str(pdf_file), str(d / tag)]
    subprocess.run(args, check=True)
    return sorted(d.glob(f"{tag}-*.png"))


def table(ax, rows, cols, widths, fs=9):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, colWidths=widths, cellLoc="center", loc="upper center")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
    return t


# ---------------------------------------------------------------- metrics --
def spike_metrics(W, lab, t, c, gt):
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
    rec, pur = float(np.mean(L == c)), float(np.sum(L == c) / max(m.sum(), 1))
    return dict(n=int(m.sum()), rec=rec, pur=pur, f1=2 * rec * pur / max(rec + pur, 1e-9),
                isi=float(np.mean(isi < 3) * 100) if isi.size else 0.0, snr=float(snr(W[m])), dp=float(dp), lr=float(lr), iso=float(iso))


PYFLAGS = {}
PYRAW = {}


def cluster_report_pages(u, S):
    """MicroWireSorter cluster report (waveforms, density, ISI, presence + cluster-metrics summary)
    for GT, python and wave_clus in the same style; generated once per channel and sort."""
    from mcws.visualization.plot_report import make_cluster_report
    import logging; logging.disable(logging.INFO)
    out = {}
    for k in ("GT", "python", "wave_clus"):
        name = {"GT": "GT", "python": u["python_arm"], "wave_clus": u["wave_clus"][0]}[k]
        d = CO / "cluster_reports" / f"ch{u['ch']}" / name; d.mkdir(parents=True, exist_ok=True)
        if not list(d.glob("report_*.pdf")):
            if k == "python":
                data = PYRAW[u["id"]]
            else:
                W, lab, t, _ = S[k]
                data = {"cluster_class": np.column_stack([lab, t]), "spikes": W, "label": f"{name} ch{u['ch']}", "par": {"sr": 30000.0},
                        "features": PCA(10, random_state=0).fit_transform(W)}   # metrics in 10 PCs of its own waveforms
            make_cluster_report({1: data} if k != "python" else {0: data}, calc_metrics=True, plot_ccg=False,
                                exclude_cluster_0=True, clusters_per_page=6, save=True, output_dir=str(d) + "/")
        out[k] = (name, [p for f in sorted(d.glob("report_*.pdf")) for p in pdf_pages(f, f"{u['ch']}_{name}")])
    return out


def python_fate(u, S, times, matched_cls):
    """per spike of a set: where it sits in python (newest sort), as category labels.
    In python's matched cluster: how it got there; otherwise: why it is not there."""
    W, pl, pt, _ = S["python"]; F = PYFLAGS[u["id"]]
    o = np.argsort(pt); ts = pt[o]
    k = np.clip(np.searchsorted(ts, times), 1, len(ts) - 1)
    k = np.where(np.abs(ts[k - 1] - times) < np.abs(ts[k] - times), k - 1, k)
    det = np.abs(ts[k] - times) <= 0.5; j = o[k]
    cat = []
    for d, i in zip(det, j):
        if not d:
            cat.append("not detected by python"); continue
        c = pl[i]
        if c == matched_cls:
            cat.append("python c%d: rescued" % c if F["rescued"][i] else "python c%d: template-forced" % c if F["forced"][i]
                       else "python c%d: SPC clustered" % c)
        elif c > 0:
            cat.append(f"in another python cluster: c{c}" + (" (rescued into it)" if F["rescued"][i] else ""))
        elif F["quarantined"][i]:
            why = ("within-channel" if F["within"][i] else "refractory" if F["refractory"][i] else "bundle" if F["bundle"][i] else "other")
            cat.append(f"python cluster 0: quarantined ({why}), not rescued")
        else:
            cat.append("python cluster 0: not quarantined (unassigned)")
    return np.array(cat), det, j


def wave_clus_fate(u, S, times, matched_cls):
    """per spike: where it sits in wave_clus (no quarantine in wave_clus)"""
    W, wl, wt, _ = S["wave_clus"]
    o = np.argsort(wt); ts = wt[o]
    k = np.clip(np.searchsorted(ts, times), 1, len(ts) - 1)
    k = np.where(np.abs(ts[k - 1] - times) < np.abs(ts[k] - times), k - 1, k)
    det = np.abs(ts[k] - times) <= 0.5; j = o[k]
    cat = np.array(["not detected by wave_clus" if not d else f"wave_clus c{wl[i]} (matched)" if wl[i] == matched_cls
                    else "wave_clus cluster 0 (unassigned)" if wl[i] == 0 else f"in another wave_clus cluster: c{wl[i]}"
                    for d, i in zip(det, j)])
    return cat, det, j


def fate_page(pdf, title, u, S, times, src_W, src_t, matched_cls, where="python"):
    """waveforms of a spike set split by python_fate (python's waveform when python detected the spike,
    else the source sort's)"""
    cat, det, j = (python_fate if where == "python" else wave_clus_fate)(u, S, times, matched_cls)
    if not len(cat):
        return
    Wp = S[where][0]
    so = np.argsort(src_t); sts = src_t[so]
    ks = np.clip(np.searchsorted(sts, times), 1, len(sts) - 1)
    ks = so[np.where(np.abs(sts[ks - 1] - times) < np.abs(sts[ks] - times), ks - 1, ks)]
    labels, counts = np.unique(cat, return_counts=True); order = np.argsort(-counts)
    n = len(order); nc = min(4, n); nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=PAGE, squeeze=False)
    tt = np.arange(64) / 30.0
    for a_, oi in zip(axs.ravel(), order):
        m = cat == labels[oi]
        X = np.vstack([Wp[j[i]] if det[i] else src_W[ks[i]] for i in np.flatnonzero(m)])
        pick = rng.choice(len(X), min(200, len(X)), replace=False)
        a_.plot(tt, X[pick].T, color="#1f77b4" if det[m].all() else "#888888", alpha=0.08, lw=0.6)
        a_.plot(tt, X.mean(0), color="k", lw=1.8)
        a_.set_title(f"{labels[oi]}\nn={counts[oi]:,} ({100 * counts[oi] / len(cat):.1f}%)", fontsize=9); a_.set_xlabel("ms")
    for a_ in axs.ravel()[n:]:
        a_.axis("off")
    fig.suptitle(textwrap.fill(title, 110), fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); pdf.savefig(fig); plt.close(fig)


def sorts_for(u):
    """{name: (W, labels, times, matched cluster)} for GT, python (newest second pass), wave_clus run"""
    gt = gt_times(u)
    g = sio.loadmat(OUT / "gt" / f"gt_{u['ch']}.mat")
    out = {"GT": (g["spikes"].astype(float), g["cluster_class"][:, 0].astype(int), g["cluster_class"][:, 1], u["gt_class"])}
    sp = second_pass_dir(u["run"])
    x = load_spikes(str(sorted(sp.glob("times_*"))[-1])); x = x[next(iter(x))]
    pl, pt = x["cluster_class"][:, 0].astype(int), x["cluster_class"][:, 1]
    L = pl[gt_index(pt, gt)]
    out["python"] = (np.asarray(x["spikes"], float), pl, pt, int(np.bincount(L[L > 0]).argmax()))
    PYRAW[u["id"]] = x
    q = x["quarantine"]; flag = lambda v: np.asarray(v).astype(bool).ravel() if v is not None else np.zeros(len(pl), bool)
    PYFLAGS[u["id"]] = dict(forced=flag(x.get("forced")), mahal=flag(x.get("mahal_reassigned")), rescued=flag(q.get("rescued")),
                            within=flag(q.get("within_channel")), refractory=flag(q.get("refractory_period")),
                            bundle=flag(q.get("bundle_collision")), quarantined=flag(q.get("all")))
    w = u["wave_clus"][0]; m = sio.loadmat(OUT / "wc" / f"{w}_ch{u['ch']}.mat")
    wl, wt = m["cluster_class"][:, 0].astype(int), m["cluster_class"][:, 1]
    L = wl[gt_index(wt, gt)]
    out["wave_clus"] = (m["spikes"].astype(float), wl, wt, int(np.bincount(L[L > 0]).argmax()))
    return out, gt


# ------------------------------------------------------------------ build --
def build(u, py_report_cache):
    ch, g, tok, w = u["ch"], u["gt_class"], u["tok"], u["wave_clus"][0]
    root = SC / u["folder"]
    gtpy = root / f"validation_gt_vs_{u['python_arm']}{tok}"
    gtwc = next((p for p in (root / w / f"validation_gt_vs_{w}{tok}", root / f"validation_gt_vs_{w}{tok}") if p.exists()), None)
    wcpy = root / w / f"validation_{w}_vs_python_sd1{tok}"
    S, gt = sorts_for(u)
    M = {k: spike_metrics(W, lab, t, c, gt) for k, (W, lab, t, c) in S.items()}
    R = {r["sort"]: r for r in read_csv(OUT / "raster_metrics.csv") if r["unit"] == u["id"]}
    diag = read_csv(gtpy / "diagnostics" / f"diagnostics_summary_ch{ch}_gt{g}.csv")
    unit = f"ch{ch} GT class {g}"
    pdf_path = CO / f"report_{u['id']}.pdf"
    with PdfPages(pdf_path) as pdf:
        # 1 summary
        fig = plt.figure(figsize=PAGE)
        fig.text(0.03, 0.965, f"{unit}  ({u['folder']})", fontsize=15, weight="bold", va="top")
        fig.text(0.03, 0.925, textwrap.fill(
            f"Compared: GT class {g} (hand selected, {M['GT']['n']:,} spikes);  python {u['python_arm']} cluster {S['python'][3]};  "
            f"{w} cluster {S['wave_clus'][3]}.  Each is the cluster holding the most GT spikes (0.5 ms matching).  In wave_clus vs "
            f"python, wave_clus is the reference and python sd1 the arm.", 135), fontsize=9.5, va="top")
        ax = fig.add_axes([0.03, 0.57, 0.94, 0.24])
        fmt = lambda k, v, d=1: "-" if not np.isfinite(v) else f"{v:.{d}f}"
        rows = [[k, f"c{S[k][3]}", f"{M[k]['n']:,}", fmt(k, 100 * M[k]['rec']), fmt(k, 100 * M[k]['pur']), fmt(k, M[k]['f1'], 3),
                 fmt(k, M[k]['isi'], 2), fmt(k, M[k]['snr'], 2), fmt(k, M[k]['dp'], 2), fmt(k, M[k]['lr'], 3), fmt(k, M[k]['iso'], 0)]
                for k in ("GT", "python", "wave_clus")]
        fig.text(0.03, 0.835, "Spike metrics", fontsize=12, weight="bold")
        table(ax, rows, ["sort", "cluster", "spikes", "GT\nrecovered %", "purity %", "F1", "ISI\n< 3 ms %", "SNR", "d'", "L-ratio", "isolation\ndistance"],
              [0.1, 0.07, 0.09, 0.11, 0.08, 0.07, 0.09, 0.07, 0.07, 0.08, 0.08])
        fig.text(0.03, 0.535, "Responses on the unit's stimulus list (validation rasters)", fontsize=12, weight="bold")
        ax = fig.add_axes([0.03, 0.27, 0.94, 0.25])
        rk = {"GT": "GT", "python": "python", "wave_clus": w}
        rrows = []
        for k in ("GT", "python", "wave_clus"):
            r = R.get(rk[k])
            if r:
                rrows.append([k, r["n_stim"], r["n_responsive"], f"{float(r['mean_zscore']):.2f}", f"{float(r['mean_median_post']):.2f}",
                              f"{float(r['mu_base']):.2f}", f"{float(r['median_sign_p']):.4f}", f"{float(r['mean_n_response']):.1f}",
                              f"{1000 * float(r['mean_n_response']) / max(float(r['cluster_total']), 1):.1f}"])
        if rrows:
            table(ax, rrows, ["sort", "stimuli", "responsive\n(p_test, z >= 3)", "mean z", "mean\nmedian post", "baseline",
                              "median\nsign p", "spikes\nin window", "in window per\n1,000 spikes"],
                  [0.08, 0.07, 0.16, 0.08, 0.12, 0.08, 0.11, 0.12, 0.14])
        py = next((r for r in diag if r.get("sort") == u["python_arm"]), None)
        if py:
            keys = [k for k in py if k.startswith("n_") and k != "n_matchedCluster" and py[k] not in ("", "0", "0.0", "NaN")]
            def nice(k):
                k = k.lower()
                for a, b in (("othercluster", "in another python cluster"), ("secondpeak", "second-peak rule"), ("lowamplitude", "low-amplitude rule"),
                             ("widthoutside", "width rule (3-15 samples)"), ("within_ch_other", "other within-channel rule"), ("refractory", "refractory rule"),
                             ("unassigned", "unassigned, not quarantined"), ("bundle", "bundle collision"), ("not_detected", "not detected")):
                    if a in k:
                        return b
                return k[2:].replace("_", " ")
            lost = ";  ".join(f"{nice(k)} {int(float(py[k])):,} ({100 * float(py[k]) / max(M['GT']['n'], 1):.1f}%)" for k in keys)
            fig.text(0.03, 0.22, "Where python loses GT spikes", fontsize=12, weight="bold")
            fig.text(0.03, 0.19, textwrap.fill(lost or "nothing", 140), fontsize=9.5, va="top")
        fig.text(0.03, 0.05, textwrap.fill("ISI < 3 ms and L-ratio: lower is better.  d', L-ratio and isolation distance measure a cluster "
                                           "against the other clusters of its own sort, so they change with how many clusters a sort has; "
                                           "recovered, purity, ISI and SNR are measured the same way for every sort.", 150), fontsize=8.5, color="0.3")
        pdf.savefig(fig); plt.close(fig)

        # 2 spikes
        fig, axs = plt.subplots(2, 4, figsize=PAGE, gridspec_kw=dict(height_ratios=[1.3, 1]))
        tt = np.arange(64) / 30.0
        for j, k in enumerate(("GT", "python", "wave_clus")):
            W, lab, t, c = S[k]; X = W[lab == c]
            pick = rng.choice(len(X), min(300, len(X)), replace=False)
            a = axs[0, j]; a.plot(tt, X[pick].T, color=COL[k], alpha=0.06, lw=0.6)
            mu, sd = X.mean(0), X.std(0)
            a.fill_between(tt, mu - sd, mu + sd, color=COL[k], alpha=0.25); a.plot(tt, mu, color="k" if k != "GT" else "w", lw=1.8)
            a.set_title(f"{k} c{c}\nn={len(X):,}, SNR {M[k]['snr']:.1f}", fontsize=10); a.set_xlabel("ms")
            isi = np.diff(np.sort(t[lab == c])); a = axs[1, j]
            a.hist(isi[isi < 100], bins=np.arange(0, 102, 2), color=COL[k], alpha=0.8)
            a.axvline(3, color="0.4", ls=":"); a.set_title(f"ISI: {M[k]['isi']:.2f}% < 3 ms", fontsize=10); a.set_xlabel("ms")
            axs[0, 3].plot(tt, mu, color=COL[k], lw=2, label=f"{k} c{c}")
        lim = [min(a.get_ylim()[0] for a in axs[0, :3]), max(a.get_ylim()[1] for a in axs[0, :3])]
        for a in axs[0, :]: a.set_ylim(lim)
        axs[0, 3].legend(fontsize=8); axs[0, 3].set_title("means", fontsize=10); axs[1, 3].axis("off")
        fig.suptitle(f"{unit}: the three matched clusters", fontsize=13, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)

        # 3 GT fate: the newest python sort only (the diagnostics figure also shows its first-pass input)
        if py:
            cats = [("n_matchedCluster", "matched cluster", "#339a4c"), ("n_otherCluster", "other cluster", "#8fbf8f"),
                    ("n_within_ch_SecondPeak_promRatio_2_", "within-ch: second peak (prom ratio < 2)", "#d9403a"),
                    ("n_within_ch_LowAmplitude_bottom5__", "within-ch: low amplitude (bottom 5%)", "#f08a35"),
                    ("n_within_ch_WidthOutside3_15Samples", "within-ch: width outside 3-15 samples", "#b5651d"),
                    ("n_within_ch_Other", "within-ch: other", "#f2cc66"), ("n_refractory", "refractory", "#7b4fa6"),
                    ("n_bundle", "bundle", "#8f8fdf"), ("n_unassigned_notQuarantined_", "unassigned (not quarantined)", "#a8a8a8"),
                    ("n_notDetected", "not detected", "#404040")]
            vals = [(lbl, c, float(py.get(k) or 0)) for k, lbl, c in cats]
            tot = max(sum(v for _, _, v in vals), 1)
            fig = plt.figure(figsize=PAGE); ax = fig.add_axes([0.1, 0.55, 0.86, 0.18])
            x0 = 0
            for lbl, c, v in vals:
                if v <= 0:
                    continue
                ax.barh(0, 100 * v / tot, left=x0, color=c, edgecolor="k", lw=0.5, label=f"{lbl}: {int(v):,} ({100 * v / tot:.1f}%)")
                if 100 * v / tot > 4:
                    ax.text(x0 + 50 * v / tot, 0, f"{int(v):,}", ha="center", va="center", color="w", fontsize=9)
                x0 += 100 * v / tot
            ax.set_xlim(0, 100); ax.set_yticks([0]); ax.set_yticklabels([f"python c{S['python'][3]}"])
            ax.set_xlabel("% of GT spikes")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=9, frameon=False)
            fig.text(0.03, 0.95, f"{unit}: where every GT spike ends up in python ({u['python_arm']})", fontsize=13, weight="bold", va="top")
            fig.text(0.03, 0.9, "Rescued spikes count under the cluster they ended in; quarantined spikes under the first rule that caught "
                                "them (within-channel, then refractory, then bundle).", fontsize=9.5, va="top")
            pdf.savefig(fig); plt.close(fig)

        # 3b waveforms and metrics of every cluster, same report style for the three sorts
        for k, (name, pages) in cluster_report_pages(u, S).items():
            mc = S[k][3]
            if k == "wave_clus":     # wave_clus's own figures (temperature map), then the metrics summary page
                for f in sorted((root / w).glob("fig2print_*.png")):
                    image_page(pdf, f, f"ch{ch} wave_clus ({w}): wave_clus output ({f.stem}) -- matched cluster c{mc}")
                pages = pages[-1:]
            for p in pages:
                image_page(pdf, p, f"ch{ch} {k} ({name}): cluster waveforms and metrics -- matched cluster c{mc}")
        # 3c python clustering and second pass (as in the appendix)
        import json, subprocess
        sp = second_pass_dir(u["run"])
        if sp and (sp / "decisions.json").exists():
            dec = json.loads((sp / "decisions.json").read_text())
            fig = plt.figure(figsize=PAGE)
            fig.text(0.03, 0.965, f"ch{ch} python clustering and second pass ({u['python_arm']})", fontsize=14, weight="bold", va="top")
            fig.text(0.03, 0.925, textwrap.fill(f"Matched cluster for {unit}: python c{S['python'][3]}.  Second pass source: {dec.get('source', '')}", 150),
                     fontsize=9, va="top")
            ax = fig.add_axes([0.03, 0.6, 0.94, 0.27])
            why = lambda c: (c.get("gmm") or {}).get("why", "") if isinstance(c.get("gmm"), dict) else ""
            table(ax, [[f"c{c['cluster']}", c["decision"], f"{c['n']:,}", f"{c['isi_pct']:.2f}", why(c)] for c in dec["clusters"]],
                  ["cluster", "decision", "spikes", "ISI<3ms %", "reason"], [0.1, 0.15, 0.12, 0.12, 0.51])
            fig.text(0.03, 0.56, "Each cluster before -> after the second pass", fontsize=12, weight="bold")
            ax = fig.add_axes([0.03, 0.08, 0.94, 0.46])
            fi = lambda v: "-" if v is None else f"{v:.2f}"
            table(ax, [[f"c{r['cluster']}", r["origin"], f"{r['n_before']:,} -> {r['n_after']:,}", f"{fi(r['isi_before'])} -> {fi(r['isi_after'])}",
                        f"{r['cores']:,}", f"{r['forced']:,}", f"{r['rescued']:,}", f"{r['mahal']:,}"] for r in dec["plan"]["table"]],
                  ["cluster", "from", "spikes", "ISI<3ms %", "SPC cores", "template-matched", "rescued", "Mahalanobis moved"],
                  [0.08, 0.16, 0.16, 0.16, 0.1, 0.12, 0.1, 0.12])
            pdf.savefig(fig); plt.close(fig)
            image_page(pdf, gtpy / "diagnostics" / f"diag_ch{ch}_gt{g}_second_pass.png", f"{unit}: GT spikes before -> after the second pass")
            for k_, png in enumerate(pdf_pages(sp / "before_after.pdf", f"{ch}_before_after")):
                image_page(pdf, png, f"ch{ch} python second pass: before -> after ({k_ + 1})")
            if (sp / "second_pass_report.pdf").exists():
                n_rep = int(subprocess.run(["pdfinfo", str(sp / "second_pass_report.pdf")], capture_output=True, text=True).stdout.split("Pages:")[1].split()[0])
                for png in pdf_pages(sp / "second_pass_report.pdf", f"{ch}_sp_report", 1, 1) + pdf_pages(sp / "second_pass_report.pdf", f"{ch}_sp_after", n_rep, n_rep):
                    image_page(pdf, png, f"ch{ch} python second pass report: " + ("summary" if "sp_report" in png.name else "after acting"))
        # 4 wave_clus vs python sd1
        def pngs(o, ref, arm, full=True):
            o = Path(o); f = []
            f += sorted(o.glob(f"class_match_ch*_{ref}_vs_{arm}.png")) + (sorted(o.glob(f"class_match_ch*_{arm}_vs_{ref}.png")) if full else [])
            f += sorted((o / "stimulus_rasters" / arm).glob("*gt_vs_arm_summary.png")) + sorted((o / "stimulus_rasters" / arm).glob("*sign_test.png"))
            f += sorted((o / "venn_three_set").glob(f"venn3_*_{arm}_cluster.png"))
            if full:
                f += sorted((o / "cluster_composition").glob(f"composition_*_{arm}.png")) + sorted((o / "pie_venn").glob(f"pie_venn_*_{arm}.png"))
                f += sorted((o / "stimulus_rasters").glob("*/*resp_window*.png"))
                f += sorted((o / "waveform_breakdowns").glob(f"{arm}_class*.png"))   # this class only, not channel-wide
            return f
        for p in pngs(wcpy, w, "python_sd1"):
            image_page(pdf, p, f"{unit} -- {w} vs python sd1 (python = arm): {p.stem}")
        # 5 GT vs python (full), then GT vs wave_clus: the class match only
        for p in pngs(gtpy, "GT", "python_sd1"):
            image_page(pdf, p, f"{unit} -- GT vs python: {p.stem}")
        # 6 wave_clus vs python exclusive rasters, then the other classes (subset_sets.py / subset_sets.m)
        import json
        mf = CO / "sets" / u["id"] / "manifest.json"
        if mf.exists():
            man = json.loads(mf.read_text())
            def raster_pages(m):
                return sorted((Path(m["dir"]) / "subset_rasters" / "python_sd1" / m["name"]).glob("*.png"))
            src = {"python_only": "python", "wave_clus_only": "wave_clus", "gtpy_gt_only": "GT", "gtpy_python_only": "python"}
            for grp, names in (("wave_clus vs python, exclusive", ("python_only", "wave_clus_only")),
                               ("GT vs python, exclusive (addition)", ("gtpy_gt_only",))):
                for m in [m for m in man["sets"] if m["name"] in names]:
                    times = sio.loadmat(m["file"])["times"].ravel()
                    sW, _, st_, _ = S[src[m["name"]]]
                    where = "wave_clus" if m["name"] == "python_only" else "python"
                    fate_page(pdf, f"{unit} -- {grp}: {m['desc']} (n={m['n']:,}): where these spikes are in "
                                   + (f"wave_clus ({w})" if where == "wave_clus" else f"python ({u['python_arm']})"),
                              u, S, times, sW, st_, S[where][3], where)
                    for p in raster_pages(m):
                        image_page(pdf, p, f"{unit} -- {grp}: {m['desc']} (n={m['n']:,})")
            if man["other"]:
                oth = man["other"]
                fig, axs = plt.subplots(1, len(oth), figsize=PAGE, squeeze=False); axs = axs[0]
                for a_, o in zip(axs, oth):
                    W, lab, t, c = S[o["sort"]]; X = W[lab == o["cls"]]
                    pick = rng.choice(len(X), min(300, len(X)), replace=False)
                    a_.plot(tt, X[pick].T, color=COL[o["sort"]], alpha=0.06, lw=0.6)
                    a_.plot(tt, X.mean(0), color="k", lw=1.8, label=f"{o['sort']} c{o['cls']} mean")
                    a_.plot(tt, W[lab == c].mean(0), color="0.5", lw=1.5, ls="--", label=f"matched {o['sort']} c{c} mean")
                    other_name = "wave_clus" if o["sort"] == "python" else "python"
                    a_.set_title(f"{o['sort']} c{o['cls']} (n={o['n']:,})\nholds {100*o['share']:.0f}% of {other_name} c{S[other_name][3]}", fontsize=9)
                    a_.set_xlabel("ms"); a_.legend(fontsize=7)
                fig.suptitle(f"{unit}: wave_clus vs python -- other classes holding >= 5% of the other sort's matched cluster", fontsize=12, weight="bold")
                fig.tight_layout(rect=(0, 0, 1, 0.93)); pdf.savefig(fig); plt.close(fig)
                for o in oth:
                    m = next(m for m in man["sets"] if m["name"] == f"{o['sort']}_c{o['cls']}")
                    for p in raster_pages(m):
                        image_page(pdf, p, f"{unit} -- other class: {m['desc']}")
    dst = root / f"report_{u['id']}.pdf"
    shutil.copy2(pdf_path, dst)
    import subprocess
    n = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout.split("Pages:")[1].split()[0]
    print(f"{u['id']}: {n} pages -> {dst}", flush=True)


_rep = {}
def py_report(u):
    """python cluster report (MicroWireSorter make_cluster_report) of the newest sort, regenerated once per channel"""
    ch = u["ch"]
    if ch not in _rep:
        from mcws.visualization.plot_report import make_cluster_report
        import logging; logging.disable(logging.INFO)
        d = CO / f"pyreport_ch{ch}"; d.mkdir(exist_ok=True)
        if not list(d.glob("report_*.pdf")):
            res = load_spikes(str(sorted(second_pass_dir(u["run"]).glob("times_*"))[-1]))
            make_cluster_report(res, calc_metrics=True, plot_ccg=False, exclude_cluster_0=True, clusters_per_page=6, save=True, output_dir=str(d) + "/")
        _rep[ch] = [p for f in sorted(d.glob("report_*.pdf")) for p in pdf_pages(f, f"py_ch{ch}")]
    return _rep[ch]


if __name__ == "__main__":
    want = sys.argv[1:]
    for u in CFG["units"]:
        if u.get("python_class") or u.get("wc_class") or (want and u["id"] not in want):
            continue
        build(u, py_report)
