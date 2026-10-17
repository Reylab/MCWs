"""Appendix PDF for one channel: every number and figure behind the overview
(build_overview.py), for that channel. Read-only on all data; writes
<out_dir>/appendix_ch<ch>.pdf.

usage: build_pack.py <ch>
"""
import csv, glob, json, shutil, subprocess, sys, textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, SC, channels, second_pass_dir
from pvw_scorecard import scorecard, mahal_net, replay_gt
INV = OUT
ch = int(sys.argv[1]); C = channels()[ch]
root = SC / C["folder"]
out = OUT; (out / "pages").mkdir(parents=True, exist_ok=True)
PY = C["units"][0]["python_arm"]
IDS = [u["id"] for u in C["units"]]


def pv(u):
    """this unit's GT vs python validation folder"""
    return root / f"validation_gt_vs_{u['python_arm']}{u['tok']}{u.get('py_tag', '')}"


def ulab(u):
    return (f"GT class {u['gt_class']}" + (f" vs python c{u['python_class']}" if u.get("python_class") else "")
            + (f", wave_clus c{u['wc_class']}" if u.get("wc_class") else ""))
sp_dir = second_pass_dir(C["run"])
dec = json.loads((sp_dir / "decisions.json").read_text())
_ldf = INV / "ld" / f"ch{ch}" / "logdet_replay.json"
ld = json.loads(_ldf.read_text()) if _ldf.exists() else None   # log-det replay (optional; logdet_replay.py)

plt.rcParams.update({"font.family": "DejaVu Sans"})
PAGE = (11, 8.5)


# ------------------------------------------------------------------ numbers --
def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def table(ax, cells, cols, y, widths, fs=8):
    t = ax.table(cellText=cells, colLabels=cols, cellLoc="center", colWidths=widths, bbox=[0.0, y - 0.05 * (len(cells) + 1), 1.0, 0.05 * (len(cells) + 1)])
    t.auto_set_font_size(False); t.set_fontsize(fs)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
    return y - 0.05 * (len(cells) + 1) - 0.04


def text_page(pdf, title, blocks):
    """blocks: list of ('h', text) | ('p', text) | ('t', (cells, cols, widths))"""
    fig = plt.figure(figsize=PAGE); ax = fig.add_axes([0.05, 0.04, 0.9, 0.88]); ax.axis("off")
    fig.text(0.05, 0.955, title, fontsize=15, weight="bold", va="top")
    y = 1.0
    for kind, b in blocks:
        if y < 0.08:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=PAGE); ax = fig.add_axes([0.05, 0.04, 0.9, 0.88]); ax.axis("off")
            fig.text(0.05, 0.955, title + " (cont.)", fontsize=15, weight="bold", va="top"); y = 1.0
        if kind == "h":
            ax.text(0, y, b, fontsize=11, weight="bold", va="top", transform=ax.transAxes); y -= 0.045
        elif kind == "p":
            w = textwrap.fill(b, 150); n = w.count("\n") + 1
            ax.text(0, y, w, fontsize=8.5, va="top", transform=ax.transAxes, linespacing=1.4); y -= 0.028 * n + 0.015
        else:
            cells, cols, widths = b
            y = table(ax, cells, cols, y, widths)
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, png, title, caption=""):
    img = plt.imread(str(png))
    fig = plt.figure(figsize=PAGE)
    fig.text(0.04, 0.965, title, fontsize=12, weight="bold", va="top")
    top = 0.93
    if caption:
        w = textwrap.fill(caption, 170)
        fig.text(0.04, 0.935, w, fontsize=8, va="top", color="#333333")
        top = 0.92 - 0.018 * (w.count("\n") + 1)
    ax = fig.add_axes([0.02, 0.02, 0.96, top - 0.03]); ax.imshow(img); ax.axis("off")
    pdf.savefig(fig); plt.close(fig)


def pdf_pages(src, tag, first=1, last=None):
    stem = out / "pages" / tag
    cmd = ["pdftoppm", "-r", "80", "-png", "-f", str(first)] + (["-l", str(last)] if last else []) + [str(src), str(stem)]
    subprocess.run(cmd, check=True)
    return sorted((out / "pages").glob(f"{tag}-*.png"))


def find_one(pat):
    m = sorted(glob.glob(pat))
    return Path(m[0]) if m else None


def wave_clus_runs():
    """the wave_clus run(s) chosen for this channel in channels.json"""
    names = sorted({w for u in C["units"] for w in u["wave_clus"]})
    return [root / n for n in names if (root / n).is_dir()]


def finished(name, tok):
    for c in (root / name / f"validation_gt_vs_{name}{tok}", root / f"validation_gt_vs_{name}{tok}"):
        if (c / "pdf_reports" / "validation_report.pdf").exists():
            return c
    return None


def finished_py(name, tok):
    for c in (root / name / f"validation_{name}_vs_python_sd1{tok}", root / f"validation_{name}_vs_python_sd1{tok}"):
        if (c / "pdf_reports" / "validation_report.pdf").exists():
            return c
    return None


# -------------------------------------------------------------------- build --
pdf_path = out / f"appendix_ch{ch}.pdf"
rm = [r for r in read_csv(INV / "raster_metrics.csv") if r["unit"] in IDS] if (INV / "raster_metrics.csv").exists() else []
wm = [r for r in read_csv(INV / "wc_python_match.csv") if r["unit"] in IDS]
fair = {(r["wave_clus"], r["unit"]): r["fair"].lower() in ("1", "true") for r in wm}
pc = [r for r in read_csv(INV / "par_compare.csv") if r["channel"] == C["folder"] and r["same"] in ("0", "false", "False")]
with PdfPages(pdf_path) as pdf:
    # ---- scorecard: one row per candidate problem, measured on this unit, with a verdict
    for u in C["units"]:
        rows, gtn = scorecard(u)
        text_page(pdf, f"ch{ch} {ulab(u)} ({gtn:,} spikes): scorecard", [
            ("p", "Each row is one thing in the python sort that could cost or misplace the GT unit's spikes, measured on this unit. "
                  "Verdicts use the same fixed rules on every channel (pvw_scorecard.py). wave_clus run compared: "
                  + ", ".join(u["wave_clus"]) + " (the run most similar to both GT and python)."),
            ("t", ([[a, b, c] for a, b, c in rows], ["candidate problem", "measured on this unit", "verdict"], [0.33, 0.37, 0.30]))])
    # ---- summary
    blocks = [("p", f"Channel folder: {root}.  Python = {PY} "
                    + (f"(trial {CFG['trial']['name']}: {CFG['trial']['within_channel']}, Mahalanobis {CFG['trial']['mahalanobis']}; same on every channel; {CFG['trial']['chosen_because']})  " if CFG.get('trial') else "(newest second pass, sd1: rescue std 1->1, re-match std 1->3).  ")
                    + f"Sort output: {sp_dir}.  "
                    "Everything here was computed read-only from the saved sorts; no pipeline code was changed.")]
    for u in C["units"]:
        g = u["gt_class"]
        S = read_csv(pv(u) / "diagnostics" / f"diagnostics_summary_ch{ch}_gt{g}.csv")
        blocks += [("h", f"{ulab(u)}: recall / purity of the matched cluster, every sort"),
                   ("t", ([[r["sort"], r["matched_cluster"], r["cluster_size"], f"{float(r['detected_pct']):.1f}",
                            f"{float(r['recall_pct']):.1f}", f"{float(r['purity_pct']):.1f}", f"{float(r['dt_median_ms']):.3f}"] for r in S],
                          ["sort", "matched cluster", "cluster size", "detected %", "GT recovered %", "purity %", "time offset ms"],
                          [0.34, 0.11, 0.11, 0.1, 0.12, 0.1, 0.12]))]
        py = [r for r in S if r["kind"] == "python"]
        fate_cols = [k for k in S[0] if k.startswith("n_")]
        blocks += [("h", f"{ulab(u)}: where its spikes end up in python"),
                   ("t", ([[r["sort"]] + [str(int(float(r[k]))) if r[k] not in ("", "NaN") else "" for k in fate_cols] for r in py],
                          ["sort"] + [k[2:].replace("_", " ")[:22] for k in fate_cols], [0.2] + [0.8 / len(fate_cols)] * len(fate_cols)))]
        mn = mahal_net(u)
        blocks += [("p", f"Mahalanobis step in the newest python sort: {mn['moved']:,} spikes moved on the channel; for {ulab(u)} "
                         f"(python c{mn['best']}): GT spikes moved INTO it {mn['into']:,}, GT spikes moved AWAY from it {mn['away']:,}, "
                         f"non-GT spikes moved INTO it {mn['bad_in']:,}.")]
    if ld:
        lv = ld["variants"]
        cells = []
        for u in C["units"]:
            R = replay_gt(u)
            for v, nm in (("as_run", "as run (Mahalanobis)"), ("logdet", "Mahalanobis + log-det"), ("euclidean", "Euclidean only")):
                c, rec, pur = R[v]
                cells.append([nm, ulab(u), f"c{c}", f"{rec:.3f}", f"{pur:.3f}", f"{lv[v]['n_mahal']:,}"])
        blocks += [("h", "Second-pass act step replayed three ways (same input sort and seed; as-run reproduces the saved sort "
                         f"{lv['as_run']['labels_equal_saved']:.0%})"),
                   ("t", (cells, ["variant", "unit", "python cluster", "GT recovered", "purity", "Mahalanobis moves"],
                          [0.24, 0.24, 0.12, 0.13, 0.12, 0.15]))]
    blocks += [("p", "Second-pass decisions: " + ", ".join(f"c{c['cluster']} {c['decision']} (n={c['n']:,}, ISI {c['isi_pct']:.2f}%)" for c in dec["clusters"]))]
    blocks += [("h", "Second pass, per cluster (before -> after)"),
               ("t", ([[f"c{r['cluster']}", r["origin"], f"{r['n_before']:,} -> {r['n_after']:,}",
                        (f"{r['isi_before']:.2f}" if r["isi_before"] is not None else "-") + " -> " + (f"{r['isi_after']:.2f}" if r["isi_after"] is not None else "-"),
                        f"{r['cores']:,}", f"{r['forced']:,}", f"{r['rescued']:,}", f"{r['mahal']:,}"] for r in dec["plan"]["table"]],
                      ["cluster", "from", "spikes", "ISI<3ms %", "cores", "matched", "rescued", "Mahal moved"],
                      [0.08, 0.14, 0.18, 0.16, 0.11, 0.11, 0.11, 0.11]))]
    if wm:
        blocks += [("h", f"wave_clus runs vs python: each side's cluster closest to GT; fair when they overlap >= {CFG['fair_min']} both ways"),
                   ("t", ([[r["unit"], r["wave_clus"], f"c{r['wc_class']} ({int(r['wc_size']):,})",
                            f"c{r['python_class']} ({int(r['python_size']):,})",
                            f"{float(r['recall']):.2f}", f"{float(r['precision']):.2f}",
                            "FAIR" if r["fair"].lower() in ("1", "true") else "low match"] for r in wm],
                          ["unit", "wave_clus run", "wave_clus GT cluster", "python cluster", "wc spikes in python", "python spikes in wc", "comparison"],
                          [0.13, 0.13, 0.15, 0.15, 0.15, 0.15, 0.14]))]
    if rm:
        blocks += [("h", "Raster numbers per sort (z-score is not comparable between sorts: near-zero baselines inflate it)"),
                   ("t", ([[r["unit"], r["sort"], r["n_stim"], f"{float(r['mean_zscore']):.2f}", f"{float(r['mean_n_response']):.1f}",
                            f"{float(r['mean_median_post']):.2f}", f"{float(r['mu_base']):.3f}", r["n_responsive"], f"{float(r['median_sign_p']):.2g}",
                            r["cluster_total"]] for r in rm],
                          ["unit", "sort", "stimuli", "mean z", "spikes in window", "median post", "baseline mu", "responsive", "sign p (median)", "cluster size"],
                          [0.12, 0.16, 0.06, 0.07, 0.1, 0.1, 0.1, 0.09, 0.11, 0.09]))]
    if pc:
        seen = {}
        for r in pc:
            if r["param"] != "detection_date":
                seen.setdefault(r["param"], set()).add(f"{r['wave_clus']} vs {r['python']}")
        blocks += [("h", "Parameters that differ (wave_clus vs python), shared names only; all other shared parameters are equal"),
                   ("p", "; ".join(f"{k}: {' | '.join(sorted(v))}" for k, v in seen.items())
                         + ".  Note: wave_clus randomseed 0 means a new random seed each run, so neither sorter is fixed.")]
    text_page(pdf, f"ch{ch}: python vs wave_clus -- numbers", blocks)

    # ---- count-free cluster quality (trial_eval.py): python as run, tuned trial, wave_clus, GT
    tm = read_csv(CFG["trial_metrics"]) if CFG.get("trial_metrics") and Path(CFG["trial_metrics"]).exists() else []
    tm = [r for r in tm if r["unit"] in IDS and (r["sort"] in ("python as_run", "python width_logdet", "GT") or not r["sort"].startswith("python"))]
    if tm:
        lbl = {"python as_run": "python as run", "python width_logdet": "python tuned (width 25 + log-det)"}
        text_page(pdf, f"ch{ch}: cluster quality (does not grow with spike count)", [
            ("p", "Each unit's matched cluster in each sort. Recovered / purity / F1 against GT; ISI violations; SNR; d', L-ratio and "
                  "isolation distance in the same feature space for every sort (10 principal components of that sort's waveforms). "
                  "Lower ISI and L-ratio are better; higher everything else."),
            ("t", ([[r["unit"], lbl.get(r["sort"], r["sort"]), f"c{r['cluster']}", f"{int(r['n']):,}", f"{100*float(r['recovered']):.1f}",
                     f"{100*float(r['purity']):.1f}", f"{float(r['f1']):.3f}", f"{float(r['isi_pct']):.2f}", f"{float(r['snr']):.2f}",
                     f"{float(r['d_prime']):.2f}", f"{float(r['l_ratio']):.3f}", f"{float(r['iso_dist']):.0f}"] for r in tm],
                   ["unit", "sort", "cluster", "spikes", "recovered %", "purity %", "F1", "ISI<3ms %", "SNR", "d'", "L-ratio", "iso dist"],
                   [0.09, 0.2, 0.05, 0.07, 0.07, 0.07, 0.06, 0.07, 0.06, 0.06, 0.07, 0.07]))])

    # ---- GT: the hand-selected clusters themselves
    gp = OUT / "gt" / f"gt_clusters_ch{ch}.png"
    if gp.exists():
        image_page(pdf, gp, f"ch{ch} GT (hand selected): every class's waveforms and ISIs",
                   "The hand-selected sort the comparisons are scored against; the GT unit(s) analysed are marked.")
    for p in sorted((root / "hand selected" / "metrics").glob("*_plots*.png")):
        image_page(pdf, p, f"ch{ch} GT (hand selected): cluster report ({p.name})")

    # ---- diagnostics + log-det
    for u in C["units"]:
        g = u["gt_class"]
        dg = pv(u) / "diagnostics"
        for f, cap in (("recall_purity", "Matched cluster per sort: share of the GT unit it holds, and share of it that is GT. Big wave_clus clusters win on recall by merging."),
                       ("gt_fate", "Every GT spike in the python sorts (before / after the second pass): matched cluster, other cluster, quarantined by reason, unassigned, not detected."),
                       ("lost_waveforms", "Python's own waveforms of the GT spikes each quarantine rule threw out (and did not rescue), against the GT spikes it kept (dashed)."),
                       ("second_pass", "GT spikes' cluster before -> after the second pass, how the movers were placed, cluster spread (the acceptance radius scales with it)."),
                       ("time_offset", "Nearest-spike time minus GT time; python sits 1 sample (0.033 ms) early on every channel.")):
            p = dg / f"diag_ch{ch}_gt{g}_{f}.png"
            if p.exists():
                image_page(pdf, p, f"{ulab(u)}: {f.replace('_', ' ')}", cap)
    if (INV / "ld" / f"ch{ch}" / f"logdet_ch{ch}.png").exists():
      image_page(pdf, INV / "ld" / f"ch{ch}" / f"logdet_ch{ch}.png", "Mahalanobis log-determinant replay",
               "The second-pass act step (re-match + rescue) replayed from the same input and seed: as run, with log|Sigma| added to the "
               "Mahalanobis comparison (a Gaussian likelihood), and with no Mahalanobis step. Cluster sizes, ISI violations, GT recovery and purity.")

    # ---- second pass outputs
    for k, png in enumerate(pdf_pages(sp_dir / "before_after.pdf", "before_after")):
        image_page(pdf, png, f"Second pass: before -> after ({k + 1})", "From before_after.pdf (temperature trees before/after, then each cluster before -> after: spikes + template, density, ISI).")
    if (sp_dir / "second_pass_report.pdf").exists():     # trial sorts (trial_replay.py) save before_after.pdf only
        n_rep = int(subprocess.run(["pdfinfo", str(sp_dir / "second_pass_report.pdf")], capture_output=True, text=True).stdout.split("Pages:")[1].split()[0])
        for png in pdf_pages(sp_dir / "second_pass_report.pdf", "sp_report", first=1, last=1) + pdf_pages(sp_dir / "second_pass_report.pdf", "sp_after", first=n_rep, last=n_rep):
            image_page(pdf, png, "Second pass report: " + ("summary" if "sp_report" in png.name else "after acting"))
    rep = find_one(str(sp_dir / "times_*" / "metrics" / "report_*.pdf"))
    if rep:
        for png in pdf_pages(rep, "py_metrics", first=1, last=3):
            image_page(pdf, png, f"Python sort ({PY}): cluster metrics report", "MicroWireSorter metrics report of this python sort.")

    # ---- GT vs newest python validation
    for u in C["units"]:
        V = pv(u)
        items = [("class_match_ch*_GT_vs_python_sd1.png", "class match: GT -> python"), ("class_match_ch*_python_sd1_vs_GT.png", "class match: python -> GT"),
                 ("venn_three_set/venn3_*_detections.png", "venn: all detections"), ("venn_three_set/venn3_*_cluster.png", "venn: GT class vs matched cluster"),
                 ("venn_three_set/venn3_*_full.png", "venn: all clustered spikes"), ("cluster_composition/composition_ch*.png", "cluster composition"),
                 ("cluster_composition/composition_mahal_*.png", "cluster composition: Mahalanobis"), ("pie_venn/pie_venn_*.png", "pie / venn"),
                 ("waveform_breakdowns/*not_rescued_by_property.png", "waveforms: quarantined, not rescued, by property"),
                 ("waveform_breakdowns/*_rescued_by_property.png", "waveforms: rescued, by property"),
                 ("response_waveforms/*.png", "response-window waveforms"),
                 ("stimulus_rasters/gt/*fig 1.png", "rasters: GT"), ("stimulus_rasters/python_sd1/*fig 1.png", "rasters: python 20261012"),
                 ("stimulus_rasters/python_sd1/*gt_vs_arm_summary.png", "GT vs python per-stimulus summary"),
                 ("stimulus_rasters/python_sd1/*sign_test.png", "per-stimulus sign test")]
        for pat, t in items:
            for p in sorted(V.glob(pat))[:1]:
                image_page(pdf, p, f"{ulab(u)} -- GT vs python: {t}")

    # ---- wave_clus runs
    shown = set()
    for w in wave_clus_runs():
        figs = sorted(w.glob("fig2print_*.png"))
        for p in figs:
            image_page(pdf, p, f"{w.name}: wave_clus output ({p.name})")
        wf = OUT / "wc" / f"wc_forced_{w.name}_ch{ch}.png"
        if wf.exists():
            image_page(pdf, wf, f"{w.name}: clustered by SPC (forced = 0) vs template-forced (forced = 1)",
                       "Each wave_clus cluster's spikes split by how they got there: SPC clustering, or wave_clus's template forcing "
                       "of the spikes SPC left out. Right column: both means +/- sd. The cluster matched to GT is marked.")
        for u in C["units"]:
            if w.name not in u["wave_clus"]:
                continue
            g = u["gt_class"]
            tagf = "FAIR comparison" if fair.get((w.name, u["id"])) else "low match with python -- not a fair comparison"
            v = finished(w.name, u["tok"])
            if v and v not in shown:          # GT vs wave_clus does not depend on the python class
                shown.add(v)
                for pat, t in (("stimulus_rasters/*/*fig 1.png", "rasters"), ("stimulus_rasters/*/*gt_vs_arm_summary.png", "GT vs wave_clus summary"),
                               ("venn_three_set/venn3_*_cluster.png", "venn: GT class vs matched cluster")):
                    for p in sorted(x for x in v.glob(pat) if "/gt/" not in str(x))[:1]:
                        image_page(pdf, p, f"{w.name} GT class {g}: {t}")
            vp = finished_py(w.name, u["tok"] + u.get("py_tag", "") + u.get("wc_tag", ""))
            if vp:
                for pat, t in (("class_match_*.png", "class match"),
                               ("stimulus_rasters/python_sd1/*fig 1.png", "python rasters"),
                               ("stimulus_rasters/python_sd1/*gt_vs_arm_summary.png", "wave_clus vs python summary"),
                               ("stimulus_rasters/python_sd1/*sign_test.png", "per-stimulus sign test"),
                               ("cluster_composition/composition_*.png", "python cluster composition"),
                               ("waveform_breakdowns/*not_rescued_by_property.png", "python: quarantined, not rescued"),
                               ("venn_three_set/venn3_*_cluster.png", "venn: wave_clus cluster vs python cluster"),
                               ("venn_three_set/venn3_*_detections.png", "venn: wave_clus vs python detections")):
                    for p in sorted(vp.glob(pat))[:2 if pat.startswith("class_match") else 1]:
                        image_page(pdf, p, f"{w.name} vs python c{mahal_net(u)['best']} ({ulab(u)}): {t} [{tagf}]")

shutil.rmtree(out / "pages")
n = int(subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout.split("Pages:")[1].split()[0])
print(f"{pdf_path}  ({n} pages)")
