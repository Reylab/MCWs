"""Cross-channel overview of the python-vs-wave_clus analysis, organised by
question (each question answered on every channel side by side):

  1  Headline per GT unit
  2  Where does python lose GT spikes?
  3  Is the within-channel rejection too tight?
  4  Mahalanobis and the log-determinant
  5  The second pass
  6  wave_clus vs python (fair pairs only)
  7  Rasters
  8  Spike times and parameters

Everything is read from the outputs of the earlier steps (README.md); nothing
is recomputed except the Mahalanobis net counts. -> <out_dir>/overview.pdf
"""
import csv, json, sys, textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, SC, channels, second_pass_dir
from pvw_scorecard import mahal_net, replay_gt

PAGE = (11, 8.5)
PY = CFG["python_arm"]
U = CFG["units"]
plt.rcParams.update({"font.family": "DejaVu Sans"})


def read_csv(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def unit_name(u):
    return (f"ch{u['ch']} GT{u['gt_class']}" + (f" vs python c{u['python_class']}" if u.get("python_class") else "")
            + (f", wave_clus c{u['wc_class']}" if u.get("wc_class") else ""))


def diag_dir(u):
    return SC / u["folder"] / f"validation_gt_vs_{u['python_arm']}{u['tok']}{u.get('py_tag', '')}" / "diagnostics"


def diag_summary(u):
    return read_csv(diag_dir(u) / f"diagnostics_summary_ch{u['ch']}_gt{u['gt_class']}.csv")


# ------------------------------------------------------------- page helpers --
class Doc:
    def __init__(self, pdf):
        self.pdf = pdf; self.page = 0

    def _new(self, title):
        fig = plt.figure(figsize=PAGE)
        fig.text(0.05, 0.955, title, fontsize=15, weight="bold", va="top")
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.86]); ax.axis("off")
        return fig, ax

    def _save(self, fig):
        self.page += 1
        fig.text(0.95, 0.015, str(self.page), ha="right", fontsize=8, color="#555")
        fig.text(0.05, 0.015, "python vs wave_clus analysis -- overview", fontsize=7.5, color="#555")
        self.pdf.savefig(fig); plt.close(fig)

    def text(self, title, blocks):
        fig, ax = self._new(title); y = 1.0
        for kind, b in blocks:
            need = 0.05 if kind != "t" else 0.05 * (len(b[0]) + 1) + 0.04
            if y - need < 0.02:
                self._save(fig); fig, ax = self._new(title + " (cont.)"); y = 1.0
            if kind == "h":
                ax.text(0, y, b, fontsize=11, weight="bold", va="top", transform=ax.transAxes); y -= 0.045
            elif kind == "p":
                w = textwrap.fill(b, 150)
                ax.text(0, y, w, fontsize=8.8, va="top", transform=ax.transAxes, linespacing=1.4)
                y -= 0.029 * (w.count("\n") + 1) + 0.015
            else:
                cells, cols, widths = b
                h = 0.05 * (len(cells) + 1)
                t = ax.table(cellText=cells, colLabels=cols, cellLoc="center", colWidths=widths, bbox=[0, y - h, 1, h])
                t.auto_set_font_size(False); t.set_fontsize(8)
                for (r, c), cell in t.get_celld().items():
                    if r == 0:
                        cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
                y -= h + 0.04
        self._save(fig)

    def images(self, title, items, caption="", ncols=1):
        """items: [(png, label)] -- up to 2 per page (stacked) or a 2x2 grid with ncols=2."""
        per = 2 if ncols == 1 else 4
        items = [(p, l) for p, l in items if p and Path(p).exists()]
        for i in range(0, len(items), per):
            fig = plt.figure(figsize=PAGE)
            fig.text(0.04, 0.965, title, fontsize=13, weight="bold", va="top")
            top = 0.93
            if caption:
                w = textwrap.fill(caption, 175)
                fig.text(0.04, 0.937, w, fontsize=8, va="top", color="#333")
                top = 0.925 - 0.017 * (w.count("\n") + 1)
            chunk = items[i:i + per]
            nrow = len(chunk) if ncols == 1 else int(np.ceil(len(chunk) / 2))
            h = (top - 0.04) / max(nrow, 1)
            for k, (p, l) in enumerate(chunk):
                r, c = (k, 0) if ncols == 1 else divmod(k, 2)
                w_ = 0.94 if ncols == 1 else 0.465
                ax = fig.add_axes([0.03 + c * 0.485, top - (r + 1) * h + 0.01, w_, h - 0.035])
                ax.imshow(plt.imread(str(p))); ax.axis("off")
                ax.set_title(l, fontsize=9, weight="bold", loc="left")
            self._save(fig)


# --------------------------------------------------------------------- data --
wm = read_csv(OUT / "wc_python_match.csv")
rm = read_csv(OUT / "raster_metrics.csv")
cov = read_csv(OUT / "covariance_check.csv")
pc = read_csv(OUT / "par_compare.csv")
ld = {ch: json.loads((OUT / "ld" / f"ch{ch}" / "logdet_replay.json").read_text())
      for ch in channels() if (OUT / "ld" / f"ch{ch}" / "logdet_replay.json").exists()}
S = {unit_name(u): diag_summary(u) for u in U}
MN = {unit_name(u): mahal_net(u) for u in U}


def sort_row(u, kind):
    rows = S[unit_name(u)]
    name = u["python_arm"] if kind == "after" else u["python_before"]
    r = [x for x in rows if x["sort"] == name]
    return r[0] if r else None


def fair_runs(u):
    return [r for r in wm if r["unit"] == u["id"] and r["fair"].lower() in ("1", "true")]


# the 9 GT-fate count columns follow the 8 sort columns, in python_loss_figures's order:
# matched, other cluster, within-ch second peak, within-ch low amplitude,
# within-ch width, within-ch other, refractory, bundle, unassigned, not detected
FATE = {"matched": 0, "otherc": 1, "second": 2, "low": 3, "width": 4, "other": 5, "refr": 6, "bundle": 7}


def fate_key(u, short):
    r = sort_row(u, "after")
    if not r:
        return 0
    v = list(r.values())[8 + FATE[short]]
    return int(float(v)) if v not in ("", "NaN") else 0


# -------------------------------------------------------------------- build --
pdf_path = OUT / "overview.pdf"
with PdfPages(pdf_path) as pdf:
    D = Doc(pdf)
    D.text("python vs wave_clus -- overview", [
        ("p", ("Python sort = the tuned trial (same parameters on every channel): " if CFG.get("trial") else "Python sort = each channel's newest second pass (rescue sd1): ") + ", ".join(sorted({u['python_arm'] for u in U})) + ".  GT units: " +
              ", ".join(f"{unit_name(u)} ({u['folder']})" for u in U) + "."),
        ("p", "Every comparison uses, on each side, the cluster closest to the GT unit (0.5 ms spike matching): python's GT cluster, "
              "each wave_clus run's GT cluster.  A wave_clus run is a FAIR comparison with python only when those two clusters "
              f"overlap >= {CFG['fair_min']:.0%} both ways; low-match runs (wave_clus merged or split the unit differently) are listed "
              "but not used for head-to-head numbers."),
        ("p", "Read in order: 1 headline, 2-5 why python loses or moves spikes (quarantine, Mahalanobis, second pass), "
              "6-7 the fair wave_clus comparisons and rasters, 8 timing / parameters.  Per-channel appendices "
              "(appendix_ch<ch>.pdf, same folder) hold every figure for one channel; README.md maps every file."),
        ("p", "Nothing in either sorter was changed; the log-determinant and Euclidean variants are offline replays of the second "
              "pass's act step from the same input and seed (the as-run replay reproduces the saved sort exactly).")])

    # 0 tuned trial: what changed, why, and count-free quality (only for PVW_CONFIG=channels_tuned.json)
    if CFG.get("trial") and Path(CFG.get("trial_metrics", "")).exists():
        T = CFG["trial"]; ids = {u["id"] for u in U}
        M = {}
        for r in read_csv(CFG["trial_metrics"]):
            if r["unit"] in ids:
                M.setdefault(r["unit"], {})["as" if r["sort"] == "python as_run" else "tu" if r["sort"] == f"python {T['name']}"
                                         else "gt" if r["sort"] == "GT" else "wc" if not r["sort"].startswith("python") else r["sort"]] = r
        unfl = {}
        for u in U:
            jp = sorted((Path(CFG["recluster_runs"]) / u["run"] / CFG["second_pass_subdir"]).glob("second_pass_*/*/trial_params.json"))
            if jp: unfl[u["ch"]] = json.loads(jp[-1].read_text())["unflagged"]
        f = lambda r, k, sc=1, fmt="{:.1f}": fmt.format(sc * float(r[k]))
        cells, wins = [], {"F1": 0, "purity": 0, "ISI": 0, "SNR": 0, "dprime": 0, "L-ratio": 0}
        for u in U:
            m = M.get(u["id"]);
            if not m or not {"as", "tu", "wc"} <= set(m): continue
            a_, t_, w_ = m["as"], m["tu"], m["wc"]
            wins["F1"] += float(t_["f1"]) > float(w_["f1"]); wins["purity"] += float(t_["purity"]) > float(w_["purity"])
            wins["ISI"] += float(t_["isi_pct"]) < float(w_["isi_pct"]); wins["SNR"] += float(t_["snr"]) > float(w_["snr"])
            wins["dprime"] += float(t_["d_prime"]) > float(w_["d_prime"]); wins["L-ratio"] += float(t_["l_ratio"]) < float(w_["l_ratio"])
            cells.append([unit_name(u), w_["sort"], f"{float(a_['f1']):.3f} -> {float(t_['f1']):.3f}", f"{float(w_['f1']):.3f}",
                          f"{f(t_,'recovered',100)} / {f(t_,'purity',100)}", f"{f(w_,'recovered',100)} / {f(w_,'purity',100)}",
                          f"{f(t_,'isi_pct',1,'{:.2f}')} / {f(w_,'isi_pct',1,'{:.2f}')}", f"{f(t_,'snr',1,'{:.1f}')} / {f(w_,'snr',1,'{:.1f}')}",
                          f"{f(t_,'d_prime',1,'{:.2f}')} / {f(w_,'d_prime',1,'{:.2f}')}", f"{unfl.get(u['ch'], 0):,}"])
        n = len(cells)
        D.text(f"0  Tuned python ({T['name']}): what changed and why it is better than wave_clus", [
            ("p", f"ONE parameter set, identical on every channel: within-channel waveform rule {T['within_channel']} "
                  f"(only change from the sorter's config: width upper limit 15 -> 25 samples); Mahalanobis reassignment with the "
                  f"log-determinant added (full Gaussian likelihood instead of distance only).  Chosen from 6 trials because it had "
                  f"{T['chosen_because']}.  The sorter's code was not changed: the trial is an offline replay of the second pass from "
                  f"the same input and seed, saved as {CFG['units'][0]['python_arm']}."),
            ("p", "Why each change helps.  (1) WIDTH 15 -> 25: the within-channel rule rejected real spikes whose waveform is "
                  "15-25 samples wide (ch328: 12.8% of the GT unit).  Those spikes are not artifacts: they match the GT waveform, "
                  "and putting them back raised ch328 GT recovery 81 -> 92% with purity unchanged.  (2) LOG-DETERMINANT: plain "
                  "Mahalanobis distance favours BROAD clusters (a wide covariance makes every spike look 'close'), so after a split "
                  "it pulled GT spikes into the broad neighbour (ch333: GT recovery 52%).  Adding log|Sigma| charges broad clusters "
                  "for their spread, which is the actual Gaussian likelihood; ch333 recovery 52 -> 87%.  On channels without that "
                  "problem both changes are neutral (worst change in F1 -0.012, ch368)."),
            ("p", "Why spike counts are not used here: rasters and response scores grow with cluster size, so a bigger (merged) "
                  "wave_clus cluster scores higher without being a better unit.  The columns below do not grow with size.  "
                  "'tuned / wave_clus' pairs; GT recovered % / purity % from 0.5 ms spike matching; ISI = % intervals < 3 ms "
                  "(lower = cleaner); SNR = peak / noise of the cluster's waveforms; d' = LDA separation of the cluster from the rest "
                  "of its own sort (10 PCs)."),
            ("t", (cells, ["GT unit", "wave_clus run", "F1 py run->tuned", "F1 wc", "rec/pur tuned", "rec/pur wc",
                           "ISI% tuned/wc", "SNR tuned/wc", "d' tuned/wc", "un-flagged"],
                   [0.08, 0.17, 0.12, 0.06, 0.09, 0.09, 0.1, 0.09, 0.1, 0.08])),
            ("p", "un-flagged = spikes on the channel that the width limit 15 -> 25 took out of quarantine (they then go through the "
                  "normal rescue / re-match like any other spike)."),
            ("p", f"Tuned python better than wave_clus in {wins['F1']}/{n} units on F1, {wins['purity']}/{n} on purity, "
                  f"{wins['ISI']}/{n} on ISI violations, {wins['SNR']}/{n} on SNR, {wins['dprime']}/{n} on d', {wins['L-ratio']}/{n} on L-ratio.  "
                  "d', L-ratio and isolation distance measure a cluster against the OTHER clusters of its own sort, so they change with "
                  "how many clusters a sort has and how they are spread; they are shown but are not a like-for-like comparison.  "
                  "GT recovery, purity and ISI are measured the same way for both sorts."),
        ])

    # 1 headline
    cells = []
    for u in U:
        b, a = sort_row(u, "before"), sort_row(u, "after")
        fr = fair_runs(u)
        best = max(fr, key=lambda r: float(r["recall"]) + float(r["precision"])) if fr else None
        wrow = [x for x in S[unit_name(u)] if best and x["sort"] == best["wave_clus"]]
        R = replay_gt(u)
        cells.append([unit_name(u),
                      f"{float(b['recall_pct']):.0f} / {float(b['purity_pct']):.0f}" if b else "-",
                      f"{float(a['recall_pct']):.0f} / {float(a['purity_pct']):.0f}" if a else "-",
                      f"{100*R['logdet'][1]:.0f} / {100*R['logdet'][2]:.0f}" if R else "-",
                      (f"{best['wave_clus']}: {float(wrow[0]['recall_pct']):.0f} / {float(wrow[0]['purity_pct']):.0f}" if wrow else "none fair"),
                      f"{fate_key(u,'second')+fate_key(u,'low')+fate_key(u,'width')+fate_key(u,'other')+fate_key(u,'refr'):,}",
                      f"{MN[unit_name(u)]['away']:,} / {MN[unit_name(u)]['bad_in']:,}"])
    D.text("1  Headline per GT unit", [
        ("p", ("GT recovered % / purity % of the GT-matched cluster.  'python as run' = the newest second pass as the sorter ran it; 'python tuned' = the trial; " if CFG.get("trial") else "GT recovered % / purity % of the GT-matched cluster.  'python Oct 7' = before the second pass; 'python newest' = after; ") +
              "'+ log-det' = newest replayed with the log-determinant.  Best fair wave_clus = highest recall+purity among fair runs."),
        ("t", (cells, ["GT unit", CFG.get("before_name", "python Oct 7"), CFG.get("after_name", "python newest"), "+ log-det", "best fair wave_clus",
                       "lost to quarantine", "Mahal away / in"],
               [0.1, 0.12, 0.13, 0.12, 0.2, 0.16, 0.17])),
        ("p", "'lost to quarantine' = GT spikes quarantined and not rescued in the newest sort.  'Mahal away / in' = GT spikes the "
              "Mahalanobis step moved out of the GT cluster / non-GT spikes it moved into it.")])

    # 2 where python loses spikes
    D.images("2  Where does python lose GT spikes?",
             [(diag_dir(u) / f"diag_ch{u['ch']}_gt{u['gt_class']}_gt_fate.png", unit_name(u)) for u in U],
             "Every GT spike in the python sorts before / after the second pass: matched cluster, other cluster, quarantined by "
             "reason (within-channel second peak, low amplitude, other; refractory; bundle), unassigned, not detected.")

    # 3 within-channel
    cells = [[unit_name(u), f"{fate_key(u,'second'):,}", f"{fate_key(u,'low'):,}", f"{fate_key(u,'width'):,}", f"{fate_key(u,'other'):,}", f"{fate_key(u,'refr'):,}"] for u in U]
    D.text("3  Is the within-channel rejection too tight?", [
        ("p", "GT spikes python quarantined and did not rescue, by rule (newest sort).  Second-peak rule: prom_ratio 2.0 "
              "(config.yaml artifact_rejection); low-amplitude rule: low_amp_percentile 5 (bottom 5 % of every channel).  "
              "Rescue of these is std 1->1."),
        ("t", (cells, ["GT unit", "second peak (prom < 2)", "low amplitude (bottom 5%)", "width outside 3-15", "within-ch other", "refractory"],
               [0.15, 0.18, 0.18, 0.17, 0.16, 0.16]))])
    D.images("3  Waveforms of the GT spikes each rule threw out",
             [(diag_dir(u) / f"diag_ch{u['ch']}_gt{u['gt_class']}_lost_waveforms.png", unit_name(u)) for u in U],
             "Python's own waveforms of the rejected GT spikes, against the GT spikes it kept (dashed mean).")

    # 4 Mahalanobis / log-det
    cells = []
    for u in U:
        R = replay_gt(u); V = ld.get(u["ch"], {}).get("variants", {})
        if not R:
            continue
        cells.append([unit_name(u)] + [f"c{R[v][0]}: {R[v][1]:.3f} / {R[v][2]:.3f}" for v in ("as_run", "logdet", "euclidean")]
                     + [f"{V['as_run']['n_mahal']:,} / {V['logdet']['n_mahal']:,}"])
    D.text("4  Mahalanobis" + (" and the log-determinant" if cells else ""), ([
        ("p", "Mahalanobis compares squared distances in each cluster's own spread; without log|Sigma| broad clusters look close to "
              "everything.  Adding log|Sigma| makes it a Gaussian likelihood.  Replay of the second pass's act step (re-match std 1->3 "
              "+ rescue) on each channel, same input and seed."),
        ("t", (cells, ["GT unit", "as run: recovered / purity", "+ log-det", "Euclidean only", "Mahal moves as run / log-det"],
               [0.2, 0.2, 0.2, 0.2, 0.2]))] if cells else []) + [
        ("h", "Mahalanobis moves relative to the GT unit (newest sort, as run)"),
        ("t", ([[k, f"{v['moved']:,}", f"{v['into']:,}", f"{v['away']:,}", f"{v['bad_in']:,}"] for k, v in MN.items()],
               ["GT unit", "moved on channel", "GT moved into GT cluster", "GT moved away", "non-GT moved into GT cluster"],
               [0.16, 0.18, 0.22, 0.2, 0.24])),
        ("h", "Covariance estimate per cluster (newest sort): cores per feature, Gaussian check"),
        ("t", ([[f"ch{r['ch']} c{r['cluster']}", r["features"], r["cores"], r["cores_per_feature"], f"{r['median_d2']} (chi2 {r['chi2_median']})",
                 r["pct_beyond_chi2_99"], r["log_det_sigma"]] for r in cov],
               ["cluster", "features", "SPC cores", "cores / feature", "median d2 of cores", "% beyond chi2 99% (Gaussian 1%)", "log|Sigma|"],
               [0.12, 0.1, 0.12, 0.14, 0.2, 0.2, 0.12])),
        ("p", "Cores per feature below ~10 = covariance poorly estimated (log|Sigma| depends on the smallest variances).")])
    if ld:
        D.images("4  Log-determinant replay, every channel",
                 [(OUT / "ld" / f"ch{ch}" / f"logdet_ch{ch}.png", f"ch{ch}") for ch in ld], ncols=1)

    # 5 second pass
    cells = []
    for ch, C in channels().items():
        dec = json.loads((second_pass_dir(C["run"]) / "decisions.json").read_text())
        cells.append([f"ch{ch}", ", ".join(f"c{c['cluster']} {c['decision']}" for c in dec["clusters"]),
                      f"{dec['plan']['n_changed']:,}", f"{dec['plan']['n_mahal']:,}"])
    D.text("5  The second pass", [
        ("p", "Decisions of the newest second pass per channel; only a SPLIT creates a new cluster, but the re-match of every non-core "
              "spike runs on every channel."),
        ("t", (cells, ["channel", "decisions", "spikes that changed cluster", "Mahalanobis moves"], [0.1, 0.5, 0.2, 0.2]))])
    D.images("5  GT spikes before -> after the second pass",
             [(diag_dir(u) / f"diag_ch{u['ch']}_gt{u['gt_class']}_second_pass.png", unit_name(u)) for u in U])

    # 6 wave_clus vs python
    cells = [[f"ch{r['ch']} GT{r['gt_class']}", r["wave_clus"], f"c{r['wc_class']} ({int(r['wc_size']):,})",
              f"c{r['python_class']} ({int(r['python_size']):,})", f"{float(r['recall']):.2f}", f"{float(r['precision']):.2f}",
              "FAIR" if r["fair"].lower() in ("1", "true") else "low match"] for r in wm]
    D.text("6  wave_clus vs python: which comparisons are fair", [
        ("p", f"Each side's GT cluster; recall = share of the wave_clus GT cluster found in python's, precision = the reverse.  "
              f"Fair = both >= {CFG['fair_min']}."),
        ("t", (cells, ["GT unit", "wave_clus run", "wave_clus GT cluster", "python GT cluster", "recall", "precision", "comparison"],
               [0.11, 0.14, 0.17, 0.17, 0.11, 0.12, 0.14]))])
    for u in U:
        runs = fair_runs(u)
        tag = "fair pair"
        if not runs:   # no fair run: still compare response to response, with the closest wave_clus run
            cand = [r for r in wm if r["unit"] == u["id"]]
            runs = [max(cand, key=lambda r: float(r["recall"]) + float(r["precision"]))] if cand else []
            tag = "LOW MATCH (closest wave_clus run)"
        for r in runs:
            w = r["wave_clus"]; root = SC / u["folder"]
            pv = root / f"validation_gt_vs_{u['python_arm']}{u['tok']}" / "stimulus_rasters"
            wv = root / w / f"validation_gt_vs_{w}{u['tok']}" / "stimulus_rasters" / w
            g1 = sorted((pv / "gt").glob("*fig 1.png")); p1 = sorted((pv / "python_sd1").glob("*fig 1.png"))
            w1 = sorted(wv.glob("*fig 1.png"))
            D.images(f"6  {unit_name(u)}: rasters, {w} vs python -- {tag}",
                     [(g1[0] if g1 else None, f"GT class {u['gt_class']}"),
                      (p1[0] if p1 else None, f"python c{r['python_class']} (GT-matched)"),
                      (w1[0] if w1 else None, f"{w} c{r['wc_class']} (matched to python's GT cluster)")],
                     f"wave_clus c{r['wc_class']} holds {float(r['precision']):.0%} of python c{r['python_class']}'s spikes; "
                     f"python c{r['python_class']} holds {float(r['recall']):.0%} of the wave_clus cluster's spikes.")

    # 7 rasters
    keep = []
    for r in rm:
        u = next(x for x in U if x["id"] == r["unit"])
        fr = {x["wave_clus"] for x in fair_runs(u)}
        if r["sort"] in ("GT", "python") or r["sort"] in u["wave_clus"]:
            name = r["sort"] if r["sort"] in ("GT", "python") or r["sort"] in fr else r["sort"] + " (low match)"
            keep.append([unit_name(u), name, f"{float(r['mean_n_response']):.1f}", r["n_responsive"] + "/" + r["n_stim"],
                         f"{float(r['median_sign_p']):.2g}", f"{float(r['mu_base']):.3f}", f"{float(r['mean_zscore']):.1f}", r["cluster_total"]])
    D.text("7  Rasters (GT, python, the wave_clus run)", [
        ("p", "Spikes in the response window and responsive stimuli are comparable across sorts; the z-score is not "
              "(a near-zero baseline -- the per-stimulus MEDIAN baseline count -- inflates it for sparse clusters)."),
        ("t", (keep, ["GT unit", "sort", "spikes in window", "responsive", "sign p (median)", "baseline", "mean z", "cluster size"],
               [0.12, 0.16, 0.13, 0.11, 0.13, 0.11, 0.1, 0.14]))])

    # 8 timing / parameters
    diff = {}
    for r in pc:
        if r["same"] in ("0", "false", "False") and r["param"] != "detection_date":
            diff.setdefault(r["param"], set()).add(f"{r['wave_clus']} vs {r['python']}")
    D.text("8  Spike times and parameters", [
        ("p", "Python spike times are 1 sample (0.033 ms) earlier than GT and wave_clus on every channel (time_offset figures in the "
              "appendices) -- inside the 0.5 ms tolerance."),
        ("h", "Shared parameters that differ (all others equal)"),
        ("t", ([[k, " | ".join(sorted(v))[:120]] for k, v in diff.items()], ["parameter", "wave_clus vs python"], [0.2, 0.8])),
        ("p", "wave_clus randomseed 0 = a new random seed each run, so neither sorter is seeded the same way twice.")])
print(f"{pdf_path}")
