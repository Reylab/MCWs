"""Every run and result of the analysis, per GT unit, with the time its main
output file was written (folder dates in a file browser do not update when
something is added deeper down).  -> <out_dir>/where_everything_is.txt"""
import glob, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import CFG, OUT, SC, sp_root


def t(p):
    return time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))) if os.path.exists(p) else "MISSING"


def rep(d):
    return t(os.path.join(d, "pdf_reports", "validation_report.pdf"))


lines = [f"written {time.strftime('%Y-%m-%d %H:%M')}; time = when that item's main output file was written", ""]
for u in CFG["units"]:
    root = f"{SC}/{u['folder']}"; tok = u["tok"]; tag = u.get("py_tag", ""); wtag = tag + u.get("wc_tag", ""); w = u["wave_clus"][0]
    sp = sorted(glob.glob(f"{sp_root(u['run'])}/second_pass_*/*/"))
    gtwc = [d for d in (f"{root}/{w}/validation_gt_vs_{w}{tok}", f"{root}/validation_gt_vs_{w}{tok}")
            if os.path.exists(f"{d}/pdf_reports/validation_report.pdf")]
    rows = [("python second pass (sorter output)", sp[-1] if sp else "", t(sp[-1] + "decisions.json") if sp else "MISSING"),
            ("python newest, converted", f"{u['python_arm']}/", t(f"{root}/{u['python_arm']}/grapes_blanks_rescue.mat")),
            ("python before second pass, converted", f"{u['python_before']}/", t(f"{root}/{u['python_before']}/grapes_blanks_rescue.mat")),
            ("GT vs python validation", f"validation_gt_vs_{u['python_arm']}{tok}{tag}/", rep(f"{root}/validation_gt_vs_{u['python_arm']}{tok}{tag}")),
            ("  diagnostics", f"validation_gt_vs_{u['python_arm']}{tok}{tag}/diagnostics/",
             t(f"{root}/validation_gt_vs_{u['python_arm']}{tok}{tag}/diagnostics/diagnostics_summary_ch{u['ch']}_gt{u['gt_class']}.csv")),
            ("wave_clus run used", f"{w}/", t(f"{root}/{w}/grapes_blanks.mat")),
            ("GT vs wave_clus", (gtwc[0].replace(root + "/", "") + "/") if gtwc else f"{w}/validation_gt_vs_{w}{tok}/",
             rep(gtwc[0]) if gtwc else "MISSING"),
            ("wave_clus vs python (python = arm)", f"{w}/validation_{w}_vs_python_sd1{tok}{wtag}/", rep(f"{root}/{w}/validation_{w}_vs_python_sd1{tok}{wtag}")),
            ("log-det replay", f"{OUT}/ld/ch{u['ch']}/", t(f"{OUT}/ld/ch{u['ch']}/logdet_replay.json")),
            ("appendix", f"{OUT}/appendix_ch{u['ch']}.pdf", t(f"{OUT}/appendix_ch{u['ch']}.pdf"))]
    pc = (f"   python class fixed to c{u['python_class']}" if u.get("python_class") else "") + (f", wave_clus class fixed to c{u['wc_class']}" if u.get("wc_class") else "")
    lines.append(f"=== {u['folder']}  GT class {u['gt_class']}{pc}   (channel folder: {root})")
    lines += [f"  {c:12s} {a:38s} {b}" for a, b, c in rows]
    lines.append("")
lines.append(f"overview: {OUT}/overview.pdf  ({t(OUT / 'overview.pdf')})")
(OUT / "where_everything_is.txt").write_text("\n".join(lines) + "\n")
print(OUT / "where_everything_is.txt")
