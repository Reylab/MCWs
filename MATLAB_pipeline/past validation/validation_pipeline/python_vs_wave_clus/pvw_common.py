"""Shared config for the python-vs-wave_clus analysis: channels.json, paths."""
import glob, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / os.environ.get("PVW_CONFIG", "channels.json")).read_text())   # PVW_CONFIG=channels_tuned.json for the tuned trial
SC = Path(CFG["single_channel"])
OUT = Path(CFG["out_dir"])
MWS = Path("/home/ajb/Documents/GitHub/MicroWireSorter")   # MicroWireSorter repo (read-only use)


def channels():
    """{ch: {'folder', 'run', 'times_file', 'units': [unit, ...]}} from channels.json."""
    out = {}
    for u in CFG["units"]:
        c = out.setdefault(u["ch"], dict(folder=u["folder"], run=u["run"], times_file=u["times_file"], units=[]))
        c["units"].append(u)
    return out


def sp_root(run):
    """folder holding the run's second_pass_<ts>/: <recluster_runs>/<run>/<second_pass_subdir>,
    the subdir from the run's units in channels.json when set ('' = the run folder itself),
    else the top-level second_pass_subdir"""
    sub = next((u["second_pass_subdir"] for u in CFG["units"] if u["run"] == run and "second_pass_subdir" in u),
               CFG["second_pass_subdir"])
    return Path(CFG["recluster_runs"]) / run / sub


def second_pass_dir(run):
    """<sp_root(run)>/second_pass_<ts>/<label>_<ch>/ (newest)"""
    m = sorted(glob.glob(f"{sp_root(run)}/second_pass_*/*/"))
    return Path(m[-1]) if m else None
