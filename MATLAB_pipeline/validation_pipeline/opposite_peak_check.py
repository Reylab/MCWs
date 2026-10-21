"""Opposite-peak check of one python sort: per cluster, the waveforms, the two side ratios python's opposite-peak rule
tests (opposite-polarity bump / peak-to-peak; quarantined when BOTH sides are above the threshold) and the size of a
second same-polarity trough (not tested by any rule). The threshold is the one the run used (par.qc_params).
Writes <out dir>/opposite_peak_check.png and opposite_peak_check.csv.

    python opposite_peak_check.py "<times_ folder of a sort>" "<out dir>" ["<title>"]
"""
import sys, os, csv, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.environ.get('MCWS_ROOT', '/home/ajb/Documents/GitHub/MicroWireSorter'))   # MicroWireSorter
from mcws.io.spike_io import load_spikes

src, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else src
os.makedirs(out, exist_ok=True)
d = next(iter(load_spikes(src).values()))
W = np.asarray(d['spikes'], float); lab = d['cluster_class'][:, 0].astype(int)
qp = d['quarantine_properties']

def col(k):
    v = qp.get(k) if isinstance(qp, dict) else None
    if v is None:
        v = [getattr(x, k, np.nan) if not isinstance(x, dict) else x.get(k, np.nan) for x in qp]
    return np.asarray(v, float).ravel()

L, R, OB = col('left_ratio_used'), col('right_ratio_used'), col('opposite_both_sides')
par = d.get('par', {}); qc = par.get('qc_params', {}) if isinstance(par, dict) else getattr(par, 'qc_params', {})
thr = float(qc.get('opposite_peak_ratio_both_sides', np.nan) if isinstance(qc, dict) else getattr(qc, 'opposite_peak_ratio_both_sides', np.nan))
tr = -W[:, 19]
sec = np.full(len(W), np.nan)                                       # deepest other negative local minimum / main trough
for i in range(len(W)):
    w = W[i]; m = np.where((w[1:-1] < w[:-2]) & (w[1:-1] < w[2:]))[0] + 1; m = m[np.abs(m - 19) > 3]
    if m.size: sec[i] = max(0, -w[m].min()) / max(tr[i], 1e-9)
lb = np.maximum(W[:, :19].max(1), 0) / np.maximum(tr, 1e-9); rb = np.maximum(W[:, 20:].max(1), 0) / np.maximum(tr, 1e-9)

cl = sorted(k for k in set(lab.tolist()) if k > 0)
fig, ax = plt.subplots(3, len(cl), figsize=(4 * len(cl), 11), squeeze=False)
rows = [['cluster', 'n', 'rule_fired', 'median_left_ratio', 'median_right_ratio', 'left_above_thr', 'right_above_thr',
         'both_bumps_gt_0.5_trough', 'second_trough_ge_0.5_main']]
for j, k in enumerate(cl):
    m = lab == k; n = int(m.sum())
    s = np.random.default_rng(0).choice(np.where(m)[0], min(300, n), replace=False)
    ax[0, j].plot(W[s].T, color='grey', alpha=.06); ax[0, j].plot(W[m].mean(0), 'k', lw=2.5); ax[0, j].axhline(0, c='k', lw=.5)
    ax[0, j].set_title(f'c{k} (n={n}): opposite rule fired on {int(np.nansum(OB[m]))}', fontsize=9)
    e = np.linspace(0, 1, 41)
    ax[1, j].hist(L[m][np.isfinite(L[m])], e, alpha=.6, label='left'); ax[1, j].hist(R[m][np.isfinite(R[m])], e, alpha=.6, label='right')
    ax[1, j].axvline(thr, c='r', ls='--', label=f'threshold {thr:g} (both sides must pass)'); ax[1, j].legend(fontsize=7)
    ax[1, j].set_title('opposite-polarity bump / peak-to-peak', fontsize=9); ax[1, j].set_xlabel('ratio')
    ax[2, j].hist(sec[m][np.isfinite(sec[m])], np.linspace(0, 1.5, 46), color='#a75'); ax[2, j].axvline(.5, c='k', ls=':')
    ax[2, j].set_title('2nd negative trough / main trough (same polarity; not tested)', fontsize=8)
    rows.append([k, n, int(np.nansum(OB[m])), f'{np.nanmedian(L[m]):.3f}', f'{np.nanmedian(R[m]):.3f}', f'{np.nanmean(L[m] > thr):.3f}',
                 f'{np.nanmean(R[m] > thr):.3f}', f'{np.mean((lb[m] > .5) & (rb[m] > .5)):.3f}', f'{np.nanmean(sec[m] >= .5):.3f}'])
q0 = lab == 0
fig.suptitle(f'{title}\nopposite-peak rule (both sides > {thr:g} x peak-to-peak) fired on {int(np.nansum(OB))} spikes '
             f'({int(np.nansum(OB[q0]))} of them in cluster 0)', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/opposite_peak_check.png', dpi=110); plt.close(fig)
with open(f'{out}/opposite_peak_check.csv', 'w', newline='') as fh: csv.writer(fh).writerows(rows)
print(f'threshold {thr:g} | rule fired on {int(np.nansum(OB))} spikes | -> {out}')
for r in rows: print('  ', r)
