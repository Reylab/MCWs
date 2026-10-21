"""Waveform-shape descriptor screen: is a python cluster one shape or more, judged on a few shape descriptors
of its own spikes (no reference sort)? The descent into the trough is where merged units differ (a knee /
two-stage fall vs one smooth fall), so most descriptors read the first derivative dW/dt there.

Descriptors per spike (spikes aligned on the trough, as stored; derivative smoothed with Savitzky-Golay; timings and
widths at sub-sample precision by interpolation -- whole-sample values have ties the dip test reads as multimodal):
  knee index        largest bulge of the down-stroke above the straight line from trough-12 to the trough (uV)
  peak fall speed   most negative dW/dt in the descent window (uV/sample)
  fall timing       samples before the trough where that fastest fall happens
  descent duration  samples from the first dW/dt below -3 x baseline-derivative sd to the trough
  curvature peak    largest d2W/dt2 in the descent window (a pause in the fall = positive curvature)
  half-width        samples below half the trough
  rebound           largest value after the trough (uV)
Screen: Hartigan's dip test on each descriptor, Bonferroni over the 7; flagged = any corrected p < 0.01 and the
smaller mode holds >= 5 % (2-component split of that descriptor).

    python descriptor_screen.py figures <times.mat> <cluster> <out_dir> <title> [<other cluster>]
    python descriptor_screen.py screen <out.csv> <times.mat>|<cluster>|<label> ...
"""
import sys, os, csv
import numpy as np, scipy.io as sio, diptest
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NAMES = ['knee index (uV)', 'peak fall speed (uV/sample)', 'fall timing (samples before trough)',
         'descent duration (samples)', 'curvature peak', 'half-width (samples)', 'rebound (uV)']


def descriptors(W):
    W = np.asarray(W, float); n, L = W.shape
    tr = int(np.argmin(W.mean(0))); s0 = max(2, tr - 12); x = np.arange(s0, tr + 1)
    dW = savgol_filter(W, 7, 2, deriv=1, axis=1); d2W = savgol_filter(W, 7, 2, deriv=2, axis=1)
    base_sd = dW[:, :max(3, s0 - 2)].std()
    D = np.empty((n, 7))
    def cross(y, level, start, stop, step):
        """sub-sample position where y first crosses LEVEL walking from START towards STOP (linear interpolation)"""
        for k in range(start, stop, step):
            a, b = y[k], y[k + step]
            if (a - level) * (b - level) <= 0 and a != b:
                return k + step * (level - a) / (b - a)
        return float(stop)
    for i in range(n):
        w = W[i]; line = w[s0] + (w[tr] - w[s0]) * (x - s0) / (tr - s0)
        D[i, 0] = (w[s0:tr + 1] - line).max()
        seg = dW[i, s0:tr + 1]; j = int(seg.argmin())
        if 0 < j < len(seg) - 1:                                   # parabolic sub-sample minimum
            den = seg[j - 1] - 2 * seg[j] + seg[j + 1]; off = 0.5 * (seg[j - 1] - seg[j + 1]) / den if den else 0.0
        else:
            off = 0.0
        D[i, 1] = seg[j]; D[i, 2] = tr - (s0 + j + off)
        c3 = cross(dW[i], -3 * base_sd, max(1, tr - 15), tr, 1)     # fall start, looked for only in the 15 samples
                                                                   # before the trough (window-edge noise is not a fall)
        D[i, 3] = tr - c3 if c3 < tr else np.nan                   # never crosses it: no value (not 0 -- a pile at 0 fakes a dip)
        D[i, 4] = d2W[i, s0:tr + 1].max()
        h = w[tr] / 2; D[i, 5] = cross(w, h, tr, L - 1, 1) - cross(w, h, tr, 0, -1)   # sub-sample half-width
        D[i, 6] = w[tr:].max()
    return D, dW, tr, s0


def screen(D):
    out = []
    for j in range(D.shape[1]):
        x = D[:, j]; x = x[np.isfinite(x)]
        if len(x) < 20 or np.ptp(x) == 0:
            out.append(dict(name=NAMES[j], dip=np.nan, p=1.0, wmin=0.0, flag=False, modes=np.array([np.nan, np.nan]))); continue
        dstat, p = diptest.diptest(x); pc = min(1.0, p * D.shape[1])
        g = GaussianMixture(2, random_state=0).fit(x[:, None]); wmin = float(g.weights_.min())
        out.append(dict(name=NAMES[j], dip=float(dstat), p=pc, wmin=wmin, flag=pc < 0.01 and wmin >= 0.05,
                        modes=np.sort(g.means_.ravel())))
    return out


def load(times, c):
    d = sio.loadmat(times, variable_names=['spikes', 'cluster_class'])
    return d['spikes'][d['cluster_class'][:, 0] == c]


if sys.argv[1] == 'figures':
    times, c, out, title = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    other = int(sys.argv[6]) if len(sys.argv) > 6 else None
    os.makedirs(out, exist_ok=True)
    W = load(times, c); D, dW, tr, s0 = descriptors(W); S = screen(D)
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.5)); ax = ax.ravel()
    for j, s in enumerate(S):
        x = D[:, j]; x = x[np.isfinite(x)]; lo, hi = np.percentile(x, [0.5, 99.5]); e = np.linspace(lo, hi, 50)
        ax[j].hist(x, bins=e, density=True, color='#c66' if s['flag'] else '#999', alpha=.75)
        if np.ptp(x) > 0:
            xx = np.linspace(lo, hi, 300); ax[j].plot(xx, gaussian_kde(x)(xx), 'k', lw=1.5)
        if s['flag']:
            for m in s['modes']: ax[j].axvline(m, color='r', ls='--', lw=1)
        ax[j].set_title(f"{s['name']}\ndip p (Bonferroni) {s['p']:.3g}  smaller mode {s['wmin']:.0%}  "
                        f"{'-> TWO GROUPS' if s['flag'] else ''}", fontsize=9, color='r' if s['flag'] else 'k')
        ax[j].set_yticks([])
    k = D[:, 0]; q = np.quantile(k, [.25, .75]); a = ax[7]; xs = np.arange(dW.shape[1])
    for m, col, nm in [(k >= q[1], 'r', 'strongest-knee quarter'), (k <= q[0], 'b', 'no-knee quarter')]:
        idx = np.random.default_rng(0).choice(np.where(m)[0], min(150, m.sum()), replace=False)
        a.plot(xs, dW[idx].T, color=col, alpha=.05); a.plot(xs, dW[m].mean(0), color=col, lw=2.5, label=nm)
    a.axvspan(s0, tr, color='y', alpha=.15); a.axhline(0, c='k', lw=.5); a.set_xlim(0, tr + 15); a.legend(fontsize=7)
    a.set_title('the fall itself: dW/dt (knee = the fall eases, then plunges again)', fontsize=9)
    fig.suptitle(f'{title}: shape descriptors of the cluster\'s own spikes (red = two groups by the screen)', fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{out}/descriptor_screen.png', dpi=110); plt.close(fig)
    # 2-D: the two strongest descriptors
    o = np.argsort([s['p'] for s in S])[:2]
    fig, ax = plt.subplots(1, 2 if other else 1, figsize=(13 if other else 6.5, 5.2)); ax = np.atleast_1d(ax)
    ax[0].scatter(D[:, o[0]], D[:, o[1]], s=4, alpha=.35)  # NaN points are skipped by matplotlib; ax[0].set_xlabel(NAMES[o[0]]); ax[0].set_ylabel(NAMES[o[1]])
    ax[0].set_title(f'python c{c}: the two most bimodal descriptors', fontsize=9)
    if other:
        D2 = descriptors(load(times, other))[0]
        ax[1].scatter(D[:, o[0]], D[:, o[1]], s=4, alpha=.3, label=f'python c{c}')
        ax[1].scatter(D2[:, o[0]], D2[:, o[1]], s=4, alpha=.3, label=f'python c{other}'); ax[1].legend(markerscale=3, fontsize=8)
        ax[1].set_xlabel(NAMES[o[0]]); ax[1].set_title(f'same, with python c{other} on top', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{out}/descriptor_2d.png', dpi=110); plt.close(fig)
    for s in S: print(f"{s['name']:38s} dip p {s['p']:.3g}  smaller mode {s['wmin']:.2f}  {'FLAG' if s['flag'] else ''}")

elif sys.argv[1] == 'screen':
    rows = [['cluster', 'n', 'flagged', 'flagging descriptors'] + [f'p {n}' for n in NAMES]]
    for spec in sys.argv[3:]:
        times, c, label = spec.split('|'); W = load(times, int(c))
        S = screen(descriptors(W)[0])
        fl = [s['name'].split(' (')[0] for s in S if s['flag']]
        rows.append([label, len(W), 'YES' if fl else 'no', '; '.join(fl)] + [f"{s['p']:.3g}" for s in S])
        print(f"{label:34s} n {len(W):6d}  flagged {'YES' if fl else 'no ':3s}  {'; '.join(fl)}")
    with open(sys.argv[2], 'w', newline='') as fh: csv.writer(fh).writerows(rows)
