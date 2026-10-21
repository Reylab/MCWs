"""Why the second pass did (not) split a python cluster, and is it separable at all? One cluster's spikes,
coloured by the GT class each one is in. Uses the second pass's own view: its 13 selected wavelet
features, standardised, Hartigan dip test on PC 1-3 (Bonferroni), K=1 / K=2 GMM (BIC, pooled-covariance D,
smaller-component share) -- recluster_test/second_pass.py gmm_screen -- and then 1-D projections:
the top single features, the axis that best separates the GT classes (LDA), the K=2 GMM axis;
plus timing (ISI per GT class, cross-correlogram between the two largest GT groups) and waveforms.

    python cluster_split_check.py <export.mat> <out_dir> <title>
export.mat (from MATLAB, -v7): F (n x 13 features), W (n x 64 waveforms), t (ms), gt (GT class, -1 = none)
"""
import sys, os, csv
import numpy as np, scipy.io as sio, diptest
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

src, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
d = sio.loadmat(src)
F, W, t, gt = d['F'].astype(float), d['W'].astype(float), d['t'].ravel(), d['gt'].ravel().astype(int)
Z = StandardScaler().fit_transform(F); Zc = Z - Z.mean(0)
V = np.linalg.svd(Zc, full_matrices=False)[2]; PC = Zc @ V[:3].T
cls = [k for k, _ in sorted({k: np.sum(gt == k) for k in np.unique(gt)}.items(), key=lambda x: -x[1])]
lab = {k: ('no GT spike' if k == -1 else 'GT cluster 0' if k == 0 else f'GT c{k}') for k in cls}
col = dict(zip(cls, plt.cm.tab10(np.arange(len(cls)))))
rows = []

def dip_p(x):
    return float(diptest.diptest(np.asarray(x, float))[1])

def dprime(x, a, b):
    xa, xb = x[gt == a], x[gt == b]
    return abs(xa.mean() - xb.mean()) / np.sqrt((xa.var() + xb.var()) / 2)

def stacked(ax, x, ttl, bins=60):
    edges = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), bins + 1)
    ax.hist([x[gt == k] for k in cls], bins=edges, stacked=True, color=[col[k] for k in cls],
            label=[f'{lab[k]} ({np.sum(gt == k)})' for k in cls])
    ax.set_title(ttl, fontsize=9); ax.set_yticks([])

big = [k for k in cls if k > 0][:2]                       # the two largest GT classes in the cluster

# ---- page 1: the second pass's own tests ----
g1 = GaussianMixture(1, covariance_type='full', reg_covar=1e-4, random_state=0).fit(Z)
g2 = GaussianMixture(2, covariance_type='full', reg_covar=1e-4, n_init=3, random_state=0).fit(Z)
o = np.argsort(-g2.weights_); mu, cv, w = g2.means_[o], g2.covariances_[o], g2.weights_[o]
dmu = mu[0] - mu[1]; D = float(np.sqrt(dmu @ np.linalg.solve((cv[0] + cv[1]) / 2, dmu)))
dbic = g1.bic(Z) - g2.bic(Z); comp = np.argsort(o)[g2.predict(Z)]
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for j in range(3):
    p = min(1, dip_p(PC[:, j]) * 3); rows.append(['second pass', f'PC{j+1} dip p (Bonferroni)', f'{p:.3g}', '< 0.01 = split'])
    stacked(ax[0, j], PC[:, j], f'PC {j+1} of the 13 standardised features | dip p = {p:.3g} (needs < 0.01)')
ax[0, 0].legend(fontsize=7, loc='upper left')
for k in cls:
    m = gt == k; ax[1, 0].scatter(PC[m, 0], PC[m, 1], s=3, color=col[k], alpha=.5, label=lab[k])
ax[1, 0].set_title('PC1 vs PC2, coloured by GT class', fontsize=9)
for c, cc in zip([0, 1], ['grey', 'red']):
    m = comp == c; ax[1, 1].scatter(PC[m, 0], PC[m, 1], s=3, color=cc, alpha=.5, label=f'GMM K=2 component {c+1} ({w[c]:.1%})')
ax[1, 1].set_title('the K=2 GMM split (what stage 2 tested)', fontsize=9); ax[1, 1].legend(fontsize=7)
ax[1, 2].axis('off')
txt = [f'stage 1  dip on PC 1-3: p = ' + ', '.join(f'{min(1, dip_p(PC[:, j]) * 3):.2g}' for j in range(3)) + '  (needs < 0.01)  -> FAIL',
       f'stage 2  A  BIC(1) - BIC(2) = {dbic:+.0f}  (needs > 10)  -> {"PASS" if dbic > 10 else "FAIL"}',
       f'         B  separation D = {D:.2f}  (needs > 2)  -> {"PASS" if D > 2 else "FAIL"}',
       f'         C  smaller component = {w[1]:.1%}  (needs >= 5 %)  -> {"PASS" if w[1] >= .05 else "FAIL"}',
       '', 'GMM K=2 components vs GT classes (rows: component, cols: GT):']
for c in [0, 1]:
    txt.append(f'  comp {c+1}: ' + ', '.join(f'{lab[k]} {np.sum((comp == c) & (gt == k))}' for k in cls))
ax[1, 2].text(0, 1, '\n'.join(txt), va='top', family='monospace', fontsize=8.5)
rows += [['second pass', 'BIC(1)-BIC(2)', f'{dbic:.0f}', '> 10'], ['second pass', 'D', f'{D:.2f}', '> 2'],
         ['second pass', 'smaller component', f'{w[1]:.3f}', '>= 0.05']]
fig.suptitle(f'{title}: why the second pass kept it (its own tests on its own features)', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/1_second_pass_tests.png', dpi=110); plt.close(fig)

# ---- page 2: 1-D projections ----
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
dips = sorted(range(Z.shape[1]), key=lambda j: diptest.diptest(Z[:, j])[0], reverse=True)[:3]
for i, j in enumerate(dips):
    p = dip_p(Z[:, j]); dp = dprime(Z[:, j], *big) if len(big) == 2 else np.nan
    stacked(ax[0, i], Z[:, j], f'feature {j+1} (most bimodal #{i+1}) | dip p {p:.2g} | d\' {lab[big[0]]}/{lab[big[1]]} {dp:.2f}')
    rows.append(['single feature', f'feature {j+1}', f'dip p {p:.3g}', f"d' {dp:.2f}"])
ax[0, 0].legend(fontsize=7, loc='upper left')
m2 = np.isin(gt, big)
lda = LinearDiscriminantAnalysis().fit(Z[m2], gt[m2]); x = (Z @ lda.scalings_[:, 0])
stacked(ax[1, 0], x, f'best axis {lab[big[0]]} vs {lab[big[1]]} (LDA) | d\' {dprime(x, *big):.2f} | dip p {dip_p(x):.2g}')
rows.append(['LDA axis', f'{lab[big[0]]} vs {lab[big[1]]}', f"d' {dprime(x, *big):.2f}", f'dip p {dip_p(x):.3g}'])
mk = gt > 0
if len(np.unique(gt[mk])) > 2:
    ldm = LinearDiscriminantAnalysis().fit(Z[mk], gt[mk]); x2 = Z @ ldm.scalings_[:, 0]
    stacked(ax[1, 1], x2, f'best axis over all GT classes (LDA 1) | dip p {dip_p(x2):.2g}')
    rows.append(['LDA axis', 'all GT classes', f'dip p {dip_p(x2):.3g}', ''])
x3 = Z @ (dmu / np.linalg.norm(dmu))
stacked(ax[1, 2], x3, f'K=2 GMM axis (means) | dip p {dip_p(x3):.2g}')
fig.suptitle(f'{title}: one-dimensional projections, coloured by GT class', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/2_projections.png', dpi=110); plt.close(fig)

# ---- page 3: timing + waveforms ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for k in [c for c in cls if c > 0][:4]:
    isi = np.diff(np.sort(t[gt == k])); ax[0].hist(isi[isi < 50], bins=50, range=(0, 50), histtype='step', color=col[k],
        label=f'{lab[k]}: {100 * np.mean(isi < 3):.2f}% < 3 ms')
isi = np.diff(np.sort(t)); ax[0].hist(isi[isi < 50], bins=50, range=(0, 50), histtype='step', color='k', lw=2,
    label=f'whole cluster: {100 * np.mean(isi < 3):.2f}% < 3 ms')
ax[0].axvline(3, ls=':', c='k'); ax[0].legend(fontsize=7); ax[0].set_xlabel('ISI (ms)'); ax[0].set_title('ISI per GT class', fontsize=9)
if len(big) == 2:
    a, b = np.sort(t[gt == big[0]]), np.sort(t[gt == big[1]]); lags = []
    for x in a:
        i0, i1 = np.searchsorted(b, [x - 50, x + 50]); lags.extend(b[i0:i1] - x)
    ax[1].hist(lags, bins=100, range=(-50, 50), color='grey'); ax[1].axvspan(-3, 3, color='r', alpha=.1)
    ax[1].set_title(f'cross-correlogram {lab[big[0]]} x {lab[big[1]]}: no gap at 0 = independent neurons', fontsize=9)
    ax[1].set_xlabel('lag (ms)')
for k in [c for c in cls if c >= 0][:5]:
    ax[2].plot(W[gt == k].mean(0), color=col[k], lw=2, label=f'{lab[k]} ({np.sum(gt == k)})')
ax[2].legend(fontsize=7); ax[2].set_title('mean waveform per GT class', fontsize=9)
fig.suptitle(f'{title}: timing and shapes of the GT groups inside the cluster', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/3_timing_waveforms.png', dpi=110); plt.close(fig)

with open(f'{out}/split_check.csv', 'w', newline='') as fh:
    csv.writer(fh).writerows([['test', 'what', 'value', 'criterion / note']] + rows)
print('\n'.join(' | '.join(r) for r in rows))

# ---- page 4-5: how separable GT's units are INSIDE this python cluster (what python should have split) ----
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from scipy.stats import chi2

units = [k for k in cls if k > 0 and np.sum(gt == k) >= 30]
mu_ = np.isin(gt, units); y = gt[mu_]
def shape_feats(Wv):
    tr = Wv.min(1); it = Wv.argmin(1); n = Wv.shape[1]
    half = np.array([np.sum(w < m / 2) for w, m in zip(Wv, tr)])                       # samples below half the trough
    pk = np.array([w[i:].argmax() for w, i in zip(Wv, it)])                             # trough -> later peak
    reb = np.array([w[i:].max() for w, i in zip(Wv, it)])
    slope = np.array([(w[min(i + 6, n - 1)] - w[i]) / 6 for w, i in zip(Wv, it)])       # repolarisation slope
    pre = Wv[:, :max(1, int(np.median(it)) - 6)].max(1)
    return np.c_[tr, half, pk, slope, reb, pre], ['trough (uV)', 'half-width (samples)', 'trough-to-peak (samples)',
                                                  'repolarisation slope', 'rebound (uV)', 'pre-trough peak (uV)']
S, snames = shape_feats(W)
Wpc = PCA(10, random_state=0).fit_transform(W - W.mean(0))
sets = {'python 13 wavelet features': Z, 'waveform PCA (10)': Wpc, 'shape features (6)': StandardScaler().fit_transform(S),
        'wavelet + shape': np.c_[Z, StandardScaler().fit_transform(S)]}
cv = StratifiedKFold(5, shuffle=True, random_state=0); acc = {}
for nm, X in sets.items():
    acc[nm] = cross_val_score(LinearDiscriminantAnalysis(), X[mu_], y, cv=cv).mean()
    rows.append(['CV accuracy (LDA, 5-fold)', nm, f'{acc[nm]:.3f}', f'chance {np.max(np.bincount(y)) / len(y):.3f}'])

def iso_lratio(X, m):
    """isolation distance and L-ratio of the spikes M among all spikes of the cluster (Schmitzer-Torbert 2005)"""
    Xm = X[m]; c = np.cov(Xm.T) + 1e-6 * np.eye(X.shape[1]); ic = np.linalg.inv(c); d = X - Xm.mean(0)
    d2 = np.einsum('ij,jk,ik->i', d, ic, d); o = np.sort(d2[~m]); nm_ = m.sum()
    iso = o[nm_ - 1] if len(o) >= nm_ else np.nan
    lr = np.sum(1 - chi2.cdf(d2[~m], X.shape[1])) / nm_
    return iso, lr
iso_rows = []
for k in units:
    for nm in ['python 13 wavelet features', 'waveform PCA (10)']:
        iso, lr = iso_lratio(sets[nm], gt == k); iso_rows.append((k, nm, iso, lr))
        rows.append(['isolation inside the cluster', f'{lab[k]} in {nm}', f'iso dist {iso:.1f}', f'L-ratio {lr:.3f}'])

fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(2, 3, 1)
ax.barh(list(acc), list(acc.values()), color=['#999', '#57a', '#a75', '#5a5']); ax.axvline(np.max(np.bincount(y)) / len(y), ls=':', c='k')
ax.set_xlim(0, 1); ax.set_title('can GT\'s units be told apart inside python c? (CV accuracy, : = chance)', fontsize=9)
for i, v in enumerate(acc.values()): ax.text(v + .01, i, f'{v:.2f}', va='center', fontsize=8)
if len(units) > 2:
    l2 = LinearDiscriminantAnalysis(n_components=2).fit(sets['wavelet + shape'][mu_], y); P2 = l2.transform(sets['wavelet + shape'])
    ax = fig.add_subplot(2, 3, 2)
    for k in units: m = gt == k; ax.scatter(P2[m, 0], P2[m, 1], s=4, color=col[k], alpha=.5, label=lab[k])
    ax.legend(fontsize=7, markerscale=3); ax.set_title('best 2-D view (LDA 1 vs 2, wavelet + shape)', fontsize=9)
ax = fig.add_subplot(2, 3, 3); ax.axis('off')
ax.text(0, 1, 'isolation of each GT unit inside python c\n(iso dist: higher = better isolated; L-ratio: lower)\n\n' +
        '\n'.join(f'{lab[k]:<8} {nm[:22]:<22} iso {iso:6.1f}  L {lr:.3f}' for k, nm, iso, lr in iso_rows),
        va='top', family='monospace', fontsize=8)
for i, j in enumerate([0, 1, 3]):
    ax = fig.add_subplot(2, 3, 4 + i)
    lo, hi = np.percentile(S[:, j], [1, 99]); edges = np.linspace(lo, hi, 40)
    for k in units: ax.hist(S[gt == k, j], bins=edges, histtype='step', lw=1.5, color=col[k], density=True,
                            label=f'{lab[k]} d\'={abs(S[gt == k, j].mean() - S[gt == units[0], j].mean()) / np.sqrt((S[gt == k, j].var() + S[gt == units[0], j].var()) / 2):.2f}')
    ax.legend(fontsize=7); ax.set_title(f'{snames[j]} (d\' vs {lab[units[0]]})', fontsize=9)
for j in range(len(snames)):
    rows.append(['shape feature', snames[j]] + [', '.join(f'{lab[k]} {S[gt == k, j].mean():.2f}' for k in units), ''])
fig.suptitle(f'{title}: how separable GT\'s units are inside this python cluster', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/4_separability_inside_cluster.png', dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
for k in units:
    m = gt == k; ax[0].scatter(t[m] / 60000, S[m, 0], s=3, color=col[k], alpha=.5, label=lab[k])
ax[0].set_xlabel('time (min)'); ax[0].set_ylabel('trough (uV)'); ax[0].legend(fontsize=7, markerscale=3)
ax[0].set_title('trough amplitude over the recording per GT unit (drift check)', fontsize=9)
for k in units:
    m = gt == k; tb = np.linspace(t.min(), t.max(), 11); idx = np.digitize(t[m], tb)
    ax[1].plot([np.mean(idx == b) for b in range(1, 11)], color=col[k], marker='o', label=lab[k])
ax[1].set_xlabel('recording decile'); ax[1].set_ylabel('share of the unit\'s spikes'); ax[1].legend(fontsize=7)
ax[1].set_title('when each GT unit fires (all units present throughout = not one drifting unit)', fontsize=9)
fig.suptitle(f'{title}: drift', fontweight='bold'); fig.tight_layout(); fig.savefig(f'{out}/5_drift.png', dpi=110); plt.close(fig)
with open(f'{out}/split_check.csv', 'w', newline='') as fh:
    csv.writer(fh).writerows([['test', 'what', 'value', 'criterion / note']] + rows)
print('\n'.join(' | '.join(str(x) for x in r) for r in rows[-(len(sets) + 2 * len(units) + len(snames)):]))
