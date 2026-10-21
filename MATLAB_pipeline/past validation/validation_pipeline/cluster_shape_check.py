"""Is one python cluster one shape or more? Uses only the cluster's own spikes (no reference sort):
  1  Euclidean distance of every spike to the cluster template (mean waveform) vs the distance expected from
     noise alone (each spike's own pre-trough baseline noise): a single neuron sits near the noise
     distribution; two averaged shapes give a wider / bimodal distribution (dip test)
  2  feature-space Mahalanobis distance to the cluster centre (the 13 features python clusters on) vs the
     chi-square a single Gaussian cluster would give (excess tail = more than one Gaussian)
  3  residuals (spike - template): their main directions (PCA), what shape each one is, and whether the spikes
     split along them (dip test)
  4  per-sample variance across the waveform vs baseline noise variance (where the extra spread is)
  5  k-means K=2 in waveform space: the two mean shapes, their share, the silhouette

    python cluster_shape_check.py <export.mat> <out_dir> <title>
export.mat (-v7): W (n x 64 waveforms, uV), F (n x d features python clustered on)
"""
import sys, os, csv
import numpy as np, scipy.io as sio, diptest
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

src, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
d = sio.loadmat(src); W = d['W'].astype(float); F = d['F'].astype(float)
n, L = W.shape; T = W.mean(0); trough = int(np.argmin(T)); base = slice(0, max(4, trough - 8))
rows = []
dip = lambda x: diptest.diptest(np.asarray(x, float))

# ---- 1 Euclidean distance to the template vs noise ----
dist = np.linalg.norm(W - T, axis=1)
noise_sd = W[:, base].std(1)                                  # each spike's own baseline noise (uV)
noise_dist = noise_sd * np.sqrt(L)                            # distance a template + white noise spike would have
ex = dist ** 2 - noise_dist ** 2                              # distance^2 beyond noise
dd, dp = dip(dist); rows += [['template distance', 'dip statistic / p', f'{dd:.4f} / {dp:.3g}', 'p < 0.05 = bimodal'],
                            ['template distance', 'median distance / median noise-only distance', f'{np.median(dist):.1f} / {np.median(noise_dist):.1f} uV', '>1 = spread beyond noise'],
                            ['template distance', 'share of spikes with distance^2 > 2x noise', f'{np.mean(dist ** 2 > 2 * noise_dist ** 2):.3f}', '']]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
e = np.linspace(0, np.percentile(dist, 99.5), 70)
ax[0].hist(dist, bins=e, color='#57a', alpha=.8, label='spike - template'); ax[0].hist(noise_dist, bins=e, histtype='step', color='k', lw=2, label='noise only (baseline sd x sqrt(64))')
ax[0].set_title(f'Euclidean distance to the template | dip p {dp:.3g}', fontsize=9); ax[0].set_xlabel('uV'); ax[0].legend(fontsize=7)
ax[1].scatter(noise_dist, dist, s=3, alpha=.4); m = max(dist.max(), noise_dist.max()); ax[1].plot([0, m], [0, m], 'k--', lw=1)
ax[1].set_xlabel('noise-only distance'); ax[1].set_ylabel('distance to template'); ax[1].set_title('each spike: above the line = differs from the template beyond its noise', fontsize=9)
o = np.argsort(dist); q = np.array_split(o, 4)
for i, qq in enumerate(q):
    ax[2].plot(W[qq].mean(0), lw=2, label=f'distance quartile {i + 1} ({dist[qq].min():.0f}-{dist[qq].max():.0f} uV)')
ax[2].plot(T, 'k--', lw=1, label='template'); ax[2].legend(fontsize=7); ax[2].set_title('mean waveform by distance quartile (shape change = second shape)', fontsize=9)
fig.suptitle(f'{title}: distance of every spike to the cluster template (waveform space)', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/6_template_distance.png', dpi=110); plt.close(fig)

# ---- 2 feature-space Mahalanobis vs chi-square ----
Z = StandardScaler().fit_transform(F); dF = Z.shape[1]
d2 = np.einsum('ij,jk,ik->i', Z - Z.mean(0), np.linalg.pinv(np.cov(Z.T)), Z - Z.mean(0))
exp99 = chi2.ppf(.99, dF); obs99 = np.mean(d2 > exp99)
rows += [['feature Mahalanobis', f'share with d2 > chi2(99%, {dF})', f'{obs99:.3f}', 'single Gaussian: 0.010'],
         ['feature Mahalanobis', 'dip p of d2', f'{dip(d2)[1]:.3g}', '']]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
e = np.linspace(0, np.percentile(d2, 99.5), 80); ax[0].hist(d2, bins=e, density=True, color='#a75', alpha=.8, label='spikes')
ax[0].plot(e, chi2.pdf(e, dF), 'k', lw=2, label=f'chi-square({dF}) = one Gaussian'); ax[0].axvline(exp99, ls=':', c='k')
ax[0].legend(fontsize=7); ax[0].set_title(f'Mahalanobis d2 in python\'s {dF} features | {obs99:.1%} beyond the 99% line (1% expected)', fontsize=9)
qs = np.linspace(.01, .99, 99); ax[1].plot(chi2.ppf(qs, dF), np.quantile(d2, qs), 'o', ms=3); m = max(chi2.ppf(.99, dF), np.quantile(d2, .99))
ax[1].plot([0, m], [0, m], 'k--'); ax[1].set_xlabel('chi-square quantile'); ax[1].set_ylabel('observed d2 quantile')
ax[1].set_title('Q-Q: above the line = heavier tail than one Gaussian cluster', fontsize=9)
fig.suptitle(f'{title}: feature space (what python clustered on)', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/7_feature_mahalanobis.png', dpi=110); plt.close(fig)

# ---- 3 + 4 residuals and per-sample variance ----
R = W - T; pca = PCA(3, random_state=0).fit(R); P = pca.transform(R)
fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
for j in range(3):
    dj, pj = dip(P[:, j]); rows.append(['residual PCA', f'residual PC{j + 1} ({pca.explained_variance_ratio_[j]:.1%} var)', f'dip p {pj:.3g}', ''])
    ax[0, j].hist(P[:, j], bins=70, color='#5a5'); ax[0, j].set_title(f'residual PC{j + 1}: {pca.explained_variance_ratio_[j]:.1%} of residual variance | dip p {pj:.3g}', fontsize=9)
    s = np.std(P[:, j]); ax[1, j].plot(T, 'k', lw=2, label='template')
    ax[1, j].plot(T + 2 * s * pca.components_[j], 'r', label='template + 2 sd along it'); ax[1, j].plot(T - 2 * s * pca.components_[j], 'b', label='template - 2 sd')
    ax[1, j].legend(fontsize=7); ax[1, j].set_title(f'what residual PC{j + 1} changes in the shape', fontsize=9)
fig.suptitle(f'{title}: how the spikes differ from the template (PCA of spike - template)', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/8_residuals.png', dpi=110); plt.close(fig)

nv = np.mean(W[:, base].var(1))
km = KMeans(2, n_init=10, random_state=0).fit(W); lab = km.labels_; sil = silhouette_score(W, lab, sample_size=min(n, 3000), random_state=0)
sh = np.bincount(lab) / n; dk = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])
rows += [['per-sample variance', 'max variance / baseline noise variance', f'{W.var(0).max() / nv:.2f} at sample {int(W.var(0).argmax()) + 1}', '~1 = noise only'],
         ['k-means K=2 (waveform)', 'shares / silhouette / distance between means', f'{sh[0]:.2f}, {sh[1]:.2f} / {sil:.3f} / {dk:.1f} uV', 'silhouette > 0.25 = some structure']]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(W.var(0) / nv, 'k', lw=2); ax[0].axhline(1, ls=':', c='grey'); ax[0].axvline(trough, ls='--', c='r', lw=1)
ax[0].set_title('variance across spikes per sample / baseline noise variance (1 = noise only; red = trough)', fontsize=9); ax[0].set_xlabel('sample')
for k, c in zip([0, 1], ['#57a', '#a75']):
    ax[1].plot(km.cluster_centers_[k], color=c, lw=2, label=f'k-means group {k + 1} ({sh[k]:.0%})')
ax[1].plot(T, 'k--', lw=1, label='template'); ax[1].legend(fontsize=7)
ax[1].set_title(f'k-means K=2 in waveform space | silhouette {sil:.2f} | means {dk:.1f} uV apart', fontsize=9)
fig.suptitle(f'{title}: where the extra spread is, and the best two-shape split', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/9_variance_kmeans.png', dpi=110); plt.close(fig)

with open(f'{out}/shape_check.csv', 'w', newline='') as fh:
    csv.writer(fh).writerows([['test', 'what', 'value', 'note']] + rows)
print('\n'.join(' | '.join(r) for r in rows))
