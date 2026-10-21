"""Every spike that went through the reassignment in a python second pass (10_19 code, reassign_rule "combined"):
Euclidean distance, Mahalanobis distance and the combined score to each cluster that accepted it, and the
feature covariance of every cluster model the Mahalanobis distance used.

Re-runs the second pass's two template matches (re-match std 1->3 of every clean non-core spike, rescue std 1->1 of
bundle / within-channel artifacts) with python's own force_membership from the saved cores, checks it reproduces the
saved final labels, and captures the cluster models (_feature_models inputs) each match used.

  score_k = e_k / min_j e_j + m_k / min_j m_j   over the clusters j whose radius accepted the spike
  (e = Euclidean distance to the refined template, m = sqrt of the feature-space Mahalanobis d2); lowest wins.

    python reassign_scores.py "<channel folder>" "<arm subfolder>" "<out dir>"
"""
import sys, os, glob, csv
import numpy as np
RSD = float(os.environ.get('MCWS_RESCUE_SD', '1'))   # the second pass's rescue radius (RESCUE_SD[1]) of the run
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.environ.get('MCWS_ROOT', '/home/ajb/Documents/GitHub/MicroWireSorter'))   # MicroWireSorter
import mcws.utils.template_matching as TM
from mcws.preprocessing.artifact_rejection.base import exclusive_artifact_mask
from mcws.clustering.rescue import RESCUE_ARTIFACT_TYPES
from mcws.io.spike_io import load_spikes

C, SUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
arm = sorted(glob.glob(os.path.join(C, SUB, 'python_recluster_*_second_pass')))[-1]
src = open(os.path.join(arm, 'second_pass_report', 'SOURCE.txt')).read().strip()
d = next(iter(load_spikes(sorted(glob.glob(f'{src}/times_*/'))[-1]).values()))
W = np.asarray(d['spikes'], float); F = np.asarray(d['features'], float)
fs = d['cluster_class'][:, 0].astype(int)
forced = np.asarray(d['forced']).astype(bool).ravel(); rescued = np.asarray(d['quarantine']['rescued']).astype(bool).ravel()
artifact = np.asarray(d['quarantine']['all']).astype(bool).ravel()
core = (fs > 0) & ~forced & ~rescued
name = os.path.basename(C.rstrip('/'))

captured = []                                   # every (classes, features) a cluster model was built from
_orig = TM._feature_models
def _capture(classes, features):
    captured.append((np.asarray(classes).copy(), np.asarray(features).copy()))
    return _orig(classes, features)
TM._feature_models = _capture

def run(ref, labs, sel, sd):
    captured.clear()
    got, _, det = TM.force_membership(waveforms_in=W[ref], classes_in=labs.astype(np.int32), waveforms_out=W[sel],
        first_pass_sdnum=sd[0], template_sdnum=sd[1], mahal_reassign=True, amp_pct_range=None,
        features_in=F[ref], features_out=F[sel], return_mahal_moved=True, return_details=True)
    return got, det, (captured[-1] if captured else None)

final = np.where(core, fs, 0); rem = ~artifact & ~core
g1, D1, M1 = run(core, final[core], rem, (1.0, 3.0)); final[rem] = g1
resc = exclusive_artifact_mask(d, RESCUE_ARTIFACT_TYPES, n_spikes=len(W)) & (final == 0)
valid = final > 0
if resc.any():
    g2, D2, M2 = run(valid, final[valid], resc, (1.0, RSD)); final[resc] = g2
else:
    D2, M2 = None, None
same = np.mean(final == fs)
K = int(fs.max()); cols = plt.cm.tab10(np.arange(10)); col = lambda k: cols[(k - 1) % 10]
note = f're-run reproduces {same:.2%} of the saved final labels; rule: {D1.get("reassign_rule", "?")}'
print(name, note)

def table(D, sdn):
    """per accepted spike: allowed mask, e (in sd of each cluster), m, score, Euclidean choice, final"""
    e_raw = np.asarray(D['euclid_dist'], float); m = np.sqrt(np.maximum(np.asarray(D['mahal_d2'], float), 0))
    al = np.asarray(D['allowed']).astype(bool); k = min(e_raw.shape[1], m.shape[1], al.shape[1])
    e_raw, m, al = e_raw[:, :k], m[:, :k], al[:, :k]
    ec, fc = np.asarray(D['euclid_class']), np.asarray(D['final_class'])
    acc = fc > 0
    E = np.where(al, e_raw, np.inf); Mm = np.where(al, m, np.inf)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = E / E.min(1, keepdims=True) + Mm / Mm.min(1, keepdims=True)
    e_sd = e_raw / (np.asarray(D['radius'], float)[:k] / sdn)            # distance in sd of that cluster
    return dict(al=al[acc], e=e_sd[acc], e_raw=e_raw[acc], m=m[acc], S=S[acc], ec=ec[acc], fc=fc[acc], n=al[acc].sum(1),
                model_n=np.asarray(D.get('model_n', np.full(k, np.nan))), logdet=np.asarray(D.get('model_logdet', np.full(k, np.nan))))

steps = [('forcing (re-match, std 1->3)', D1, M1, 3)] + ([(f'rescue (std 1->{RSD:g})', D2, M2, RSD)] if D2 is not None else [])
rows = [['step', 'cluster', 'model_spikes', 'logdet_cov', 'trace_cov', 'accepted_alone', 'contested', 'won_euclid', 'won_final', 'net']]
for si, (sname, D, Mcap, sdn) in enumerate(steps):
    T = table(D, sdn); k = T['al'].shape[1]; tag = ['forcing', 'rescue'][si]
    fi = T['fc'] - 1; idx = np.arange(len(fi))
    # ---- 1 distributions: to the winning cluster / to the other accepting clusters ----
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    for r, (key, lab, lim) in enumerate((('e', 'Euclidean distance (sd of that cluster)', None),
                                         ('m', 'Mahalanobis distance (sqrt d2, features)', None),
                                         ('S', 'score  e/min e + m/min m  (2 = best on both)', None))):
        win = T[key][idx, fi]
        lose_mask = T['al'].copy(); lose_mask[idx, fi] = False
        hi = np.nanpercentile(np.r_[win[np.isfinite(win)], T[key][lose_mask & np.isfinite(T[key])]], 99) if lose_mask.any() else np.nanpercentile(win, 99)
        if key == 'S':
            # winners: most score exactly 2 (nearest on both) -> counted, the rest in fine bins from 2 up;
            # losers: spread well above 2 -> 100 bins from 2 to their 99th percentile
            wv = win[np.isfinite(win)]; above = wv[wv > 2 + 1e-9]
            lv = T[key][lose_mask]; lv = lv[np.isfinite(lv)]
            e_w = np.linspace(2, min(3, np.percentile(above, 99)) if above.size else 2.1, 101)
            e_l = np.linspace(2, np.percentile(lv, 99) if lv.size else 3, 101)
        else:
            e_w = e_l = np.linspace(0, max(hi, 1e-3), 60)
        for c in range(1, k + 1):
            wc = win[T['fc'] == c]; wc = wc[np.isfinite(wc)]
            if key == 'S': wc = wc[wc > 2 + 1e-9]
            if wc.size: ax[r, 0].hist(wc, e_w, histtype='step', lw=2, color=col(c), label=f'c{c} (n={wc.size})')
            lc = T[key][:, c - 1][lose_mask[:, c - 1]]; lc = lc[np.isfinite(lc)]
            if lc.size: ax[r, 1].hist(lc, e_l, histtype='step', lw=2, color=col(c), label=f'c{c} (n={lc.size})')
        ax[r, 0].set_xlabel(lab); ax[r, 1].set_xlabel(lab)
        if key == 'S':
            ax[r, 0].set_xlim(e_w[0], e_w[-1]); ax[r, 1].set_xlim(e_l[0], e_l[-1])
            ax[r, 0].text(.98, .6, f'{np.sum(wv <= 2 + 1e-9)} of {wv.size} score exactly 2 (nearest on both; not drawn)\n'
                          f'{np.sum(wv > e_w[-1])} above {e_w[-1]:.2f} (not drawn)', transform=ax[r, 0].transAxes, ha='right', fontsize=8)
            ax[r, 1].text(.98, .6, f'{np.sum(lv > e_l[-1])} of {lv.size} above {e_l[-1]:.2f} (not drawn)', transform=ax[r, 1].transAxes, ha='right', fontsize=8)
            ax[r, 0].set_title('to the cluster that GOT the spike (score above 2 only)', fontsize=9)
        ax[r, 0].set_title(f'to the cluster that GOT the spike', fontsize=9); ax[r, 1].set_title('to the other clusters that also accepted it (lost it)', fontsize=9)
        for a in ax[r]: a.legend(fontsize=7); a.set_ylabel('spikes')
    fig.suptitle(f'{name}, {SUB}, {sname}: {len(fi)} accepted spikes, {int((T["n"] > 1).sum())} accepted by 2+ clusters\n({note})', fontweight='bold', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/{si + 1}a_{tag}_distances_and_score.png', dpi=110); plt.close(fig)

    # ---- 2 bias: contested spikes (accepted by 2+ clusters) ----
    ct = T['n'] > 1
    won_e = np.bincount(T['ec'][ct], minlength=k + 1)[1:k + 1]; won_f = np.bincount(T['fc'][ct], minlength=k + 1)[1:k + 1]
    covs = {}
    if Mcap is not None:
        cls, feat = Mcap
        for c in range(1, k + 1):
            X = feat[cls == c]
            if X.shape[0] > X.shape[1] + 2:
                cv = np.cov(X.T); covs[c] = (0.7 * cv + 0.3 * np.diag(np.diag(cv)) + 1e-6 * np.eye(cv.shape[0]), X.shape[0])
    tr = np.array([np.trace(covs[c][0]) if c in covs else np.nan for c in range(1, k + 1)])
    for c in range(1, k + 1):
        rows.append([tag, c, int(T['model_n'][c - 1]) if c - 1 < len(T['model_n']) else '', f"{T['logdet'][c - 1]:.1f}", f'{tr[c - 1]:.1f}',
                     int(((T['n'] == 1) & (T['fc'] == c)).sum()), int((ct & T['al'][:, c - 1]).sum()), int(won_e[c - 1]), int(won_f[c - 1]), int(won_f[c - 1] - won_e[c - 1])])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    x = np.arange(1, k + 1)
    ax[0].bar(x - .2, won_e, .4, color='#999', label='Euclidean nearest'); ax[0].bar(x + .2, won_f, .4, color=[col(c) for c in x], label='after the combined score')
    ax[0].set_xticks(x); ax[0].set_xticklabels([f'c{c}' for c in x]); ax[0].legend(fontsize=8)
    ax[0].set_title(f'contested spikes ({int(ct.sum())}, accepted by 2+ clusters): who gets them', fontsize=9)
    net = won_f - won_e
    ax[1].scatter(T['logdet'][:k], net, s=np.clip(T['model_n'][:k] / max(T['model_n'][:k].max(), 1) * 600, 30, 600), c=[col(c) for c in x])
    for c in x: ax[1].annotate(f'c{c} (n={int(T["model_n"][c - 1])})', (T['logdet'][c - 1], net[c - 1]), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax[1].axhline(0, c='k', lw=.8); ax[1].set_xlabel('log det of the cluster model covariance (broader ->)'); ax[1].set_ylabel('net spikes gained from the score (final - Euclidean)')
    rr = f"r = {np.corrcoef(T['logdet'][:k], net)[0, 1]:.2f}" if k > 2 else 'r: n/a (2 clusters)'
    ax[1].set_title(f'net gain vs spread of the model (dot size = model spikes); {rr}', fontsize=9)
    mv = ct & (T['fc'] != T['ec'])
    if mv.any():
        ii = np.where(mv)[0]
        er = T['e_raw'][ii, T['fc'][ii] - 1] / T['e_raw'][ii, T['ec'][ii] - 1]
        mr = T['m'][ii, T['fc'][ii] - 1] / T['m'][ii, T['ec'][ii] - 1]
        ax[2].scatter(er, mr, s=5, alpha=.4, c=[col(c) for c in T['fc'][ii]])
        ax[2].axhline(1, c='k', lw=.8); ax[2].axvline(1, c='k', lw=.8)
        xx = np.linspace(1, max(er.max(), 1.01), 50); ax[2].plot(xx, 2 - xx, 'k--', lw=1, label='e ratio + m ratio = 2 (moves below)')
        ax[2].set_xlim(.95, max(er.max(), 1.01) * 1.02); ax[2].legend(fontsize=8)
    ax[2].set_xlabel('Euclidean distance: winner / Euclidean-nearest (>1 = farther in waveform)'); ax[2].set_ylabel('Mahalanobis distance: winner / Euclidean-nearest')
    ax[2].set_title(f'the {int(mv.sum())} spikes the score moved off their Euclidean-nearest cluster (colour = winner)', fontsize=9)
    fig.suptitle(f'{name}, {sname}: does the combined score favour broad clusters?', fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{OUT}/{si + 1}b_{tag}_bias.png', dpi=110); plt.close(fig)

    # ---- 3 the cluster models' covariances (what the Mahalanobis distance used) ----
    if covs:
        cl = sorted(covs); vmax = max(np.abs(covs[c][0]).max() for c in cl)
        fig, ax = plt.subplots(2, len(cl) + 1, figsize=(4.3 * (len(cl) + 1), 8.4), squeeze=False)
        for j, c in enumerate(cl):
            S_, n_ = covs[c]
            im = ax[0, j].imshow(S_, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax[0, j].set_title(f'c{c}: n={n_}, trace {np.trace(S_):.0f}, log det {np.linalg.slogdet(S_)[1]:.1f}', fontsize=9)
            ax[0, j].set_xlabel('feature'); ax[0, j].set_ylabel('feature')
            ev = np.sort(np.linalg.eigvalsh(S_))[::-1]
            ax[1, -1].semilogy(np.arange(1, len(ev) + 1), ev, '-o', ms=3, color=col(c), label=f'c{c}')
            ax[1, j].bar(np.arange(S_.shape[0]), np.diag(S_), color=col(c)); ax[1, j].set_title(f'c{c}: variance per feature (diagonal)', fontsize=9)
            ax[1, j].set_xlabel('feature')
        fig.colorbar(im, ax=ax[0, :len(cl)].tolist(), shrink=.8, label='covariance (shrunk: 0.7 full + 0.3 diag)')
        ax[0, -1].axis('off'); ax[1, -1].set_xlabel('component'); ax[1, -1].set_ylabel('eigenvalue (log)'); ax[1, -1].legend(fontsize=8)
        ax[1, -1].set_title('eigenvalues: higher curve = broader model', fontsize=9)
        fig.suptitle(f'{name}, {sname}: feature covariance of each cluster model (same colour scale; the models the Mahalanobis distance used)', fontweight='bold', fontsize=10)
        fig.savefig(f'{OUT}/{si + 1}c_{tag}_cluster_covariances.png', dpi=110, bbox_inches='tight'); plt.close(fig)
with open(f'{OUT}/clusters.csv', 'w', newline='') as fh: csv.writer(fh).writerows(rows)
print('saved', OUT)
