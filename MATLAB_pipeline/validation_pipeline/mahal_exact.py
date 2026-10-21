"""Exact Euclidean -> Mahalanobis moves for one python class of one unit (python second pass).
Re-runs the second pass's re-match (std 1->3) and rescue (std 1->1) with python's own force_membership from the saved
cores (final>0 & ~forced & ~rescued), checks it reproduces the saved final labels, then breaks down the class's
gains and losses -> <report>/10_mahal_forced_rescued/c<k>_euclid_vs_mahal.png (one figure)

    python mahal_exact.py "<channel folder>" <python class> "<report dir (for ppt or for ppt/GT c<g>)>" ["<arm subfolder>"]
"""
import sys, os, csv, glob
import numpy as np
RSD = float(os.environ.get('MCWS_RESCUE_SD', '1'))   # the second pass's rescue radius (RESCUE_SD[1]) of the run
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.environ.get('MCWS_ROOT', '/home/ajb/Documents/GitHub/MicroWireSorter'))   # MicroWireSorter
from mcws.utils.template_matching import force_membership
from mcws.preprocessing.artifact_rejection.base import exclusive_artifact_mask
from mcws.clustering.rescue import RESCUE_ARTIFACT_TYPES
from mcws.io.spike_io import load_spikes

C, KC, REP = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SUB = sys.argv[4] if len(sys.argv) > 4 else ''           # e.g. 'python second pass 10_19'
SC = os.path.dirname(C.rstrip('/'))
arm = sorted(glob.glob(os.path.join(C, SUB, 'python_recluster_*_second_pass')))[-1]

def source_dir():
    """the python second-pass times_ folder the arm was converted from"""
    s = f'{arm}/second_pass_report/SOURCE.txt'
    if os.path.exists(s):
        src = open(s).read().strip()
        t = sorted(glob.glob(f'{src}/times_*/'))
        if t: return t[-1]
    rows = list(csv.DictReader(open(f'{SC}/recluster_conversion_log.csv')))
    r = [x for x in rows if x['folder'] == os.path.basename(C.rstrip('/')) and x['arm'] == os.path.join(SUB, os.path.basename(arm)) and x['status'] == 'converted']
    return os.path.dirname(r[-1]['source'])

src = source_dir()
out = f'{REP}/10_mahal_forced_rescued'
d = next(iter(load_spikes(src).values()))
W = np.asarray(d['spikes'], float); F = np.asarray(d['features'], float)
final_saved = d['cluster_class'][:, 0].astype(int)
forced = np.asarray(d['forced']).astype(bool).ravel()
rescued = np.asarray(d['quarantine']['rescued']).astype(bool).ravel()
artifact = np.asarray(d['quarantine']['all']).astype(bool).ravel()
core = (final_saved > 0) & ~forced & ~rescued
ttl = f"{os.path.basename(C.rstrip('/'))}{' ' + os.path.basename(REP) if 'GT c' in REP else ''}, python second pass c{KC}"

def run(ref, labs, sel, sd):
    got, moved, det = force_membership(waveforms_in=W[ref], classes_in=labs.astype(np.int32), waveforms_out=W[sel],
        first_pass_sdnum=sd[0], template_sdnum=sd[1], mahal_reassign=True, amp_pct_range=None,
        features_in=F[ref], features_out=F[sel], return_mahal_moved=True, return_details=True)
    return got, det

final = np.where(core, final_saved, 0)
rem = ~artifact & ~core
g1, D1 = run(core, final[core], rem, (1.0, 3.0)); final[rem] = g1
resc = exclusive_artifact_mask(d, RESCUE_ARTIFACT_TYPES, n_spikes=len(W)) & (final == 0)
valid = final > 0
g2, D2 = run(valid, final[valid], resc, (1.0, RSD)) if resc.any() else (np.zeros(0, int), None)
final[resc] = g2
same = np.mean(final == final_saved); ndiff = int(np.sum(final != final_saved))

rows = []
for nm, sel, D, sdn in (('forcing', rem, D1, 3), ('rescue', resc, D2, RSD)):
    if D is None or 'mahal_d2' not in D: continue
    idx = np.where(sel)[0]; e = D['euclid_class']; f = D['final_class']
    for j, i in enumerate(idx):
        if e[j] != f[j] and e[j] > 0 and (e[j] == KC or f[j] == KC):
            rows.append(dict(i=i, t=d['cluster_class'][i, 1], pas=nm, euclid=int(e[j]), final=int(f[j]),
                eu_e=D['euclid_dist'][j, e[j] - 1] / D['radius'][e[j] - 1] * sdn,     # in sd of that cluster
                eu_f=D['euclid_dist'][j, f[j] - 1] / D['radius'][f[j] - 1] * sdn,
                md_e=D['mahal_d2'][j, e[j] - 1], md_f=D['mahal_d2'][j, f[j] - 1]))
K = len(D1['radius']); cols = plt.cm.tab10(np.arange(10)); col = lambda k: cols[(k - 1) % 10]
gain = [r for r in rows if r['final'] == KC]; lost = [r for r in rows if r['euclid'] == KC]
note = f're-run reproduces {same:.2%} of final labels' + (f' ({ndiff} differ)' if ndiff else '')

# ---- 3 why: Euclidean vs Mahalanobis distances, and how broad each cluster model is ----
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for kind, S, key in (('gained from', gain, 'euclid'), ('lost to', lost, 'final')):
    for k in sorted({r[key] for r in S}):
        R = [r for r in S if r[key] == k]
        mk = 'o' if kind == 'gained from' else 'x'
        ax[0].scatter([r['eu_e'] for r in R], [r['eu_f'] for r in R], s=6, marker=mk, alpha=.4, color=col(k), label=f'{kind} c{k} (n={len(R)})')
        ax[1].scatter([r['md_e'] for r in R], [r['md_f'] for r in R], s=6, marker=mk, alpha=.4, color=col(k))
for a in ax[:2]:
    m = max(a.get_xlim()[1], a.get_ylim()[1]); a.plot([0, m], [0, m], 'k--', lw=1)
ax[0].set_xlabel('Euclidean distance to the Euclidean choice (sd of that cluster)'); ax[0].set_ylabel('Euclidean distance to the final class (sd of that cluster)')
ax[0].set_title('above the line = farther (waveform) from where Mahalanobis put it', fontsize=9)
if rows: ax[0].legend(fontsize=7, markerscale=2)
ax[1].set_xlabel('Mahalanobis d2 to the Euclidean choice'); ax[1].set_ylabel('Mahalanobis d2 to the final class')
ax[1].set_title('below the line = the final class is closer in feature space (the move)', fontsize=9)
ld = D1.get('model_logdet', np.full(K, np.nan)); n_ = D1['model_n']
ax[2].bar(np.arange(1, K + 1), ld, color=[col(k) for k in range(1, K + 1)])
for k in range(K): ax[2].text(k + 1, ld[k], f'n={n_[k]}', ha='center', va='bottom', fontsize=8)
ax[2].set_xticks(np.arange(1, K + 1)); ax[2].set_xticklabels([f'c{k}' for k in range(1, K + 1)])
ax[2].set_ylabel('log det of the feature covariance (higher = broader)')
ax[2].set_title('spread of each cluster model in the forcing pass\n(the Mahalanobis argmin does not use it)', fontsize=9)
fig.suptitle(f'{ttl}: Euclidean (waveform) vs Mahalanobis (features) choice ({note}; gained {len(gain)}, lost {len(lost)})', fontweight='bold')
fig.tight_layout(); fig.savefig(f'{out}/c{KC}_euclid_vs_mahal.png', dpi=110); plt.close(fig)

print(f'{ttl} | source {src} | {note} | gained {len(gain)} lost {len(lost)}')
