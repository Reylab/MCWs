"""Per cluster of the newest python sort: features F, SPC cores n, n/F (how
well the covariance is estimated), how Gaussian the cores are (squared
Mahalanobis of the cores to their own shrunk model vs chi-square(F)), and
log|Sigma|.  -> <out_dir>/covariance_check.csv"""
import csv, glob, sys
from pathlib import Path
import numpy as np, scipy.io as sio
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvw_common import OUT, MWS, channels, second_pass_dir
sys.path.insert(0, str(MWS))
from mcws.utils.template_matching import _feature_models, _feat_mahal

rows = []
for ch, C in channels().items():
    f = sorted(glob.glob(str(second_pass_dir(C["run"]) / "times_*" / "times_*.mat")))[0]
    T = sio.loadmat(f, simplify_cells=True, variable_names=["cluster_class", "features", "forced", "quarantine"])
    lab = T["cluster_class"][:, 0].astype(int); F = np.asarray(T["features"], float)
    q = T["quarantine"]; art = np.asarray(q["all"]).astype(bool).ravel()
    resc = np.asarray(q.get("rescued", 0)).astype(bool).ravel()
    core = (lab > 0) & ~np.asarray(T["forced"]).astype(bool).ravel() & ~resc & ~art
    means, inv = _feature_models(lab[core], F[core])
    d2 = _feat_mahal(F[core], means, inv)
    nf = F.shape[1]
    for k in range(1, lab.max() + 1):
        m = lab[core] == k
        if m.sum() < 2:
            continue
        dk = d2[m, k - 1]
        rows.append(dict(ch=ch, cluster=k, features=nf, cores=int(m.sum()), cores_per_feature=round(m.sum() / nf, 1),
                         median_d2=round(float(np.median(dk)), 2), chi2_median=round(float(chi2.median(nf)), 2),
                         pct_beyond_chi2_99=round(100 * float(np.mean(dk > chi2.ppf(.99, nf))), 2),
                         log_det_sigma=round(-float(np.linalg.slogdet(inv[k - 1])[1]), 2)))
with open(OUT / "covariance_check.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(OUT / "covariance_check.csv")
