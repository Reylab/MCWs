"""Side-by-side GMM feature selection: MATLAB vs the python port, + KS, on the same clean spikes.
usage: compare_gmm.py <chdir>   (needs clean_spikes.mat + matlab_gmm.mat from gmm_matlab_dump.m)"""
import sys, time, json
import numpy as np, scipy.io as sio
sys.path.insert(0, "/home/ajb/Documents/GitHub/MicroWireSorter")
from mcws.preprocessing.feature_extraction import gmm as G
from mcws.preprocessing.feature_extraction.ks import KSFeatureExtractor

d = sys.argv[1]
spk = sio.loadmat(f"{d}/clean_spikes.mat")["spikes"].astype(float)
M = sio.loadmat(f"{d}/matlab_gmm.mat")
res = {}

# 1. wavelet coefficients
wd = G.haar_coeffs(spk, 4)
res["haar_max_abs_diff"] = float(np.abs(wd - M["wd"]).max())

# 2. metrics + selection from MATLAB's own fits (tests the ported logic exactly)
fits = [G.gmm_metrics_1d(wd[:, j], M["mu"][:, j], M["sd"][:, j], M["a"][:, j]) for j in range(wd.shape[1])]
kj_err = np.nanmax([np.nanmax(np.abs(f["kj"] - M["kj"][:, j])) if np.isfinite(f["kj"]).any() else 0 for j, f in enumerate(fits)])
md_err = max(np.abs(f["med_dist"] - M["md"][:, j]).max() for j, f in enumerate(fits))
mc_same = all(set(f["M_comp"].tolist()) == set(M["Mc"][:, j][M["Mc"][:, j] > 0].astype(int).tolist()) for j, f in enumerate(fits))
res.update(kj_max_abs_diff=float(kj_err), med_idist_max_abs_diff=float(md_err), main_peak_components_same=mc_same)
coeffs_py, info = G.select_from_fits(wd, fits)
m_coeffs = M["coeffs"].ravel().astype(int)
res["from_matlab_fits"] = dict(matlab=m_coeffs.tolist(), python=(coeffs_py + 1).tolist(), identical=bool(np.array_equal(coeffs_py + 1, m_coeffs)),
                               k_select_same=bool(np.allclose(info["k_select"], M["k_select"])),
                               medDist_select_same=bool(np.allclose(info["medDist_select"], M["md_select"])),
                               boundary_10pct_same=bool(np.allclose(info["boundary_10pct"], M["sel10"])),
                               spike_match_same=bool(np.allclose(info["spike_match"], M["sm"])))

# 3. python end to end (its own fits), and KS on the same spikes
t = time.time()
ext = G.GMMFeatureExtractor(scales=4, seed=0, verbose=False)
_, cpy, summ = ext.apply(spk)
res["python_own_fits"] = dict(coeffs=(cpy + 1).tolist(), n=len(cpy), seconds=round(time.time() - t, 1),
                              shared_with_matlab=sorted(set((cpy + 1).tolist()) & set(m_coeffs.tolist())))
ks = KSFeatureExtractor(scales=4, max_inputs=48, min_inputs=10, nd=10, verbose=False)
_, cks, _ = ks.apply(spk)
res["ks"] = dict(coeffs=sorted((np.asarray(cks) + 1).tolist()), n=len(cks),
                 shared_with_matlab_gmm=sorted(set((np.asarray(cks) + 1).tolist()) & set(m_coeffs.tolist())))
# redundancy of each set: mean |corr| between selected coefficients
def red(c):
    c = np.asarray(c) - 1
    if len(c) < 2: return float("nan")
    R = np.corrcoef(wd[:, c].T); return float(np.mean(np.abs(R[np.triu_indices(len(c), 1)])))
res["mean_abs_corr"] = dict(matlab_gmm=red(m_coeffs), python_gmm=red(cpy + 1), ks=red(np.asarray(cks) + 1))
json.dump(res, open(f"{d}/compare.json", "w"), indent=1)
print(json.dumps(res, indent=1))
