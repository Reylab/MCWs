"""KS vs GMM features, everything else identical: same detections (the run's
spikes_<ts>/), same artifact rejection, same SPC seeds as the original run,
same forcing / rescue / second pass. Writes only under
/mnt/data0/recluster_runs/<run>/gmm_feature_test/<variant>/.
usage: run_feature_test.py <run> <variant: ks|gmm> [gmm_seed]"""
import glob, logging, sys, time
from pathlib import Path
import numpy as np, scipy.io as sio
MWS = Path("/home/ajb/Documents/GitHub/MicroWireSorter")
sys.path[:0] = [str(MWS), str(MWS / "recluster_test")]
sys.path.insert(0, str(MWS))
import test_full_pipeline as TFP                      # the run's own settings (CLUSTER_KWARGS, RESCUE_*, FEATURE_KWARGS)
from mcws.io.spike_io import load_spikes
from mcws.preprocessing.artifact_rejection import apply_artifact_rejection
from mcws.preprocessing.feature_extraction.api import apply_feature_extraction
from mcws.clustering.api import apply_clustering
from mcws.clustering.rescue import apply_spike_rescue
from second_pass import run_second_pass

run, variant = sys.argv[1], sys.argv[2]
gseed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
R = Path("/mnt/data0/recluster_runs") / run
out = R / "gmm_feature_test" / (variant if variant in ("ks", "ks_allfit") else f"gmm_seed{gseed}")
out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=out / "run.log", level=logging.INFO)
spk_dir = sorted(R.glob("spikes_2026-*"))[0]
orig = sorted(R.glob("times_2026-*"))[0]                  # first (clustered) sort: its SPC seeds
cm = sio.loadmat(sorted(orig.glob("times_*.mat"))[0], squeeze_me=True, struct_as_record=False,
                 variable_names=["clustering_metadata"])["clustering_metadata"]
seeds = dict(spc_seed=int(cm.spc_seed), permut_seed=int(cm.permut_seed))
AT = ["refractory_period", "bundle_collision", "within_channel"]
t0 = time.time()
res = load_spikes(str(spk_dir))
res = apply_artifact_rejection(data=res, rejectors=AT, rejector_kwargs={"refractory_period": {"refractory_ms": 1.5}}, n_workers=1)
if variant == "ks":
    res = apply_feature_extraction(data=res, method="ks", n_workers=1, artifact_types=AT, extractor_kwargs=TFP.FEATURE_KWARGS)
elif variant == "ks_allfit":   # KS coefficient selection on ALL spikes; the quarantine mask is applied only at clustering
    res = apply_feature_extraction(data=res, method="ks", n_workers=1, artifact_types=[], extractor_kwargs=TFP.FEATURE_KWARGS)
else:
    res = apply_feature_extraction(data=res, method="gmm", n_workers=1, artifact_types=AT, extractor_kwargs={"seed": gseed})
ch = next(iter(res))
sel = np.asarray(res[ch]["selected_coeffs"]).ravel()
print(f"[{run} {variant}] features: {len(sel)} coefficients {(sel + 1).tolist()}", flush=True)
res = apply_clustering(data=res, method="spc", n_workers=1, artifact_types=AT, save=True, output_dir=str(out),
                       clusterer_kwargs={**TFP.CLUSTER_KWARGS, **seeds})
res = apply_spike_rescue(results=res, artifact_types=TFP.RESCUE_ARTIFACT_TYPES, n_workers=1, save=True,
                         output_dir=str(out), **TFP.RESCUE_KWARGS)
sp = run_second_pass(res, str(out), n_workers=1)
(out / "features.txt").write_text(f"{variant} seed {gseed}: {len(sel)} coefficients (1-based) {(sel + 1).tolist()}\nSPC seeds {seeds}\n")
print(f"[{run} {variant}] done in {time.time() - t0:.0f}s -> {out}  second pass {sp}", flush=True)
