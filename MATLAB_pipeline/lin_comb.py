import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.io import loadmat
from scipy.stats import trim_mean
from scipy.linalg import toeplitz, cho_factor, cho_solve, solve_triangular
from scipy.optimize import lsq_linear


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class RescueConfig:
    # Group detections into a chain if consecutive detections
    # are closer than this.
    chain_ms: float = 1.5
    # Spike must have no neighboring detection within this
    # interval to be used for template/noise training.
    isolated_ms: float = 2.0
    # Absolute same-unit refractory period.
    refractory_ms: float = 1.0
    # Minimum number of isolated examples required for a unit.
    min_template_spikes: int = 40
    # Fraction trimmed from EACH tail when learning template.
    trim_fraction: float = 0.10
    # Number of ACF lags used for colored-noise model.
    n_acf: int = 20
    # Diagonal covariance regularization.
    ridge: float = 1e-4
    # Retain this fraction of lowest-template-error residuals
    # when estimating noise.
    noise_keep_fraction: float = 0.60
    # Amplitude limits learned from isolated spikes.
    amp_percentiles: tuple = (1.0, 99.0)
    # Maximum inferred spikes in one chain.
    max_spikes: int = 6
    # Local timing refinement after greedy selection.
    timing_refine_samples: int = 2
    # Monte-Carlo null simulations for score threshold.
    n_null: int = 10000
    # Family-wise probability that pure noise produces
    # at least one candidate above threshold.
    alpha: float = 0.001
    # Reject rescue if whitened residual variance is enormous.
    # Ideal value is around 1 if the noise model is correct.
    max_whitened_resid_var: float = 4.0
    # Random seed.
    seed: int = 42


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class UnitModel:
    k: int
    T: np.ndarray
    a_lo: float = np.nan
    a_hi: float = np.nan
    a_mu: float = np.nan
    a_sd: float = np.nan
    n_train: int = 0
    heldout: np.ndarray | None = None


@dataclass
class RescueBank:
    """
    Every possible (unit, time) placement for a window.
    """

    A: np.ndarray
    Aw: np.ndarray
    units: np.ndarray
    tau: np.ndarray
    pos: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    den: np.ndarray
    n: int


# ============================================================
# 1. LOAD WAVE_CLUS CHANNEL
# ============================================================
def load_channel(path):
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    par = m["par"]
    times_ms = np.atleast_1d(np.asarray(m["index"], float))
    n = len(times_ms)
    spikes = np.atleast_2d(np.asarray(m["spikes"], float))
    cluster_class = np.asarray(m["cluster_class"], float)
    cluster_class = np.atleast_2d(cluster_class)
    # Depending on MATLAB squeezing/orientation.
    if cluster_class.shape[0] != n and cluster_class.shape[1] == n:
        cluster_class = cluster_class.T
    cls = cluster_class[:, 0].astype(int)
    q = m.get("quarantine", None)

    def qfield(name):
        if q is None:
            return np.zeros(n, dtype=bool)
        value = getattr(q, name, None)
        if value is None:
            return np.zeros(n, dtype=bool)
        value = np.atleast_1d(np.asarray(value, float))
        out = np.zeros(n, dtype=bool)
        out[: min(n, len(value))] = value[:n] > 0
        return out

    sr = float(par.sr)
    wpre = int(par.w_pre)
    wpost = int(par.w_post)
    s = np.round(times_ms / 1000.0 * sr).astype(np.int64)
    thresholds = np.atleast_1d(np.asarray(m["thresholds"], float))
    ch = {
        "label": str(np.atleast_1d(m["label"])[0]),
        "sr": sr,
        "wpre": wpre,
        "wpost": wpost,
        "L": wpre + wpost,
        "peak_off": wpre - 1,
        "stdmin": float(par.stdmin),
        "threshold": float(np.mean(thresholds)),
        "spikes": spikes,
        "t_ms": times_ms,
        "s": s,
        "cls": cls,
        "q_refr": qfield("refractory_period"),
    }
    # Explicitly sort everything by sample time.
    order = np.argsort(ch["s"])
    for key in ["spikes", "t_ms", "s", "cls", "q_refr"]:
        ch[key] = ch[key][order]
    return ch


# ============================================================
# 2. CHAINS AND ISOLATED EVENTS
# ============================================================
def build_chains(s, R):
    """
    Transitive connected components in time.
    If:
        s[i+1] - s[i] < R
    they belong to the same chain.
    """
    s = np.asarray(s)
    if len(s) == 0:
        return np.empty((0, 2), dtype=int)
    if len(s) == 1:
        return np.array([[0, 0]], dtype=int)
    breaks = np.where(np.diff(s) >= R)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(s) - 1]
    return np.c_[starts, ends].astype(int)


def isolated_mask(s, R):
    """
    Detection must be farther than R from both neighbors.
    """
    s = np.asarray(s)
    if len(s) <= 1:
        return np.ones(len(s), dtype=bool)
    d = np.diff(s).astype(float)
    left = np.r_[np.inf, d]
    right = np.r_[d, np.inf]
    return (left > R) & (right > R)


# ============================================================
# 3. STITCH STORED SNIPPETS
# ============================================================
def stitch(ch, i0, i1):
    """
    Recover local continuous trace covered by snippets i0..i1.
    """
    off = ch["peak_off"]
    L = ch["L"]
    s = ch["s"]
    start = int(s[i0] - off)
    stop = int(s[i1] - off + L)
    x = np.full(stop - start, np.nan, dtype=float)
    for i in range(i0, i1 + 1):
        a = int(s[i] - off - start)
        b = a + L
        new = ch["spikes"][i]
        # Check overlapping stored values are identical.
        existing = x[a:b]
        overlap = np.isfinite(existing)
        if np.any(overlap):
            scale = max(np.max(np.abs(new[overlap])), 1e-12)
            mismatch = np.max(np.abs(existing[overlap] - new[overlap])) / scale
            if mismatch > 1e-6:
                raise ValueError(
                    f"Snippet overlap mismatch "
                    f"({mismatch:.3e}) in candidates "
                    f"{i0}..{i1}"
                )
        x[a:b] = new
    if np.any(~np.isfinite(x)):
        raise ValueError(
            "Chain snippets do not continuously cover the "
            "requested window. Reduce chain_ms or use raw trace."
        )
    return x, start


def check_all_snippet_overlaps(ch):
    """
    Diagnostic: maximum mismatch between every overlapping
    neighboring snippet.
    """
    s = ch["s"]
    L = ch["L"]
    errors = []
    for i in range(len(s) - 1):
        lag = int(s[i + 1] - s[i])
        if 0 < lag < L:
            a = ch["spikes"][i, lag:]
            b = ch["spikes"][i + 1, : L - lag]
            scale = max(np.max(np.abs(a)), 1e-12)
            errors.append(np.max(np.abs(a - b)) / scale)
    return {
        "n_overlap_pairs": len(errors),
        "max_relative_error": float(max(errors)) if errors else np.nan,
    }


# ============================================================
# 4. INITIAL TEMPLATE LEARNING
# ============================================================
def learn_initial_templates(ch, iso, cfg, rng):
    """
    Learn waveform SHAPES before noise covariance is available.
    """
    units = {}
    cluster_ids = sorted(set(ch["cls"]) - {0})
    for k in cluster_ids:
        idx = np.where(iso & (ch["cls"] == k))[0]
        if len(idx) < cfg.min_template_spikes:
            continue
        p = rng.permutation(idx)
        # Half fitting, half held-out.
        nfit = (len(p) + 1) // 2
        fit_idx = p[:nfit]
        held_idx = p[nfit:]
        W = ch["spikes"][fit_idx]
        T = trim_mean(W, proportiontocut=cfg.trim_fraction, axis=0)
        norm = np.linalg.norm(T)
        if not np.isfinite(norm) or norm <= 0:
            continue
        T = T / norm
        units[k] = UnitModel(
            k=k, T=T, n_train=len(fit_idx), heldout=ch["spikes"][held_idx]
        )
    return units


# ============================================================
# 5. COLORED-NOISE MODEL
# ============================================================
class NoiseModel:
    def __init__(self, acf, ridge=1e-4):
        self.acf = np.asarray(acf, float)
        self.s2 = float(self.acf[0])
        self.ridge = ridge
        self._cov = {}
        self._cho = {}
        self._chol = {}

    def cov(self, n):
        if n not in self._cov:
            g = min(len(self.acf), n)
            acf = self.acf[:g]
            # Bartlett taper to improve covariance stability.
            taper = 1.0 - np.arange(g) / max(g, 1)
            taps = acf * taper
            first_col = np.r_[taps, np.zeros(max(0, n - g))][:n]
            C = toeplitz(first_col)
            C += self.ridge * max(self.s2, 1e-12) * np.eye(n)
            self._cov[n] = C
        return self._cov[n]

    def solve(self, n, B):
        if n not in self._cho:
            self._cho[n] = cho_factor(
                self.cov(n), lower=True, check_finite=False
            )
        return cho_solve(self._cho[n], B, check_finite=False)

    def chol(self, n):
        if n not in self._chol:
            self._chol[n] = np.linalg.cholesky(self.cov(n))
        return self._chol[n]

    def whiten(self, x):
        x = np.asarray(x, float)
        C = self.chol(len(x))
        return solve_triangular(C, x, lower=True, check_finite=False)


def estimate_noise(ch, units, iso, cfg):
    """
    Estimate colored background noise from template-subtracted
    isolated snippets.
    If raw spike-free continuous data are available, estimating
    covariance from those segments is preferable.
    """
    residuals = []
    for k, u in units.items():
        idx = np.where(iso & (ch["cls"] == k))[0]
        if len(idx) == 0:
            continue
        W = ch["spikes"][idx]
        # Initial Euclidean projection only for noise estimation.
        a = W @ u.T
        R = W - a[:, None] * u.T[None, :]
        residual_fraction = np.sum(R**2, axis=1) / np.maximum(
            np.sum(W**2, axis=1), 1e-12
        )
        cutoff = np.quantile(residual_fraction, cfg.noise_keep_fraction)
        residuals.append(R[residual_fraction <= cutoff])
    if not residuals:
        raise RuntimeError("Could not obtain residuals for noise estimation.")
    R = np.vstack(residuals)
    # Remove any tiny DC bias.
    R = R - np.mean(R)
    L = R.shape[1]
    n_acf = min(cfg.n_acf, L)
    acf = np.empty(n_acf)
    for lag in range(n_acf):
        acf[lag] = np.mean(R[:, : L - lag] * R[:, lag:])
    if not np.isfinite(acf[0]) or acf[0] <= 0:
        raise RuntimeError("Invalid estimated noise variance.")
    return NoiseModel(acf, ridge=cfg.ridge), R


# ============================================================
# 6. GLS-CONSISTENT AMPLITUDE DISTRIBUTIONS
# ============================================================
def fit_unit_amplitude_statistics(ch, units, iso, noise, cfg):
    """
    Estimate amplitude distributions using the SAME GLS metric
    used during rescue.
        a = (T' Sigma^-1 W) / (T' Sigma^-1 T)
    """
    L = ch["L"]
    for k, u in units.items():
        idx = np.where(iso & (ch["cls"] == k))[0]
        W = ch["spikes"][idx]
        SiT = noise.solve(L, u.T)
        den = float(u.T @ SiT)
        a = (W @ SiT) / max(den, 1e-12)
        lo, hi = np.percentile(a, cfg.amp_percentiles)
        mu = float(np.median(a))
        mad = float(np.median(np.abs(a - mu)))
        sd = max(1.4826 * mad, 1e-6)
        # Template direction normally means positive amplitudes.
        # Avoid allowing sign reversal if training amplitudes are
        # clearly positive.
        if mu > 0:
            lo = max(0.0, float(lo))
        u.a_lo = float(lo)
        u.a_hi = float(hi)
        u.a_mu = mu
        u.a_sd = sd


# ============================================================
# 7. TEMPLATE IDENTIFIABILITY / COHERENCE
# ============================================================
def template_coherence(units, noise):
    ks = sorted(units)
    if not ks:
        return [], np.empty((0, 0))
    T = np.vstack([units[k].T for k in ks])
    L = T.shape[1]
    Si = noise.solve(L, T.T).T
    G = T @ Si.T
    d = np.sqrt(np.maximum(np.diag(G), 1e-12))
    C = np.abs(G / np.outer(d, d))
    return ks, C


# ============================================================
# 8. BUILD TEMPLATE/TIME BANK
# ============================================================
def make_bank(units, n, noise, peak_off):
    ks = sorted(units)
    if not ks:
        raise RuntimeError("No learned units.")
    L = len(units[ks[0]].T)
    if n < L:
        raise ValueError("Rescue window shorter than template.")
    rows = []
    unit_ids = []
    tau = []
    pos = []
    lo = []
    hi = []
    for k in ks:
        T = units[k].T
        for t in range(n - L + 1):
            v = np.zeros(n)
            v[t : t + L] = T
            rows.append(v)
            unit_ids.append(k)
            tau.append(t)
            pos.append(t + peak_off)
            lo.append(units[k].a_lo)
            hi.append(units[k].a_hi)
    A = np.asarray(rows).T
    C = noise.chol(n)
    Aw = solve_triangular(C, A, lower=True, check_finite=False)
    den = np.sum(Aw * Aw, axis=0)
    return RescueBank(
        A=A,
        Aw=Aw,
        units=np.asarray(unit_ids, int),
        tau=np.asarray(tau, int),
        pos=np.asarray(pos, int),
        lo=np.asarray(lo, float),
        hi=np.asarray(hi, float),
        den=np.maximum(den, 1e-12),
        n=n,
    )


# ============================================================
# 9. NULL CALIBRATION
# ============================================================
def calibrate_score_threshold(bank, cfg, rng):
    """
    Calibrate maximum candidate score under whitened Gaussian
    noise.
    Because x_w ~ N(0, I), no covariance matrix needs to be
    generated explicitly.
    Score = reduction in whitened SSE from adding one bounded
    template to the current residual.
    """
    n = bank.n
    m = bank.Aw.shape[1]
    maxima = np.empty(cfg.n_null, dtype=float)
    # Batch to keep memory controlled.
    batch = 500
    done = 0
    while done < cfg.n_null:
        b = min(batch, cfg.n_null - done)
        Z = rng.standard_normal((b, n))
        num = Z @ bank.Aw
        amp = np.clip(
            num / bank.den[None, :], bank.lo[None, :], bank.hi[None, :]
        )
        gain = 2.0 * amp * num - amp**2 * bank.den[None, :]
        maxima[done : done + b] = np.max(gain, axis=1)
        done += b
    threshold = float(np.quantile(maxima, 1.0 - cfg.alpha))
    return threshold, maxima


# ============================================================
# 10. BOUNDED JOINT FIT
# ============================================================
def bounded_joint_fit(xw, bank, selected):
    """
    Proper bounded least squares.
    NOT:
        unconstrained LS -> np.clip()
    but:
        min ||xw - A a||^2
        subject to lo <= a <= hi
    """
    selected = list(selected)
    if len(selected) == 0:
        return (np.zeros(0), xw.copy())
    idx = np.asarray(selected, dtype=int)
    X = bank.Aw[:, idx]
    lo = bank.lo[idx]
    hi = bank.hi[idx]
    result = lsq_linear(
        X, xw, bounds=(lo, hi), method="trf", lsmr_tol="auto", max_iter=300
    )
    a = result.x
    residual = xw - X @ a
    return a, residual


# ============================================================
# 11. REFRACTORY CHECK
# ============================================================
def valid_refractory(bank, selected, refractory_samples):
    selected = list(selected)
    for i in range(len(selected)):
        ji = selected[i]
        ki = bank.units[ji]
        pi = bank.pos[ji]
        for j in range(i):
            jj = selected[j]
            kj = bank.units[jj]
            pj = bank.pos[jj]
            if ki == kj and abs(pi - pj) < refractory_samples:
                return False
    return True


# ============================================================
# 12. RESCUE ONE CHAIN
# ============================================================
def rescue_chain(x, bank, score_threshold, refractory_samples, cfg, noise):
    """
    Infer overlapping spikes in one stitched waveform.
    This is essentially bounded, noise-whitened OMP:
        select candidate
        -> jointly refit all amplitudes
        -> recompute residual
        -> select next candidate
    """
    x = np.asarray(x, float)
    xw = noise.whiten(x)
    selected = []
    accepted_scores = []
    # --------------------------------------------------------
    # Greedy selection
    # --------------------------------------------------------
    for _ in range(cfg.max_spikes):
        # IMPORTANT:
        # residual always comes from a JOINT bounded refit.
        amplitudes, residual = bounded_joint_fit(xw, bank, selected)
        num = residual @ bank.Aw
        candidate_amp = np.clip(num / bank.den, bank.lo, bank.hi)
        scores = 2.0 * candidate_amp * num - candidate_amp**2 * bank.den
        # Exact candidates already selected.
        if selected:
            scores[np.asarray(selected)] = -np.inf
        # Same-unit refractory restriction.
        for jj in selected:
            same_unit = bank.units == bank.units[jj]
            too_close = np.abs(bank.pos - bank.pos[jj]) < refractory_samples
            scores[same_unit & too_close] = -np.inf
        jbest = int(np.argmax(scores))
        best_score = float(scores[jbest])
        if not np.isfinite(best_score) or best_score <= score_threshold:
            break
        selected.append(jbest)
        accepted_scores.append(best_score)
    # --------------------------------------------------------
    # Timing refinement
    # --------------------------------------------------------
    for _ in range(2):
        changed = False
        if not selected:
            break
        _, current_resid = bounded_joint_fit(xw, bank, selected)
        current_error = float(current_resid @ current_resid)
        for p in range(len(selected)):
            old_j = selected[p]
            k = bank.units[old_j]
            old_tau = bank.tau[old_j]
            best_j = old_j
            best_error = current_error
            for dt in range(
                -cfg.timing_refine_samples, cfg.timing_refine_samples + 1
            ):
                if dt == 0:
                    continue
                target_tau = old_tau + dt
                candidates = np.where(
                    (bank.units == k) & (bank.tau == target_tau)
                )[0]
                if len(candidates) == 0:
                    continue
                new_j = int(candidates[0])
                trial = selected.copy()
                trial[p] = new_j
                # No duplicate atom.
                if len(set(trial)) != len(trial):
                    continue
                if not valid_refractory(bank, trial, refractory_samples):
                    continue
                _, r = bounded_joint_fit(xw, bank, trial)
                e = float(r @ r)
                if e < best_error - 1e-9:
                    best_error = e
                    best_j = new_j
            if best_j != old_j:
                selected[p] = best_j
                current_error = best_error
                changed = True
        if not changed:
            break
    # --------------------------------------------------------
    # Final joint amplitude fit
    # --------------------------------------------------------
    amplitudes, residual_w = bounded_joint_fit(xw, bank, selected)
    if selected:
        reconstruction_w = bank.Aw[:, selected] @ amplitudes
    else:
        reconstruction_w = np.zeros_like(xw)
    # Back to voltage space.
    C = noise.chol(len(x))
    reconstruction = C @ reconstruction_w
    residual = x - reconstruction
    spikes = []
    for j, a in zip(selected, amplitudes):
        spikes.append(
            {
                "unit": int(bank.units[j]),
                "sample_local": int(bank.pos[j]),
                "tau": int(bank.tau[j]),
                "amplitude": float(a),
            }
        )
    spikes = sorted(spikes, key=lambda z: z["sample_local"])
    whitened_error = float(residual_w @ residual_w)
    whitened_resid_var = whitened_error / max(len(x), 1)
    return {
        "spikes": spikes,
        "selected": selected,
        "accepted_scores": accepted_scores,
        "reconstruction": reconstruction,
        "residual": residual,
        "residual_w": residual_w,
        "whitened_error": whitened_error,
        "whitened_resid_var": whitened_resid_var,
        "n_spikes": len(spikes),
    }


# ============================================================
# 13. FIT COMPLETE RESCUE MODEL
# ============================================================
def fit_rescue_model(mat_path, cfg=None):
    if cfg is None:
        cfg = RescueConfig()
    rng = np.random.default_rng(cfg.seed)
    ch = load_channel(mat_path)
    sr = ch["sr"]
    R_chain = int(round(cfg.chain_ms * 1e-3 * sr))
    R_iso = int(round(cfg.isolated_ms * 1e-3 * sr))
    R_abs = int(round(cfg.refractory_ms * 1e-3 * sr))
    if R_chain <= 0:
        raise ValueError("chain_ms produces zero samples.")
    if R_abs <= 0:
        raise ValueError("refractory_ms produces zero samples.")
    chains = build_chains(ch["s"], R_chain)
    iso = isolated_mask(ch["s"], R_iso)
    units = learn_initial_templates(ch, iso, cfg, rng)
    if len(units) == 0:
        raise RuntimeError(
            "No units had enough isolated spikes " "for template learning."
        )
    noise, noise_residuals = estimate_noise(ch, units, iso, cfg)
    # Critical fix:
    # amplitude distributions now use the same GLS metric
    # used during inference.
    fit_unit_amplitude_statistics(ch, units, iso, noise, cfg)
    ks, coherence = template_coherence(units, noise)
    overlap_check = check_all_snippet_overlaps(ch)
    return {
        "channel": ch,
        "config": cfg,
        "units": units,
        "noise": noise,
        "noise_residuals": noise_residuals,
        "chains": chains,
        "isolated_mask": iso,
        "R_chain": R_chain,
        "R_iso": R_iso,
        "R_abs": R_abs,
        "unit_ids": ks,
        "coherence": coherence,
        "overlap_check": overlap_check,
        # Cache because different chain lengths need
        # different banks and null thresholds.
        "bank_cache": {},
        "threshold_cache": {},
    }


# ============================================================
# 14. GET BANK + CALIBRATION FOR A WINDOW LENGTH
# ============================================================
def get_calibrated_bank(model, n):
    if n in model["bank_cache"]:
        return (model["bank_cache"][n], model["threshold_cache"][n])
    ch = model["channel"]
    cfg = model["config"]
    units = model["units"]
    noise = model["noise"]
    bank = make_bank(units, n, noise, ch["peak_off"])
    # Deterministic but different RNG stream by window length.
    rng = np.random.default_rng(cfg.seed + 100000 + n)
    threshold, null_maxima = calibrate_score_threshold(bank, cfg, rng)
    model["bank_cache"][n] = bank
    model["threshold_cache"][n] = {
        "score_threshold": threshold,
        "null_maxima": null_maxima,
    }
    return (bank, model["threshold_cache"][n])


# ============================================================
# 15. RESCUE ALL MULTI-DETECTION CHAINS
# ============================================================
def rescue_channel(model):
    ch = model["channel"]
    cfg = model["config"]
    noise = model["noise"]
    R_abs = model["R_abs"]
    records = []
    chain_results = []
    for chain_id, (i0, i1) in enumerate(model["chains"]):
        i0 = int(i0)
        i1 = int(i1)
        size = i1 - i0 + 1
        # Ordinary isolated detection:
        # no rescue required.
        if size <= 1:
            continue
        try:
            x, s0 = stitch(ch, i0, i1)
        except ValueError as e:
            chain_results.append(
                {
                    "chain_id": chain_id,
                    "i0": i0,
                    "i1": i1,
                    "accepted": False,
                    "reason": str(e),
                }
            )
            continue
        bank, calibration = get_calibrated_bank(model, len(x))
        result = rescue_chain(
            x=x,
            bank=bank,
            score_threshold=calibration["score_threshold"],
            refractory_samples=R_abs,
            cfg=cfg,
            noise=noise,
        )
        # Conservative QC.
        accepted = (
            result["n_spikes"] > 0
            and result["whitened_resid_var"] <= cfg.max_whitened_resid_var
        )
        chain_info = {
            "chain_id": chain_id,
            "i0": i0,
            "i1": i1,
            "n_crossings": size,
            "n_inferred": result["n_spikes"],
            "score_threshold": calibration["score_threshold"],
            "whitened_resid_var": result["whitened_resid_var"],
            "accepted": accepted,
            "result": result,
            "s0": s0,
        }
        chain_results.append(chain_info)
        if not accepted:
            continue
        crossing_samples = ch["s"][i0 : i1 + 1]
        for spike in result["spikes"]:
            absolute_sample = int(s0 + spike["sample_local"])
            distance_to_crossing = int(
                np.min(np.abs(crossing_samples - absolute_sample))
            )
            records.append(
                {
                    "chain_id": chain_id,
                    "unit": spike["unit"],
                    "sample": absolute_sample,
                    "time_ms": absolute_sample / ch["sr"] * 1000.0,
                    "amplitude": spike["amplitude"],
                    "nearest_crossing_distance_samples": distance_to_crossing,
                    "has_nearby_original_crossing": distance_to_crossing <= 10,
                    "n_original_crossings": size,
                    "n_inferred_chain_spikes": result["n_spikes"],
                    "whitened_resid_var": result["whitened_resid_var"],
                }
            )
    rescued = pd.DataFrame(records)
    return rescued, chain_results


# ============================================================
# 16. BUILD FINAL EVENT TABLE
# ============================================================
def build_final_event_table(model, rescued, chain_results):
    """
    Conservative merge:
    - singleton detections remain unchanged;
    - accepted rescued chains are replaced by inferred events;
    - failed rescue chains are retained as original detections,
      marked as unresolved.
    This avoids silently deleting data when rescue fails.
    """
    ch = model["channel"]
    chain_lookup = {}
    for info in chain_results:
        chain_lookup[info["chain_id"]] = info
    rows = []
    for chain_id, (i0, i1) in enumerate(model["chains"]):
        i0 = int(i0)
        i1 = int(i1)
        size = i1 - i0 + 1
        # --------------------------------------------
        # Singleton: preserve original.
        # --------------------------------------------
        if size == 1:
            i = i0
            rows.append(
                {
                    "unit": int(ch["cls"][i]),
                    "sample": int(ch["s"][i]),
                    "time_ms": float(ch["s"][i] / ch["sr"] * 1000),
                    "amplitude": np.nan,
                    "source": "original",
                    "chain_id": chain_id,
                    "original_index": i,
                    "original_refractory_flag": bool(ch["q_refr"][i]),
                }
            )
            continue
        info = chain_lookup.get(chain_id, None)
        # --------------------------------------------
        # Accepted rescue replaces chain.
        # --------------------------------------------
        if info is not None and info.get("accepted", False):
            r = rescued[rescued["chain_id"] == chain_id]
            for _, event in r.iterrows():
                rows.append(
                    {
                        "unit": int(event["unit"]),
                        "sample": int(event["sample"]),
                        "time_ms": float(event["time_ms"]),
                        "amplitude": float(event["amplitude"]),
                        "source": "rescued",
                        "chain_id": chain_id,
                        "original_index": np.nan,
                        "original_refractory_flag": np.nan,
                    }
                )
        # --------------------------------------------
        # Rescue failed: preserve originals.
        # --------------------------------------------
        else:
            for i in range(i0, i1 + 1):
                rows.append(
                    {
                        "unit": int(ch["cls"][i]),
                        "sample": int(ch["s"][i]),
                        "time_ms": float(ch["s"][i] / ch["sr"] * 1000),
                        "amplitude": np.nan,
                        "source": "unresolved_original",
                        "chain_id": chain_id,
                        "original_index": i,
                        "original_refractory_flag": bool(ch["q_refr"][i]),
                    }
                )
    events = pd.DataFrame(rows)
    if len(events):
        events = events.sort_values("sample").reset_index(drop=True)
    return events


# ============================================================
# 17. DIAGNOSTIC SUMMARY
# ============================================================
def print_model_summary(model, rescued=None, chain_results=None):
    ch = model["channel"]
    iso = model["isolated_mask"]
    chains = model["chains"]
    units = model["units"]
    sizes = chains[:, 1] - chains[:, 0] + 1
    print(f"Channel: {ch['label']}")
    print(f"Detections: {len(ch['s']):,}")
    print(f"Wave_Clus refractory flags: " f"{ch['q_refr'].sum():,}")
    print(f"Isolated template candidates: " f"{iso.sum():,}")
    print(f"Multi-detection chains: " f"{np.sum(sizes > 1):,}")
    print(f"Learned units: " f"{sorted(units)}")
    sigma = np.sqrt(model["noise"].s2)
    print(f"Estimated noise sigma: " f"{sigma:.2f} uV")
    check = model["overlap_check"]
    print(f"Overlapping snippet pairs checked: " f"{check['n_overlap_pairs']}")
    print(
        f"Maximum relative overlap mismatch: "
        f"{check['max_relative_error']:.3e}"
    )
    # Template identifiability.
    C = model["coherence"]
    ks = model["unit_ids"]
    if len(ks) > 1:
        off = C[np.triu_indices(len(ks), 1)]
        print(
            f"Template coherence: "
            f"min={off.min():.3f}, "
            f"max={off.max():.3f}"
        )
        high = np.argwhere(np.triu(C > 0.90, k=1))
        if len(high):
            print("Highly coherent unit pairs:")
            for i, j in high:
                print(f"  U{ks[i]} / U{ks[j]}: " f"{C[i,j]:.3f}")
    if chain_results is not None:
        attempted = len(chain_results)
        accepted = sum(x.get("accepted", False) for x in chain_results)
        print(f"Rescue chains accepted: " f"{accepted}/{attempted}")
    if rescued is not None:
        print(f"Rescued inferred events: " f"{len(rescued):,}")


# ============================================================
# 18. ONE-CALL FULL PIPELINE
# ============================================================
def run_rescue_pipeline(mat_path, cfg=None):
    """
    Complete pipeline.
    Returns
    -------
    model
        Learned templates, covariance, chains, calibration caches.
    rescued
        Only events inferred from accepted rescue chains.
    final_events
        Conservative merged event table.
    """
    # ----------------------------
    # TRAIN
    # ----------------------------
    model = fit_rescue_model(mat_path, cfg)
    # ----------------------------
    # RESCUE
    # ----------------------------
    rescued, chain_results = rescue_channel(model)
    model["chain_results"] = chain_results
    # ----------------------------
    # MERGE
    # ----------------------------
    final_events = build_final_event_table(model, rescued, chain_results)
    # ----------------------------
    # REPORT
    # ----------------------------
    print_model_summary(model, rescued, chain_results)
    return (model, rescued, final_events)
