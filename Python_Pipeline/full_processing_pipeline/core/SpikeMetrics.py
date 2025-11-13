"""
Quality Metrics for Spike Sorting Pipeline - FIXED VERSION

This module provides a class-based wrapper, SpikeMetricsCalculator, to compute
a comprehensive suite of quality metrics for sorted units from 'times_*.mat' files.

FIXED: Now checks for both 'inspk' and 'features' fields in mat files

Usage:
    from SpikeMetrics import SpikeMetricsCalculator
    
    metrics_calc = SpikeMetricsCalculator(
        input_dir='/path/to/times_files',
        output_dir='/path/to/save_reports'
    )
    
    # Run on all channels
    metrics_calc.process_all_channels(channels='all')
    
    # Run on specific channels
    metrics_calc.process_all_channels(channels=[257, 263])
"""

import os
import re
import numpy as np
import glob
import traceback
from scipy.io import loadmat, savemat
from typing import Optional, Tuple, Dict, Any, Union
from numpy.typing import NDArray
from datetime import datetime
import warnings

# Import dependencies
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score as _sk_silhouette_score
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# CORE METRIC FUNCTIONS
# ============================================================================

def num_spikes(spike_times: NDArray) -> int:
    """Calculate number of spikes for each cluster in each channel."""
    return len(np.asarray(spike_times))

        
def firing_rate(
    spike_times: NDArray,
    recording_duration: float,
    start_time: float = 0.0,
    end_time: Optional[float] = None
) -> float:
    """Calculate the average firing rate of a unit."""
    spike_times = np.asarray(spike_times)
    if recording_duration <= 0:
        raise ValueError("Recording duration must be positive")
    
    if spike_times.size == 0:
        return 0.0
    
    if np.any(spike_times < 0):
        raise ValueError("Spike times cannot be negative")
    
    if end_time is None:
        end_time = recording_duration
    
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    
    mask = (spike_times >= start_time) & (spike_times <= end_time)
    n_spikes_in_window = np.sum(mask)
    window_duration = end_time - start_time
    rate = (n_spikes_in_window / window_duration) * 1000.0
    
    return float(rate)


def snr(waveforms: np.ndarray, noise_samples_n: int = 5) -> float:
    """Calculates the Signal-to-Noise Ratio (SNR) for 2D single-channel waveforms."""
    if waveforms.ndim != 2:
        raise ValueError(f"Input must be 2D (n_spikes, n_samples), but got {waveforms.ndim} dimensions")

    if waveforms.size == 0:
        return np.nan
        
    n_spikes, n_samples = waveforms.shape

    if n_samples < noise_samples_n * 2:
        noise_residuals = waveforms - np.mean(waveforms, axis=0)
        noise_level = np.std(noise_residuals)
    else:
        template = np.mean(waveforms, axis=0)
        residuals = waveforms - template
        noise_residuals = np.concatenate(
            (residuals[:, :noise_samples_n], residuals[:, -noise_samples_n:]),
            axis=1,
        )
        noise_level = np.std(noise_residuals)

    template = np.mean(waveforms, axis=0)
    peak_to_peak = np.ptp(template)

    if noise_level == 0:
        return np.inf if peak_to_peak > 0 else 0.0
        
    return peak_to_peak / noise_level


def presence_ratio(
    spike_times: NDArray[np.float64],
    recording_duration: float,
    bin_duration: float = 60000.0
) -> float:
    """Calculate the fraction of time the unit is present."""
    if recording_duration <= 0:
        raise ValueError("Recording duration must be positive")
    
    if bin_duration <= 0:
        raise ValueError("Bin duration must be positive")
    
    if spike_times.size == 0:
        return 0.0
    
    n_bins = int(np.ceil(recording_duration / bin_duration))
    if n_bins == 0:
        return 0.0
    
    bins = np.arange(0, recording_duration + bin_duration, bin_duration)
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    bins_with_spikes = np.sum(spike_counts > 0)
    ratio = bins_with_spikes / n_bins
    
    return float(ratio)


def isi(spike_times_ms: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate Inter-Spike Intervals (ISIs) from spike times."""
    if spike_times_ms.size < 2:
        return np.empty(0, dtype=np.float64)
    
    if np.any(np.isnan(spike_times_ms)) or np.any(np.isinf(spike_times_ms)):
        raise ValueError("spike_times_ms contains NaN or Inf values")
    
    if not np.all(spike_times_ms[1:] >= spike_times_ms[:-1]):
        warnings.warn("spike_times_ms was not sorted. Sorting now...")
        spike_times_ms = np.sort(spike_times_ms)
    
    return np.diff(spike_times_ms)


def cv2(isi_ms: NDArray[np.float64]) -> float:
    """Calculate CV2 (Coefficient of Variation 2) - local irregularity index."""
    if isi_ms.size < 2:
        return 0.0
    
    if np.any(np.isnan(isi_ms)) or np.any(np.isinf(isi_ms)):
        raise ValueError("isi_ms contains NaN or Inf values")
    
    if np.any(isi_ms < 0):
        raise ValueError("isi_ms cannot contain negative values")
    
    diff = np.abs(np.diff(isi_ms))
    denom = isi_ms[:-1] + isi_ms[1:]
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    cv2_values = 2.0 * diff / denom
    
    return float(np.mean(cv2_values))


def isi_violations(
    spike_times_ms: NDArray[np.float64],
    refractory_period_ms: float = 3.0,
    censored_period_ms: float = 0.0,
    recording_duration_ms: Optional[float] = None,
) -> Tuple[float, float, int]:
    """Calculate the rate of Inter-Spike Interval (ISI) violations."""
    if spike_times_ms.size == 0:
        return 0.0, 0.0, 0

    if not np.all(spike_times_ms[1:] >= spike_times_ms[:-1]):
        warnings.warn("spike_times_ms was not sorted. Sorting now...")
        spike_times_ms = np.sort(spike_times_ms)

    if censored_period_ms < 0:
        raise ValueError("censored_period_ms cannot be negative")

    if censored_period_ms >= refractory_period_ms:
        raise ValueError("censored_period_ms must be smaller than refractory_period_ms")

    dup_idx = np.where(np.diff(spike_times_ms) <= censored_period_ms)[0]
    if dup_idx.size > 0:
        spike_times_ms = np.delete(spike_times_ms, dup_idx + 1)

    isis = np.diff(spike_times_ms)
    num_violations = int(np.sum((isis > censored_period_ms) & (isis < refractory_period_ms)))

    if recording_duration_ms is None:
        recording_duration_ms = float(spike_times_ms[-1] - spike_times_ms[0])

    n_spikes = spike_times_ms.size
    if n_spikes <= 1 or recording_duration_ms <= 0:
        return 0.0, 0.0, num_violations

    total_rate = n_spikes / recording_duration_ms
    violation_time = 2.0 * n_spikes * (refractory_period_ms - censored_period_ms)

    if violation_time <= 0 or total_rate == 0:
        return 0.0, 0.0, num_violations

    violation_rate = num_violations / violation_time
    fp_rate = (violation_rate / total_rate)
    
    return float(violation_rate), float(fp_rate), int(num_violations)


def amplitude_cutoff(
    waveforms: NDArray[np.float64],
    num_histogram_bins: int = 500,
    histogram_smoothing_value: int = 3,
) -> float:
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes."""
    if waveforms.size == 0:
        return np.nan

    amplitudes = np.ptp(waveforms, axis=1)
    
    if amplitudes.size < 2:
        return 0.0
    
    hist_density, bin_edges = np.histogram(
        amplitudes, bins=num_histogram_bins, density=True
    )
    pdf = gaussian_filter1d(hist_density, histogram_smoothing_value)
    support = bin_edges[:-1]

    peak_idx = int(np.argmax(pdf))
    
    if peak_idx == 0:
        return 0.5 

    right_segment = pdf[peak_idx:]
    target = pdf[0]
    rel_idx = int(np.argmin(np.abs(right_segment - target)))
    G = peak_idx + rel_idx

    bin_size = float(np.mean(np.diff(support)))
    fraction_missing = float(np.sum(pdf[G:]) * bin_size)

    return float(min(fraction_missing, 0.5))


def mahalanobis_metrics(
    features: NDArray[np.float64],
    labels: NDArray[np.int32],
    target_cluster: int,
) -> Tuple[float, float]:
    """Isolation distance and L-ratio, Schmitzer-Torbert style."""
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have same length")

    target_mask = labels == target_cluster
    if not np.any(target_mask):
        raise ValueError(f"Cluster {target_cluster} not found.")

    pcs_for_this = features[target_mask, :]
    pcs_for_other = features[~target_mask, :]

    n_self = pcs_for_this.shape[0]
    n_other = pcs_for_other.shape[0]
    dof = features.shape[1]

    if n_self < dof + 1 or n_other < 1:
        return np.nan, np.nan

    try:
        mean_val = np.expand_dims(np.mean(pcs_for_this, axis=0), axis=0)
        cov = np.cov(pcs_for_this.T)
        VI = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    except ValueError:
        return np.nan, np.nan

    mahal_other = cdist(
        mean_val,
        pcs_for_other,
        metric="mahalanobis",
        VI=VI
    )[0]
    mahal_other = np.sort(mahal_other)

    n = min(n_self, n_other)

    l_ratio = np.sum(1.0 - chi2.cdf(mahal_other**2, dof)) / float(n_self)
    isolation_distance = (mahal_other[n - 1] ** 2) if n >= 1 else np.nan

    return float(isolation_distance), float(l_ratio)


def d_prime_lda(
    features: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    target_cluster: int,
) -> float:
    """LDA-based d' (Hill / cortex-lab style)."""
    if features.shape[0] != cluster_labels.shape[0]:
        raise ValueError("features and cluster_labels must have the same length")

    if target_cluster not in cluster_labels:
        raise ValueError(f"Target cluster {target_cluster} not found")

    X = features
    y = (cluster_labels == target_cluster)

    if y.sum() < 2 or (~y).sum() < 2:
        return 0.0

    try:
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X, y)
    except Exception:
        return np.nan

    this_proj = X_lda[y]
    other_proj = X_lda[~y]

    mu1 = np.mean(this_proj)
    mu2 = np.mean(other_proj)
    s1 = np.std(this_proj)
    s2 = np.std(other_proj)

    pooled = np.sqrt(0.5 * (s1**2 + s2**2))
    if pooled == 0:
        return 0.0

    dprime = (mu1 - mu2) / pooled
    return float(abs(dprime))


def nearest_neighbor_metrics(
    features: NDArray[np.float64],
    labels: NDArray[np.int32],
    target_cluster: int,
    max_spikes_for_nn: int = 10000,
    n_neighbors: int = 5,
) -> tuple[float, float]:
    """kNN-based hit/miss rates for a target cluster"""
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same length")
    if target_cluster not in labels:
        raise ValueError(f"target_cluster {target_cluster} not found in labels")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    is_target = labels == target_cluster
    X_target = features[is_target]
    X_other = features[~is_target]
    
    n_target = X_target.shape[0]
    n_other = X_other.shape[0]

    if n_target < 2 or n_other < 1:
        return 1.0, 0.0

    if (n_target + n_other) > max_spikes_for_nn:
        frac_target = n_target / (n_target + n_other)
        n_target_new = int(frac_target * max_spikes_for_nn)
        n_other_new = max_spikes_for_nn - n_target_new
        
        rng = np.random.default_rng()
        X_target = X_target[rng.choice(n_target, n_target_new, replace=False)]
        X_other = X_other[rng.choice(n_other, n_other_new, replace=False)]
        n_target = n_target_new
        
    X = np.concatenate((X_target, X_other), axis=0)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
    nn.fit(X)
    dists, inds = nn.kneighbors(X)

    target_neighbors = inds[:n_target, 1:].flatten()
    other_neighbors = inds[n_target:, 1:].flatten()

    hit_rate = float(np.mean(target_neighbors < n_target)) if target_neighbors.size else 1.0
    miss_rate = float(np.mean(other_neighbors < n_target)) if other_neighbors.size else 0.0

    return hit_rate, miss_rate


def silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    target_cluster: Optional[int] = None,
    metric: str = "euclidean",
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    return_matrix: bool = False,
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]:
    """Computes silhouette score, optionally as a pairwise matrix."""
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same length")
    if metric not in {"euclidean", "manhattan", "cosine"}:
        raise ValueError(f"Unsupported metric: {metric}")

    unique_labels = np.array(sorted(np.unique(labels)))
    if unique_labels.size < 2:
        nan_scores = np.full(unique_labels.size, np.nan)
        nan_matrix = np.full((unique_labels.size, unique_labels.size), np.nan)
        if target_cluster is None:
            return (nan_scores, nan_matrix) if return_matrix else nan_scores
        else:
            return (np.nan, nan_matrix) if return_matrix else np.nan

    idx = np.arange(features.shape[0])
    if sample_size is not None and sample_size < features.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=sample_size, replace=False)
    X = features[idx]
    y = labels[idx]
    
    unique_labels = np.array(sorted(np.unique(y)))
    K = unique_labels.size
    
    if K < 2:
        nan_scores = np.full(np.unique(labels).size, np.nan)
        nan_matrix = np.full((np.unique(labels).size, np.unique(labels).size), np.nan)
        if target_cluster is None:
            return (nan_scores, nan_matrix) if return_matrix else nan_scores
        else:
            return (np.nan, nan_matrix) if return_matrix else np.nan

    SS = np.full((K, K), np.nan, dtype=float)

    def _score_row(i_idx: int):
        row = [np.nan] * K
        i_lab = unique_labels[i_idx]
        for j_idx in range(i_idx + 1, K):
            j_lab = unique_labels[j_idx]
            mask = (y == i_lab) | (y == j_lab)
            if mask.sum() > 2 and np.unique(y[mask]).size > 1:
                try:
                    row[j_idx] = _sk_silhouette_score(X[mask], y[mask], metric=metric)
                except Exception:
                    row[j_idx] = np.nan
        return row

    if n_jobs != 1:
        try:
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_score_row)(i_idx) for i_idx in range(K)
            )
        except Exception:
            rows = [_score_row(i_idx) for i_idx in range(K)]
    else:
        rows = [_score_row(i_idx) for i_idx in range(K)]

    for i_idx, row in enumerate(rows):
        for j_idx, val in enumerate(row):
            SS[i_idx, j_idx] = val

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_by_col = np.nanmin(SS, axis=0)
        min_by_row = np.nanmin(SS, axis=1)
        per_cluster = np.array([np.nanmin([a, b]) for a, b in zip(min_by_col, min_by_row)])

    final_scores = np.full(np.unique(labels).size, np.nan)
    orig_labels = np.array(sorted(np.unique(labels)))
    sampled_label_map = {lab: score for lab, score in zip(unique_labels, per_cluster)}
    
    for i, orig_lab in enumerate(orig_labels):
        if orig_lab in sampled_label_map:
            final_scores[i] = sampled_label_map[orig_lab]

    if target_cluster is None:
        result = final_scores
    else:
        pos = np.where(orig_labels == target_cluster)[0]
        if pos.size == 0:
            raise ValueError(f"Target cluster {target_cluster} not found in labels")
        result = float(final_scores[int(pos[0])])

    return (result, SS) if return_matrix else result


def compute_cluster_metrics(data: dict, 
                            exclude_cluster_0: bool = True, 
                            n_jobs: int = -1, 
                            n_neighbors: int = 5, 
                            bin_duration_ms: float = 60000.0,
                            refractory_period_ms: float = 3.0,
                            censored_period_ms: float = 0.0
                            ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute comprehensive quality metrics for spike sorting clusters."""
    
    # Extract data
    cluster_class = data['cluster_class']
    waveforms = data['spikes']
    features = data['features']  # Changed from 'inspk' to 'features'
    
    # Get cluster information
    cluster_ids = cluster_class[:, 0].astype(int)
    spike_times_ms = cluster_class[:, 1]
    unique_clusters = np.unique(cluster_ids)
    
    if spike_times_ms.size > 0:
        recording_duration_ms = spike_times_ms.max()
    else:
        recording_duration_ms = 0.0
    
    def compute_metrics_for_cluster(cluster_id):
        mask = cluster_ids == cluster_id
        cluster_spike_times_ms = spike_times_ms[mask]
        cluster_waveforms = waveforms[mask]
        
        metrics = {'cluster_id': cluster_id}
        
        metrics['num_spikes'] = num_spikes(cluster_spike_times_ms)
        metrics['firing_rate'] = firing_rate(cluster_spike_times_ms, recording_duration_ms)
        metrics['snr'] = snr(cluster_waveforms)
        metrics['presence_ratio'] = presence_ratio(
            cluster_spike_times_ms, recording_duration_ms, bin_duration=bin_duration_ms
        )
        
        metrics['amplitude_cutoff'] = amplitude_cutoff(cluster_waveforms)
        
        isis = isi(cluster_spike_times_ms)
        metrics['cv2'] = cv2(isis) if len(isis) > 1 else 0.0
        
        viol_rate, fp_rate, viol_count = isi_violations(
            cluster_spike_times_ms, 
            refractory_period_ms=refractory_period_ms,
            censored_period_ms=censored_period_ms,
            recording_duration_ms=recording_duration_ms
        )
        metrics['isi_violation_rate'] = viol_rate
        metrics['isi_contamination_fraction'] = fp_rate
        metrics['isi_violations_count'] = viol_count
        
        if len(unique_clusters) > 1:
            try:
                labels_full = cluster_ids.astype(np.int32)
                n_in_cluster = int(np.sum(labels_full == cluster_id))
                if n_in_cluster < 2:
                    raise ValueError("Too few spikes in target cluster for isolation metrics.")
                
                tc = int(cluster_id)
                
                isolation_distance, l_ratio = mahalanobis_metrics(
                    features, labels_full, target_cluster=tc
                )
                metrics["isolation_distance"], metrics["l_ratio"] = isolation_distance, l_ratio
                
                metrics["d_prime"] = d_prime_lda(
                    features, labels_full, target_cluster=tc
                )
                
                hit, miss = nearest_neighbor_metrics(
                    features, labels_full, target_cluster=tc, n_neighbors=n_neighbors
                )
                metrics["nearest_neighbor_hit_rate"] = hit
                metrics["nearest_neighbor_miss_rate"] = miss
                
            except Exception as e:
                metrics['isolation_distance'] = np.nan
                metrics['l_ratio'] = np.nan
                metrics['d_prime'] = np.nan
                metrics['nearest_neighbor_hit_rate'] = np.nan
                metrics['nearest_neighbor_miss_rate'] = np.nan
        else:
            metrics['isolation_distance'] = np.nan
            metrics['l_ratio'] = np.nan
            metrics['d_prime'] = np.nan
            metrics['nearest_neighbor_hit_rate'] = np.nan
            metrics['nearest_neighbor_miss_rate'] = np.nan
        
        return metrics
    
    if exclude_cluster_0:
        metric_clusters = [cid for cid in unique_clusters if cid != 0]
    else:
        metric_clusters = list(unique_clusters)
    
    print(f"Computing ALL metrics for {len(metric_clusters)} clusters with parallelism (n_jobs={n_jobs})...\n")
    
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_metrics_for_cluster)(cid) for cid in metric_clusters
    )
    
    if not results:
        print("No clusters found to compute metrics for.")
        return pd.DataFrame(), np.array([])
        
    all_metrics = sorted(results, key=lambda m: list(metric_clusters).index(m['cluster_id']))
    df_metrics = pd.DataFrame(all_metrics).set_index('cluster_id')
    
    if exclude_cluster_0:
        mask_valid = cluster_ids != 0
        if np.sum(mask_valid) > 0:
            features_valid = features[mask_valid]
            cluster_ids_valid = cluster_ids[mask_valid]
        else:
            features_valid, cluster_ids_valid = None, None
    else:
        features_valid = features
        cluster_ids_valid = cluster_ids
    
    if features_valid is not None and np.unique(cluster_ids_valid).size > 1:
        scores, SS = silhouette_score(features_valid, cluster_ids_valid, return_matrix=True, n_jobs=n_jobs)
        score_map = {lab: score for lab, score in zip(np.unique(cluster_ids_valid), scores)}
        df_metrics['silhouette_score'] = df_metrics.index.map(score_map)
    else:
        print("Skipping silhouette score (not enough clusters or data).")
        df_metrics['silhouette_score'] = np.nan
        SS = np.array([])
    
    print("✓ All metrics computed successfully!")
    
    return df_metrics, SS


# ============================================================================
# MAIN METRICS CALCULATOR CLASS
# ============================================================================

class SpikeMetricsCalculator:
    """
    Class to find 'times_*.mat' files, compute quality metrics for all clusters,
    and save the aggregated results to a single CSV file.
    
    FIXED: Now handles both 'inspk' and 'features' field names in mat files.
    """
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None,
                 n_jobs: int = -1,
                 n_neighbors: int = 5,
                 bin_duration_ms: float = 60000.0,
                 refractory_period_ms: float = 3.0,
                 censored_period_ms: float = 0.0,
                 verbose: bool = True):
        """Initialize the SpikeMetricsCalculator."""
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.verbose = verbose
        
        self.metric_params = {
            'exclude_cluster_0': True,
            'n_jobs': n_jobs,
            'n_neighbors': n_neighbors,
            'bin_duration_ms': bin_duration_ms,
            'refractory_period_ms': refractory_period_ms,
            'censored_period_ms': censored_period_ms
        }
        
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            self._log(f"Created output directory: {self.output_dir}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _parse_channels(self, channels: Union[str, int, list]) -> list:
        """Parse channel specification and return list of channel numbers."""
        all_times_files = glob.glob(os.path.join(self.input_dir, 'times_*.mat'))
        all_available_channels = set()
        
        for f in all_times_files:
            basename = os.path.basename(f)
            match1 = re.search(r'times_.*_(\d+)\.mat$', basename)
            match2 = re.search(r'times_ch(\d+)\.mat$', basename)
            
            if match1:
                all_available_channels.add(int(match1.group(1)))
            elif match2:
                all_available_channels.add(int(match2.group(1)))
                
        if not all_available_channels:
            self._log(f"Warning: No 'times_*.mat' or 'times_ch*.mat' files found in {self.input_dir}")
            return []

        if isinstance(channels, str) and channels.lower() == 'all':
            return sorted(list(all_available_channels))
        
        if isinstance(channels, int):
            channels = [channels]
        
        if isinstance(channels, (list, tuple)):
            requested = set(channels)
            available = set(all_available_channels)
            
            to_process = sorted(list(requested.intersection(available)))
            not_found = sorted(list(requested - available))
            
            if not_found:
                self._log(f"Warning: Requested channels not found: {not_found}")
            if not to_process:
                self._log(f"Warning: No requested channels found in directory.")
            
            return to_process
        
        self._log(f"Warning: Unknown channel type {type(channels)}, using all channels")
        return sorted(list(all_available_channels))

    def _find_file_for_channel(self, channel_num: int) -> Optional[str]:
        """Find the times file for a specific channel."""
        times_files = glob.glob(os.path.join(self.input_dir, f'times_*_{channel_num}.mat'))
        if times_files:
            return times_files[0]
            
        times_files_ch = glob.glob(os.path.join(self.input_dir, f'times_ch{channel_num}.mat'))
        if times_files_ch:
            return times_files_ch[0]
            
        return None

    def process_single_channel(self, channel_num: int) -> Optional[pd.DataFrame]:
        """Run metrics computation on a single channel's times file."""
        file_path = self._find_file_for_channel(channel_num)
        
        if file_path is None:
            self._log(f"Warning: No 'times_*_{channel_num}.mat' or 'times_ch{channel_num}.mat' file found")
            return None
            
        try:
            self._log(f"\n{'='*60}")
            self._log(f"Channel {channel_num} - Computing Quality Metrics")
            self._log(f"File: {os.path.basename(file_path)}")
            self._log(f"{'='*60}")
            
            mat_data = loadmat(file_path)
            
            # --- FIXED: Check for both 'inspk' and 'features' ---
            if 'spikes' not in mat_data:
                self._log("Error: 'spikes' (waveforms) field not found.")
                return None
            if 'cluster_class' not in mat_data:
                self._log("Error: 'cluster_class' field not found.")
                return None
            
            # Check for features - can be either 'inspk' or 'features'
            if 'inspk' in mat_data:
                features = mat_data['inspk'].astype(np.float64)
                self._log("  Using 'inspk' field for features")
            elif 'features' in mat_data:
                features = mat_data['features'].astype(np.float64)
                self._log("  Using 'features' field for features")
            else:
                self._log("Error: Neither 'inspk' nor 'features' field found.")
                self._log("  Available fields: " + str([k for k in mat_data.keys() if not k.startswith('__')]))
                return None
                
            waveforms = mat_data['spikes'].astype(np.float64)
            cluster_class = mat_data['cluster_class']
            
            # Ensure cluster_class is (n_spikes, 2)
            if cluster_class.shape[0] == 2 and cluster_class.shape[1] > 2:
                 cluster_class = cluster_class.T
            
            if cluster_class.shape[1] < 2:
                self._log(f"Error: 'cluster_class' has shape {cluster_class.shape}, expected (n, 2).")
                return None
            
            if not all(s.shape[0] == cluster_class.shape[0] for s in [waveforms, features]):
                self._log(f"Error: Mismatch in spike count. Waveforms: {waveforms.shape[0]}, Features: {features.shape[0]}, Classes: {cluster_class.shape[0]}")
                return None
            
            # --- Run Metrics Computation ---
            data_dict = {
                'cluster_class': cluster_class,
                'spikes': waveforms,
                'features': features  # Always use 'features' key internally
            }
            
            df_metrics, _ = compute_cluster_metrics(data_dict, **self.metric_params)
            
            if df_metrics.empty:
                self._log("No clusters found to compute metrics for.")
                return None
                
            df_metrics['channel_id'] = channel_num
            
            self._log(f"✓ Successfully computed metrics for {len(df_metrics)} clusters.")
            return df_metrics

        except Exception as e:
            self._log(f"ERROR processing channel {channel_num}: {e}")
            traceback.print_exc()
            return None

    def process_all_channels(self, channels: Union[str, int, list] = 'all') -> Optional[pd.DataFrame]:
        """Process spike data for all specified channels and save to a single CSV."""
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            self._log(f"No channels specified or found in {self.input_dir}")
            return None
        
        self._log(f"\n{'#'*70}")
        self._log(f"SPIKE QUALITY METRICS PIPELINE")
        self._log(f"{'#'*70}")
        self._log(f"Input directory: {self.input_dir}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Total channels to process: {len(channel_nums)}")
        self._log(f"Parameters: {self.metric_params}")
        self._log(f"{'#'*70}\n")
        
        all_metrics_dfs = []
        successful = 0
        failed = 0
        
        for i, ch_num in enumerate(channel_nums, 1):
            df_channel = self.process_single_channel(ch_num)
            if df_channel is not None:
                all_metrics_dfs.append(df_channel)
                successful += 1
            else:
                failed += 1
                
        self._log(f"\n{'#'*70}")
        self._log(f"METRICS COMPUTATION COMPLETE")
        self._log(f"{'#'*70}")
        self._log(f"Successful: {successful}/{len(channel_nums)}")
        self._log(f"Failed: {failed}/{len(channel_nums)}")
        
        if not all_metrics_dfs:
            self._log("No metrics were generated.")
            self._log(f"{'#'*70}\n")
            return None
            
        final_df = pd.concat(all_metrics_dfs, ignore_index=False)
        
        cols = ['channel_id'] + [c for c in final_df.columns if c != 'channel_id']
        final_df = final_df.reset_index().rename(columns={'index': 'cluster_id'})
        cols = ['channel_id', 'cluster_id'] + [c for c in final_df.columns if c not in ['channel_id', 'cluster_id']]
        final_df = final_df[cols]
        
        save_path = os.path.join(self.output_dir, 'spike_quality_metrics.csv')
        try:
            final_df.to_csv(save_path, index=False, float_format='%.4f')
            self._log(f"✓ Aggregated metrics saved to: {save_path}")
        except Exception as e:
            self._log(f"Error saving aggregated CSV: {e}")
            self._log(f"Attempting to save to backup: 'metrics_backup.csv'")
            final_df.to_csv('metrics_backup.csv', index=False, float_format='%.4f')
            
        self._log(f"{'#'*70}\n")
        
        return final_df


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    """Example usage of the SpikeMetricsCalculator class."""
    
    base_dir = os.path.dirname(__file__)
    times_file_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(times_file_dir):
        print(f"Example directory not found: {times_file_dir}")
        print("Please run clustering first to generate 'times_*.mat' files.")
    else:
        print("\n" + "="*70)
        print("EXAMPLE: Compute metrics for ALL channels")
        print("="*70)
        
        metrics_calc = SpikeMetricsCalculator(
            input_dir=times_file_dir,
            output_dir=times_file_dir,
            n_jobs=-1,
            refractory_period_ms=3.0
        )
        all_metrics_df = metrics_calc.process_all_channels(channels='all')
        
        if all_metrics_df is not None:
            print("\nAggregated Metrics Head:")
            print(all_metrics_df.head())