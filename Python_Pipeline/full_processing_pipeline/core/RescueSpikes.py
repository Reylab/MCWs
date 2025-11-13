"""
Rescue Spikes with Template Matching - FIXED VERSION

FIXED: Now works with waveclus spike file format (Spike_Time, Label fields)
       instead of requiring 'index' and 'properties' fields.

This module provides a class-based wrapper, SpikeRescuer, to find
spikes that were not clustered and re-classify them using template 
matching against already-clustered units.

"""

import os
import re
import numpy as np
import glob
import traceback
from scipy.io import loadmat, savemat
from typing import Optional, Tuple, Union, Literal
from numpy.typing import NDArray
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CORE TEMPLATE MATCHING FUNCTIONS
# ============================================================================

def build_templates(
    classes: NDArray[np.int32],
    waveforms: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build template waveforms and variance metrics for each cluster."""
    if classes.shape[0] != waveforms.shape[0]:
        raise ValueError("classes and waveforms must have the same number of rows")
    if classes.size == 0:
        raise ValueError("classes array cannot be empty")
    
    valid_classes = classes[classes > 0]
    if valid_classes.size == 0:
        if np.any(classes > 1000):
            print("Warning: classes array contains no valid clusters (only 0 or artifact labels). Cannot build templates.")
            n_samples = waveforms.shape[1]
            return np.empty((0, n_samples)), np.empty(0), np.empty((0, n_samples))
        else:
            raise ValueError("classes array contains no valid clusters (all labels <= 0)")
        
    max_class = int(np.max(valid_classes))
    n_samples = waveforms.shape[1]
    
    templates = np.zeros((max_class, n_samples))
    maxdist = np.zeros(max_class)
    pointdist = np.zeros((max_class, n_samples))
    
    for i in range(1, max_class + 1):
        mask = classes == i
        cluster_waveforms = waveforms[mask, :]
        
        if cluster_waveforms.shape[0] == 0:
            continue
            
        templates[i-1, :] = np.mean(cluster_waveforms, axis=0)
        maxdist[i-1] = np.sqrt(np.sum(np.var(cluster_waveforms, axis=0, ddof=0)))
        pointdist[i-1, :] = np.sqrt(np.var(cluster_waveforms, axis=0, ddof=0))
    
    return templates, maxdist, pointdist


def nearest_neighbor(
    x: NDArray[np.float64],
    templates: NDArray[np.float64],
    maxdist: NDArray[np.float64],
    pointdist: Optional[NDArray[np.float64]] = None,
    pointlimit: Optional[int] = None,
    k: Optional[int] = None,
    metric: Literal['euclidean', 'correlation'] = 'euclidean'
) -> Union[int, NDArray[np.int32]]:
    """Find the nearest neighbor template for a spike waveform."""
    if x.ndim != 1:
        raise ValueError("x must be a 1D array (waveform)")
    if templates.shape[1] != x.shape[0]:
        raise ValueError(f"x (shape {x.shape}) and templates (shape {templates.shape}) must have compatible dimensions")
    if templates.shape[0] != maxdist.shape[0]:
        raise ValueError("templates and maxdist must have same number of templates")
    
    n_templates = templates.shape[0]
    
    if n_templates == 0:
        return 0 
    
    if metric == 'euclidean':
        distances = np.sqrt(np.sum((templates - x[np.newaxis, :]) ** 2, axis=1))
        conforming = np.where(distances < maxdist)[0]
    elif metric == 'correlation':
        x_std = np.std(x)
        if x_std == 0:
            distances = np.full(n_templates, np.inf) 
            conforming = np.array([], dtype=int)
        else:
            template_stds = np.std(templates, axis=1)
            valid_templates = template_stds > 0
            
            correlations = np.zeros(n_templates)
            if np.any(valid_templates):
                valid_template_arr = templates[valid_templates]
                correlations[valid_templates] = np.array([
                    np.corrcoef(x, valid_template_arr[i, :])[0, 1] 
                    for i in range(valid_template_arr.shape[0])
                ])
            
            distances = 1.0 - correlations
            conforming = np.where(distances < maxdist)[0]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if pointdist is not None:
        if pointlimit is None:
            pointlimit = np.inf
        
        pointwise_conforming = []
        for i in conforming: 
            n_deviations = np.sum(np.abs(x - templates[i, :]) > pointdist[i, :])
            if n_deviations < pointlimit:
                pointwise_conforming.append(i)
        
        conforming = np.array(pointwise_conforming, dtype=int)
    
    if len(conforming) == 0:
        return 0
    
    if k is not None:
        sorted_indices = np.argsort(distances[conforming])
        k_nearest = sorted_indices[:min(len(sorted_indices), k)]
        return conforming[k_nearest] + 1
    else:
        nearest_idx = np.argmin(distances[conforming])
        return int(conforming[nearest_idx] + 1)


def force_membership(
    waveforms_in: NDArray[np.float64],
    classes_in: NDArray[np.int32],
    waveforms_out: NDArray[np.float64],
    template_sdnum: float = 3.0,
    template_k: Optional[int] = None,
    template_k_min: int = 1,
    use_pointdist: bool = False,
    pointlimit: Optional[int] = None,
    metric: Literal['euclidean', 'correlation'] = 'euclidean'
) -> NDArray[np.int32]:
    """Classify unclassified spikes using waveform template matching."""
    if waveforms_in.shape[0] != classes_in.shape[0]:
        raise ValueError("waveforms_in and classes_in must have same number of rows")
    if waveforms_out.shape[0] == 0:
        return np.array([], dtype=np.int32)
    if waveforms_in.shape[1] != waveforms_out.shape[1]:
        raise ValueError("waveforms_in and waveforms_out must have same number of samples")
    
    n_spikes = waveforms_out.shape[0]
    class_out = np.zeros(n_spikes, dtype=np.int32)
    
    try:
        templates, sd, pd = build_templates(classes_in, waveforms_in)
    except ValueError as e:
        print(f"Warning: {e}. Cannot build templates. All rescue spikes will be unclassified.")
        return class_out
    
    if templates.shape[0] == 0:
        print("Warning: Template building returned no templates. Cannot classify.")
        return class_out
        
    if metric == 'euclidean':
        sd_scaled = template_sdnum * sd
    elif metric == 'correlation':
        sd_scaled = 1.0 - template_sdnum
    else:
        raise ValueError(f"Unknown metric: {metric}")

    pd_to_use = pd if use_pointdist else None
    
    for i in range(n_spikes):
        if template_k is not None:
            neighbors = nearest_neighbor(
                waveforms_out[i, :], templates, sd_scaled,
                pointdist=pd_to_use, pointlimit=pointlimit,
                k=template_k, metric=metric
            )
            
            if isinstance(neighbors, np.ndarray) and len(neighbors) >= template_k_min:
                unique, counts = np.unique(neighbors, return_counts=True)
                class_out[i] = unique[np.argmax(counts)]
            else:
                class_out[i] = 0
        else:
            class_out[i] = nearest_neighbor(
                waveforms_out[i, :], templates, sd_scaled,
                pointdist=pd_to_use, pointlimit=pointlimit,
                metric=metric
            )
    
    return class_out


def rescue_spikes_with_templates(
    data: dict,
    rescue: dict,
    time_tolerance: float = 1e-6,
    verbose: bool = True,
    **match_params
) -> dict:
    """
    Rescue quarantined spikes by template matching.
    
    Parameters:
    -----------
    data : dict
        Classified data with keys:
        - 'cluster_class': (n_spikes, 2) array [label, spike_time]
        - 'spikes': (n_spikes, n_samples) waveforms
        - 'inspk' or 'features': (n_spikes, n_features) features (optional)
    rescue : dict
        Rescue candidates with keys:
        - 'times': (n_rescue_spikes,) spike times
        - 'spikes': (n_rescue_spikes, n_samples) waveforms
        - 'inspk' or 'features': (n_rescue_spikes, n_features) features (optional)
    time_tolerance : float
        Tolerance for matching spike times (ms)
    **match_params : dict
        Parameters for force_membership
    
    Returns:
    --------
    updated_data : dict
        Updated data dictionary with rescued spikes added
    """
    # Extract classified data
    cluster_class = data['cluster_class']
    data_waveforms = data['spikes']
    data_labels = cluster_class[:, 0].astype(np.int32)
    data_times = cluster_class[:, 1]
    
    # Extract rescue candidates
    rescue_times = rescue['times']
    rescue_waveforms = rescue['spikes']
    
    if verbose:
        print(f"\nTemplate Matching Parameters:")
        for k, v in match_params.items():
            print(f"  {k}: {v}")
    
    # Find rescue spikes that are NOT already in classified data
    rescue_mask = np.ones(len(rescue_times), dtype=bool)
    for i, rt in enumerate(rescue_times):
        if np.any(np.abs(data_times - rt) < time_tolerance):
            rescue_mask[i] = False
    
    truly_unclassified_times = rescue_times[rescue_mask]
    truly_unclassified_waveforms = rescue_waveforms[rescue_mask, :]
    
    if verbose:
        print(f"\nFiltering rescue candidates:")
        print(f"  Total rescue candidates: {len(rescue_times)}")
        print(f"  Already classified: {np.sum(~rescue_mask)}")
        print(f"  Truly unclassified: {len(truly_unclassified_times)}")
    
    if len(truly_unclassified_times) == 0:
        if verbose:
            print("  No new spikes to rescue.")
        return data.copy()
    
    # Run template matching
    if verbose:
        print(f"\nRunning template matching on {len(truly_unclassified_times)} spikes...")
    
    rescue_labels = force_membership(
        waveforms_in=data_waveforms,
        classes_in=data_labels,
        waveforms_out=truly_unclassified_waveforms,
        **match_params
    )
    
    # Count rescued spikes
    n_rescued = np.sum(rescue_labels > 0)
    
    if verbose:
        print(f"\nRescue Results:")
        print(f"  Total matched to templates: {n_rescued}")
        print(f"  Remain unclassified: {len(rescue_labels) - n_rescued}")
        
        if n_rescued > 0:
            unique_rescued, counts = np.unique(rescue_labels[rescue_labels > 0], return_counts=True)
            print(f"\n  Rescued spikes by cluster:")
            for label, count in zip(unique_rescued, counts):
                print(f"    Cluster {label}: {count} spikes")
    
    # Merge rescued spikes back into data
    new_cluster_class = np.column_stack([
        rescue_labels.astype(float),
        truly_unclassified_times.astype(float)
    ])
    
    updated_cluster_class = np.vstack([cluster_class, new_cluster_class])
    updated_waveforms = np.vstack([data_waveforms, truly_unclassified_waveforms])
    
    # Sort by spike time
    sort_idx = np.argsort(updated_cluster_class[:, 1])
    updated_cluster_class = updated_cluster_class[sort_idx, :]
    updated_waveforms = updated_waveforms[sort_idx, :]
    
    # Update features if present
    updated_features = None
    if 'inspk' in data or 'features' in data:
        data_key = 'inspk' if 'inspk' in data else 'features'
        rescue_key = 'inspk' if 'inspk' in rescue else 'features' if 'features' in rescue else None
        
        if rescue_key is not None:
            rescue_features = rescue[rescue_key][rescue_mask, :]
            updated_features = np.vstack([data[data_key], rescue_features])
            updated_features = updated_features[sort_idx, :]
    
    # Create updated dictionary
    updated_data = data.copy()
    updated_data['cluster_class'] = updated_cluster_class
    updated_data['spikes'] = updated_waveforms
    updated_data['cluster_labels'] = updated_cluster_class[:, 0].astype(np.uint32)
    updated_data['spike_times'] = updated_cluster_class[:, 1].astype(np.float64)
    updated_data['n_spikes'] = np.array([len(updated_cluster_class)], dtype=np.uint32)
    
    if updated_features is not None:
        if 'inspk' in data:
            updated_data['inspk'] = updated_features
        if 'features' in data:
            updated_data['features'] = updated_features
    
    return updated_data

# ============================================================================
# MAIN SPIKE RESCUER CLASS - FIXED FOR WAVECLUS FORMAT
# ============================================================================

class SpikeRescuer:
    """
    Rescue quarantined spikes using template matching.
    
    HOW IT WORKS:
    1. Loads times file (classified spikes with cluster labels)
    2. Loads spike file with ALL detected spikes:
       - Uses 'spikes_all' + 'index_all' (all detected spikes)
       - NOT just 'spikes' + 'index' (only preserved spikes)
    3. Identifies quarantined spikes by comparing spike times:
       - Spikes in spike file NOT in times file = quarantined
    4. Performs template matching on quarantined spikes
    5. Merges successfully matched spikes back into times file
    
    SPIKE FILE STRUCTURE:
    - spikes_all (8999): ALL detected spikes including quarantined
    - index_all (8999): Times for ALL spikes
    - spikes (8323): Only preserved spikes (subset)
    - index (8323): Times for preserved spikes only
    - Quarantined = spikes_all - spikes (e.g., 676 spikes)
    
    The quarantined spikes are typically removed due to:
    - Collision artifacts (mask_bundle_coll)
    - Possible artifacts (possible_artifact)
    - ISI violations
    - Low amplitude, etc.
    """
    
    def __init__(self, times_dir: str, spikes_dir: str, 
                 output_dir: Optional[str] = None,
                 template_sdnum: float = 3.0,
                 metric: Literal['euclidean', 'correlation'] = 'euclidean',
                 time_tolerance: float = 1e-6,
                 rescue_label_threshold: int = 0,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize the SpikeRescuer.
        
        Args:
            times_dir: Directory containing 'times_*.mat' files (classified data).
            spikes_dir: Directory containing '*_spikes.mat' files (all spikes).
            output_dir: Directory to save updated 'times_*.mat' files (default: times_dir).
            template_sdnum: Number of std devs for acceptance (default: 3.0).
            metric: 'euclidean' or 'correlation' (default: 'euclidean').
            time_tolerance: Tolerance in ms for matching spike times (default: 1e-6).
            rescue_label_threshold: Spikes with Label <= this value are rescue candidates (default: 0).
            verbose: Print progress (default: True).
            **kwargs: Additional args for force_membership.
        """
        self.times_dir = times_dir
        self.spikes_dir = spikes_dir
        self.output_dir = output_dir if output_dir else times_dir
        self.verbose = verbose
        self.time_tolerance = time_tolerance
        self.rescue_label_threshold = rescue_label_threshold
        
        self.match_params = {
            'template_sdnum': template_sdnum,
            'metric': metric,
            **kwargs
        }
        
        if not os.path.exists(self.times_dir):
            raise ValueError(f"Times directory does not exist: {self.times_dir}")
        if not os.path.exists(self.spikes_dir):
            raise ValueError(f"Spikes directory does not exist: {self.spikes_dir}")
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            self._log(f"Created output directory: {self.output_dir}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _find_files_for_channel(self, channel_num: int) -> Tuple[Optional[str], Optional[str]]:
        """Find the times and spikes files for a specific channel."""
        
        # Find times file
        times_file_path = None
        times_files = glob.glob(os.path.join(self.times_dir, f'times_*_{channel_num}.mat'))
        if times_files:
            times_file_path = times_files[0]
        else:
            times_files_ch = glob.glob(os.path.join(self.times_dir, f'times_ch{channel_num}.mat'))
            if times_files_ch:
                times_file_path = times_files_ch[0]
                
        # Find spikes file
        spikes_file_path = None
        spikes_files = glob.glob(os.path.join(self.spikes_dir, f'*_{channel_num}_spikes.mat'))
        if spikes_files:
            spikes_file_path = spikes_files[0]
            
        return times_file_path, spikes_file_path

    def _parse_channels(self, channels: Union[str, int, list]) -> list:
        """Parse channel spec and return channels present in both dirs."""
        
        # Find channels with times files
        all_times_files = glob.glob(os.path.join(self.times_dir, 'times_*.mat'))
        times_channels = set()
        for f in all_times_files:
            basename = os.path.basename(f)
            match1 = re.search(r'times_.*_(\d+)\.mat$', basename)
            match2 = re.search(r'times_ch(\d+)\.mat$', basename)
            if match1:
                times_channels.add(int(match1.group(1)))
            elif match2:
                times_channels.add(int(match2.group(1)))
        
        # Find channels with spikes files
        all_spike_files = glob.glob(os.path.join(self.spikes_dir, '*_spikes.mat'))
        spikes_channels = set()
        for f in all_spike_files:
            basename = os.path.basename(f)
            match = re.search(r'_(\d+)_spikes\.mat$', basename)
            if match:
                spikes_channels.add(int(match.group(1)))
                
        all_available_channels = times_channels.intersection(spikes_channels)

        if not all_available_channels:
            self._log(f"Warning: No channels found in *both* directories")
            return []

        if isinstance(channels, str) and channels.lower() == 'all':
            return sorted(list(all_available_channels))
        
        if isinstance(channels, int):
            channels = [channels]
        
        if isinstance(channels, (list, tuple)):
            requested = set(channels)
            to_process = sorted(list(requested.intersection(all_available_channels)))
            not_found = sorted(list(requested - all_available_channels))
            
            if not_found:
                self._log(f"Warning: Requested channels not found in both dirs: {not_found}")
            if not to_process:
                self._log(f"Warning: No requested channels found in both dirs.")
            
            return to_process
        
        self._log(f"Warning: Unknown channel type {type(channels)}, using all channels")
        return sorted(list(all_available_channels))

    def process_single_channel(self, channel_num: int) -> bool:
        """Run spike rescue on a single channel."""
        times_file, spikes_file = self._find_files_for_channel(channel_num)
        
        if times_file is None:
            self._log(f"Warning: No 'times_*.mat' file found for channel {channel_num}. Skipping.")
            return False
        if spikes_file is None:
            self._log(f"Warning: No '*_spikes.mat' file found for channel {channel_num}. Skipping.")
            return False
            
        try:
            self._log(f"\n{'='*60}")
            self._log(f"Channel {channel_num} - Spike Rescue")
            self._log(f"  Templates File: {os.path.basename(times_file)}")
            self._log(f"  Spikes Source:  {os.path.basename(spikes_file)}")
            self._log(f"{'='*60}")
            
            # --- 1. Load Classified Data (times file) ---
            times_data = loadmat(times_file)
            
            if 'cluster_class' not in times_data or 'spikes' not in times_data:
                self._log("Error: 'times' file is missing 'cluster_class' or 'spikes'. Skipping.")
                return False
            
            cluster_class = times_data['cluster_class']
            if cluster_class.shape[0] == 2 and cluster_class.shape[1] > 2:
                cluster_class = cluster_class.T
            
            # Get features - check both field names
            features_key = None
            if 'inspk' in times_data:
                features_key = 'inspk'
            elif 'features' in times_data:
                features_key = 'features'
            
            data_dict = {
                'cluster_class': cluster_class,
                'spikes': times_data['spikes'].astype(np.float64)
            }
            
            if features_key:
                data_dict[features_key] = times_data[features_key]
            
            # Preserve other fields from times file
            for key in times_data.keys():
                if not key.startswith('__') and key not in data_dict:
                    data_dict[key] = times_data[key]
            
            data_labels = cluster_class[:, 0].astype(np.int32)
            data_times = cluster_class[:, 1]
            
            self._log(f"Loaded {len(data_times)} classified spikes from times file.")

            # --- 2. Load Rescue Source (spike file - FIXED) ---
            spike_data = loadmat(spikes_file)
            
            # CRITICAL: Use spikes_ALL and index_ALL (all detected spikes)
            # NOT just 'spikes' and 'index' (which are only preserved spikes)
            
            # Check for waveforms - prefer spikes_all over spikes
            if 'spikes_all' in spike_data:
                all_waveforms = spike_data['spikes_all'].astype(np.float64)
                self._log(f"  Using 'spikes_all' field ({len(all_waveforms)} total detected spikes)")
            elif 'spikes' in spike_data:
                all_waveforms = spike_data['spikes'].astype(np.float64)
                self._log(f"  Using 'spikes' field ({len(all_waveforms)} spikes)")
            else:
                self._log("Error: Neither 'spikes_all' nor 'spikes' field found in spike file. Skipping.")
                return False
            
            # Check for spike times - prefer index_all over index
            if 'index_all' in spike_data:
                all_times = spike_data['index_all'].flatten()
                self._log(f"  Using 'index_all' field ({len(all_times)} times)")
            elif 'Spike_Time' in spike_data:
                all_times = spike_data['Spike_Time'].flatten()
                self._log(f"  Using 'Spike_Time' field ({len(all_times)} times)")
            elif 'index' in spike_data:
                all_times = spike_data['index'].flatten()
                self._log(f"  Using 'index' field ({len(all_times)} times)")
            else:
                self._log("Error: No time field found (tried index_all, Spike_Time, index). Skipping.")
                self._log(f"  Available fields: {[k for k in spike_data.keys() if not k.startswith('__')]}")
                return False
            
            if len(all_waveforms) != len(all_times):
                self._log(f"Error: Mismatch between waveforms ({len(all_waveforms)}) and times ({len(all_times)}). Skipping.")
                return False
            
            # Identify rescue candidates by comparing against classified data
            # Quarantined = spikes in spike file that are NOT in times file
            self._log(f"\n  Comparing spike times to find quarantined spikes...")
            
            quarantined_mask = np.ones(len(all_times), dtype=bool)
            
            # Mark spikes that are already classified
            for i, st in enumerate(all_times):
                if np.any(np.abs(data_times - st) < self.time_tolerance):
                    quarantined_mask[i] = False
            
            n_total = len(all_times)
            n_already_classified = np.sum(~quarantined_mask)
            n_quarantined = np.sum(quarantined_mask)
            
            self._log(f"  Total spikes in spike file: {n_total}")
            self._log(f"  Already classified in times file: {n_already_classified}")
            self._log(f"  Quarantined (rescue candidates): {n_quarantined} ({n_quarantined/n_total*100:.1f}%)")
            
            if np.sum(quarantined_mask) == 0:
                self._log("  No quarantined spikes found. All spikes are already classified.")
                return True

            rescue_times = all_times[quarantined_mask]
            rescue_waveforms = all_waveforms[quarantined_mask, :]
            
            rescue_dict = {
                'times': rescue_times,
                'spikes': rescue_waveforms
            }
            
            # Add features if available - check for features_all or inspk_all first
            if 'features_all' in spike_data:
                rescue_dict['features'] = spike_data['features_all'][quarantined_mask, :]
                self._log(f"  Using 'features_all' for rescue spike features")
            elif 'inspk_all' in spike_data:
                rescue_dict['inspk'] = spike_data['inspk_all'][quarantined_mask, :]
                self._log(f"  Using 'inspk_all' for rescue spike features")
            elif 'inspk' in spike_data:
                rescue_dict['inspk'] = spike_data['inspk'][quarantined_mask, :]
                self._log(f"  Using 'inspk' for rescue spike features")
            elif 'features' in spike_data:
                rescue_dict['features'] = spike_data['features'][quarantined_mask, :]
                self._log(f"  Using 'features' for rescue spike features")
            else:
                self._log(f"  No features found in spike file (optional)")
            
            self._log(f"\n  Ready to rescue {len(rescue_times)} spikes.")

            # --- 3. Run Rescue ---
            updated_data = rescue_spikes_with_templates(
                data=data_dict,
                rescue=rescue_dict,
                time_tolerance=self.time_tolerance,
                verbose=self.verbose,
                **self.match_params
            )
            
            # --- 4. Save Results ---
            if self.output_dir != self.times_dir:
                output_path = os.path.join(self.output_dir, os.path.basename(times_file))
            else:
                output_path = times_file
            
            n_added = updated_data['cluster_class'].shape[0] - data_dict['cluster_class'].shape[0]
            
            # Add rescue stats
            stats = {
                'total_candidates': len(rescue_times),
                'total_rescued': n_added,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': self.match_params['metric'],
                'template_sdnum': self.match_params['template_sdnum']
            }
            updated_data['spike_rescue_stats'] = stats
            
            savemat(output_path, updated_data, do_compression=True)
            self._log(f"\n✓ Successfully rescued {n_added} spikes for channel {channel_num}.")
            self._log(f"✓ Updated file saved to: {os.path.basename(output_path)}")
            
            return True

        except Exception as e:
            self._log(f"\n✗ ERROR processing channel {channel_num}: {e}")
            traceback.print_exc()
            return False

    def process_all_channels(self, channels: Union[str, int, list] = 'all'):
        """Process spike data for all specified channels."""
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            self._log(f"No channels specified or found to process.")
            return
        
        self._log(f"\n{'#'*70}")
        self._log(f"SPIKE RESCUE PIPELINE - FIXED VERSION")
        self._log(f"{'#'*70}")
        self._log(f"Times (Data) Dir: {self.times_dir}")
        self._log(f"Spikes (Rescue) Dir: {self.spikes_dir}")
        self._log(f"Output Dir: {self.output_dir}")
        self._log(f"Total channels to process: {len(channel_nums)}")
        self._log(f"Rescue threshold: Label <= {self.rescue_label_threshold}")
        self._log(f"Parameters: {self.match_params}")
        self._log(f"{'#'*70}\n")
        
        successful = 0
        failed = 0
        
        for i, ch_num in enumerate(channel_nums, 1):
            self._log(f"[{i}/{len(channel_nums)}] Processing channel {ch_num}...")
            if self.process_single_channel(ch_num):
                successful += 1
            else:
                failed += 1
                
        self._log(f"\n{'#'*70}")
        self._log(f"SPIKE RESCUE COMPLETE")
        self._log(f"{'#'*70}")
        self._log(f"Successful: {successful}/{len(channel_nums)}")
        self._log(f"Failed: {failed}/{len(channel_nums)}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"{'#'*70}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    """Example usage of the fixed SpikeRescuer class."""
    
    import sys
    from pathlib import Path
    
    # Example paths
    base_dir = Path().resolve()
    times_dir = str(base_dir / 'output')
    spikes_dir = str(base_dir / 'output')
    
    print("="*70)
    print("SPIKE RESCUER - FIXED FOR WAVECLUS FORMAT")
    print("="*70)
    print("\nThis version works with waveclus spike files that have:")
    print("  - 'Spike_Time' instead of 'index'")
    print("  - 'Label' field for identifying unclassified spikes")
    print("  - No 'properties' field required")
    print("="*70)
    
    if len(sys.argv) > 1:
        channel = int(sys.argv[1])
        channels_to_process = [channel]
    else:
        channels_to_process = 'all'
    
    rescuer = SpikeRescuer(
        times_dir=times_dir,
        spikes_dir=spikes_dir,
        output_dir=times_dir,
        template_sdnum=3.0,
        metric='euclidean',
        rescue_label_threshold=0  # Rescue spikes with Label <= 0
    )
    
    rescuer.process_all_channels(channels=channels_to_process)