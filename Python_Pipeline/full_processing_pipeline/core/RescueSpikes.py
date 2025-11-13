"""
Rescue Spikes with Template Matching

This module provides a class-based wrapper, SpikeRescuer, to find
spikes that were quarantined (e.g., 'preserved' == 0) and re-classify
them using template matching against already-clustered units.

It reads from both the '*_spikes.mat' (source of all spikes) and 
'times_*.mat' (source of classified templates) files.

Author: Masoud Khani
Date: November 5, 2025
Version: 1.0.0
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
warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore correlation NaN warnings

# ============================================================================
# CORE TEMPLATE MATCHING FUNCTIONS
# (From template_matching.py, included here to be self-contained)
# ============================================================================

def build_templates(
    classes: NDArray[np.int32],
    waveforms: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Build template waveforms and variance metrics for each cluster.
    """
    if classes.shape[0] != waveforms.shape[0]:
        raise ValueError("classes and waveforms must have the same number of rows")
    if classes.size == 0:
        raise ValueError("classes array cannot be empty")
    
    valid_classes = classes[classes > 0]
    if valid_classes.size == 0:
        if np.any(classes > 1000): # Handle artifact/noise labels
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
    """
    Find the nearest neighbor template for a spike waveform.
    """
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
        return conforming[k_nearest] + 1  # 1-based indexing
    else:
        nearest_idx = np.argmin(distances[conforming])
        return int(conforming[nearest_idx] + 1)  # 1-based indexing


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
    """
    Classify unclassified spikes using waveform template matching.
    """
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
        return class_out # Return all zeros
    
    if templates.shape[0] == 0:
        print("Warning: Template building returned no templates. Cannot classify.")
        return class_out # Return all zeros
        
    # Scale distance threshold by sdnum
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

# ============================================================================
# MAIN SPIKE RESCUER CLASS
# ============================================================================

class SpikeRescuer:
    """
    Class to rescue quarantined spikes by matching them to classified templates.
    
    Finds 'times_*.mat' (classified data) and '*_spikes.mat' (all spikes),
    loads them, performs template matching on spikes where 'preserved == 0',
    and merges the newly classified spikes back into the 'times_*.mat' file.
    """
    
    def __init__(self, times_dir: str, spikes_dir: str, 
                 output_dir: Optional[str] = None,
                 template_sdnum: float = 3.0,
                 metric: Literal['euclidean', 'correlation'] = 'euclidean',
                 time_tolerance: float = 1e-6,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize the SpikeRescuer.
        
        Args:
            times_dir (str): Directory containing 'times_*.mat' files (classified data).
            spikes_dir (str): Directory containing '*_spikes.mat' files (all spikes).
            output_dir (str, optional): Directory to save updated 'times_*.mat' files.
                                        Defaults to times_dir (in-place update).
            template_sdnum (float): Number of std devs for acceptance (3.0) OR
                                    min correlation threshold (e.g., 0.9) if metric='correlation'.
            metric (str): 'euclidean' or 'correlation' (default 'euclidean').
            time_tolerance (float): Tolerance in ms for matching spike times (default 1e-6).
            verbose (bool): Print progress.
            **kwargs: Additional args for force_membership (template_k, use_pointdist, etc.)
        """
        self.times_dir = times_dir
        self.spikes_dir = spikes_dir
        self.output_dir = output_dir if output_dir else times_dir
        self.verbose = verbose
        self.time_tolerance = time_tolerance
        
        # Store matching parameters
        self.match_params = {
            'template_sdnum': template_sdnum,
            'metric': metric,
            **kwargs # Pass through template_k, use_pointdist, etc.
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
        """Finds the times and spikes files for a specific channel."""
        
        # 1. Find times file (classified data)
        times_file_path = None
        times_files = glob.glob(os.path.join(self.times_dir, f'times_*_{channel_num}.mat'))
        if times_files:
            times_file_path = times_files[0]
        else:
            times_files_ch = glob.glob(os.path.join(self.times_dir, f'times_ch{channel_num}.mat'))
            if times_files_ch:
                times_file_path = times_files_ch[0]
                
        # 2. Find spikes file (all spikes + properties)
        spikes_file_path = None
        spikes_files = glob.glob(os.path.join(self.spikes_dir, f'*_{channel_num}_spikes.mat'))
        if spikes_files:
            spikes_file_path = spikes_files[0]
            
        return times_file_path, spikes_file_path

    def _parse_channels(self, channels: Union[str, int, list]) -> list:
        """Parse channel spec and return channels present in *both* dirs."""
        
        # Find all channels with times files
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
        
        # Find all channels with spikes files
        all_spike_files = glob.glob(os.path.join(self.spikes_dir, '*_spikes.mat'))
        spikes_channels = set()
        for f in all_spike_files:
            basename = os.path.basename(f)
            match = re.search(r'_(\d+)_spikes\.mat$', basename)
            if match:
                spikes_channels.add(int(match.group(1)))
                
        # Find the intersection
        all_available_channels = times_channels.intersection(spikes_channels)

        if not all_available_channels:
            self._log(f"Warning: No channels found in *both* {self.times_dir} and {self.spikes_dir}")
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
        """
        Run spike rescue on a single channel.
        """
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
            
            # --- 1. Load Classified Data (for templates) ---
            data_file = loadmat(times_file)
            if 'cluster_class' not in data_file or 'spikes' not in data_file:
                self._log("Error: 'times' file is missing 'cluster_class' or 'spikes'. Skipping.")
                return False
            
            cluster_class = data_file['cluster_class']
            if cluster_class.shape[0] == 2 and cluster_class.shape[1] > 2:
                 cluster_class = cluster_class.T
            
            data_dict = {
                'cluster_class': cluster_class,
                'spikes': data_file['spikes'].astype(np.float64),
                'inspk': data_file.get('inspk') # Optional
            }
            data_clusters = cluster_class[:, 0].astype(np.int32)
            data_times = cluster_class[:, 1]
            data_waveforms = data_dict['spikes']

            # --- 2. Load Rescue Source (Quarantined Spikes) ---
            rescue_file = loadmat(spikes_file)
            if 'spikes' not in rescue_file or 'index' not in rescue_file or 'properties' not in rescue_file:
                self._log("Error: 'spikes' file is missing 'spikes', 'index', or 'properties'. Skipping.")
                return False
            
            all_waveforms = rescue_file['spikes'].astype(np.float64)
            all_times = rescue_file['index'].flatten()
            
            # Check for 'preserved' field in properties
            if 'preserved' not in rescue_file['properties'].dtype.names:
                self._log("Error: 'properties' field in 'spikes' file is missing 'preserved'. Skipping.")
                return False
                
            preserved_mask = rescue_file['properties']['preserved'].flatten().astype(int)
            quarantined_mask = (preserved_mask == 0)
            
            if np.sum(quarantined_mask) == 0:
                self._log("No quarantined spikes (preserved=0) found to rescue. Skipping.")
                return True

            rescue_times = all_times[quarantined_mask]
            rescue_waveforms = all_waveforms[quarantined_mask, :]
            
            rescue_dict = {'times': rescue_times, 'spikes': rescue_waveforms}
            
            # Add features if they exist
            if 'inspk' in rescue_file:
                rescue_dict['inspk'] = rescue_file['inspk'][quarantined_mask, :]
            
            self._log(f"Found {len(data_times)} classified spikes (for templates).")
            self._log(f"Found {len(rescue_times)} quarantined spikes to rescue.")

            # --- 3. Run Rescue Function ---
            # This is the function from your provided script
            updated_data = rescue_spikes_with_templates(
                data=data_dict,
                rescue=rescue_dict,
                time_tolerance=self.time_tolerance,
                verbose=self.verbose,
                **self.match_params
            )
            
            # --- 4. Save Results ---
            # The rescue function already copies over 'par' and other fields
            # We just need to save the returned dictionary
            
            # Determine output path
            if self.output_dir != self.times_dir:
                output_path = os.path.join(self.output_dir, os.path.basename(times_file))
            else:
                output_path = times_file # Overwrite in-place
            
            # Add rescue stats
            n_added = updated_data['cluster_class'].shape[0] - data_dict['cluster_class'].shape[0]
            stats = {
                'total_quarantined': len(rescue_times),
                'total_rescued': n_added,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            updated_data['spike_rescue_stats'] = stats
            
            savemat(output_path, updated_data, do_compression=True)
            self._log(f"Successfully rescued {n_added} spikes for channel {channel_num}.")
            self._log(f"Updated file saved to: {output_path}")
            
            return True

        except Exception as e:
            self._log(f"ERROR processing channel {channel_num}: {e}")
            traceback.print_exc()
            return False

    def process_all_channels(self, channels: Union[str, int, list] = 'all'):
        """
        Process spike data for all specified channels.
        
        Args:
            channels: 'all', a single channel int, or a list of channel ints.
        """
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            self._log(f"No channels specified or found to process.")
            return
        
        self._log(f"\n{'#'*70}")
        self._log(f"SPIKE RESCUE PIPELINE")
        self._log(f"{'#'*70}")
        self._log(f"Times (Data) Dir: {self.times_dir}")
        self._log(f"Spikes (Rescue) Dir: {self.spikes_dir}")
        self._log(f"Output Dir: {self.output_dir}")
        self._log(f"Total channels to process: {len(channel_nums)}")
        self._log(f"Parameters: {self.match_params}")
        self._log(f"{'#'*70}\n")
        
        successful = 0
        failed = 0
        
        for i, ch_num in enumerate(channel_nums, 1):
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
    """
    Example usage of the SpikeRescuer class.
    """
    
    base_dir = os.path.dirname(__file__) # Or set your absolute path
    
    # Dir with 'times_*.mat' files
    times_file_dir = os.path.join(base_dir, 'output') 
    # Dir with '*_spikes.mat' files
    spike_file_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(times_file_dir) or not os.path.exists(spike_file_dir):
        print(f"Example directory not found: {times_file_dir} or {spike_file_dir}")
        print("Please run spike detection and clustering first.")
    else:
        # ========================================
        # EXAMPLE: Rescue spikes for all channels
        # ========================================
        print("\n" + "="*70)
        print("EXAMPLE: Rescue all quarantined spikes in all channels")
        print("="*70)
        
        rescuer = SpikeRescuer(
            times_dir=times_file_dir,
            spikes_dir=spike_file_dir,
            output_dir=times_file_dir, # Overwrite existing times_ files
            template_sdnum=3.0, 
            metric='euclidean'
        )
        rescuer.process_all_channels(channels='all')