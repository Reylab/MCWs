"""
Template Matching for Spike Sorting

This module provides a class-based wrapper for template-matching, designed to
run on the output of a spike clustering pipeline (i.e., on 'times_*.mat' files). 
It finds unclassified spikes (label 0) and assigns them to existing clusters 
based on waveform similarity.

Usage:
    from TemplateMatching import TemplateMatcher
    
    # Initialize the matcher
    matcher = TemplateMatcher(
        input_dir='/path/to/times_files',  # Directory containing times_*.mat
        template_sdnum=3.0,
        metric='euclidean'
    )
    
    # Match unclassified spikes in ALL channels
    matcher.process_all_channels(channels='all')
    
    # Match unclassified spikes in specific channels
    matcher.process_all_channels(channels=[257, 263])
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
# ============================================================================

def build_templates(
    classes: NDArray[np.int32],
    waveforms: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Build template waveforms and variance metrics for each cluster.
    
    Args:
        classes: Array of cluster labels with shape (n_spikes,).
        waveforms: Waveform matrix with shape (n_spikes, n_samples).
    
    Returns:
        Tuple containing:
            - templates: (n_clusters, n_samples) - mean waveform per cluster
            - maxdist: (n_clusters,) - std dev of Euclidean distance from mean
            - pointdist: (n_clusters, n_samples) - std dev along each time point
    """
    if classes.shape[0] != waveforms.shape[0]:
        raise ValueError("classes and waveforms must have the same number of rows")
    if classes.size == 0:
        raise ValueError("classes array cannot be empty")
    
    # Find max class, ignoring potential artifact labels like 500, 1000, 9999
    valid_classes = classes[classes > 0]
    if valid_classes.size == 0:
        # Check for legacy noise cluster 9999
        if np.any(classes == 9999): 
             # This file might only have noise, which is not an error
             print("Warning: classes array contains no valid clusters (only 0 or 9999). Cannot build templates.")
             # Return empty arrays with correct dimensions
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
        # Using ddof=0 to match MATLAB's var(..., 1) which divides by N
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
    Find the nearest neighbor template for a spike waveform using distance constraints.
    
    Args:
        x: Waveform of the spike to classify with shape (n_samples,).
        templates: Template waveforms with shape (n_clusters, n_samples).
        maxdist: Max allowed distance or (1 - min_correlation)
    
    Returns:
        int or array: Index (1-based) of nearest template(s) or 0 if none conform.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array (waveform)")
    if templates.shape[1] != x.shape[0]:
        raise ValueError(f"x (shape {x.shape}) and templates (shape {templates.shape}) must have compatible dimensions")
    if templates.shape[0] != maxdist.shape[0]:
        raise ValueError("templates and maxdist must have same number of templates")
    
    n_templates = templates.shape[0]
    
    if n_templates == 0:
        return 0 # No templates to match against
    
    if metric == 'euclidean':
        distances = np.sqrt(np.sum((templates - x[np.newaxis, :]) ** 2, axis=1))
        conforming = np.where(distances < maxdist)[0]
    elif metric == 'correlation':
        # Compute Pearson correlation for each template
        # Handle zero-variance waveforms (constant lines) which cause NaN
        x_std = np.std(x)
        if x_std == 0:
            distances = np.full(n_templates, np.inf) # Cannot correlate
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
            # For correlation, maxdist is (1 - min_corr_threshold)
            conforming = np.where(distances < maxdist)[0]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if pointdist is not None:
        if pointlimit is None:
            pointlimit = np.inf
        
        pointwise_conforming = []
        for i in conforming: # Only check templates that already conform
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
    if waveforms_in.shape[0] == 0:
        print("Warning: No classified spikes provided (waveforms_in is empty). Cannot build templates.")
        return np.zeros(waveforms_out.shape[0], dtype=np.int32)
    if waveforms_out.shape[0] == 0:
        print("No unclassified spikes to process.")
        return np.array([], dtype=np.int32)
    if waveforms_in.shape[1] != waveforms_out.shape[1]:
        raise ValueError("waveforms_in and waveforms_out must have same number of samples")
    
    n_spikes = waveforms_out.shape[0]
    class_out = np.zeros(n_spikes, dtype=np.int32)
    
    try:
        templates, sd, pd = build_templates(classes_in, waveforms_in)
    except ValueError as e:
        print(f"Warning: {e}. Cannot build templates.")
        return class_out # Return all zeros
    
    # Scale distance threshold by sdnum
    if metric == 'euclidean':
        sd_scaled = template_sdnum * sd
    elif metric == 'correlation':
        # For correlation, template_sdnum is the threshold (e.g., 0.9)
        # maxdist becomes (1 - threshold)
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


def match_spikes_to_templates(
    spike_times: NDArray[np.float64],
    waveforms: NDArray[np.float64],
    classes: NDArray[np.int32],
    template_sdnum: float = 3.0,
    **kwargs
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    """
    High-level convenience function for waveform-based template matching.
    """
    if not all(s.shape[0] == spike_times.shape[0] for s in [waveforms, classes]):
        raise ValueError("spike_times, waveforms, and classes must all have same length")
    
    classified_mask = classes > 0
    unclassified_mask = classes == 0
    
    waveforms_classified = waveforms[classified_mask, :]
    classes_classified = classes[classified_mask]
    waveforms_unclassified = waveforms[unclassified_mask, :]
    
    new_labels = force_membership(
        waveforms_classified,
        classes_classified,
        waveforms_unclassified,
        template_sdnum=template_sdnum,
        **kwargs
    )
    
    all_classes = classes.copy()
    all_classes[unclassified_mask] = new_labels
    
    still_unclassified = all_classes == 0
    unclassified_times = spike_times[still_unclassified]
    
    return all_classes, unclassified_times

# ============================================================================
# MAIN TEMPLATE MATCHER CLASS
# ============================================================================

class TemplateMatcher:
    """
    Class to find and update cluster assignments for unclassified spikes
    by matching them to templates built from classified spikes.
    
    Finds 'times_*.mat' files, loads them, performs template matching on
    spikes with label 0, and updates the files in-place.
    """
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None,
                 template_sdnum: float = 3.0,
                 template_k: Optional[int] = None,
                 template_k_min: int = 1,
                 use_pointdist: bool = False,
                 pointlimit: Optional[int] = None,
                 metric: Literal['euclidean', 'correlation'] = 'euclidean',
                 verbose: bool = True):
        """
        Initialize the Template Matcher.
        
        Args:
            input_dir (str): Directory containing 'times_*.mat' files.
            output_dir (str, optional): Directory to save updated files. 
                                        Defaults to input_dir (in-place update).
            template_sdnum (float): Number of std devs for acceptance (3.0) OR
                                    min correlation threshold (e.g., 0.9) if metric='correlation'.
            template_k (int, optional): Number of neighbors for k-NN voting.
            template_k_min (int): Min neighbors for k-NN (default 1).
            use_pointdist (bool): Use point-wise deviation constraints (default False).
            pointlimit (int, optional): Max allowed samples outside pointdist.
            metric (str): 'euclidean' or 'correlation' (default 'euclidean').
            verbose (bool): Print progress.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.verbose = verbose
        
        # Store matching parameters
        self.match_params = {
            'template_sdnum': template_sdnum,
            'template_k': template_k,
            'template_k_min': template_k_min,
            'use_pointdist': use_pointdist,
            'pointlimit': pointlimit,
            'metric': metric
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
        # --- EDIT: Look for times_*.mat files ---
        all_times_files = glob.glob(os.path.join(self.input_dir, 'times_*.mat'))
        all_available_channels = set()
        
        for f in all_times_files:
            basename = os.path.basename(f)
            # --- START OF FIX: Use two regex patterns ---
            # Pattern 1: e.g., times_mLAMY08_raw_273.mat -> 273
            match1 = re.search(r'times_.*_(\d+)\.mat$', basename)
            # Pattern 2: e.g., times_ch328.mat -> 328
            match2 = re.search(r'times_ch(\d+)\.mat$', basename)
            
            if match1:
                all_available_channels.add(int(match1.group(1)))
            elif match2:
                all_available_channels.add(int(match2.group(1)))
            # --- END OF FIX ---
                
        if not all_available_channels:
            self._log(f"Warning: No 'times_*.mat' or 'times_ch*.mat' files found in {self.input_dir}")
            return []
        # --- END EDIT ---

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
        # --- START OF FIX: Check both patterns ---
        # Pattern 1: e.g., times_..._328.mat
        times_files = glob.glob(os.path.join(self.input_dir, f'times_*_{channel_num}.mat'))
        if times_files:
            return times_files[0] # Return the first match
            
        # Pattern 2: e.g., times_ch328.mat
        times_files_ch = glob.glob(os.path.join(self.input_dir, f'times_ch{channel_num}.mat'))
        if times_files_ch:
            return times_files_ch[0] # Return this match
            
        return None
        # --- END OF FIX ---

    def process_single_channel(self, channel_num: int) -> bool:
        """
        Run template matching on a single channel's times file.
        
        Args:
            channel_num (int): Channel number to process.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        file_path = self._find_file_for_channel(channel_num)
        
        if file_path is None:
            # --- EDIT: Updated warning message ---
            self._log(f"Warning: No 'times_*_{channel_num}.mat' or 'times_ch{channel_num}.mat' file found")
            return False
            # --- END EDIT ---
            
        try:
            self._log(f"\n{'='*60}")
            self._log(f"Channel {channel_num} - Template Matching")
            self._log(f"File: {os.path.basename(file_path)}")
            self._log(f"{'='*60}")
            
            mat_data = loadmat(file_path)
            
            # --- 1. Load Data ---
            if 'spikes' not in mat_data:
                self._log("Error: 'spikes' (waveforms) field not found.")
                return False
            if 'cluster_class' not in mat_data:
                self._log("Error: 'cluster_class' field not found.")
                return False
                
            waveforms = mat_data['spikes'].astype(np.float64)
            cluster_class = mat_data['cluster_class']
            
            # Ensure cluster_class is (n_spikes, 2)
            if cluster_class.shape[0] == 2 and cluster_class.shape[1] > 2: # Check if it's (2, N)
                 cluster_class = cluster_class.T # Transpose if saved in MATLAB format
            
            if cluster_class.shape[1] < 2:
                self._log(f"Error: 'cluster_class' has shape {cluster_class.shape}, expected (n, 2).")
                return False
            
            if waveforms.shape[0] != cluster_class.shape[0]:
                self._log(f"Error: Mismatch in spike count. Waveforms: {waveforms.shape[0]}, Classes: {cluster_class.shape[0]}")
                return False
                
            classes = cluster_class[:, 0].astype(np.int32)
            spike_times = cluster_class[:, 1].astype(np.float64)

            # --- 2. Check for work ---
            classified_mask = classes > 0
            unclassified_mask = classes == 0
            
            n_classified = np.sum(classified_mask)
            n_unclassified = np.sum(unclassified_mask)
            
            if n_classified == 0:
                self._log("Skipping: No classified spikes found to build templates.")
                return True # Not a failure, just nothing to do
            if n_unclassified == 0:
                self._log("Skipping: No unclassified spikes (label 0) found to match.")
                return True # Not a failure, just nothing to do

            self._log(f"Found {n_classified} classified spikes (templates) and {n_unclassified} unclassified spikes.")

            # --- 3. Run Template Matching ---
            self._log(f"Running template matching with metric: {self.match_params['metric']}...")
            
            waveforms_in = waveforms[classified_mask, :]
            classes_in = classes[classified_mask]
            waveforms_out = waveforms[unclassified_mask, :]
            
            new_labels_unclassified = force_membership(
                waveforms_in, classes_in, waveforms_out, **self.match_params
            )
            
            # --- 4. Update and Save Results ---
            updated_classes = classes.copy()
            updated_classes[unclassified_mask] = new_labels_unclassified
            
            n_newly_classified = np.sum(new_labels_unclassified > 0)
            n_still_unclassified = np.sum(new_labels_unclassified == 0)
            
            # Update the original .mat data structure
            mat_data['cluster_class'][:, 0] = updated_classes
            
            # Add stats
            stats = {
                'total_processed': n_unclassified,
                'newly_classified': n_newly_classified,
                'still_unclassified': n_still_unclassified,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            mat_data['template_matching_stats'] = stats
            
            # Determine output path
            if self.output_dir != self.input_dir:
                output_path = os.path.join(self.output_dir, os.path.basename(file_path))
            else:
                output_path = file_path # Overwrite in-place
                
            savemat(output_path, mat_data, do_compression=True)
            
            self._log(f"Successfully processed channel {channel_num}.")
            self._log(f"  Newly classified: {n_newly_classified}")
            self._log(f"  Still unclassified (label 0): {n_still_unclassified}")
            self._log(f"  Updated file saved to: {output_path}")
            
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
            self._log(f"No channels specified or found in {self.input_dir}")
            return
        
        self._log(f"\n{'#'*70}")
        self._log(f"TEMPLATE MATCHING PIPELINE")
        self._log(f"{'#'*70}")
        self._log(f"Input directory: {self.input_dir}")
        self._log(f"Output directory: {self.output_dir}")
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
        self._log(f"TEMPLATE MATCHING COMPLETE")
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
    Example usage of the TemplateMatcher class.
    """
    
    # --- EDIT: Set directory to where 'times_*.mat' files are ---
    base_dir = os.path.dirname(__file__) # Or set your absolute path
    times_file_dir = os.path.join(base_dir, 'output') # Assumes 'output' dir
    
    if not os.path.exists(times_file_dir):
        print(f"Example directory not found: {times_file_dir}")
        print("Please run clustering first to generate 'times_*.mat' files.")
    else:
        # ========================================
        # EXAMPLE 1: Match all channels
        # ========================================
        print("\n" + "="*70)
        print("EXAMPLE 1: Match all unclassified spikes in all channels")
        print("="*70)
        
        matcher_all = TemplateMatcher(
            input_dir=times_file_dir,
            template_sdnum=3.0,  # Use 3 standard deviations
            metric='euclidean'
        )
        matcher_all.process_all_channels(channels='all')
        
        
        # ========================================
        # EXAMPLE 2: Match specific channels with correlation
        # ========================================
        print("\n" + "="*70)
        print("EXAMPLE 2: Match specific channels [257, 263] using correlation")
        print("="*70)
        
        matcher_specific = TemplateMatcher(
            input_dir=times_file_dir,
            template_sdnum=0.9,  # Corresponds to a 0.9 correlation threshold
            metric='correlation'
        )
        matcher_specific.process_all_channels(channels=[257, 263])