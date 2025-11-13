"""
SPC Clustering Module for Spike Sorting Pipeline
Performs Super-Paramagnetic Clustering on extracted features.

Based on WaveClus SPC clustering implementation.

Features:
- Process all channels or specific channels
- Works with both *_spikes.mat and ch*_features_*.mat files
- Automatic file detection and feature extraction
- Progress tracking and detailed logging
- Updates files with cluster assignments
- Generates times_*.mat files in WaveClus format

Usage:
    from Clustering import SPCClustering
    
    # Initialize clustering
    clusterer = SPCClustering(input_dir='/path/to/features')
    
    # Cluster all channels
    clusterer.process_all_channels(channels='all')
    
    # Cluster specific channels
    clusterer.process_all_channels(channels=[257, 263])
"""

import os
import numpy as np
import glob
import re
import traceback
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try to import WaveClus SPC clustering
try:
    from pywaveclus.clustering import SPC_clustering
    WAVECLUS_AVAILABLE = True
except ImportError:
    WAVECLUS_AVAILABLE = False
    print("Warning: WaveClus not available. Using fallback clustering method.")


class SPCClustering:
    """
    SPC (Super-Paramagnetic Clustering) for spike sorting.
    
    Processes feature files or spike files with pre-extracted features
    and assigns cluster labels to each spike.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing feature files (ch*_features_*.mat) or spike files (*_spikes.mat)
    output_dir : str, optional
        Directory to save clustering results (default: same as input_dir, updates files in-place)
    times_output_dir : str, optional
        Directory to save times_*.mat files (default: auto-detected)
        - If input_dir ends with 'features': saves to parent directory (e.g., .../output/)
        - Otherwise: saves to input_dir/output/
    min_clus : int
        Minimum cluster size (default: 20)
    max_clus : int
        Maximum number of clusters (default: 20)
    temperature : float
        SPC temperature parameter (default: 0.201)
    knn : int
        Number of nearest neighbors (default: 11)
    generate_times_files : bool
        Whether to generate times_*.mat files (default: True)
    verbose : bool
        Print detailed progress messages (default: True)
    """
    
    def __init__(self, input_dir, output_dir=None, times_output_dir=None, min_clus=20, max_clus=20,
                 temperature=0.201, knn=11, generate_times_files=True, verbose=True):
        """Initialize SPC clustering."""
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.verbose = verbose
        self.generate_times_files = generate_times_files
        
        # Validate input directory exists
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # SPC parameters
        self.min_clus = min_clus
        self.max_clus = max_clus
        self.temperature = temperature
        self.knn = knn
        
        # Set times output directory
        if times_output_dir is not None:
            self.times_output_dir = times_output_dir
        else:
            # Default: if input_dir ends with 'features', save times to parent 'output' dir
            # Otherwise, create 'output' subdirectory in input_dir
            if self.input_dir.rstrip('/').endswith('features'):
                # /path/to/output/features -> /path/to/output
                self.times_output_dir = os.path.dirname(self.input_dir.rstrip('/'))
            else:
                # /path/to/data -> /path/to/data/output
                self.times_output_dir = os.path.join(self.input_dir, 'output')
        
        # Only create output directory if it's different from input and doesn't exist
        if self.output_dir != self.input_dir and not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except PermissionError:
                self._log(f"Warning: Cannot create output directory {self.output_dir}")
                self._log(f"         Using input directory instead: {self.input_dir}")
                self.output_dir = self.input_dir
        
        # Create times output directory if needed
        if self.generate_times_files and not os.path.exists(self.times_output_dir):
            try:
                os.makedirs(self.times_output_dir, exist_ok=True)
                self._log(f"Created times output directory: {self.times_output_dir}")
            except PermissionError:
                self._log(f"Warning: Cannot create times output directory {self.times_output_dir}")
                self._log(f"         Using feature directory instead: {self.output_dir}")
                self.times_output_dir = self.output_dir
    
    def _log(self, msg):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(msg)
    
    def _parse_channels(self, channels):
        """
        Parse channel specification and return list of channel numbers.
        
        Parameters:
        -----------
        channels : 'all', list, int, or None
            Channel selection specification
            
        Returns:
        --------
        list : Channel numbers to process
        """
        if isinstance(channels, str) and channels.lower() == 'all':
            # Look for both feature files and spike files
            feature_files = glob.glob(os.path.join(self.input_dir, 'ch*_features_*.mat'))
            spike_files = glob.glob(os.path.join(self.input_dir, '*_spikes.mat'))
            
            channel_nums = set()
            
            # Extract from feature files
            for f in feature_files:
                basename = os.path.basename(f)
                # ch257_features_gmm.mat -> 257
                match = re.search(r'ch(\d+)_features', basename)
                if match:
                    channel_nums.add(int(match.group(1)))
            
            # Extract from spike files
            for f in spike_files:
                basename = os.path.basename(f)
                # mLAMY08_raw_273_spikes.mat -> 273
                match = re.search(r'_(\d+)_spikes\.mat$', basename)
                if match:
                    channel_nums.add(int(match.group(1)))
            
            return sorted(list(channel_nums))
        
        elif isinstance(channels, int):
            return [channels]
        
        elif isinstance(channels, (list, tuple)):
            return sorted(list(set(channels)))
        
        elif channels is None:
            return self._parse_channels('all')
        
        else:
            self._log(f"Warning: Unknown channel type {type(channels)}, using all channels")
            return self._parse_channels('all')
    
    def _find_files_for_channel(self, channel_num):
        """
        Find feature file or spike file for a specific channel.
        
        Returns:
        --------
        tuple : (file_path, file_type) where file_type is 'feature' or 'spike'
        """
        # First look for feature files (preferred)
        feature_files = glob.glob(os.path.join(self.input_dir, f'ch{channel_num}_features_*.mat'))
        if feature_files:
            return feature_files[0], 'feature'
        
        # Then look for spike files
        spike_files = glob.glob(os.path.join(self.input_dir, f'*_{channel_num}_spikes.mat'))
        if spike_files:
            return spike_files[0], 'spike'
        
        return None, None
    
    def _spc_clustering_fallback(self, features):
        """
        Fallback SPC clustering implementation when WaveClus is not available.
        Uses a simplified approach based on connectivity and temperature.
        
        Parameters:
        -----------
        features : ndarray
            Feature matrix (n_spikes, n_features)
            
        Returns:
        --------
        labels : ndarray
            Cluster assignments (0 = noise/unassigned)
        metadata : dict
            Clustering metadata
        """
        n_spikes = features.shape[0]
        
        if n_spikes < self.min_clus:
            self._log(f"  Warning: Only {n_spikes} spikes, less than min_clus={self.min_clus}")
            return np.zeros(n_spikes, dtype=int), {'n_clusters': 0}
        
        # Compute pairwise distances
        self._log(f"  Computing pairwise distances...")
        distances = squareform(pdist(features, metric='euclidean'))
        
        # Find k-nearest neighbors
        self._log(f"  Finding {self.knn} nearest neighbors...")
        knn_indices = np.argsort(distances, axis=1)[:, 1:self.knn+1]  # Exclude self
        
        # Build connectivity matrix based on temperature threshold
        connectivity = np.zeros((n_spikes, n_spikes), dtype=bool)
        
        for i in range(n_spikes):
            for j in knn_indices[i]:
                dist = distances[i, j]
                # Temperature-based connection probability
                # Lower temperature = more selective connections
                threshold = self.temperature * np.median(distances[distances > 0])
                if dist < threshold:
                    connectivity[i, j] = True
                    connectivity[j, i] = True
        
        # Find connected components
        self._log(f"  Finding connected components...")
        n_components, labels = connected_components(connectivity, directed=False)
        
        # Filter small clusters (mark as noise with label 0)
        self._log(f"  Filtering clusters by size (min_clus={self.min_clus})...")
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Reassign labels: 0 for noise, 1+ for valid clusters
        new_labels = np.zeros_like(labels)
        cluster_id = 1
        
        for label, count in zip(unique_labels, counts):
            if count >= self.min_clus:
                new_labels[labels == label] = cluster_id
                cluster_id += 1
            else:
                new_labels[labels == label] = 0  # Mark as noise
        
        n_clusters = cluster_id - 1
        n_noise = np.sum(new_labels == 0)
        
        self._log(f"  Found {n_clusters} clusters, {n_noise} noise spikes")
        
        metadata = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': [np.sum(new_labels == i) for i in range(1, cluster_id)],
            'temperature': self.temperature,
            'knn': self.knn,
            'method': 'fallback_spc'
        }
        
        return new_labels, metadata
    
    def _spc_clustering_waveclus(self, features):
        """
        Use WaveClus SPC clustering implementation.
        
        Parameters:
        -----------
        features : ndarray
            Feature matrix (n_spikes, n_features)
            
        Returns:
        --------
        labels : ndarray
            Cluster assignments
        metadata : dict
            Clustering metadata
        """
        # WaveClus expects a dictionary of features per channel
        features_dict = {0: features}  # Use dummy channel 0
        
        self._log(f"  Running WaveClus SPC clustering...")
        labels_dict, metadata = SPC_clustering(features_dict)
        
        # Extract labels for our dummy channel
        labels = labels_dict[0]
        
        # Count clusters and noise
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels > 0])
        n_noise = np.sum(labels == 0)
        
        self._log(f"  Found {n_clusters} clusters, {n_noise} noise spikes")
        
        metadata['method'] = 'waveclus_spc'
        
        return labels, metadata
    
    def _cluster_features(self, features):
        """
        Perform SPC clustering on features.
        
        Parameters:
        -----------
        features : ndarray
            Feature matrix (n_spikes, n_features)
            
        Returns:
        --------
        labels : ndarray
            Cluster assignments (0 = noise)
        metadata : dict
            Clustering metadata
        """
        if WAVECLUS_AVAILABLE:
            try:
                return self._spc_clustering_waveclus(features)
            except Exception as e:
                self._log(f"  Warning: WaveClus clustering failed: {e}")
                self._log(f"  Falling back to simplified SPC implementation...")
                return self._spc_clustering_fallback(features)
        else:
            return self._spc_clustering_fallback(features)
    
    def _create_times_file(self, mat_data, channel_num, labels, file_path):
        """
        Create times_*.mat file with clustering results.
        
        Simplified version that creates a clean times file with clustering data.
        
        Parameters:
        -----------
        mat_data : dict
            Loaded mat file data
        channel_num : int
            Channel number
        labels : ndarray
            Cluster labels
        file_path : str
            Original file path
            
        Returns:
        --------
        bool : True if successful
        """
        try:
            # Extract spike times
            spike_times = mat_data.get('Spike_Time', mat_data.get('index', np.arange(len(labels)))).flatten()
            
            # Ensure spike_times matches label length
            if len(spike_times) != len(labels):
                spike_times = np.arange(len(labels))
            
            # Create cluster_class: (label, spike_time) pairs
            cluster_class = np.column_stack([labels.astype(float), spike_times.astype(float)])
            
            # Create times file name
            base_name = os.path.basename(file_path).replace('_features_gmm.mat', '').replace('_spikes.mat', '')
            times_filename = f"times_{base_name}.mat"
            times_path = os.path.join(self.times_output_dir, times_filename)
            
            # Create times data dictionary
            times_data = {
                'cluster_labels': labels.astype(np.uint32),
                'spike_times': spike_times.astype(np.float64),
                'cluster_class': cluster_class,
                'channel_id': np.array([channel_num], dtype=np.uint32),
                'n_clusters': np.array([np.max(labels)], dtype=np.uint32),
                'n_spikes': np.array([len(labels)], dtype=np.uint32),
                'temperature': np.array([self.temperature], dtype=np.float32),
                'min_cluster_size': np.array([self.min_clus], dtype=np.uint32)
            }
            
            # Add waveforms and features if available
            if 'spikes' in mat_data:
                times_data['spikes'] = mat_data['spikes']
            
            if 'spikes_features' in mat_data:
                times_data['features'] = mat_data['spikes_features']
            elif 'inspk' in mat_data:
                times_data['features'] = mat_data['inspk']
            
            # Save times file
            savemat(times_path, times_data, do_compression=True)
            
            self._log(f"  ✓ Times file created: {times_filename}")
            return True
            
        except Exception as e:
            self._log(f"  ✗ Failed to create times file: {e}")
            traceback.print_exc()
            return False
    
    def process_single_channel(self, channel_num):
        """
        Process a single channel's feature data.
        
        Parameters:
        -----------
        channel_num : int
            Channel number to process
            
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        file_path, file_type = self._find_files_for_channel(channel_num)
        
        if file_path is None:
            self._log(f"Warning: No file found for channel {channel_num}")
            return False
        
        try:
            self._log(f"\n{'='*60}")
            self._log(f"Channel {channel_num} - SPC Clustering")
            self._log(f"File: {os.path.basename(file_path)} ({file_type} file)")
            self._log(f"{'='*60}")
            
            # Load data
            mat_data = loadmat(file_path)
            
            # Extract features based on file type
            if file_type == 'feature':
                # Feature file: ch257_features_gmm.mat
                if 'spikes_features' in mat_data:
                    features = mat_data['spikes_features'].astype(np.float64)
                elif 'features' in mat_data:
                    features = mat_data['features'].astype(np.float64)
                else:
                    self._log("Error: No 'spikes_features' or 'features' field found")
                    return False
            else:
                # Spike file: mLAMY08_raw_273_spikes.mat
                if 'features' in mat_data:
                    features = mat_data['features'].astype(np.float64)
                elif 'inspk' in mat_data:
                    features = mat_data['inspk'].astype(np.float64)
                else:
                    self._log("Error: No 'features' or 'inspk' field found")
                    return False
            
            n_spikes, n_features = features.shape
            self._log(f"Loaded {n_spikes} spikes with {n_features} features")
            
            if n_spikes == 0:
                self._log("No spikes to cluster. Skipping.")
                return False
            
            # Perform clustering
            labels, metadata = self._cluster_features(features)
            
            # Create cluster_class mapping (label, spike_index pairs)
            # This matches WaveClus output format
            if 'index' in mat_data:
                spike_indices = mat_data['index'].flatten()
            elif 'Spike_Time' in mat_data:
                spike_indices = mat_data['Spike_Time'].flatten()
            else:
                # Use sequential indices if no timing info
                spike_indices = np.arange(n_spikes)
            
            # cluster_class: [(label, spike_time), ...]
            cluster_class = np.column_stack([labels, spike_indices])
            
            # Update mat_data with clustering results
            mat_data['cluster_labels'] = labels
            mat_data['cluster_class'] = cluster_class
            mat_data['n_clusters'] = metadata['n_clusters']
            mat_data['clustering_metadata'] = metadata
            
            # Determine output path
            if self.output_dir != self.input_dir:
                # Save to different directory
                output_filename = os.path.basename(file_path)
                output_path = os.path.join(self.output_dir, output_filename)
            else:
                # Update file in-place
                output_path = file_path
            
            # Save results
            savemat(output_path, mat_data, do_compression=True)
            
            self._log(f"Saved clustering results to: {os.path.basename(output_path)}")
            self._log(f"Clusters: {metadata['n_clusters']}")
            
            # Print cluster summary
            unique_labels = np.unique(labels)
            for label in unique_labels:
                count = np.sum(labels == label)
                if label == 0:
                    self._log(f"  Noise: {count} spikes")
                else:
                    self._log(f"  Cluster {label}: {count} spikes")
            
            # Generate times file if requested
            if self.generate_times_files:
                self._log(f"Generating times file...")
                self._create_times_file(mat_data, channel_num, labels, file_path)
            
            return True
            
        except Exception as e:
            self._log(f"ERROR processing channel {channel_num}: {e}")
            traceback.print_exc()
            return False
    
    def process_all_channels(self, channels='all'):
        """
        Process spike data for specified channels.
        
        Parameters:
        -----------
        channels : 'all', list, int, or None
            Channels to process:
            - 'all': Process all channels (default)
            - [257, 263]: Process specific channels
            - 290: Process single channel
            - None: Process all channels
        
        Examples:
        ---------
        # Process all channels
        clusterer = SPCClustering(input_dir)
        clusterer.process_all_channels(channels='all')
        
        # Process specific channels
        clusterer.process_all_channels(channels=[257, 263, 290])
        
        # Process range
        clusterer.process_all_channels(channels=list(range(290, 301)))
        """
        # Parse channels
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            self._log(f"No channels specified or found in {self.input_dir}")
            return
        
        # Summary header
        self._log(f"\n{'#'*70}")
        self._log(f"SPC CLUSTERING PIPELINE")
        self._log(f"{'#'*70}")
        self._log(f"Input directory: {self.input_dir}")
        self._log(f"Output directory: {self.output_dir}")
        if self.generate_times_files:
            self._log(f"Times output directory: {self.times_output_dir}")
        self._log(f"Channels: {channel_nums}")
        self._log(f"Total channels: {len(channel_nums)}")
        self._log(f"Parameters:")
        self._log(f"  Min cluster size: {self.min_clus}")
        self._log(f"  Max clusters: {self.max_clus}")
        self._log(f"  Temperature: {self.temperature}")
        self._log(f"  K-nearest neighbors: {self.knn}")
        self._log(f"  Generate times files: {self.generate_times_files}")
        self._log(f"  Method: {'WaveClus SPC' if WAVECLUS_AVAILABLE else 'Fallback SPC'}")
        self._log(f"{'#'*70}\n")
        
        # Process each channel
        successful = 0
        failed = 0
        
        for i, ch_num in enumerate(channel_nums, 1):
            self._log(f"[{i}/{len(channel_nums)}] Processing channel {ch_num}...")
            
            if self.process_single_channel(ch_num):
                successful += 1
            else:
                failed += 1
        
        # Summary footer
        self._log(f"\n{'#'*70}")
        self._log(f"CLUSTERING COMPLETE")
        self._log(f"{'#'*70}")
        self._log(f"Successful: {successful}/{len(channel_nums)}")
        self._log(f"Failed: {failed}/{len(channel_nums)}")
        self._log(f"Output directory: {self.output_dir}")
        if self.generate_times_files:
            self._log(f"Times files saved to: {self.times_output_dir}")
        self._log(f"{'#'*70}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of SPCClustering with times file generation
    """
    
    # Define directories
    feature_dir = '/media/sEEG_DATA/Tests/Matlab sorting pipeline/Tapasi/MCWs_Pipeline/full_processing_pipeline/output/features'
    spike_dir = '/media/sEEG_DATA/Tests/Matlab sorting pipeline/Tapasi/MCWs_Pipeline/full_processing_pipeline/output'
    
    # ========================================
    # EXAMPLE 1: Cluster feature files with times generation
    # ========================================
    print("\n" + "="*70)
    print("EXAMPLE 1: Cluster feature files and generate times files")
    print("="*70)
    
    clusterer = SPCClustering(
        input_dir=feature_dir,
        min_clus=20,
        max_clus=20,
        temperature=0.201,
        knn=11,
        generate_times_files=True,  # Enable times file generation
        verbose=True
    )
    
    # Process all channels
    clusterer.process_all_channels(channels='all')
    
    # ========================================
    # EXAMPLE 2: Cluster specific channels
    # ========================================
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Cluster specific channels with times files")
    print("="*70)
    
    clusterer = SPCClustering(
        input_dir=feature_dir,
        min_clus=15,
        generate_times_files=True,
        verbose=True
    )
    
    # Process specific channels
    clusterer.process_all_channels(channels=[257, 263])
    """
    
    # ========================================
    # EXAMPLE 3: Without times file generation
    # ========================================
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Cluster without generating times files")
    print("="*70)
    
    clusterer = SPCClustering(
        input_dir=feature_dir,
        min_clus=20,
        generate_times_files=False,  # Disable times file generation
        verbose=True
    )
    
    clusterer.process_all_channels(channels='all')
    """