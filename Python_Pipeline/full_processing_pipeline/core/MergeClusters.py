"""
Cluster Merging and Reporting Module

This module provides a class-based wrapper, ClusterMerger, to merge 
specified clusters from 'times_*.mat' files, renumber them contiguously,
and generate a new set of quality reports and data files.

Usage:
    from core.MergeClusters import ClusterMerger
    
    # Define merge groups: [[target, source1, source2], [target, source1], ...]
    merge_groups = [[1, 3, 5], [2, 7]] 

    merger = ClusterMerger(
        input_dir='/path/to/times_files',
        output_dir='/path/to/merged_times_files',
        report_dir='/path/to/merged_reports'
    )
    
    # Merge and report for all channels
    merger.process_all_channels(channels='all', merge_groups=merge_groups)
    
    # Merge and report for specific channels
    merger.process_all_channels(channels=[257, 263], merge_groups=merge_groups)
"""

import os
import re
import numpy as np
import glob
import traceback
from scipy.io import loadmat, savemat
import copy
from typing import Optional, Tuple, Union, Literal, List, Dict, Any
import warnings
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# --- ASSUMPTIONS ---
# 1. Assumes a 'PlotReport.py' file exists in 'core/' containing 'make_cluster_report'
# 2. Assumes 'make_cluster_report' returns: figs, df_metrics, SS
try:
    from core.PlotReport import make_cluster_report
except ImportError:
    print("Warning: Could not import 'make_cluster_report' from 'core.PlotReport'.")
    print("The ClusterMerger class will be defined, but reporting will fail.")
    print("Please ensure 'core/PlotReport.py' exists and is correctly structured.")
    # Define a placeholder function to allow the class to be defined
    def make_cluster_report(data, **kwargs):
        print("--- FAKE make_cluster_report ---")
        return [], pd.DataFrame(), np.array([])
# ---

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# MAIN CLUSTER MERGER CLASS
# ============================================================================

class ClusterMerger:
    """
    Class to find 'times_*.mat' files, merge specified clusters,
    renumber them, save new .mat files, and generate new reports.
    """

    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 report_dir: str,
                 calc_metrics: bool = True,
                 verbose: bool = True, 
                 **report_kwargs):
        """
        Initialize the ClusterMerger.
        
        Args:
            input_dir (str): Directory containing 'times_*.mat' files.
            output_dir (str): Directory to save *new* '..._merged.mat' files.
            report_dir (str): Directory to save *new* PDF reports.
            calc_metrics (bool): Whether to calculate metrics for the report.
            verbose (bool): Print progress messages.
            **report_kwargs: Additional args for make_cluster_report 
                              (e.g., clusters_per_page=6, refractory_ms=3.0)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.verbose = verbose
        self.calc_metrics = calc_metrics
        self.report_kwargs = report_kwargs
        
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
            
        # Create output directories if they don't exist
        for dir_path in [self.output_dir, self.report_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                self._log(f"Created directory: {dir_path}")

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

    def _find_file_for_channel(self, channel_num: int) -> Tuple[Optional[str], Optional[str]]:
        """Find the times file and its base name for a specific channel."""
        times_files = glob.glob(os.path.join(self.input_dir, f'times_*_{channel_num}.mat'))
        if times_files:
            # e.g., times_mLTP01_raw_257.mat -> times_mLTP01_raw_257
            base_name = os.path.basename(times_files[0]).replace('.mat', '')
            return times_files[0], base_name
            
        times_files_ch = glob.glob(os.path.join(self.input_dir, f'times_ch{channel_num}.mat'))
        if times_files_ch:
            # e.g., times_ch328.mat -> times_ch328
            base_name = os.path.basename(times_files_ch[0]).replace('.mat', '')
            return times_files_ch[0], base_name
            
        return None, None

    def _merge_data(self, data: Dict[str, Any], merge_groups: List[List[int]], channel_num: int) -> Dict[str, Any]:
        """
        Applies merging and renumbering logic to a single channel's data.
        This is the core logic from your merge_and_report function.
        """
        # Deep copy the data to avoid modifying the original
        new_data = copy.deepcopy(data)
        
        # Extract cluster_class
        cluster_class = new_data['cluster_class'].copy()
        cluster_ids = cluster_class[:, 0].astype(int)
        
        self._log("\n=� Original cluster distribution:")
        unique_original = np.unique(cluster_ids)
        for cid in unique_original:
            count = np.sum(cluster_ids == cid)
            self._log(f"   Cluster {cid}: {count:5d} spikes ({count/len(cluster_ids)*100:5.1f}%)")
        
        # Apply merges
        if merge_groups:
            self._log("\n= Applying merges...")
            for merge_group in merge_groups:
                if len(merge_group) < 2:
                    self._log(f"   Skipping {merge_group} (need at least 2 clusters to merge)")
                    continue
                
                target_cluster = merge_group[0]
                source_clusters = merge_group[1:]
                
                self._log(f"\n   Merging {merge_group} � {target_cluster}")
                
                total_merged = 0
                for source_cluster in source_clusters:
                    mask = (cluster_ids == source_cluster)
                    n_spikes = np.sum(mask)
                    if n_spikes > 0:
                        cluster_ids[mask] = target_cluster
                        total_merged += n_spikes
                        self._log(f"     {n_spikes:5d} spikes from cluster {source_cluster} � {target_cluster}")
                    else:
                        self._log(f"     Cluster {source_cluster} not found (0 spikes)")
                
                self._log(f"   Total merged into cluster {target_cluster}: {total_merged} spikes")
        else:
            self._log("\n   No merges specified.")
        
        # Renumber clusters to be contiguous from 0 to N
        self._log("\n= Renumbering clusters to be contiguous (0 to N)...")
        unique_clusters = np.sort(np.unique(cluster_ids))
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        
        self._log("   Cluster mapping (old � new):")
        for old_id, new_id in cluster_mapping.items():
            old_count = np.sum(cluster_ids == old_id)
            self._log(f"     {old_id:3d} � {new_id:3d}   ({old_count:5d} spikes)")
        
        # Apply renumbering
        new_cluster_ids = np.array([cluster_mapping[cid] for cid in cluster_ids])
        cluster_class[:, 0] = new_cluster_ids
        
        # Update the data
        new_data['cluster_class'] = cluster_class
        
        self._log("\n= Final cluster distribution:")
        unique_merged = np.unique(new_cluster_ids)
        for cid in unique_merged:
            count = np.sum(new_cluster_ids == cid)
            self._log(f"   Cluster {cid}: {count:5d} spikes ({count/len(new_cluster_ids)*100:5.1f}%)")
        
        self._log(f"\n Cluster count: {len(unique_original)} � {len(unique_merged)}")
        
        return new_data


    def process_single_channel(self, channel_num: int, merge_groups: List[List[int]]) -> bool:
        """
        Run merging and reporting on a single channel's times file.
        """
        file_path, base_name = self._find_file_for_channel(channel_num)
        
        if file_path is None:
            self._log(f"Warning: No 'times_*_{channel_num}.mat' file found. Skipping.")
            return False
            
        try:
            self._log(f"\n{'='*70}")
            self._log(f"Channel {channel_num} - Cluster Merging")
            self._log(f"File: {os.path.basename(file_path)}")
            self._log(f"{'='*70}")
            
            mat_data = loadmat(file_path)
            
            # --- 1. Perform Merge and Renumbering ---
            new_data = self._merge_data(mat_data, merge_groups, channel_num)
            
            # --- 2. Save new .mat file ---
            output_mat_filename = f"{base_name}_merged.mat"
            output_mat_path = os.path.join(self.output_dir, output_mat_filename)
            savemat(output_mat_path, new_data, do_compression=True)
            self._log(f"\nSaved merged data to: {output_mat_path}")
            
            # --- 3. Generate and Save Report ---
            self._log("\n=� Generating cluster report for merged data...")
            figs, df_metrics, SS = make_cluster_report(
                new_data, 
                calc_metrics=self.calc_metrics, 
                **self.report_kwargs
            )
            
            if not figs:
                self._log("   No figures generated by report.")
            else:
                output_pdf_filename = f"{base_name}_merged_report.pdf"
                output_pdf_path = os.path.join(self.report_dir, output_pdf_filename)
                
                with PdfPages(output_pdf_path) as pdf:
                    for fig in figs:
                        pdf.savefig(fig)
                        plt.close(fig) # Close figure to save memory
                self._log(f"Saved merged report to: {output_pdf_path}")

            return True

        except Exception as e:
            self._log(f"ERROR processing channel {channel_num}: {e}")
            traceback.print_exc()
            return False

    def process_all_channels(self, channels: Union[str, int, list] = 'all', 
                             merge_groups: Optional[List[List[int]]] = None):
        """
        Process spike data for all specified channels.
        
        Args:
            channels: 'all', a single channel int, or a list of channel ints.
            merge_groups: List of lists, e.g., [[3,4,5], [2,7]]. 
                          If None, will just renumber.
        """
        if merge_groups is None:
            merge_groups = []
            
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            self._log(f"No channels specified or found in {self.input_dir}")
            return
        
        self._log(f"\n{'#'*70}")
        self._log(f"CLUSTER MERGING & REPORTING PIPELINE")
        self._log(f"{'#'*70}")
        self._log(f"Input directory: {self.input_dir}")
        self._log(f"Output MAT directory: {self.output_dir}")
        self._log(f"Output PDF directory: {self.report_dir}")
        self._log(f"Total channels to process: {len(channel_nums)}")
        self._log(f"Merge Groups: {merge_groups if merge_groups else 'None'}")
        self._log(f"{'#'*70}\n")
        
        successful = 0
        failed = 0
        
        for i, ch_num in enumerate(channel_nums, 1):
            if self.process_single_channel(ch_num, merge_groups):
                successful += 1
            else:
                failed += 1
                
        self._log(f"\n{'#'*70}")
        self._log(f"MERGE & REPORT COMPLETE")
        self._log(f"{'#'*70}")
        self._log(f"Successful: {successful}/{len(channel_nums)}")
        self._log(f"Failed: {failed}/{len(channel_nums)}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Report directory: {self.report_dir}")
        self._log(f"{'#'*70}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of the ClusterMerger class.
    """
    
    # Define the directory where your 'times_*.mat' files are located
    base_dir = os.path.dirname(__file__) 
    times_file_dir = os.path.join(base_dir, 'output') 
    
    # Define output directories
    merged_mat_dir = os.path.join(base_dir, 'output_merged_mat')
    merged_report_dir = os.path.join(base_dir, 'output_merged_reports')
    
    if not os.path.exists(times_file_dir):
        print(f"Example input directory not found: {times_file_dir}")
        print("Please run clustering first to generate 'times_*.mat' files.")
    else:
        # ========================================
        # EXAMPLE 1: Merge specific channels
        # ========================================
        print("\n" + "="*70)
        print("EXAMPLE 1: Merge clusters [1, 3] and [2, 5] for channel 328")
        print("="*70)
        
        # Define the merge rules
        merge_rules = [
            [1, 3], # Merge cluster 3 into cluster 1
            [2, 5]  # Merge cluster 5 into cluster 2
        ]
        
        merger = ClusterMerger(
            input_dir=times_file_dir,
            output_dir=merged_mat_dir,
            report_dir=merged_report_dir,
            calc_metrics=True,
            clusters_per_page=6 # Example kwarg for make_cluster_report
        )
        
        merger.process_all_channels(channels=[328], merge_groups=merge_rules)
        
        
        # ========================================
        # EXAMPLE 2: Renumber all channels without merging
        # ========================================
        print("\n" + "="*70)
        print("EXAMPLE 2: Renumber all channels without merging")
        print("="*70)
        
        merger_renumber = ClusterMerger(
            input_dir=times_file_dir,
            output_dir=merged_mat_dir,
            report_dir=merged_report_dir,
            calc_metrics=False # Skip metrics, just renumber and plot
        )
        
        # Pass merge_groups=None (or empty list) to only renumber
        merger_renumber.process_all_channels(channels='all', merge_groups=None)