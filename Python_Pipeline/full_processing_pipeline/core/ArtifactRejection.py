import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks, peak_prominences
from multiprocessing import Pool, cpu_count
import glob
import traceback


class ArtifactRejection():
    """
    Artifact Rejection class for spike data quality control.
    Implements the workflow from the uploaded diagram.
    """
    
    def __init__(self, input_dir, output_dir=None, backup_dir=None):
        """
        Initialize artifact rejection.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing *_spikes.mat files
        output_dir : str, optional
            Directory to save processed files (default: same as input_dir)
        backup_dir : str, optional
            Directory to backup original files (default: input_dir/backup)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.backup_dir = backup_dir if backup_dir else os.path.join(input_dir, 'backup')
        
        # Default thresholds (can be modified)
        self.prom_threshold = 2.0
        self.width_upper_threshold = 15.0
        self.width_lower_threshold = 3.0
        self.low_amp_percentile = 5.0
        self.other_prom_min_ratio = 0.01  # 1% of peak amplitude
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def set_thresholds(self, prom_ratio=None, width_upper=None, width_lower=None, 
                       low_amp_percentile=None, other_prom_min_ratio=None):
        """Set artifact rejection thresholds."""
        if prom_ratio is not None:
            self.prom_threshold = prom_ratio
        if width_upper is not None:
            self.width_upper_threshold = width_upper
        if width_lower is not None:
            self.width_lower_threshold = width_lower
        if low_amp_percentile is not None:
            self.low_amp_percentile = low_amp_percentile
        if other_prom_min_ratio is not None:
            self.other_prom_min_ratio = other_prom_min_ratio
    
    def _parse_channels(self, channels):
        """
        Parse channel specification and return list of channel numbers to process.
        
        Parameters:
        -----------
        channels : 'all', list, int, or None
            Channel selection specification:
            - 'all': Process all channels found in input directory
            - [257, 263]: Process specific channels
            - 257: Process single channel
            - list(range(290, 301)): Process range of channels
            
        Returns:
        --------
        list : Channel numbers to process
        """
        if isinstance(channels, str) and channels.lower() == 'all':
            # Get all spike files and extract channel numbers
            spike_files = glob.glob(os.path.join(self.input_dir, '*_spikes.mat'))
            channel_nums = []
            for f in spike_files:
                # Extract channel number from filename (format: label_channelnum_spikes.mat)
                basename = os.path.basename(f)
                parts = basename.replace('_spikes.mat', '').split('_')
                if len(parts) >= 2:
                    try:
                        ch_num = int(parts[-1])
                        channel_nums.append(ch_num)
                    except ValueError:
                        continue
            return sorted(list(set(channel_nums)))
        
        elif isinstance(channels, int):
            # Single channel
            return [channels]
        
        elif isinstance(channels, (list, tuple)):
            # List of channels
            return sorted(list(set(channels)))
        
        elif channels is None:
            # None = all channels
            return self._parse_channels('all')
        
        else:
            print(f"Warning: Unknown channel specification type {type(channels)}. Using all channels.")
            return self._parse_channels('all')
    
    def _find_spike_file(self, channel_num):
        """
        Find spike file for a specific channel.
        
        Parameters:
        -----------
        channel_num : int
            Channel number to find
            
        Returns:
        --------
        str or None : Path to spike file, or None if not found
        """
        spike_files = glob.glob(os.path.join(self.input_dir, '*_spikes.mat'))
        
        for f in spike_files:
            basename = os.path.basename(f)
            # Check if filename ends with {channel_num}_spikes.mat
            if basename.endswith(f'{channel_num}_spikes.mat'):
                return f
        
        return None
    
    def process_all_channels(self, channels='all', parallel=True):
        """
        Process spike files for specified channels.
        
        Parameters:
        -----------
        channels : 'all', list, int, optional
            Channels to process:
            - 'all': Process all channels (default)
            - [257, 263]: Process specific channels
            - 257: Process single channel
            - list(range(290, 301)): Process range 290-300 (inclusive)
        parallel : bool
            Use parallel processing for spike metrics calculation
            
        Examples:
        ---------
        # Process all channels
        artifact_rej.process_all_channels(channels='all', parallel=True)
        
        # Process specific channels
        artifact_rej.process_all_channels(channels=[257, 263, 264], parallel=True)
        
        # Process single channel
        artifact_rej.process_all_channels(channels=290, parallel=True)
        
        # Process range of channels
        artifact_rej.process_all_channels(channels=list(range(290, 301)), parallel=True)
        """
        # Parse channel specification
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            print(f"No channels specified or found in {self.input_dir}")
            return
        
        print(f"Selected channels: {channel_nums}")
        print(f"Found {len(channel_nums)} channels to process")
        print(f"Thresholds: Prom>{self.prom_threshold}, Width>{self.width_lower_threshold} & <{self.width_upper_threshold}")
        print(f"Low Amp Percentile: {self.low_amp_percentile}%, Other Prom Min: {self.other_prom_min_ratio*100}%\n")
        
        # Process each channel
        for ch_num in channel_nums:
            spike_file = self._find_spike_file(ch_num)
            
            if spike_file is None:
                print(f"\nWarning: No spike file found for channel {ch_num}")
                continue
            
            filename = os.path.basename(spike_file)
            print(f"\n{'='*60}")
            print(f"Processing channel {ch_num}: {filename}")
            print(f"{'='*60}")
            
            try:
                self.process_single_channel(spike_file, parallel=parallel)
            except Exception as e:
                print(f"ERROR processing channel {ch_num} ({filename}): {e}")
                traceback.print_exc()
                continue
    
    def process_single_channel(self, spike_file, parallel=True):
        """
        Process a single channel's spike file.
        
        Parameters:
        -----------
        spike_file : str
            Path to *_spikes.mat file
        parallel : bool
            Use parallel processing for metrics calculation
        """
        # Load spike file
        mat_data = loadmat(spike_file)
        
        # Extract data - prioritize spikes_all if it exists
        if 'spikes_all' in mat_data:
            spikes_all = mat_data['spikes_all'].astype(np.float64)
            index_all = mat_data['index_all'].flatten() if 'index_all' in mat_data else np.arange(spikes_all.shape[0])
            data_source = 'spikes_all'
        else:
            # Fallback to spikes if spikes_all doesn't exist
            spikes_all = mat_data['spikes'].astype(np.float64)
            index_all = mat_data['index'].flatten()
            data_source = 'spikes'
        
        num_spikes = spikes_all.shape[0]
        print(f"Loaded {num_spikes} spikes (using {data_source})")
        
        if num_spikes == 0:
            print("No spikes to process. Skipping.")
            return
        
        # Check for bundle collision field - must match spikes_all size
        if 'possible_artifact' in mat_data:
            bundle_coll = mat_data['possible_artifact'].flatten().astype(bool)
            bundle_field = 'possible_artifact'
        elif 'mask_bundle_coll' in mat_data:
            bundle_coll = mat_data['mask_bundle_coll'].flatten().astype(bool)
            bundle_field = 'mask_bundle_coll'
        else:
            print("Warning: 'possible_artifact' or 'mask_bundle_coll' field not found. Creating empty array.")
            bundle_coll = np.zeros(num_spikes, dtype=bool)
            bundle_field = 'created'
        
        # Ensure bundle_coll matches spikes_all size
        if bundle_coll.shape[0] != num_spikes:
            print(f"Warning: {bundle_field} size ({bundle_coll.shape[0]}) != spikes_all size ({num_spikes})")
            print(f"Resizing {bundle_field} to match spikes_all...")
            # Pad or truncate to match
            if bundle_coll.shape[0] < num_spikes:
                bundle_coll = np.pad(bundle_coll, (0, num_spikes - bundle_coll.shape[0]), 
                                    mode='constant', constant_values=0)
            else:
                bundle_coll = bundle_coll[:num_spikes]
            print(f"Resized {bundle_field} to {bundle_coll.shape[0]}")
        
        # Verify sizes match before proceeding
        assert bundle_coll.shape[0] == num_spikes, \
            f"Size mismatch after processing: {bundle_field}={bundle_coll.shape[0]}, spikes_all={num_spikes}"
        
        # Calculate spike metrics
        print(f"Calculating metrics for {num_spikes} spikes...")
        metrics = self.calculate_all_metrics(spikes_all, parallel=parallel)
        
        # Apply artifact rejection logic
        print("Applying artifact rejection logic...")
        mask_quarantined, mask_preserved = self.apply_artifact_rejection(
            metrics, bundle_coll, spikes_all
        )
        
        # Create final filtered arrays
        mask_keep = ~mask_quarantined & ~bundle_coll
        spikes_filtered = spikes_all[mask_keep]
        index_filtered = index_all[mask_keep]
        
        num_preserved = np.sum(mask_keep)
        num_quarantined = np.sum(mask_quarantined)
        num_bundle_coll = np.sum(bundle_coll)
        num_both = np.sum(mask_quarantined & bundle_coll)
        
        print(f"\nResults:")
        print(f"  Total spikes: {num_spikes}")
        print(f"  Preserved: {num_preserved} ({num_preserved/num_spikes*100:.1f}%)")
        print(f"  Quarantined: {num_quarantined} ({num_quarantined/num_spikes*100:.1f}%)")
        print(f"  Bundle collision: {num_bundle_coll} ({num_bundle_coll/num_spikes*100:.1f}%)")
        print(f"  Both quarantined & bundle: {num_both}")
        
        # Update mat_data with new fields
        mat_data['spikes_all'] = spikes_all
        mat_data['index_all'] = index_all
        mat_data['spikes'] = spikes_filtered
        mat_data['index'] = index_filtered
        mat_data['mask_quarantined'] = mask_quarantined.astype(int)
        mat_data['bundle_coll'] = bundle_coll.astype(int)
        
        # Add quarantine properties
        mat_data['quarantine_properties'] = self.create_metrics_struct(metrics)
        
        # Save updated file
        output_path = os.path.join(self.output_dir, os.path.basename(spike_file))
        savemat(output_path, mat_data, do_compression=True)
        print(f"Saved updated file to: {output_path}")
    
    def calculate_all_metrics(self, spikes, parallel=True):
        """
        Calculate metrics for all spikes.
        
        Parameters:
        -----------
        spikes : ndarray
            Spike waveforms (num_spikes x num_samples)
        parallel : bool
            Use parallel processing
            
        Returns:
        --------
        dict : Dictionary containing all metrics
        """
        num_spikes = spikes.shape[0]
        
        if parallel and num_spikes > 100:
            print(f"  Using {cpu_count()} CPU cores for parallel processing...")
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(self.calculate_spike_metrics, 
                                  [spikes[i, :] for i in range(num_spikes)])
        else:
            results = [self.calculate_spike_metrics(spikes[i, :]) for i in range(num_spikes)]
        
        # Convert results to arrays
        results = np.array(results)
        
        metrics = {
            'prominence_ratio': results[:, 0],
            'num_peaks': results[:, 1],
            'prominence_sample_20': results[:, 2],
            'other_prominence': results[:, 3],
            'width': results[:, 4],
            'peak_sample_20': results[:, 5],
            'other_peak_loc': results[:, 6],
            'peak_pos_max': results[:, 7],
            'prominence_pos_max': results[:, 8],
            'ratio_20_max_pr': results[:, 9],
            'ratio_20_max_amp': results[:, 10]
        }
        
        return metrics
    
    def calculate_spike_metrics(self, waveform):
        """
        Calculate detailed metrics for a single spike waveform.
        
        Parameters:
        -----------
        waveform : ndarray
            Single spike waveform
            
        Returns:
        --------
        tuple : 11 metrics values
        """
        try:
            # Peak Detection
            inverted_waveform = -waveform
            all_peaks, _ = find_peaks(inverted_waveform)
            num_peaks = len(all_peaks)

            # Initialize all return values
            ratio = np.nan
            prominence_sample_20 = np.nan
            other_prominence = np.nan
            width = np.nan
            peak_sample_20 = np.nan
            other_peak_loc = np.nan
            peak_pos_max = np.nan
            prominence_pos_max = np.nan
            ratio_20_max_pr = np.nan
            ratio_20_max_amp = np.nan

            if num_peaks == 0:
                return (np.nan, 0, np.nan, np.nan, np.nan, np.nan, np.nan, 
                       np.nan, np.nan, np.nan, np.nan)
            
            prominences, left_bases, right_bases = peak_prominences(inverted_waveform, all_peaks)
            
            # Calculate positive peaks and prominences
            pos_peaks, _ = find_peaks(waveform)
            if len(pos_peaks) > 0:
                pos_prominences, _, _ = peak_prominences(waveform, pos_peaks)
                pos_peak_amplitudes = waveform[pos_peaks]
                max_pos_amp_idx = np.argmax(pos_peak_amplitudes)
                peak_pos_max = pos_peak_amplitudes[max_pos_amp_idx]
                prominence_pos_max = pos_prominences[max_pos_amp_idx]
            
            TARGET_PEAK_INDEX = 19  # Sample 20 (0-indexed)
            
            # Check if a peak exists at the target index
            target_peak_mask = (all_peaks == TARGET_PEAK_INDEX)
            
            if np.any(target_peak_mask):
                local_peak_idx = np.where(target_peak_mask)[0][0]
                target_prominence = prominences[local_peak_idx]
                prominence_sample_20 = target_prominence
                peak_sample_20 = waveform[TARGET_PEAK_INDEX]

                other_prominences_arr = prominences[~target_peak_mask]
                other_peaks_arr = all_peaks[~target_peak_mask]
                
                if len(other_prominences_arr) > 0:
                    max_prom_idx_local = np.argmax(other_prominences_arr)
                    max_other_prominence = other_prominences_arr[max_prom_idx_local]
                    other_prominence = max_other_prominence
                    other_peak_loc = other_peaks_arr[max_prom_idx_local] + 1
                    
                    if np.isnan(max_other_prominence):
                        ratio = np.nan
                    elif max_other_prominence > 0:
                        ratio = target_prominence / max_other_prominence
                    else:
                        ratio = np.inf
                else:
                    ratio = np.nan
            else:
                ratio = np.nan
                if num_peaks > 0:
                    max_prom_idx_local = np.argmax(prominences)
                    other_prominence = prominences[max_prom_idx_local]
                    other_peak_loc = all_peaks[max_prom_idx_local] + 1

            # Calculate Width
            if np.any(target_peak_mask):
                main_peak_idx = TARGET_PEAK_INDEX
                peak_voltage = waveform[main_peak_idx]
                baseline = np.mean(np.concatenate((waveform[:5], waveform[-5:])))
                half_amplitude = baseline + (peak_voltage - baseline) / 2
                x_indices = np.arange(len(waveform))

                left_candidates = np.where(waveform[:main_peak_idx+1] > half_amplitude)[0]
                if len(left_candidates) > 0:
                    cross_idx = left_candidates[-1]
                    if cross_idx < main_peak_idx:
                        y_vals = waveform[cross_idx : cross_idx+2]
                        x_vals = x_indices[cross_idx : cross_idx+2]
                        left_idx = np.interp(half_amplitude, [y_vals[1], y_vals[0]], 
                                            [x_vals[1], x_vals[0]])
                    else:
                        left_idx = np.nan
                else:
                    left_idx = np.nan

                right_candidates = np.where(waveform[main_peak_idx:] > half_amplitude)[0]
                if len(right_candidates) > 0:
                    cross_idx_local = right_candidates[0]
                    cross_idx_global = main_peak_idx + cross_idx_local
                    if cross_idx_global > 0:
                        y_vals = waveform[cross_idx_global-1 : cross_idx_global+1]
                        x_vals = x_indices[cross_idx_global-1 : cross_idx_global+1]
                        right_idx = np.interp(half_amplitude, y_vals, x_vals)
                    else:
                        right_idx = np.nan
                else:
                    right_idx = np.nan
                    
                if not np.isnan(left_idx) and not np.isnan(right_idx):
                    width = right_idx - left_idx
            
            # Calculate ratios
            if not np.isnan(prominence_sample_20) and not np.isnan(prominence_pos_max):
                if prominence_pos_max > 0:
                    ratio_20_max_pr = prominence_sample_20 / prominence_pos_max
                elif prominence_pos_max == 0:
                    ratio_20_max_pr = np.inf
            
            if not np.isnan(peak_sample_20) and not np.isnan(peak_pos_max):
                if peak_pos_max != 0:
                    ratio_20_max_amp = peak_sample_20 / peak_pos_max
                elif peak_pos_max == 0:
                    if peak_sample_20 == 0:
                        ratio_20_max_amp = np.nan
                    else:
                        ratio_20_max_amp = np.inf * np.sign(peak_sample_20)
            
            return (ratio, num_peaks, prominence_sample_20, other_prominence, width, 
                   peak_sample_20, other_peak_loc, peak_pos_max, prominence_pos_max, 
                   ratio_20_max_pr, ratio_20_max_amp)

        except Exception:
            traceback.print_exc()
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                   np.nan, np.nan, np.nan, np.nan)
    
    def apply_artifact_rejection(self, metrics, bundle_coll, spikes):
        """
        Apply artifact rejection logic based on the workflow diagram.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing all spike metrics
        bundle_coll : ndarray (bool)
            Bundle collision flags
        spikes : ndarray
            Spike waveforms (for amplitude threshold)
            
        Returns:
        --------
        mask_quarantined : ndarray (bool)
            True for quarantined spikes
        mask_preserved : ndarray (bool)
            True for preserved spikes
        """
        num_spikes = len(metrics['prominence_ratio'])
        
        # Extract metrics
        prominence_ratio = metrics['prominence_ratio']
        num_peaks = metrics['num_peaks']
        other_prominence = metrics['other_prominence']
        width = metrics['width']
        peak_sample_20 = metrics['peak_sample_20']
        
        # Calculate channel-wide amplitude threshold
        abs_peak_amps = np.abs(peak_sample_20)
        valid_abs_peak_amps = abs_peak_amps[np.isfinite(abs_peak_amps)]
        
        if len(valid_abs_peak_amps) > 0:
            amp_threshold = np.percentile(valid_abs_peak_amps, self.low_amp_percentile)
        else:
            amp_threshold = 0
            print(f"  Warning: No valid peak amplitudes for threshold calculation")
        
        print(f"  Amplitude threshold ({self.low_amp_percentile}th percentile): {amp_threshold:.2f}")
        
        # Identify low amplitude spikes
        mask_low_amp = np.abs(peak_sample_20) < amp_threshold
        mask_low_amp[np.isnan(mask_low_amp)] = False
        
        # Find auto-quarantined spikes
        mask_nan_width = np.isnan(width)
        mask_nan_prom_ratio = np.isnan(prominence_ratio)
        
        # Find "invalid" spikes that can be preserved
        with np.errstate(invalid='ignore'):
            mask_one_peak = (num_peaks == 1)
            mask_failed_prom_rule = (other_prominence < (self.other_prom_min_ratio * np.abs(peak_sample_20)))
        
        mask_one_peak[np.isnan(mask_one_peak)] = False
        mask_failed_prom_rule[np.isnan(mask_failed_prom_rule)] = False
        
        # Modified auto-quarantine logic
        mask_quarantine_bad_ratio = mask_nan_prom_ratio & ~mask_one_peak
        mask_auto_quarantine = mask_nan_width | mask_quarantine_bad_ratio
        
        # Find "Valid" spikes
        mask_invalid_but_potentially_preserved = (mask_one_peak | mask_failed_prom_rule)
        mask_for_plotting = ~(mask_auto_quarantine | mask_invalid_but_potentially_preserved)
        
        # Find valid spikes meeting thresholds
        with np.errstate(invalid='ignore'):
            mask_valid_and_meets_thresholds = (mask_for_plotting) & \
                                              (prominence_ratio > self.prom_threshold) & \
                                              (width < self.width_upper_threshold) & \
                                              (width > self.width_lower_threshold)
        mask_valid_and_meets_thresholds[np.isnan(mask_valid_and_meets_thresholds)] = False
        
        # Special case: 1-peak OR fail 0.01% rule can be preserved if pass width test
        with np.errstate(invalid='ignore'):
            mask_exception_case = (mask_one_peak | mask_failed_prom_rule)
            mask_passes_width_test = ~np.isnan(width) & \
                                     (width < self.width_upper_threshold) & \
                                     (width > self.width_lower_threshold)
            mask_exception_and_valid_width = mask_exception_case & mask_passes_width_test
        
        mask_exception_and_valid_width[np.isnan(mask_exception_and_valid_width)] = False
        
        # Define Preserved (base)
        mask_preserved_base = mask_valid_and_meets_thresholds | mask_exception_and_valid_width
        
        # Final preserved mask (must not be auto-quarantined OR low-amp)
        mask_preserved = mask_preserved_base & ~mask_auto_quarantine & ~mask_low_amp
        
        # Quarantined = NOT preserved
        mask_quarantined = ~mask_preserved
        
        print(f"  Auto-quarantined (bad ratio/width): {np.sum(mask_auto_quarantine)}")
        print(f"  Low amplitude: {np.sum(mask_low_amp)}")
        print(f"  One-peak exceptions preserved: {np.sum(mask_exception_and_valid_width)}")
        
        return mask_quarantined, mask_preserved
    
    def create_metrics_struct(self, metrics):
        """
        Create a structured array for quarantine properties.
        Includes low_low_amp_spike flag based on 5th percentile of peak amplitudes.
        """
        # Calculate low amplitude spike flag (peak_sample_20 < 5th percentile)
        peak_sample_20 = metrics['peak_sample_20']
        valid_peaks = peak_sample_20[np.isfinite(peak_sample_20)]
        
        if len(valid_peaks) > 0:
            low_amp_threshold = np.percentile(np.abs(valid_peaks), self.low_amp_percentile)
            low_low_amp_spike = np.abs(peak_sample_20) < low_amp_threshold
        else:
            low_low_amp_spike = np.zeros(len(peak_sample_20), dtype=bool)
        
        # Handle NaN values
        low_low_amp_spike[np.isnan(low_low_amp_spike)] = False
        
        metrics_array = np.core.records.fromarrays(
            [
                metrics['prominence_ratio'],
                metrics['num_peaks'],
                metrics['prominence_sample_20'],
                metrics['other_prominence'],
                metrics['width'],
                metrics['peak_sample_20'],
                metrics['other_peak_loc'],
                metrics['peak_pos_max'],
                metrics['prominence_pos_max'],
                low_low_amp_spike.astype(int)
            ],
            names='prominence_ratio,num_peaks,prominence_sample_20,other_prominence,width,peak_sample_20,other_peak_loc,peak_pos_max,prominence_pos_max,low_low_amp_spike'
        )
        return metrics_array


# Example usage
if __name__ == '__main__':
    # Initialize artifact rejection
    artifact_rej = ArtifactRejection(
        input_dir='/media/sEEG_DATA/Tests/Matlab sorting pipeline/Tapasi/MCWs_Pipeline/full_processing_pipeline/output',
        output_dir=None,  # None = same as input
        backup_dir=None   # None = input_dir/backup
    )
    
    # Optionally set custom thresholds
    artifact_rej.set_thresholds(
        prom_ratio=2.0,
        width_upper=15.0,
        width_lower=3.0,
        low_amp_percentile=5.0,
        other_prom_min_ratio=0.01
    )
    
    # ========================================
    # CHANNEL SELECTION OPTIONS
    # ========================================
    
    # Option 1: Process all channels
    # artifact_rej.process_all_channels(channels='all', parallel=True)
    
    # Option 2: Process specific channels
    # artifact_rej.process_all_channels(channels=[257, 263, 264], parallel=True)
    
    # Option 3: Process single channel
    # artifact_rej.process_all_channels(channels=290, parallel=True)
    
    # Option 4: Process range of channels
    artifact_rej.process_all_channels(channels=list(range(290, 301)), parallel=True)