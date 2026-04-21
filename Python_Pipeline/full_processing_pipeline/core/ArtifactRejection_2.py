import numpy as np
from scipy.signal import find_peaks, peak_prominences
import traceback
import logging
 
from mcws.core.registry import register_artifact_rejector
from .base import BaseArtifactRejector, set_artifact_column
 
try:
    from mcws.config import get_artifact_rejection_params
except ImportError:
    def get_artifact_rejection_params():
        return {}
 
logger = logging.getLogger(__name__)
 
 
def _process_channel_artifacts(ch_id, data, rejector_params):
    """
    Worker function for parallel within-channel artifact rejection.
 
    Parameters
    ----------
    ch_id : int
        Channel identifier.
    data : dict
        Channel data dict with 'spikes', 'possible_artifact', etc.
    rejector_params : dict
        Rejector thresholds (prom_threshold, width_upper_threshold, etc.).
 
    Returns
    -------
    tuple or None
        (ch_id, mask_quarantined, quarantine_properties, combined_artifact_mask) or None if skipped.
    """
    spikes = data.get('spikes', np.array([]))
    if spikes.ndim != 2 or spikes.shape[0] == 0:
        return None
 
    # Create a temporary rejector to use its methods
    rejector = WithinChannelRejector.__new__(WithinChannelRejector)
    rejector.prom_threshold = rejector_params['prom_threshold']
    rejector.width_upper_threshold = rejector_params['width_upper_threshold']
    rejector.width_lower_threshold = rejector_params['width_lower_threshold']
    rejector.low_amp_percentile = rejector_params['low_amp_percentile']
    rejector.other_prom_min_ratio = rejector_params['other_prom_min_ratio']
 
    metrics = rejector.calculate_all_metrics(spikes, parallel=False)
    mask_quarantined, _ = rejector.apply_artifact_rejection(metrics, spikes)
    quarantine_properties = rejector.create_metrics_struct(metrics)
 
    return (ch_id, mask_quarantined.astype(bool), quarantine_properties)
 
 
@register_artifact_rejector("within_channel")
class WithinChannelRejector(BaseArtifactRejector):
    """
    Implements comprehensive within-channel artifact rejection based on waveform
    morphology, peak width, and prominence ratios.
    """
    artifact_name = "within_channel"
 
    def __init__(
        self,
        prom_ratio=None,
        width_upper=None,
        width_lower=None,
        low_amp_percentile=None,
        other_prom_min_ratio=None,
        parallel=False,
        n_workers=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.parallel = parallel
        self.n_workers = n_workers
 
        cfg = get_artifact_rejection_params()
        self.prom_threshold = prom_ratio if prom_ratio is not None else cfg.get('prom_ratio', 2.0)
        self.width_upper_threshold = width_upper if width_upper is not None else cfg.get('width_upper', 15.0)
        self.width_lower_threshold = width_lower if width_lower is not None else cfg.get('width_lower', 3.0)
        self.low_amp_percentile = low_amp_percentile if low_amp_percentile is not None else cfg.get('low_amp_percentile', 5.0)
        self.other_prom_min_ratio = other_prom_min_ratio if other_prom_min_ratio is not None else cfg.get('other_prom_min_ratio', 0.01)
 
    def _get_params(self):
        """Return rejector parameters as a plain dict (picklable for joblib)."""
        return {
            'prom_threshold': self.prom_threshold,
            'width_upper_threshold': self.width_upper_threshold,
            'width_lower_threshold': self.width_lower_threshold,
            'low_amp_percentile': self.low_amp_percentile,
            'other_prom_min_ratio': self.other_prom_min_ratio,
        }
 
    def apply(self, results, recording=None, n_workers=None, **kwargs):
        if n_workers is None:
            n_workers = self.n_workers
 
        n_channels = len(results)
        use_parallel = (n_workers != 1) and (n_channels > 1)
        params = self._get_params()
 
        if use_parallel:
            from joblib import Parallel, delayed
            channel_results = Parallel(n_jobs=n_workers, backend="loky", prefer="processes")(
                delayed(_process_channel_artifacts)(ch_id, data, params)
                for ch_id, data in results.items()
            )
        else:
            channel_results = [
                _process_channel_artifacts(ch_id, data, params)
                for ch_id, data in results.items()
            ]
 
        # Reassemble
        for res in channel_results:
            if res is None:
                continue
            ch_id, mask_quarantined, quarantine_properties = res
            results[ch_id]['mask_quarantined'] = mask_quarantined
            results[ch_id]['quarantine_properties'] = quarantine_properties
            # Write dedicated column and refresh aggregate possible_artifact.
            set_artifact_column(results[ch_id], self.artifact_name, mask_quarantined)
 
        return results
 
    def calculate_all_metrics(self, spikes, parallel=False):
        from mcws.utils.accel import batch_find_peaks_and_prominences
        return batch_find_peaks_and_prominences(spikes)
    
    def calculate_spike_metrics(self, waveform):
        try:
            inverted_waveform = -waveform
            all_peaks, _ = find_peaks(inverted_waveform)
            num_peaks = len(all_peaks)
 
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
            
            pos_peaks, _ = find_peaks(waveform)
            if len(pos_peaks) > 0:
                pos_prominences, _, _ = peak_prominences(waveform, pos_peaks)
                pos_peak_amplitudes = waveform[pos_peaks]
                max_pos_amp_idx = np.argmax(pos_peak_amplitudes)
                peak_pos_max = pos_peak_amplitudes[max_pos_amp_idx]
                prominence_pos_max = pos_prominences[max_pos_amp_idx]
            
            TARGET_PEAK_INDEX = 19
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
 
            # Width
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
    
    def apply_artifact_rejection(self, metrics, spikes):
        prominence_ratio = metrics['prominence_ratio']
        num_peaks = metrics['num_peaks']
        other_prominence = metrics['other_prominence']
        width = metrics['width']
        peak_sample_20 = metrics['peak_sample_20']
        
        abs_peak_amps = np.abs(peak_sample_20)
        valid_abs_peak_amps = abs_peak_amps[np.isfinite(abs_peak_amps)]
        
        if len(valid_abs_peak_amps) > 0:
            amp_threshold = np.percentile(valid_abs_peak_amps, self.low_amp_percentile)
        else:
            amp_threshold = 0
        
        mask_low_amp = np.abs(peak_sample_20) < amp_threshold
        mask_low_amp[np.isnan(mask_low_amp)] = False
        
        mask_nan_width = np.isnan(width)
        mask_nan_prom_ratio = np.isnan(prominence_ratio)
        
        with np.errstate(invalid='ignore'):
            mask_one_peak = (num_peaks == 1)
            mask_failed_prom_rule = (other_prominence < (self.other_prom_min_ratio * np.abs(peak_sample_20)))
        
        mask_one_peak[np.isnan(mask_one_peak)] = False
        mask_failed_prom_rule[np.isnan(mask_failed_prom_rule)] = False
        
        mask_quarantine_bad_ratio = mask_nan_prom_ratio & ~mask_one_peak
        mask_auto_quarantine = mask_nan_width | mask_quarantine_bad_ratio
        
        mask_invalid_but_potentially_preserved = (mask_one_peak | mask_failed_prom_rule)
        mask_for_plotting = ~(mask_auto_quarantine | mask_invalid_but_potentially_preserved)
        
        with np.errstate(invalid='ignore'):
            mask_valid_and_meets_thresholds = (mask_for_plotting) & \
                                              (prominence_ratio > self.prom_threshold) & \
                                              (width < self.width_upper_threshold) & \
                                              (width > self.width_lower_threshold)
        mask_valid_and_meets_thresholds[np.isnan(mask_valid_and_meets_thresholds)] = False
        
        with np.errstate(invalid='ignore'):
            mask_exception_case = (mask_one_peak | mask_failed_prom_rule)
            mask_passes_width_test = ~np.isnan(width) & \
                                     (width < self.width_upper_threshold) & \
                                     (width > self.width_lower_threshold)
            mask_exception_and_valid_width = mask_exception_case & mask_passes_width_test
        
        mask_exception_and_valid_width[np.isnan(mask_exception_and_valid_width)] = False
        
        mask_preserved_base = mask_valid_and_meets_thresholds | mask_exception_and_valid_width
        mask_preserved = mask_preserved_base & ~mask_auto_quarantine & ~mask_low_amp
        mask_quarantined = ~mask_preserved
        
        return mask_quarantined, mask_preserved
    
    def create_metrics_struct(self, metrics):
        peak_sample_20 = metrics['peak_sample_20']
        valid_peaks = peak_sample_20[np.isfinite(peak_sample_20)]
        
        if len(valid_peaks) > 0:
            low_amp_threshold = np.percentile(np.abs(valid_peaks), self.low_amp_percentile)
            low_low_amp_spike = np.abs(peak_sample_20) < low_amp_threshold
        else:
            low_low_amp_spike = np.zeros(len(peak_sample_20), dtype=bool)
        
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
 
 
def _calculate_spike_metrics_static(waveform):
    """
    Module-level static version of calculate_spike_metrics for joblib compatibility.
    Joblib cannot pickle bound methods, so this is a standalone function.
    """
    try:
        inverted_waveform = -waveform
        all_peaks, _ = find_peaks(inverted_waveform)
        num_peaks = len(all_peaks)
 
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
 
        pos_peaks, _ = find_peaks(waveform)
        if len(pos_peaks) > 0:
            pos_prominences, _, _ = peak_prominences(waveform, pos_peaks)
            pos_peak_amplitudes = waveform[pos_peaks]
            max_pos_amp_idx = np.argmax(pos_peak_amplitudes)
            peak_pos_max = pos_peak_amplitudes[max_pos_amp_idx]
            prominence_pos_max = pos_prominences[max_pos_amp_idx]
 
        TARGET_PEAK_INDEX = 19
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
 
        # Width
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