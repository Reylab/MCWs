import os
import re
import numpy as np
import scipy.io
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
import traceback
import struct
import math
from scipy import signal
from scipy.io import loadmat, savemat
from scipy import interpolate
import datetime
from core.parameter_functions.Parameters import par
import pyqtgraph as pg
import glob
import multiprocessing
import sys


class Spikes():
    def __init__(self, par, nsx):
        self.par = par
        self.nsx = nsx
    
    def process_spikes_from_notebook(self, all_channels, neg_channels, pos_channels, param):
        """
        Helper function to be called from the Jupyter Notebook.
        This function contains the logic from your notebook cell.
        """
        
        # ========================================
        # PROCESS CHANNEL SELECTION
        # ========================================
        def parse_channels(channel_spec, all_channels):
            """Convert channel specification to channel objects."""
            if isinstance(channel_spec, str) and channel_spec.lower() == 'all':
                return all_channels
            elif channel_spec is None or (isinstance(channel_spec, list) and len(channel_spec) == 0):
                return np.array([], dtype=object)
            else:
                if isinstance(channel_spec, int):
                    channel_spec = [channel_spec]
                return np.array([ch for ch in all_channels if ch[0] in channel_spec], dtype=object)

        # Parse neg_channels and pos_channels
        neg_thr_channels = parse_channels(neg_channels, all_channels)
        pos_thr_channels = parse_channels(pos_channels, all_channels)

        print(f"neg_thr_channels: {[ch[0] for ch in neg_thr_channels] if len(neg_thr_channels) > 0 else 'None'}")
        print(f"pos_thr_channels: {[ch[0] for ch in pos_thr_channels] if len(pos_thr_channels) > 0 else 'None'}")

        # ========================================
        # RUN SPIKE DETECTION
        # ========================================
        print('\nStarting spike detection...')
        
        if param.parallel:
            # Process negative detection channels
            if neg_thr_channels.size:
                param.detection = 'neg'
                self.par = param
                print(f"Processing {len(neg_thr_channels)} channels with negative detection...")
                with multiprocessing.Pool(processes=10) as pool:
                    pool.map(self.get_spikes, neg_thr_channels, chunksize=1)
                    pool.close()
                    pool.join()
            
            # Process positive detection channels
            if pos_thr_channels.size:
                param.detection = 'pos'
                self.par = param
                print(f"Processing {len(pos_thr_channels)} channels with positive detection...")
                with multiprocessing.Pool(processes=10) as pool:
                    pool.map(self.get_spikes, pos_thr_channels, chunksize=1)
                    pool.close()
                    pool.join()
            
            # Process 'both' detection channels
            param.detection = 'both'
            self.par = param
            both_thr_channels_nums = np.setdiff1d(
                np.setdiff1d(
                    np.array([ch[0] for ch in all_channels]),
                    np.array([ch[0] for ch in neg_thr_channels])
                ),
                np.array([ch[0] for ch in pos_thr_channels]) if pos_thr_channels.size else []
            )
            
            if both_thr_channels_nums.size:
                both_thr_channels = np.array([ch for ch in all_channels if ch[0] in both_thr_channels_nums], dtype=object)
                print(f"Processing {len(both_thr_channels)} channels with both detection...")
                with multiprocessing.Pool(processes=10) as pool:
                    pool.map(self.get_spikes, both_thr_channels, chunksize=1)
                    pool.close()
                    pool.join()
            
            print('Spike detection done!')
            
        else:
            # Process negative detection channels
            if neg_thr_channels.size:
                param.detection = 'neg'
                self.par = param
                print(f"Processing {len(neg_thr_channels)} channels with negative detection...")
                for channel in neg_thr_channels:
                    self.get_spikes(channel)
            
            # Process positive detection channels
            if pos_thr_channels.size:
                param.detection = 'pos'
                self.par = param
                print(f"Processing {len(pos_thr_channels)} channels with positive detection...")
                for channel in pos_thr_channels:
                    self.get_spikes(channel)
            
            # Process 'both' detection channels
            param.detection = 'both'
            self.par = param
            both_thr_channels_nums = np.setdiff1d(
                np.setdiff1d(
                    np.array([ch[0] for ch in all_channels]),
                    np.array([ch[0] for ch in neg_thr_channels])
                ),
                np.array([ch[0] for ch in pos_thr_channels]) if pos_thr_channels.size else []
            )
            
            if both_thr_channels_nums.size:
                both_thr_channels = np.array([ch for ch in all_channels if ch[0] in both_thr_channels_nums], dtype=object)
                print(f"Processing {len(both_thr_channels)} channels with both detection...")
                for channel in both_thr_channels:
                    self.get_spikes(channel)
            
            print('Spike detection done!')

    def get_spikes(self, channel):
        """
        Extract spikes from a single channel.
        Called by multiprocessing pool or notebook.
        """
        channel_num = channel[0][0] if isinstance(channel[0], np.ndarray) else channel[0]
        par = self.par
        nsx = self.nsx
        
        ichan = np.nonzero(nsx['chan_ID'] == channel)[1][0]
        ch_type = nsx['ext'][0, ichan][0]
        sr = int(nsx['sr'][0, ichan][0])
        lts = int(nsx['lts'][0, ichan])
        unit = nsx['unit'][0, ichan][0]
        label = nsx['label'][0, ichan][0]
        conversion = nsx['conversion'][0, ichan][0][0]
        output_name = nsx['output_name'][0, ichan][0]
        dc = float(nsx['dc'][0, ichan])
        
        # Get .NC filepaths
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/input/' + str(output_name) + str(ch_type)
        
        # Get x data
        f1 = open(filepath, 'rb')
        binary_data = f1.read()
        format_string = '<' + 'h' * (len(binary_data) // 2)
        unpacked_data = struct.unpack(format_string, binary_data)
        x_raw = np.array(unpacked_data) * conversion + dc
        f1.close()
        
        tmax_str = getattr(par, 'tmax', 'all')
        if tmax_str == 'all':   
            tmax = len(x_raw) / sr
        else:
            tmax = tmax_str
            
        tmin = getattr(par, 'tmin', 0)
        
        max_segments = math.ceil((tmax - tmin) / (par.segments_length * 60))
        slen = par.segments_length * 60 * par.sr
        
        # Start at tmin
        start_sample = int(tmin * sr)
        x_raw_from_tmin = x_raw[start_sample:]
        x_raws = np.array_split(x_raw_from_tmin, np.arange(slen, len(x_raw_from_tmin), slen, dtype=int))
        
        max_segments = len(x_raws)
        
        spike_file = []
        current_par = self.par
        current_par.process_info = {}
        
        try:
            dir = os.path.dirname(os.path.dirname(__file__))
            file = dir + '/output/process_info.mat'
            metadata = loadmat(file)
            
            # Find the index for this channel
            mat_ch_ids = metadata['process_info']['ch_Id'][0, 0].ravel()
            ch_idx_in_mat = np.where(mat_ch_ids == channel_num)[0]
            
            if ch_idx_in_mat.size > 0:
                ch_idx = ch_idx_in_mat[0]
                current_par.process_info = {
                    'chID': metadata['process_info']['ch_Id'][0, 0][0, ch_idx],
                    'sos': metadata['process_info']['sos'][0, 0][0, ch_idx],
                    'freqs_notch': metadata['process_info']['freqs_notch'][0, 0][0, ch_idx],
                    'BwHz_notch': metadata['process_info']['BwHz_notch'][0, 0][0, ch_idx]
                }
                s = current_par.process_info['sos']
            else:
                print(f"Warning: No process_info found for channel {channel_num}. Proceeding without notch.")
                current_par.process_info = np.array([])
                s = []

            spike_file = glob.glob(pathname=dir + f'/output/{label}_{channel_num}_spikes.mat')
        except Exception as e:
            print(f"Warning: Could not load process_info.mat. Proceeding without notch filters. Error: {e}")
            current_par.process_info = np.array([])
            s = []
        
        threshold = []
        if spike_file != []:
            print(f"Spike file already exists for channel {channel_num}, skipping...")
            return
        else:
            index = []
            spikes = np.array([])
            possible_artifact = np.array([])
            
            for i, x_raw_segment in enumerate(x_raws): 
                new_spikes, aux_th, new_indexs, remove_counter, new_possible_artifact = self.amp_detect(x_raw_segment, current_par, s)
                new_time_indexs = self.index2ts(new_indexs, sr, i, slen, tmin)
                index.extend(new_time_indexs)
                
                # Check if new_possible_artifact is an array
                if not isinstance(new_possible_artifact, np.ndarray):
                    print(f"Warning: amp_detect returned 'possible_artifact' as a {type(new_possible_artifact)}, not np.array. Converting.")
                    new_possible_artifact = np.array(new_possible_artifact)
                
                if not possible_artifact.size:
                    possible_artifact = new_possible_artifact
                elif new_possible_artifact.size > 0:
                    possible_artifact = np.concatenate((possible_artifact, new_possible_artifact), axis=0)
                
                if not spikes.size:
                    spikes = new_spikes
                elif new_spikes.size > 0:
                    spikes = np.concatenate((spikes, new_spikes), axis=0)
                
                threshold.append(aux_th)

            current_par.detection_date = str(datetime.datetime.now())
            
            lplot = min(math.floor(60 * sr), len(x_raws[0]))
            xf_detect = self.spike_detection_filter(x_raws[0][0:lplot], current_par, ichan)
            sub = max(1, math.floor(lplot / par.cont_plot_samples))
            psegment = xf_detect[::sub]
            sr_psegment = sr / sub
            
            mdic = {
                "par": current_par,
                "threshold": threshold,
                "index": index,
                "spikes": spikes,
                "possible_artifact": possible_artifact,
                "psegment": psegment,
                "sr_psegment": sr_psegment
            }
            dir = os.path.dirname(os.path.dirname(__file__))
            savemat(dir + f'/output/{label}_{channel_num}_spikes.mat', mdic)
            print(f"Saved spike file for channel {channel_num} with {spikes.shape[0]} spikes")

    def index2ts(self, indexs, sr, count, slen, tmin):
        """Convert sample indices to timestamps in milliseconds."""
        ts_list = []
        tmin_samples = int(tmin * sr)
        for i, index in enumerate(indexs):
            # Add offset for segment AND tmin
            ts = ((index + 1) + (count * slen) + tmin_samples) / sr * 1000
            ts_list.append(ts)
        return ts_list

    def amp_detect(self, x, par, s):
        """
        Amplitude-based spike detection using findpeaks algorithm.
        Exact replica of MATLAB logic with proper continue handling.
        """
        sr = par.sr
        w_pre = par.w_pre
        w_post = par.w_post
        
        if 'ref_ms' in list(vars(par).keys()):
            ref = math.floor(par.ref_ms * par.sr / 1000)
        else:
            ref = par.ref
        
        if 'minus_one' in list(vars(par).keys()):
            offset = 1 if par.minus_one else 0
        else:
            offset = 0
        
        detect = par.detection
        stdmin = par.stdmin
        stdmax = par.stdmax

        # Filter for sorting (finer alignment)
        if par.sort_order > 0:
            xf = self.filt_signal(x=x, order=par.sort_order, fmin=par.sort_fmin, fmax=par.sort_fmax,
                                  sr=sr, par=par, s=s)
        else:
            if par.preprocessing and isinstance(par.process_info, dict) and 'sos' in par.process_info:
                xf = signal.sosfiltfilt(s, x, padlen=3 * (s.shape[0]))
            else:
                xf = x
        
        # Filter for detection (threshold crossing)
        if par.detect_order > 0:
            xf_detect = self.filt_signal(x=x, order=par.detect_order, fmin=par.detect_fmin, fmax=par.detect_fmax, sr=sr, par=par, s=s)
        else:
            if par.preprocessing and isinstance(s, np.ndarray) and s.size > 0:
                xf_detect = signal.sosfiltfilt(s, x, padlen=3 * (s.shape[0]))
            elif par.preprocessing and isinstance(s, list) and len(s) > 0:
                xf_detect = signal.sosfiltfilt(np.array(s), x, padlen=3 * (len(s)))
            else:
                xf_detect = x
        
        noise_std_detect = np.median(np.absolute(xf_detect)) / 0.6745
        noise_std_sorted = np.median(np.absolute(xf)) / 0.6745
        thr = stdmin * noise_std_detect
        thrmax = stdmax * noise_std_sorted
        
        index = []
        possible_artifact = []
        sample_ref = math.floor(ref / 2)

        # ===== SPIKE DETECTION LOGIC =====
        match detect:
            case 'pos':
                # ===== POSITIVE DETECTION =====
                xaux = np.nonzero(xf_detect[w_pre + 1:len(xf_detect) - w_post - 2 - sample_ref] > thr)[0] + w_pre + 1
                xaux0 = -np.inf
                
                for i in range(len(xaux)):
                    if xaux[i] >= xaux0 + ref:
                        # Expanded window for findpeaks
                        start_idx = max(0, int(xaux[i]) - 10)
                        end_idx = min(len(xf), int(xaux[i]) + int(sample_ref) + 10)
                        search_window = xf[start_idx:end_idx]
                        
                        if search_window.size > 0:
                            locs, _ = signal.find_peaks(search_window)
                            
                            if len(locs) > 0:
                                # Get peak heights manually
                                pks = search_window[locs]
                                iM = int(np.argmax(pks))
                                peak_loc_in_window = locs[iM]
                                peak_idx = start_idx + peak_loc_in_window
                                
                                # Artifact check
                                possible_artifact.append(xf_detect[xaux[i] - 1] > thr)
                                index.append(peak_idx)
                                xaux0 = peak_idx
            
            case 'neg':
                # ===== NEGATIVE DETECTION =====
                xaux = np.nonzero(xf_detect[w_pre + 1:len(xf_detect) - w_post - 2 - sample_ref] < -thr)[0] + w_pre + 1
                xaux0 = -np.inf
                
                for i in range(len(xaux)):
                    if xaux[i] >= xaux0 + ref:
                        # Expanded window for findpeaks
                        start_idx = max(0, int(xaux[i]) - 10)
                        end_idx = min(len(xf), int(xaux[i]) + int(sample_ref) + 10)
                        search_window = -xf[start_idx:end_idx]  # Invert signal
                        
                        if search_window.size > 0:
                            locs, _ = signal.find_peaks(search_window)
                            
                            if len(locs) > 0:
                                # Get peak heights manually
                                pks = search_window[locs]
                                iM = int(np.argmax(pks))
                                peak_loc_in_window = locs[iM]
                                peak_idx = start_idx + peak_loc_in_window
                                
                                # Artifact check
                                possible_artifact.append(xf_detect[xaux[i] - 1] < -thr)
                                index.append(peak_idx)
                                xaux0 = peak_idx
            
            case 'both':
                # ===== BOTH POSITIVE AND NEGATIVE DETECTION =====
                xaux = np.nonzero(np.absolute(xf_detect[w_pre + 1:len(xf_detect) - w_post - 2 - sample_ref]) > thr)[0] + w_pre + 1
                xaux0 = -np.inf
                
                for i in range(len(xaux)):
                    if xaux[i] >= xaux0 + ref:
                        # Expanded window for findpeaks
                        start_idx = max(0, int(xaux[i]) - 10)
                        end_idx = min(len(xf), int(xaux[i]) + int(sample_ref) + 10)
                        search_window = np.absolute(xf[start_idx:end_idx])
                        
                        if search_window.size > 0:
                            locs, _ = signal.find_peaks(search_window)
                            
                            if len(locs) > 0:
                                # Get peak heights manually
                                pks = search_window[locs]
                                iM = int(np.argmax(pks))
                                peak_loc_in_window = locs[iM]
                                peak_idx = start_idx + peak_loc_in_window
                                
                                # Artifact check
                                possible_artifact.append(np.absolute(xf_detect[xaux[i] - 1]) > thr)
                                index.append(peak_idx)
                                xaux0 = peak_idx

        # ===== EXTRACT SPIKE WAVEFORMS =====
        nspk = len(index)
        ls = w_pre + w_post
        
        spikes_to_keep = []
        index_to_keep = []
        artifact_to_keep = []
        remove_counter = 0
        
        # Pad xf if necessary
        if len(xf) < len(x) + w_post:
            xf = np.concatenate((xf, np.zeros(w_post + len(x) - len(xf))))
        
        if nspk > 0:
            for i in range(nspk):
                idx = int(index[i])
                start = max(0, idx - w_pre)
                end = min(len(xf), idx + w_post + 1)
                
                if end - start > 1:
                    # Check spike amplitude
                    if idx < len(xf) and max(np.absolute(xf[start:end])) < thrmax:
                        end_spike = min(idx + w_post + 3, len(xf))
                        start_spike = max(idx - w_pre - 1, 0)
                        spike_data = xf[start_spike:end_spike]
                        
                        # Pad if necessary
                        if len(spike_data) < ls + 4:
                            spike_data = np.concatenate((spike_data, np.zeros(ls + 4 - len(spike_data))))
                        
                        spikes_to_keep.append(spike_data[:ls + 4])
                        index_to_keep.append(index[i])
                        artifact_to_keep.append(possible_artifact[i])
                    else:
                        remove_counter += 1
                else:
                    remove_counter += 1
        
        # Convert lists to arrays
        if len(spikes_to_keep) > 0:
            spikes = np.array(spikes_to_keep)
            index = np.array(index_to_keep, dtype=int)
            possible_artifact = np.array(artifact_to_keep)
        else:
            # Return properly shaped empty arrays
            spikes = np.array([]).reshape(0, ls + 4)
            index = np.array([], dtype=int)
            possible_artifact = np.array([])
        
        # ===== OPTIONAL INTERPOLATION =====
        match par.interpolation:
            case 'n':
                if spikes.shape[0] > 0:
                    spikes = spikes[:, 2:-2]
            case 'y':
                spikes = self.int_spikes(spikes=spikes, par=par)
        
        return spikes, thr, index, remove_counter, possible_artifact

    def int_spikes(self, spikes, par):
        """Interpolate spike waveforms for better alignment."""
        w_pre = par.w_pre
        w_post = par.w_post
        ls = w_pre + w_post
        detect = par.detection
        int_factor = par.int_factor
        nspk = spikes.shape[0]
        
        if nspk == 0:
            return np.array([]).reshape(0, ls)
        
        extra = int((spikes.shape[1] - ls) / 2)
        s = np.arange(1, spikes.shape[1] + 1)
        ints = np.arange(1 / int_factor, spikes.shape[1] + (1 / int_factor), 1 / int_factor)
        spikes1 = np.zeros(shape=(nspk, ls))
        
        intspikes = np.zeros((nspk, len(ints)))
        for i, spike in enumerate(spikes):
            spl = interpolate.splrep(s, spike)
            intspikes[i, :] = interpolate.splev(ints, spl, der=0)
        
        # 0-based indexing for Python
        start_idx = (w_pre + extra - 1) * int_factor 
        end_idx = (w_pre + extra + 1) * int_factor
        search_window = intspikes[:, start_idx:end_idx]

        match detect:
            case 'pos':
                iaux = np.argmax(a=search_window, axis=1)
            case 'neg':
                iaux = np.argmin(a=search_window, axis=1)
            case 'both':
                iaux = np.argmax(a=np.absolute(search_window), axis=1)
        
        iaux = iaux + start_idx
        
        for i in range(nspk):
            start_spike = iaux[i] - (w_pre * int_factor) + int_factor
            end_spike = iaux[i] + (w_post * int_factor)
            spikes1[i, :] = intspikes[i, np.arange(start_spike, end_spike + 1, int_factor)]
        
        return spikes1

    def filt_signal(self, x, order, fmin, fmax, sr, par, s):
        """Design and apply Elliptic bandpass filter."""
        b, a = signal.ellip(N=order, rp=0.1, rs=40, btype='bandpass', Wn=np.array([fmin, fmax]) * 2 / sr)
        
        if par.preprocessing and isinstance(par.process_info, dict) and 'sos' in par.process_info:
            sos = signal.tf2sos(b, a)
            try:
                sos = np.concatenate((par.process_info['sos'], sos), axis=0)   
            except:
                sos = sos
            filtered = signal.sosfiltfilt(sos, x, padlen=3 * (sos.shape[0]))
        else:
            filtered = signal.filtfilt(b, a, x, padlen=3 * (max(len(a), len(b)) - 1))
        
        return filtered
    
    def spike_detection_filter(self, x, par, ch):
        """Apply detection filter to signal."""
        sr = par.sr
        fmin_detect = par.detect_fmin
        fmax_detect = par.detect_fmax

        if par.detect_order > 0:
            b, a = signal.ellip(par.detect_order, 0.1, 40, np.array([fmin_detect, fmax_detect]) * 2 / sr, btype='bandpass')

            if par.preprocessing and isinstance(par.process_info, dict) and 'sos' in par.process_info:
                s = signal.tf2sos(b, a)
                try:
                    sos = par.process_info['sos']
                    sos = np.concatenate((sos, s), axis=0)
                except:
                    sos = s
                xf_detect = signal.sosfiltfilt(sos, x, padlen=3 * (sos.shape[0]))
            else:
                xf_detect = signal.filtfilt(b, a, x, padlen=3 * (max(len(a), len(b)) - 1))
        else:
            if par.preprocessing and not (isinstance(par.process_info, dict) and 'sos' in par.process_info):
                print(f'Warning: preprocessing enabled but no process_info for channel {ch}.')
            xf_detect = x
        
        return xf_detect


if __name__ == '__main__':
    print("Running spike detection script directly...")
    
    NSx_file_path = os.path.abspath(glob.glob("input/NSx.mat")[0])
    metadata = loadmat(NSx_file_path)
    nsx = metadata['NSx']
    
    param = par()
    param.detection = 'neg'
    param.sr = 30000
    param.detect_fmin = 300
    param.detect_fmax = 3000
    param.auto = 0
    param.mVmin = 50
    param.w_pre = 20                
    param.w_post = 44               
    param.min_ref_per = 1.5               
    param.ref = np.floor(param.min_ref_per * param.sr / 1000)            
    param.factor_thr = 5
    param.detect_order = 4
    param.sort_order = 2
    param.sort_fmin = 300
    param.sort_fmax = 3000
    param.stdmin = 5
    param.stdmax = 50
    param.ref_ms = 1.5
    param.preprocessing = True
    param.interpolation = 'n'
    param.segments_length = 5
    param.tmax = 'all'
    param.tmin = 0
    param.cont_plot_samples = 60000
    param.parallel = True
    param.int_factor = 2
    param.minus_one = 0
    
    spike = Spikes(par=param, nsx=nsx)
    
    # Get all available channels
    if param.micros:
        all_channels = nsx['chan_ID'][0][list(set(np.where(nsx['unit']=='uV')[1]) & 
                                              set(np.where(nsx['sr']==30000)[1]))]
    else:
        all_channels = nsx['chan_ID'][0][list(set(np.where(nsx['sr']==2000)[1]))]

    specific_channels = 'all'
    neg_channels = 'all'
    pos_channels = None
    
    spike.process_spikes_from_notebook(all_channels, neg_channels, pos_channels, param)