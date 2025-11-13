import os
from scipy.io import loadmat
import numpy as np
import scipy.signal as signal
from core.config import config
import struct
import math
import csv
import pyqtgraph as pg
from core.graphing import graphing
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt, QThreadPool, pyqtSignal
import sys
from core.ui import MainWindow
import concurrent.futures
import multiprocessing
import pyqtgraph as pg
from scipy.io import savemat
import core.ThreadWorker as TW
import queue


class filtering():
    def __init__(self, save_fig=False, show_img=True, direc_resus_bae=os.path.dirname(os.path.realpath(__file__)),
                 resus_folder_name='spectra', direc_raw=os.path.dirname(os.path.realpath(__file__)), with_NoNotch=False,
                 time_plot_duration=1, freq_line=60, parallel=False, k_periodograms=200, notch_filter=True, spectrum_resolution=0.5):
        self.save_fig = save_fig
        self.show_img = show_img
        self.direc_resus_bae = direc_resus_bae
        self.resus_folder_name = resus_folder_name
        self.direc_raw = direc_raw
        self.with_NoNotch = with_NoNotch
        self.time_plot_duration = time_plot_duration
        self.freq_line = freq_line
        self.parallel = parallel
        self.k_periodograms = k_periodograms
        self.notch_filter = notch_filter
        self.spectrum_resolution = spectrum_resolution

    def new_check_lfp_power_NSX(self, nsx_file, channels='all'):
        """
        Check LFP power for specified channels.
        
        Parameters:
        -----------
        nsx_file : dict
            NSx data structure
        channels : 'all', list, int, optional
            - 'all': Process all channels (default)
            - list: Specific channel numbers [257, 263, 290]
            - int: Single channel number 257
            
        Examples:
        ---------
        filter.new_check_lfp_power_NSX(metadata, channels='all')
        filter.new_check_lfp_power_NSX(metadata, channels=[257, 263])
        filter.new_check_lfp_power_NSX(metadata, channels=257)
        filter.new_check_lfp_power_NSX(metadata, channels=list(range(290, 301)))
        """
        self.session = os.path.split(self.direc_raw)[1]
        self.span_smooth = 21
        self.db_thr = 5
        self.freqs_comp = [300, 1000, 2000, 3000, 6000]

        self.nsx = nsx_file['NSx']
        self.freq_priority = nsx_file['freq_priority'][0, :].tolist()

        self.conf_table = config
        
        # ========== CHANNEL FILTERING ==========
        # Get all available channels from NSx
        all_channels_raw = self.nsx['chan_ID'][0]
        
        if isinstance(channels, str) and channels.lower() == 'all':
            # Use all channels - keep original format
            selected_channels = all_channels_raw
            selected_nums = [ch[0] for ch in all_channels_raw]
            print(f"Processing all {len(selected_channels)} channels: {selected_nums}")
        else:
            # Convert to list if single channel
            if isinstance(channels, int):
                channels = [channels]
            
            # Filter channels - KEEP ORIGINAL CHANNEL OBJECTS
            selected_channels = np.array([ch for ch in all_channels_raw if ch[0] in channels], dtype=object)
            
            #print(f"Requested channels: {channels}")
            print(f"Processing {len(selected_channels)} channels: {[ch[0] for ch in selected_channels]}")
            
            # If no matches, warn and use all
            if len(selected_channels) == 0:
                print(f"WARNING: No matching channels found!")
                print(f"Available channels: {[ch[0] for ch in all_channels_raw]}")
                print("Using all channels instead.")
                selected_channels = all_channels_raw  
        # ========================================
        
        nchannels = len(selected_channels)
        ch_ID = []
        sos = []
        freqs_notch = []
        BwHz_notch = []
        g = []
       
        if self.parallel: 
            # GUI Thread
            results = []
            with multiprocessing.Pool(processes=10) as pool:
                results = pool.imap(self.new_check_lfp_power, selected_channels, chunksize=10) 
                pool.close()
                pool.join()
            
            app = QApplication([])
            for result in results:
                graph = graphing(result[1], result[2], result[3], result[4], result[5], result[6], 
                               result[7], result[8], result[9], result[10], result[11], result[12], 
                               result[13], result[14], result[15], result[16], result[17], result[18])
                graph.plot_notches()
                info_temp = result[0]
                if info_temp != []:
                    ch_ID.append(info_temp[0][0][0])
                    sos.append(info_temp[1].tolist())
                    freqs_notch.append(info_temp[2].tolist())
                    BwHz_notch.append(info_temp[3].tolist())
        else:            
            for i, channel in enumerate(selected_channels):
                print(f"Processing channel {channel[0]} ({i+1}/{nchannels})")
                result = self.new_check_lfp_power(channel)
                graph = graphing(result[1], result[2], result[3], result[4], result[5], result[6],
                               result[7], result[8], result[9], result[10], result[11], result[12],
                               result[13], result[14], result[15], result[16], result[17], result[18])
                info = result[0]
                ch_ID.append(info[0][0][0])
                sos.append(info[1].tolist())
                freqs_notch.append(info[2].tolist())
                BwHz_notch.append(info[3].tolist())
        
        filename = os.path.dirname(os.path.dirname(__file__)) + '/output/' + "process_info.mat"
        struct = {
            'ch_Id': ch_ID,
            'sos': sos,
            'freqs_notch': freqs_notch,
            'BwHz_notch': BwHz_notch
        }
        savemat(filename, {'process_info': struct})
        print(f"\nProcessing complete! Results saved to {filename}")
        #print(f"Processed {len(ch_ID)} channels: {ch_ID}")

    def new_check_lfp_power(self, channel):
        ch_type = []
        extra_title = []
        freq_sr = [set(np.where(freq == self.nsx['sr'])[1]) for freq in self.freq_priority]
        freq_idx = np.nonzero(freq_sr)[0][0]
        selected = list(set(np.where((self.nsx['chan_ID'] == channel) == True)[1]) & freq_sr[freq_idx])
        if selected == None:
            raise ValueError('there are no channels in NSX')
        elif len(np.nonzero(selected)) > 1:
            posch = max(selected)
        else:
            posch = selected[0]
        ch_type = self.nsx['ext'][0, posch][0]
        sr = self.nsx['sr'][0, posch][0][0]
        lts = int(self.nsx['lts'][0, posch])
        unit = self.nsx['unit'][0, posch][0]
        label = self.nsx['label'][0, posch][0]
        conversion = self.nsx['conversion'][0, posch]
        output_name = self.nsx['output_name'][0, posch][0]
        try:
            dc = self.nsx['dc'][0, posch]
        except:
            dc = 0
        if ch_type == []:
            raise ValueError(f"channel ' {channel} ' not parsed")
        if sr == 30000:
            self.freqs_comp = [300, 1000, 2000, 3000, 6000]
            identifier = 'uV'
        elif sr == 7500:
            self.freqs_comp = [300, 1000, 2000, 3000]
        else:
            self.freqs_comp = [300, 990]
        if sr == 2000:
            identifier = 'mV'

        conf = self.conf_table['lfp_par']['freq'][str(sr)][identifier]
        if self.notch_filter == False:
            conf['notch_Q'] = []
            conf['notch_width'] = []
        nofilter = conf['notch_q'] == [] and conf['notch_width'] == [] and conf['filter_stop'] == [] and conf['filter_order'] == []
        b_pass = []
        a_pass = []
        max_freq_line = sr / 2
        min_freq_line = 0
        if conf['filter_stop'] != []:
            min_freq_line = conf['filter_stop'][0]
            max_freq_line = conf['filter_stop'][1]
            ord, wn = signal.ellipord(conf['filter_pass'] * 2 / sr, conf['filter_stop'] * 2 / sr, conf['filter_Rp'], conf['filter_Rs'])
            b_pass, a_pass = signal.ellip(ord, conf['filter_Rp'], conf['filter_Rs'], wn)
        elif conf['filter_order'] != []:
            min_freq_line = conf['filter_pass'][0]
            max_freq_line = conf['filter_pass'][1]
            b_pass, a_pass = signal.ellip(N=conf['filter_order'], rp=conf['filter_Rp'], rs=conf['filter_Rs'], 
                                         Wn=np.array(conf['filter_pass']) * 2 / sr, btype='bandpass')
        n = int(2 ** np.ceil(np.log2(sr / self.spectrum_resolution)))
        samples_spectrum = min(self.k_periodograms * n, lts)
        samples_timeplot = min(self.time_plot_duration * 60 * sr, lts)
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/input/' + str(output_name) + str(ch_type)
        f1 = open(filepath, 'rb')
        idx = lts - max(samples_spectrum, samples_timeplot)
        f1.seek(idx * 2)
        binary_data = f1.read()
        format_string = '<' + 'h' * (len(binary_data) // 2)
        unpacked_data = struct.unpack(format_string, binary_data)
        x_raw = unpacked_data * conversion + dc
        x_raw = x_raw[0]
        f1.close()
        
        # Possibly do an "average" of the first 2.5 mins and the last 2.5 min
        f, pxx = signal.welch(fs=sr, x=x_raw, window=signal.windows.barthann(n), noverlap=0, 
                             return_onesided=True, scaling='density', detrend=False)
        pxx_nofilter_db = 10 * np.log10(pxx)
        
        if nofilter == True:
            extra_title.append('No filter')
        else:
            extra_title.append('Filtered')
        if conf['filter_stop'] == [] and conf['filter_order'] == []:
            x = x_raw
        else:
            x = signal.filtfilt(b=b_pass, a=a_pass, x=x_raw, padlen=3 * (max(len(a_pass), len(b_pass)) - 1))
            extra_title.append(f"Band-pass:{conf['filter_pass'][0]}-{conf['filter_pass'][1]} Hz.")
        f_filtered, pxx_filtered = signal.welch(fs=sr, x=x, window=signal.windows.barthann(n), noverlap=0, 
                                               return_onesided=True, detrend=False)
        if self.with_NoNotch:
            x_NoNotch = x
        pxx_db = 10 * np.log10(pxx_filtered)
        pxx_slideavg = signal.medfilt(volume=pxx_db, kernel_size=self.span_smooth)
        pxx_thr_db = pxx_slideavg + self.db_thr

        low_freqs = np.where(f_filtered <= min_freq_line)[0][-1]
        high_freqs = np.where(f_filtered >= max_freq_line)[0][0]
        pxx_thr_db[0:low_freqs + 1] = pxx_thr_db[low_freqs]
        pxx_thr_db[high_freqs:len(pxx_thr_db)] = pxx_thr_db[high_freqs]
        pow_comp = np.zeros(len(self.freqs_comp))
        for i, freq_comp in enumerate(self.freqs_comp):
            indf = np.where(f > freq_comp)[0]
            pow_comp[i] = pxx_slideavg[indf[0]]
        inds_freqs_search_notch = range(19, len(f) - 1)
        supra_thr = np.where(pxx_db[inds_freqs_search_notch] > pxx_thr_db[inds_freqs_search_notch])[0] + inds_freqs_search_notch[0]
        used_notches = np.array([])
        if supra_thr.size:
            max_amp4notch = max(pxx_db[inds_freqs_search_notch] - pxx_thr_db[inds_freqs_search_notch])
            temp_supra = np.where(np.diff(supra_thr) > 1)[0]

            inds_to_explore = np.zeros(shape=(supra_thr[temp_supra + 1].size + supra_thr[0].size))
            inds_to_explore[0] = supra_thr[0]
            inds_to_explore[1:len(inds_to_explore)] = supra_thr[temp_supra + 1]
            inds_to_explore = inds_to_explore.astype(int)
            bw_notches = []
            if not temp_supra.size:
                sample_above = np.array([supra_thr.size])
            else:
                diff = np.diff(temp_supra)
                last_value = supra_thr.size - np.where(supra_thr == inds_to_explore[-1])[0]
                sample_above = np.zeros(shape=(temp_supra[0].size + diff.size + last_value.size))
                sample_above[0] = temp_supra[0] + 1
                sample_above[1:sample_above.size - 1] = diff
                sample_above[-1] = last_value
                sample_above = sample_above.astype(int)
            for j, idx in enumerate(inds_to_explore):
                if sample_above[j] > 1:
                    iaux = max(pxx_db[inds_to_explore[j]:inds_to_explore[j] + sample_above[j]])
                    center_sample = np.mean(range(idx, idx + sample_above[j]))
                    ind_max = np.where(pxx_db == max(pxx_db[idx:idx + sample_above[j]]))[0]
                    if np.remainder(center_sample, 1) == 0.5 and (pxx_db[math.floor(center_sample)] > pxx_db[math.ceil(center_sample)]):
                        center_sample = math.floor(center_sample)
                    else:
                        center_sample = math.ceil(center_sample)
                    amp4notch = pxx_db[ind_max][0] - pxx_thr_db[ind_max][0]
                    used_notches = np.append(used_notches, f[center_sample])
                    bw_notches = np.append(bw_notches, (f[1] - f[0]) * sample_above[j] * 2 * amp4notch / max_amp4notch)

        info = np.array([])
        if conf['notch_q'] != [] or conf['notch_width'] != [] and supra_thr.size:
            z = np.array([])
            p = np.array([])
            k = 1
            for i, notch in enumerate(used_notches):
                w = notch / (sr / 2)
                if notch < 295:
                    max_bw = 3
                else:
                    max_bw = 5
                bw_notches[i] = min(max(bw_notches[i], 1), max_bw)
                bw = bw_notches[i] / (sr / 2)
                b_notch, a_notch = signal.iirnotch(w, w / bw)
                zi, pi, ki = signal.tf2zpk(b_notch, a_notch)
                k = k * ki
                z = np.append(z, zi)
                p = np.append(p, pi)
            if z.size:
                extra_title.append('#notches: ' + str(len(used_notches)) + '.')
                s = signal.zpk2sos(z, p, k)
                x_bf = x[:, np.newaxis]
                x = signal.sosfiltfilt(s, x_bf, padlen=3 * (s.shape[0]), axis=0)
                info = [channel, s, used_notches, bw_notches]
            f_filtered, pxx_filtered = signal.welch(fs=sr, x=x, window=signal.windows.barthann(n), noverlap=0, 
                                                   return_onesided=True, detrend=False, axis=0)
            
        naverages = math.floor(samples_spectrum / n)
        if conf['freqs_fit'] != []:
            flog_data1 = np.log(f)
            fit1 = self.logfit(f, pxx, conf['freqs_fit'][0], conf['freqs_fit'][1])
            fit2 = self.logfit(f, pxx, conf['freqs_fit'][1], conf['freqs_fit'][2])
            if len(conf['freqs_fit']) == 4:
                fit3 = self.logfit(f, pxx, conf['freqs_fit'][2], conf['freqs_fit'][3])
        log_info = ''
        if conf['freqs_fit'] != []:
            log_info = '%d-%d Hz: \u03B1 = %1.2f (r^2 = %1.2f)' % (conf['freqs_fit'][0], conf['freqs_fit'][1], fit1['k1'], fit1['r2'])
            log_info = log_info + '\n%d-%d Hz: \u03B1 = %1.2f (r^2 = %1.2f)' % (conf['freqs_fit'][1], conf['freqs_fit'][2], fit2['k1'], fit2['r2'])
            if len(conf['freqs_fit']) == 4:
                log_info = '%d-%d Hz: \u03B1 = %1.2f (r^2 = %1.2f)' % (conf['freqs_fit'][0], conf['freqs_fit'][1], fit1['k1'], fit1['r2'])
                log_info = log_info + '\n%d-%d Hz: \u03B1 = %1.2f (r^2 = %1.2f)' % (conf['freqs_fit'][1], conf['freqs_fit'][2], fit2['k1'], fit2['r2'])
                log_info = log_info + '\n%d-%d Hz: \u03B1 = %1.2f (r^2 = %1.2f)' % (conf['freqs_fit'][2], conf['freqs_fit'][3], fit3['k1'], fit3['r2'])
        thr = None
        if conf['threshold'] != []:
            sigma = np.median(np.abs(x)) / 0.6745
            thr = conf['threshold'] * sigma
            if self.with_NoNotch:
                sigma_NoNotch = np.median(np.abs(x_NoNotch)) / 0.6745
                thr_NoNotch = conf['threshold'] * sigma_NoNotch
                thr = thr_NoNotch
            extra_title.append(f"thr = sigma*{conf['threshold']} = " + "{:.2f}".format(thr) + f"{str(unit)}. Power at ")
            for i, freq in enumerate(self.freqs_comp):
                extra_title.append(f"{freq/1000}, ")
            extra_title.append('KHz = ')
            for i, pow in enumerate(pow_comp):
                if i == len(pow_comp) - 1:
                    extra_title.append("{:.2f} ".format(pow))
                else:
                    extra_title.append("{:.2f}, ".format(pow))
            extra_title.append('dB')
        title = f"Spectrum of ch {channel[0][0]} ({output_name}) session {self.session}. sr {sr} Hz ({naverages} averaged periodograms of length {n} with barthannwin)"
        subtitle = ''
        for extra in extra_title:
            subtitle = subtitle + str(extra)
        outputpath = self.direc_raw + '/output/' + f"spectrum_{channel[0][0]}_{label}"
        pxx_filtered_db = 10 * np.log10(pxx_filtered)
        
        return info, f, pxx_db, pxx_thr_db, used_notches, pxx_filtered_db, f_filtered, pxx_nofilter_db, x_raw, x, sr, samples_timeplot, title, outputpath, subtitle, log_info, thr, conf, str(unit)

    def logfit(self, freqs, s, f1, f2):
        power_fit = np.log10(s[np.where(freqs > f1)[0][0]:np.where(freqs >= f2)[0][0]])
        freqs_fit = np.log10(freqs[np.where(freqs > f1)[0][0]:np.where(freqs >= f2)[0][0]])
        p_all1 = np.polyfit(freqs_fit, power_fit, 1)
        k1 = p_all1[0]
        loga = p_all1[1]
        model = k1 * freqs_fit + loga
        r2 = max(0, 1 - sum((power_fit - model) ** 2) / sum((power_fit - np.mean(power_fit)) ** 2))
        my_dict = {'k1': k1, 'loga': loga, 'r2': r2}
        return my_dict
    
    def calculate_quality_metrics(self, x, x_raw, pxx, pxx_filtered):
        time_power_raw = np.sum([sample ** 2 for sample in x_raw])
        frequency_power_raw = np.sum([sample ** 2 for sample in pxx]) / len(pxx)
        time_power_filtered = np.sum([sample ** 2 for sample in x])
        frequency_power_filtered = np.sum([sample ** 2 for sample in pxx_filtered]) / len(pxx_filtered)
        time_power_raw_rms = np.sqrt(time_power_raw / len(x_raw))
        frequency_power_raw_rms = np.sqrt(frequency_power_raw / len(pxx))
        time_power_filtered_rms = np.sqrt(time_power_filtered / len(x))
        frequency_power_filtered_rms = np.sqrt(frequency_power_filtered / len(pxx_filtered))
        Q_raw = frequency_power_raw_rms / time_power_raw_rms
        Q_filtered = frequency_power_filtered_rms / time_power_filtered_rms
        Q_raw2filtered = frequency_power_filtered_rms / time_power_raw_rms

        return Q_raw, Q_filtered, Q_raw2filtered
    
    def coherence_calculation(self, n, x_filtered, y_filtered):
        freqs, coherence = signal.coherence(x=x_filtered.ravel(), y=y_filtered.ravel(), fs=30000, 
                                            window=signal.windows.barthann(n), noverlap=0, detrend=False)
        return freqs, coherence