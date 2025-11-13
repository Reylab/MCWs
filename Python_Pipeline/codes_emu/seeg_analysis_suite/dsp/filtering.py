# Author: Sunil Mathew
# Date: 10 Jan 2024
# Filtering class for signal processing
import traceback
import threading
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import pyqtgraph as pg


class Filtering():
    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(Filtering, cls).__new__(cls)
    #     return cls.instance
    
    def __init__(self) -> None:
        # Bandpass filter parameters
        self.lowcut = 300.0  # Lower cutoff frequency in Hz
        self.highcut = 3000.0  # Upper cutoff frequency in Hz
        self.filt_order = 4 # Order
        self.rp = 0.1 # 0.07 Maximum ripple in passband in dB
        self.rs = 40.0 # 20 Minimum attenuation in stopband in dB
        self.fs = 30000  # Sampling frequency in Hz
        self.waveform_length=40

        # Notch filter parameters
        self.basic_notches = [60, 120]
        self.notch_width = 1
        self.span_smooth = 21
        self.db_thr = 10 # dB
        self.init_bp_custom_filter()
        self.filt_lock = threading.Lock()

    #region Signal filtering (bandpass, notch, power spectrum)

    def init_bp_custom_filter(self):
        """
        Initialize the bandpass filter with custom parameters.
        """
        bOK = False
        self.k = 1 # Gain
        self.z = np.array([]) # Zeros
        self.p = np.array([]) # Poles

        for freq in self.basic_notches:
            b, a = signal.iirnotch(w0=freq, Q=freq/self.notch_width, fs=self.fs)
            z, p, k = signal.tf2zpk(b, a)
            self.z = np.append(self.z, z)
            self.p = np.append(self.p, p)
            self.k *= k

        # TBD: Compute order for a ellip bandpass filter based on the lowcut and highcut frequencies
        if self.lowcut > self.highcut:
            print('Lowcut frequency is greater than highcut frequency. Check the filter parameters.')
            return bOK
        if self.lowcut > self.fs or self.highcut > self.fs:
            print(f'Cutoff frequency {self.lowcut}-{self.highcut} is greater than sampling frequency of {self.fs}. Check the filter parameters.')
            return bOK
        Wn = [self.lowcut * 2 / self.fs, self.highcut * 2 / self.fs]
        z, p, k = signal.ellip(N=self.filt_order, rp=self.rp, rs=self.rs, 
                               Wn=Wn, btype='bandpass', analog=False, output='zpk') # Bandpass filter
        self.z = np.append(self.z, z)
        self.p = np.append(self.p, p)
        self.k *= k

        self.sos = signal.zpk2sos(z=self.z, p=self.p, k=self.k) # Second-order sections representation

        bOK = True
        return bOK
    
    def init_spk_detection_filter(self, filt_order=4, notches_info=None):
        """
        Initialize the bandpass filter for spike detection.
        """
        self.k = 1 # Gain
        self.z = np.array([]) # Zeros
        self.p = np.array([]) # Poles
        self.rp_det = 0.1 # 0.07 Maximum ripple in passband in dB
        self.rs_det = 40.0 # 20 Minimum attenuation in stopband in dB
        
        # TBD: Compute order for a ellip bandpass filter based on the lowcut and highcut frequencies
        if self.lowcut > self.highcut:
            print('Lowcut frequency is greater than highcut frequency. Check the filter parameters.')
            return
        Wn = [self.lowcut * 2 / self.fs, self.highcut * 2 / self.fs]
        self.sos = signal.ellip(N=filt_order, rp=self.rp_det, rs=self.rs_det, 
                                              Wn=Wn, btype='bandpass', analog=False, output='sos') # Bandpass filter
        # First value of first row of self.SOS is the gain
        self.k = self.sos[0, 0]
        # Divide the first three values of first row by the gain
        self.sos[0, 0:3] /= self.k
        if notches_info is not None:
            self.k *= notches_info['G']
            self.sos = np.vstack((notches_info['SOS'], self.sos))
            self.sos[0, 0:3] = self.sos[0, 0:3] * self.k
            self.spk_b, self.spk_a, self.zi, self.n_fact = self.compute_initial_conditions(self.sos)
        else:
            self.spk_b, self.spk_a = signal.ellip(N=filt_order, rp=self.rp_det, rs=self.rs_det,
                                                    Wn=Wn, btype='bandpass', analog=False, output='ba')

    def compute_initial_conditions(self, sos):
        b1, a1 = signal.sos2tf(sos)
        # Normalize b1
        max_b1 = np.max(np.abs(b1))
        # print(max_b1)
        b1 = b1 / max_b1
        # Last nonzero coefficient of b1
        b1_last_nonzero = np.max(np.nonzero(b1))
        # print(b1_last_nonzero)

        # Normalize a1
        max_a1 = np.max(np.abs(a1))
        # print(max_a1)
        a1 = a1 / max_a1
        # Last nonzero coefficient of a1
        a1_last_nonzero = np.max(np.nonzero(a1))

        # filter order n is maximum of the last nonzero coefficient of a1 and b1
        ord = max(b1_last_nonzero, a1_last_nonzero)
        # print(ord)
        n_fact = max(1, 3 * ord)
        # print(n_fact)

        a1 = sos[:, 3:6].T
        # print(b1)
        b1 = sos[:, 0:3].T
        # print(a1)

        # Compute initial conditions to remove DC offset at beginning and end of
        # filtered sequence.  Use sparse matrix to solve linear system for initial
        # conditions zi, which is the vector of states for the filter b(z)/a(z) in
        # the state-space formulation of the filter.
        L = b1.shape[1]  # Number of sections
        zi = np.zeros((2, L))  # Initialize zi with zeros

        for ii in range(L):
            rhs = (b1[1:3, ii] - b1[0, ii] * a1[1:3, ii])
            zi[:, ii] = np.linalg.solve(np.eye(2) - np.vstack((-a1[1:3, ii], [1, 0])).T, rhs)
        # for value in zi[0, :]:
        #     print(f"{value:.6f}", end=' ')
        return b1, a1, zi, n_fact
        
    def FilterX_fc(self, b, a, x, zi, direction='forward'):
        if direction == 'forward':
            y, zf = signal.lfilter(b, a, x, zi=zi)
        else:
            y, zf = signal.lfilter(b, a, x[::-1], zi=zi)
            y = y[::-1]
        return y, zf

    def ff_one_chan(self, x):
        """
        Filters the input signal x using the filter coefficients in b and a.
        The initial conditions of the filter are set to z.
        """
        L = self.spk_b.shape[1]
        for ii in range(L):
            b = self.spk_b[:, ii]
            a = self.spk_a[:, ii]
            zi = self.zi[:, ii]
            # Calculate xt
            xt = -x[self.n_fact:0:-1] + 2 * x[0]
            
            # Perform the first filter operation
            _, zo = self.FilterX_fc(b=b, a=a, x=xt, zi=zi * xt[0])
            yc2, zo = self.FilterX_fc(b=b, a=a, x=x, zi=zo)
            
            xt = -x[-2:-self.n_fact-2:-1] + 2 * x[-1]
            yc3, zo = self.FilterX_fc(b, a, xt, zo)
            
            # Reverse the sequence
            _, zo = self.FilterX_fc(b=b, a=a, x=yc3, zi=zi * yc3[-1], direction='reverse')
            x = self.FilterX_fc(b=b, a=a, x=yc2, zi=zo, direction='reverse')[0]
        
        return x

    def bandpass_filter(self, data):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data
    
    def notch_filter(self, data):
        filtered_data = signal.sosfilt(self.sos, data)
        return filtered_data
    
    def custom_bp_filter(self, data, channel=None, fs=30000):
        self.fs = fs
        bOK = self.init_bp_custom_filter()
        if not bOK:
            return None
        filtered_data = signal.sosfiltfilt(self.sos, data)
        return filtered_data
    
    def spk_detection_filter(self, data, channel, filt_order=4):
        try:
            self.filt_lock.acquire()
            notches_info = None
            if hasattr(self, 'notches_info'):
                ch_id = int(channel.split('_')[-1])
                for info in self.notches_info:
                    if info['chID'] == ch_id:
                        notches_info = info
                        break
            self.init_spk_detection_filter(notches_info=notches_info, filt_order=filt_order)
            if notches_info is not None:
                filtered_data = self.ff_one_chan(data)
            else:
                filtered_data = signal.filtfilt(self.spk_b, self.spk_a, data, 
                                                padtype = 'odd', padlen=3*(max(len(self.spk_b),len(self.spk_a))-1))
            return filtered_data
        except:
            print(f'Filtering failed for {channel}')
            print(traceback.print_exc())
        finally:
            self.filt_lock.release()
     
    def calculate_notches(self, pxx_db, f):
        """
        Calculates notches for every channel.
        """
        try:
        # Smooth the spectrum
            # pxx_db_smooth = signal.medfilt(pxx_db, self.span_smooth)
            pxx_db_smooth = gaussian_filter1d(pxx_db, sigma=50)

            # Add self.db_thr to smooth spectrum
            pxx_db_smooth += self.db_thr

            # Plot the smoothed spectrum
            
            self.psd_smooth_curve.setData(x=f, y=pxx_db_smooth)

            # Check if pxx_db is above pxx_db_smooth to detect notches
            notches = np.where(pxx_db > pxx_db_smooth, 1, 0)

            # Find amplitude of the largest notch
            notches_amp = pxx_db[notches == 1] - pxx_db_smooth[notches == 1]
            max_notch_idx = np.argmax(notches_amp)
            max_notch_amp = notches_amp[max_notch_idx]

            print(f'Max notch: {max_notch_amp:.2f} dB at {f[max_notch_idx]:.2f} Hz.')

            # Find the indices of the notches that are greater than half of the largest notch
            notches_idx = np.where(notches_amp > (max_notch_amp * 0.5))[0]

            # remove notch lines from before
            for line in self.notch_lines:
                self.psd_graph_1.removeItem(line)

            # Plot vertical dotted lines at the location of the notches
            for notch_idx in notches_idx:
                line = self.psd_graph_1.addLine(x=f[notch_idx], pen=pg.mkPen('k', width=1, style=pg.QtCore.Qt.DashLine))
                self.notch_lines.append(line)
        except:
            print(traceback.print_exc())

    #endregion Signal filtering (bandpass, notch, power spectrum)

    def remove_packet_loss_data(self, data):
        pak_lost = data == np.iinfo(np.int16).min
        data[pak_lost] = 0

        return data
    

if __name__ == '__main__':
    import os

    filtering = Filtering()
    filtered_data = filtering.spk_detection_filter(data=raw_data, channel=ch_name)