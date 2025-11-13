# Author: Sunil Mathew
# Date: 10 Jan 2024
# Calculate power spectral density of the neural data
# based on codes_emu/cx_lfp.m

import traceback
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from qtpy import QtWidgets, QtCore
import pyqtgraph as pg

from core.config import config

class PowerSpectrum():

    def __init__(self, psd_view, c) -> None:
        self.psd_view = psd_view
        self.c = c

        # Power spectrum parameters
        self.fs = 30000
        self.freq_min = 300 # Hz
        self.freq_max = 3000 # Hz
        self.psd_1 = 10000
        self.psd_2 = 300
        self.init_fft_params()

        # Notch filter parameters
        # self.basic_notches = [60, 120]
        # self.notch_width = 1
        self.span_smooth = 21
        self.db_thr = 10 # dB

        self.init_power_spectrum_layout()

    def init_fft_params(self):
        self.fft_length = 2**np.ceil(np.log2(self.fs*2))/self.fs
        self.nfft = int(self.fft_length * self.fs)
        self.n_blocks = 5
        self.n_samples = self.n_blocks * self.fft_length * self.fs
        self.barthannwin = signal.windows.barthann(self.nfft)

    def init_power_spectrum_layout(self):
        """
        Initializes the power spectrum plots.
        """
        self.psd_widget = QtWidgets.QWidget()
        self.psd_layout = QtWidgets.QVBoxLayout()
        self.psd_widget.setLayout(self.psd_layout)

        # Controls
        self.psd_params_layout = QtWidgets.QHBoxLayout()
        self.psd_layout.addLayout(self.psd_params_layout)
        # Add a spacer
        self.psd_params_layout.addStretch()

        for param in config['psd']['cmb_params']:
            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(QtWidgets.QLabel(param))
            cmb = QtWidgets.QComboBox(self.psd_widget)
            cmb.setObjectName(f'cmb{param}')
            cmb.addItems(config['psd']['cmb_params'][param])
            cmb.currentIndexChanged.connect(self.update_psd_params)
            vLyt.addWidget(cmb)
            self.psd_params_layout.addLayout(vLyt)
        
        self.init_psd_view_spb()

        # Calculate notches button
        self.btn_calculate_notches = QtWidgets.QPushButton('Calculate Notches')
        self.btn_calculate_notches.clicked.connect(self.calculate_notches)
        self.psd_params_layout.addWidget(self.btn_calculate_notches)

        self.glw_psd = pg.GraphicsLayoutWidget()
        self.psd_layout.addWidget(self.glw_psd)
                
        # 0-10kHz
        self.psd_graph_1 = self.glw_psd.addPlot()
        self.psd_graph_1.showGrid(x=True, y=True, alpha=0.5)
        self.psd_graph_1.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_graph_1.setLabel('left', 'Power Spectral Density', units='dB/Hz')
        self.psd_graph_1.setTitle("Power Spectral Density", color='k', size='10pt')

        self.psd_curve_1 = self.psd_graph_1.plot() # psd
        self.psd_curve_1.setPen(color='b', width=1.5)
        self.psd_curve_2 = self.psd_graph_1.plot() # filtered psd
        self.psd_curve_2.setPen(color='r', width=1.5)

        self.glw_psd.nextRow()

        # 0-300 Hz
        self.psd_graph_2 = self.glw_psd.addPlot()
        self.psd_graph_2.showGrid(x=True, y=True, alpha=0.5)
        self.psd_graph_2.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_graph_2.setLabel('left', 'Power Spectral Density', units='dB/Hz')
        self.psd_graph_2.setTitle("Power Spectral Density", color='k', size='10pt')  

        self.psd_curve_300_1 = self.psd_graph_2.plot() # psd
        self.psd_curve_300_1.setPen(color='b', width=1.5)
        self.psd_curve_300_2 = self.psd_graph_2.plot() # filtered psd
        self.psd_curve_300_2.setPen(color='r', width=1.5)

        # db_thr line
        self.psd_smooth_curve = self.psd_graph_1.plot()
        self.psd_smooth_curve.setPen(color='g', width=1.5)

        self.notch_lines = []

        # glw_layout = self.glw.ci.layout
        # glw_layout.setColumnStretchFactor(0, 2)
        # glw_layout.setColumnStretchFactor(1, 1)

        self.psd_view.addWidget(self.psd_widget)

    def init_psd_view_spb(self):
        """
            Initialize spinboxes for the signal viewer. This is called by init_data
        """
        spb_params = config['psd']['spb_params']
        for param in spb_params:
            # Spinbox for clustering parameters, use the config values to figure out if doublespinbox or spinbox
            if isinstance(spb_params[param][0], float):                       
                spb = QtWidgets.QDoubleSpinBox(self.psd_widget)               
            else:
                spb = QtWidgets.QSpinBox(self.psd_widget)

            spb.setObjectName(f'spb_psd_{param}')
            spb.setRange(spb_params[param][1], spb_params[param][2])
            spb.setValue(spb_params[param][0])
            spb.setSingleStep(spb_params[param][3])
            spb.valueChanged.connect(self.update_psd_view_spb_params)
            lbl = QtWidgets.QLabel(param)

            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(lbl)
            vLyt.addWidget(spb)
            self.psd_params_layout.addLayout(vLyt)

    def update_psd_view_spb_params(self, value=None):
        """
         Update psd filt order params. This is called when the user changes the value of a spinbox
         
         Args:
         	 value: The value of the spinbox
        """
        try:
            spb = self.psd_widget.sender()
            param = spb.objectName().split('spb_psd_')[1]
            value = spb.value()
            setattr(self,param,value)
            self.c.sig_view_filt_param.emit(param, value)
        except:
            print(traceback.format_exc())

    def update_psd_params(self, idx):
        """
        Updates the power spectrum parameters based on user input.
        """
        cmb = self.psd_widget.sender()
        param = cmb.objectName().replace('cmb', '')
        value = cmb.currentText().split(' ')[0]
        if cmb.currentText().split(' ')[1] == 'Hz':
            value = int(value)
        elif cmb.currentText().split(' ')[1] == 'kHz':
            value = int(value) * 1000
        setattr(self, param, int(value))

        if param == 'psd_1':
            self.psd_graph_1.setXRange(0, self.psd_1)
        elif param == 'psd_2':
            self.psd_graph_2.setXRange(0, self.psd_2)

    def plot_power_spectrum_nsx(self, ch_label, data, filtered_data, fs):
        """
        Plots the power spectrum of a selected channel. 
        There are two plots: one with most of the frequency range(10kHz) and one with 0 to 300 Hz.
        """
        if fs != self.fs:
            self.fs = fs
            self.init_fft_params()
        
        f1, Pxx_den_1 = signal.welch(x=data, fs=fs, window=self.barthannwin, nfft=self.nfft)
        f2, Pxx_den_2 = signal.welch(x=filtered_data, fs=fs, window=self.barthannwin, nfft=self.nfft)

        pxx_db = 10*np.log10(Pxx_den_1) # Converted to dB/Hz from Watts/Hz
        pxx_db_filtered = 10*np.log10(Pxx_den_2)

        self.psd_curve_1.setData(x=f1, y=pxx_db) 
        self.psd_curve_2.setData(x=f2, y=pxx_db_filtered)
        self.psd_graph_1.setXRange(0, self.psd_1)
        self.psd_graph_1.setYRange(-50, 50)
        self.psd_graph_1.setTitle(f'Power Spectral Density {ch_label} (0-{self.psd_1} kHz)', color='k', size='10pt')
        

        # Get index of 300 - 3000 Hz
        # idx = np.where((f1 >= self.freq_min) & (f1 <= self.freq_max))
        idx = np.where((f1 <= self.freq_max))
        f1 = f1[idx]
        f2 = f2[idx]
        pxx_db = pxx_db[idx]
        pxx_db_filtered = pxx_db_filtered[idx]

        self.psd_curve_300_1.setData(x=f1, y=pxx_db) # Converted to dB/Hz from Watts/Hz
        self.psd_curve_300_2.setData(x=f2, y=pxx_db_filtered)
        self.psd_graph_2.setXRange(0, self.psd_2)    
        self.psd_graph_2.setYRange(-30, 50)          
        self.psd_graph_2.setTitle(f'Power Spectral Density {ch_label} (0-{self.psd_2} Hz)', color='k', size='10pt')

        idx = np.where(f1 >= self.freq_min)
        self.calculate_notches(pxx_db_filtered[idx], f2[idx])

    def plot_power_spectrum(self):
        """
        Plots the power spectrum of a selected channel. 
        There are two plots: one with most of the frequency range(10kHz) and one with 0 to 300 Hz.
        """
        if not self.psd:
            return

        ch_label = self.disp_chs[self.current_entity].label
        data = self.seq_info.data[self.current_entity]
        filtered_data = self.custom_bp_filter(data=data)
        
        f1, Pxx_den_1 = signal.welch(x=data, fs=self.fs, window=self.barthannwin, nfft=len(data))
        f2, Pxx_den_2 = signal.welch(x=filtered_data, fs=self.fs, window=self.barthannwin, nfft=len(data))

        pxx_db = 10*np.log10(Pxx_den_1) # Converted to dB/Hz from Watts/Hz
        pxx_db_filtered = 10*np.log10(Pxx_den_2)
        idx = np.where((f1 <= self.psd_1))
        f1 = f1[idx]
        f2 = f2[idx]
        pxx_db = pxx_db[idx]
        pxx_db_filtered = pxx_db_filtered[idx]

        self.psd_curve_1.setData(x=f1, y=pxx_db) 
        self.psd_curve_2.setData(x=f2, y=pxx_db_filtered)
        self.psd_graph_1.setXRange(0, self.psd_1)
        self.psd_graph_1.setTitle(f'Power Spectral Density {ch_label} (0-{self.psd_1} kHz)', color='k', size='10pt')
        

        # Get index of 300 - 3000 Hz
        # idx = np.where((f1 >= self.freq_min) & (f1 <= self.freq_max))
        idx = np.where((f1 <= self.psd_2))
        f1 = f1[idx]
        f2 = f2[idx]
        pxx_db = pxx_db[idx]
        pxx_db_filtered = pxx_db_filtered[idx]

        self.psd_curve_300_1.setData(x=f1, y=pxx_db) # Converted to dB/Hz from Watts/Hz
        self.psd_curve_300_2.setData(x=f2, y=pxx_db_filtered)
        self.psd_graph_2.setXRange(0, self.psd_2)              
        self.psd_graph_2.setTitle(f'Power Spectral Density {ch_label} (0-{self.psd_2} Hz)', color='k', size='10pt')

        idx = np.where(f1 >= self.freq_min)
        self.calculate_notches(pxx_db_filtered[idx], f2[idx])

    def calculate_notches(self, pxx_db=None, f=None):
        """
        Calculates notches for every channel.
        """
        try:
            if pxx_db is None or f is None:
                pxx_db = self.psd_curve_300_2.yData
                f = self.psd_curve_300_2.xData
            # Smooth the spectrum
            # pxx_db_smooth = signal.medfilt(pxx_db, self.span_smooth)
            pxx_db_smooth = gaussian_filter1d(pxx_db, sigma=50)

            # Add self.db_thr to smooth spectrum
            pxx_db_smooth += self.db_thr

            # Plot the smoothed spectrum
            
            self.psd_smooth_curve.setData(x=f, y=pxx_db_smooth)

            # Check if pxx_db is above pxx_db_smooth to detect notches
            notches = np.where(pxx_db > pxx_db_smooth, 1, 0)

            if not np.any(notches):
                print('No notches detected.')
                return

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