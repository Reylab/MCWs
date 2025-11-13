import pyqtgraph as pg
import PyQt5 as qt
import numpy as np
import math
import scipy.signal as signal
import struct
import csv
import glob
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import pyqtgraph as pg
from PyQt5.QtGui import QPixmap
from scipy.io import loadmat
from core.parameter_functions.Parameters import par
import pyqtgraph.exporters
from PyQt5 import QtWidgets

class PlotBundles():
    def __init__(self) -> None:
        pass
    
    def plot(self, par, nsx_file, channels='all', notchfilter=0, filt_order=4, detect='neg', tmin=10):
        nsx = nsx_file
        bundles = np.unique(nsx['bundle'])
        sr = nsx['sr'].ravel()
        bundle_list = nsx['bundle'].ravel()
        is_micro = nsx['is_micro'].ravel()
        lts = nsx['lts'].ravel()
        chan_ID = nsx['chan_ID'].ravel()
        output_name = nsx['output_name'].ravel()
        ext = nsx['ext'].ravel()
        label = nsx['label'].ravel()
        conversion = nsx['conversion'].ravel()
        rec_length = 120
        
        # ========== CHANNEL FILTERING ==========
        if isinstance(channels, str) and channels.lower() == 'all':
            # Use all channels
            channel_mask = np.ones(len(chan_ID), dtype=bool)
            print(f"Plotting all {len(chan_ID)} channels")
        else:
            # Convert to list if single channel
            if isinstance(channels, int):
                channels = [channels]
            
            # Find indices of requested channels
            channel_mask = np.isin(chan_ID, channels)
            print(f"Requested channels: {channels}")
            print(f"Found {np.sum(channel_mask)} matching channels")
            
            # If no matches, warn and use all
            if np.sum(channel_mask) == 0:
                print(f"WARNING: No matching channels found. Available channels: {chan_ID}")
                print("Using all channels instead.")
                channel_mask = np.ones(len(chan_ID), dtype=bool)
        # ========================================
        
        app = QApplication([])
        
        for ibun, bundle in enumerate(bundles):
            if 'Mic' in bundle[0] or bundle == 'Photo_analog':
                continue
            
            # Get channels for this bundle, filtered by channel_mask
            idx_chan_all = np.nonzero(bundle_list == bundle)[0]
            idx_chan = idx_chan_all[channel_mask[idx_chan_all]]
            
            # Skip bundle if no channels match
            if len(idx_chan) == 0:
                print(f"Skipping bundle {bundle[0]} - no matching channels")
                continue
            
            print(f"Plotting bundle {bundle[0]} with {len(idx_chan)} channels: {chan_ID[idx_chan]}")
            
            main_window = QMainWindow()
            main_window.setGeometry(100, 60, 1920, 1080)
            central_widget = QtWidgets.QWidget()
            w = QtWidgets.QGridLayout()
            central_widget.setLayout(w)
            main_window.setCentralWidget(central_widget)
            
            posch = idx_chan[0]
            if is_micro[idx_chan[0]]:
                par.sr = 30000
                par.detect_fmin = 300
                par.detect_fmax = 3000
                par.auto = 0
                mVmin = 50
                w_pre = 20                       
                w_post = 44                     
                min_ref_per = 1.5                                    
                ref = np.floor(min_ref_per * par.sr / 1000)                  
                par.ref = ref
                factor_thr = 5
            elif sr[posch] != 30000:
                par.sr = 2000
                par.detect_fmax = 120
                par.detect_fmin = 1
                par.auto = 1
            
            folderName = os.path.dirname(os.path.dirname(__file__))
            label_item = QtWidgets.QLabel('%s. bundle %s. fmin %d Hz. fmax %d Hz' % 
                                         (folderName, bundle[0], par.detect_fmin, par.detect_fmax))
            w.addWidget(label_item, 0, 0)
            
            max_subplots = len(idx_chan)
            vmin = np.zeros(max_subplots)
            vmax = np.zeros(max_subplots)
            channel_text_macro = np.zeros(max_subplots)
            
            if lts[posch] < par.sr * tmin:
                print('tmin is smaller than the recording length')
            else:
                min_record = par.sr * tmin
            
            max_record = math.floor(min(lts[posch], min_record + par.sr * rec_length))
            tmax = int(max_record / par.sr)
            b, a = signal.ellip(N=filt_order, rp=0.1, rs=40, 
                               Wn=np.array([par.detect_fmin, par.detect_fmax]) * 2 / (par.sr), 
                               btype='bandpass')
            j = 0
            
            for ichan, posch in enumerate(idx_chan):
                title = ''
                channel1 = chan_ID[posch]
                
                if nsx['dc'].size:
                    dc = nsx['dc'].ravel()[posch]
                else:
                    dc = 0
                
                filepath = os.path.dirname(os.path.dirname(__file__)) + '/input/' + \
                          str(output_name[posch][0]) + str(ext[posch][0])
                f1 = open(filepath, 'rb')
                f1.seek((min_record - 1) * 2)
                binary_data = f1.read()
                format_string = '<' + 'h' * (len(binary_data) // 2)
                unpacked_data = struct.unpack(format_string, binary_data)
                x_raw = unpacked_data * conversion[posch] + dc
                x_raw = x_raw[0][0:max_record - min_record + 1]
                f1.close()

                if notchfilter:
                    filename = os.path.dirname(os.path.dirname(__file__)) + "/output/process_info.mat"
                    mat = loadmat(filename)
                    process_info = mat['process_info']
                    ch_ID = process_info['ch_Id'][0, 0]
                    idx = np.nonzero(ch_ID == channel1)[1]
                    s = signal.tf2sos(b, a)
                    try:
                        sos = process_info['sos'][0, 0][0, idx][0]
                        freqs_notch = process_info['freqs_notch'][0, 0][0]
                        BwHz_notch = process_info['BwHz_notch'][0, 0][0, idx][0][0]
                        sos = sos.copy(order='C')
                        sos = np.concatenate((sos, s))
                    except:
                        sos = []
                        freqs_notch = []
                        BwHz_notch = []
                        sos = s    
                    xd = signal.sosfiltfilt(sos, x_raw, padlen=3 * (len(sos)), axis=0)
                else:
                    xd = signal.filtfilt(b, a, x_raw, padlen=3 * (max(len(a), len(b)) - 1))
                
                if is_micro[posch]:
                    thr = factor_thr * np.median(np.abs(xd)) / 0.6745
                    thrmax = 10 * thr
                    vlim = mVmin
                
                # Plotting
                eje = np.linspace(tmin, tmax, len(xd))
                p = pg.PlotWidget()
                p.setBackground('w')
                p.plot(eje, xd, pen=pg.mkPen('b'))
                p.setXRange(tmin, tmax)
                ax = p.getAxis('left')
                vmin[j] = 1.05 * np.percentile(xd, 0.5)
                vmax[j] = 1.05 * np.percentile(xd, 99.5)
                
                p.setLabel(axis='left', text=f"Ch {chan_ID[posch]}")
                
                if is_micro[posch]:
                    match detect:
                        case 'pos':
                            xaux = np.nonzero((xd[w_pre + 1:len(xd) - w_post - 2] > thr) & 
                                            (np.abs(xd[w_pre + 1:len(xd) - w_post - 2]) < thrmax))[0] + w_pre + 1
                            p.addLine(y=thr, pen=pg.mkPen('r'))
                            if not par.auto:
                                p.setYRange(-vlim, 2 * vlim)
                        case 'neg':
                            xaux = np.nonzero((xd[w_pre + 1:len(xd) - w_post - 2] < -thr) & 
                                            (np.abs(xd[w_pre + 1:len(xd) - w_post - 2]) < thrmax))[0] + w_pre + 1
                            p.addLine(y=-thr, pen=pg.mkPen('r'))
                            if not par.auto:
                                p.setYRange(-2 * vlim, vlim)
                        case 'both':
                            xaux = np.nonzero((np.absolute(xd[w_pre + 1:len(xd) - w_post - 2]) > thr) & 
                                            (np.absolute(xd[w_pre + 1:len(xd) - w_post - 2]) < np.absolute(thrmax)))[0] + w_pre + 1
                            p.addLine(y=thr, pen=pg.mkPen('r'))
                            p.addLine(y=-thr, pen=pg.mkPen('r'))
                            if not par.auto:
                                p.setYRange(-2 * vlim, 2 * vlim)
                    
                    nspk = np.count_nonzero(np.diff(xaux) > ref) + 1
                    if (nspk > np.ceil(rec_length / 20) and nspk < rec_length * 60):
                        p.setLabel(axis='left', text=f"Ch {chan_ID[posch][0][0]}")
                    
                    if notchfilter == 0 or len(np.nonzero(ch_ID == channel1)) == 0:
                        title = '%s. %d spikes. thr = %2.2f' % (label[posch], nspk, thr)
                    elif len(np.nonzero(ch_ID == channel1)) > 0:
                        posch_notch = np.nonzero(ch_ID == channel1)[1]
                        if not posch_notch.size:
                            cant_notches1l = 0
                            cant_notches1h = 0
                        else:
                            cant_notches1h = np.count_nonzero((freqs_notch[posch_notch][0] > par.detect_fmin) & 
                                                             (freqs_notch[posch_notch][0] < par.detect_fmax))
                            cant_notches1l = np.count_nonzero(freqs_notch[posch_notch][0] < par.detect_fmin)
                        title = '%s. %d spikes. %d and %d notches applied below and above %d Hz' % \
                               (label[posch][0], nspk, cant_notches1l, cant_notches1h, par.detect_fmin) + title
                        p.setTitle(title)
                else:
                    if notchfilter == 1:
                        posch_notch = np.nonzero(ch_ID == channel1)[1]
                        if not posch_notch.size:
                            cant_notches1l = 0
                            cant_notches1h = 0
                        else:
                            cant_notches1l = np.sum(freqs_notch[posch_notch][0] < 200)
                            cant_notches1h = np.sum(freqs_notch[posch_notch][0] and freqs_notch[posch_notch][0] < 1000)
                        channel_text_macro[j] = '%s. %d and %d notches applied below and above %d Hz' % \
                                               (label[posch], cant_notches1l, cant_notches1h, 200) + title
                    else:
                        factor_thr = 5
                        thr = factor_thr * np.median(np.abs(xd)) / 0.6745
                        title = title + '%s. thr = %2.2f' % (label[posch], thr)
                
                p.setLabel(axis='bottom', text='Time (sec)')
                w.addWidget(p, ichan + 1, 0)

            if not is_micro[posch] and par.auto:
                textbox_string = str((tmax - tmin) / 3 + tmin)
                text2 = pg.TextItem(str(textbox_string))
                p.addItem(text2)
            
            if is_micro[posch]:
                outfile = '%s_%s_filtorder%dwithnothes%s' % (bundle[0], filt_order, notchfilter, detect)
            else:
                outfile = '%s_%s_filtorder%d_withnotches%s' % (bundle[0], filt_order, notchfilter, detect)

            outpath = folderName + '/output/' + outfile + '.png'
            pixmap = QPixmap(main_window.size())
            main_window.render(pixmap)
            pixmap.save(outpath)
            main_window.destroy()
            
            print(f"Saved: {outpath}")


if __name__ == '__main__':
    NSx_file_path = os.path.abspath(glob.glob("input/NSx.mat")[0])
    metadata = loadmat(NSx_file_path)
    nsx = metadata['NSx']
    param = par()
    plt = PlotBundles()
    
    # To select specific channels:
    specific_channels = 'all'  # Use all channels
    # specific_channels = [290, 291, 292, 293]  # Use specific channels
    # specific_channels = 290  # Use single channel
    # specific_channels = list(range(290, 301))  # Use range of channels
    
    plt.plot(nsx_file=nsx, par=param, channels=specific_channels, notchfilter=1)