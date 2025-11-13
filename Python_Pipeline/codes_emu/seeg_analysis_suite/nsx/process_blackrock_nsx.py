import os
import concurrent.futures
import shutil

import traceback
import itertools
import numpy as np
import scipy.io
from scipy import signal, linalg
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from spclustering import SPC, plot_temperature_plot

import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.dockarea.Dock import Dock

import psutil
import subprocess
import sys
import threading
from multiprocessing import Process
import datetime
import time
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'neuroshare/pyns')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'brpylib')))
sys.path.append(os.path.dirname(__file__))

from brpylib import NsxFile
from nsfile import NSFile
from nsentity import EntityType 
from qtpy import QtGui, QtCore, QtWidgets
from config import config
from sequence_info import SequenceInfo
from ripple_map_file import RippleElectrodeInfo, RippleMapFile
from core.responses import ResponseViewer
from core.seeg_3d import SEEG_3D
from core.utils import get_elapsed_time, get_time_hrs_mins_secs_ms

class BlackrockNsx():
    def __init__(self) -> None:
        pass

    def read_brk_ns_file(self, filename):
        self.nsfile = NsxFile(filename)

        # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
        cont_data = self.nsfile.getdata(elec_ids='all', start_time_s='all', data_time_s='all', downsample=1, full_timestamps=True)

        # Close the nsx file now that all data is out
        # self.nsfile.close()

        # Plot the data channel
        seg_id = 0
        plot_chan = 1
        hdr_idx = cont_data["elec_ids"].index(plot_chan)
        ch_idx  = cont_data["elec_ids"].index(plot_chan)
        t = cont_data["data_headers"][seg_id]["Timestamp"] / cont_data["samp_per_s"]
        
        # self.clear_layout()
        pw = pg.PlotWidget()
        self.lyt.addWidget(pw)
        curve = pw.plot()
        curve.setPen(color='b', width=1.5)
        pw.setLabel('bottom', 'Time', units='s')
        pw.setXRange(0, t[-1])
        # pw.setTitle("Analog data for {0}".format(self.nsfile.extended_headers[hdr_idx]['ElectrodeLabel']), color='k', size='10pt')
        curve.setData(x=t, y=cont_data["data"][seg_id][ch_idx])


        # plt.plot(t, cont_data["data"][seg_id][ch_idx])
        # plt.axis([t[0], t[-1], min(cont_data["data"][seg_id][ch_idx]), max(cont_data["data"][seg_id][ch_idx])])
        # plt.locator_params(axis="y", nbins=20)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Output (" + self.nsfile.extended_headers[hdr_idx]['Units'] + ")")
        # plt.title(self.nsfile.extended_headers[hdr_idx]['ElectrodeLabel'])
        # plt.show()