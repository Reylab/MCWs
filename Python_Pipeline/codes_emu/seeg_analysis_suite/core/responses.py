"""
Author: Sunil Mathew
Date: 05 December 2023

This class is for displaying the raster plot of the responses of the neurons for stimulus presentation.
As the experiment progresses the raster plot will be updated. Each stimulus will be shown multiple times 
and each row of the raster plot will represent the response of the neuron to that stimulus. The stimulus 
will be ranked based on a response metric and the top 40 stimuli will be shown along with its raster plot. 
At the end of the subscreening the user can select stimuli to be kept or removed for the next subscreening.

"""
import os
import threading
import shutil
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
import numpy as np
import traceback
import imageio
import time
from core.utils import get_elapsed_time, get_time_hrs_mins_secs_ms
from config import config
from scipy.signal.windows import gaussian

from utils import custom_symbol

import torch
import pandas as pd
from ai.rasternet import RasterDataset, RasterNet

import concurrent.futures

class Stimulus():
    def __init__(self, id=0, img=None, lbl=None) -> None:
        self.stim_img = img
        if lbl is None:
            self.lbl = 'img_00.jpg'
        else:
            self.lbl = lbl
        self.id = id
        self.ch_responses = {}
        self.channel = None
        self.cluster = None
        self.z_thr = 2.5
        self.raster_x_range = [-1000, 2000] # seconds

        self.sort_criteria = 'id' # id, rank, z_score, duration, onset

    def add_dummy_responses(self):
        """Add dummy responses for testing"""
        t = np.arange(0, 3, 1/30000, dtype=np.float32)
        
        for i in range(15):
            response = np.zeros_like(t)
            # randomly add spkikes to response
            spike_indices = np.random.randint(0, len(response), size=10)
            response[spike_indices] = 1
            self.responses.append(response)

    def set_channel(self, channel):
        self.channel = channel

    def set_cluster(self, cluster):
        self.cluster = cluster

    def get_img_item(self):
        if self.stim_img is None:
            return pg.ImageItem(np.random.rand(160, 160))
        else:
            return pg.ImageItem(self.stim_img)
        
    def get_short_lbl_txt(self):
        # remove extension
        lbl = os.path.splitext(self.lbl)[0]
        # if label is too long, truncate the middle part and put ellipsis
        if len(lbl) > 25:
            lbl = lbl[:15] + '...' + self.lbl[-7:]
        else:
            lbl = lbl

        return lbl

    def get_img_lbl(self):
        lbl = self.get_short_lbl_txt()
        return pg.LabelItem(lbl)

    def get_scr_id_lbl(self):
        return pg.LabelItem(f'SCR:{self.scr} ID:{self.id}')

    def get_ch_clus_lbl(self):
        return pg.LabelItem(f'CH:{self.channel} CLS:{self.cluster}')
    
    def get_rank(self):
        if len(self.ch_responses) == 0:
            return 0
        return self.ch_responses[self.channel].rank[self.cluster]
    
    def get_z_score(self):
        if len(self.ch_responses) == 0:
            return 0
        return self.ch_responses[self.channel].z_score[self.cluster]

    def get_rank_z_lbl(self):
        return pg.LabelItem(f'R:{self.get_rank()} Z:{self.get_z_score():.2f}')
    
    def get_rnet_score_lbl(self):
        # return pg.LabelItem(f'RNET:{self.get_rnet_score():.2f}')
        return pg.LabelItem(f'RNET:{self.get_rnet_prob():.0f}%')
    
    def get_rnet_score(self):
        if len(self.ch_responses) == 0:
            return 0
        return self.ch_responses[self.channel].rnet_score[self.cluster]
    
    def get_rnet_prob(self):
        if len(self.ch_responses) == 0:
            return 0
        return self.ch_responses[self.channel].keep[self.cluster][1] * 100
    
    def get_response_count(self):
        if len(self.ch_responses) == 0:
            return 0
        return len(self.ch_responses[self.channel].responses)
    
    def get_response_count_lbl(self):
        return pg.LabelItem(f'ID:{self.id} REP:{self.get_response_count()}')

    def __lt__(self, other):
        if self.sort_criteria == 'id':
            return self.id < other.id
        elif self.sort_criteria == 'rank':
            # return self.rank < other.rank
            return self.ch_responses[self.channel].rank[self.cluster] < other.ch_responses[self.channel].rank[other.cluster]
        elif self.sort_criteria == 'z_score':
            # return self.z_score < other.z_score
            return self.ch_responses[self.channel].z_score[self.cluster] < other.ch_responses[self.channel].z_score[other.cluster]
        elif self.sort_criteria == 'duration':
            # return self.duration < other.duration
            return self.ch_responses[self.channel].duration[self.cluster] < other.ch_responses[self.channel].duration[other.cluster]
        elif self.sort_criteria == 'onset':
            # return self.onset < other.onset
            return self.ch_responses[self.channel].onset[self.cluster] < other.ch_responses[self.channel].onset[other.cluster]
        elif self.sort_criteria == 'rnet':
            return self.ch_responses[self.channel].rnet_score[self.cluster] < other.ch_responses[self.channel].rnet_score[other.cluster]

class ChannelResponses():
    """
    Holds information for displaying one raster plots of responses to a stimulus in a single channel(electrode).
    """
    def __init__(self, c, channel, responses, clusters=None, cluster_labels=None, gauss_win=None) -> None:
        self.c = c
        self.channel = channel
        self.responses = responses # Contains spike trains for each time the stimulus was presented
        self.clusters = clusters
        self.cluster_labels = cluster_labels
        
        self.z_score = {}
        self.rank = {}        
        self.ifr = {}
        self.raster_x_range = [-1000, 2000] # seconds
        self.tons         = {}
        self.onset        = {}
        self.duration     = {}
        self.good_lat     = {}
        self.ifr_thr      = {}
        self.ifr_max      = {}
        self.p_value_sign = {}
        self.p_value_thr  = {}
        self.p_test       = {}
        self.median_post  = {}
        self.min_spk_test = {}

        self.keep = {}
        self.remove = {}
        self.meh = {}
        self.rnet_score = {}
        self.raster_class = {}
        self.initial_rank = {}

        if gauss_win is not None:
            for cluster in clusters:
                self.set_keep(cluster, 'none')
                self.compute_ifr(gauss_win=gauss_win, cluster=cluster)

    def compute_ifr(self, gauss_win, cluster='mu'):
        """Calculate the instantaneous firing rate of the responses"""
        self.ifr[cluster] = []
        n_trials = len(self.responses)
        if n_trials == 0:
            return
        
        # Ignore items that are not numpy arrays
        spikes = []
        for res_idx, response in enumerate(self.responses):
            if type(response) == np.ndarray:
                if cluster == 'mu':
                    spikes.append(response)
                else:
                    response = response[self.cluster_labels[res_idx] == int(cluster)]
                    spikes.append(response)
        
        if len(spikes) == 0:
            return
        spikes = np.concatenate(spikes)

        # Histogram of spikes
        spike_timeline = np.histogram(spikes, bins=np.arange(-1000, 2000, 1))[0]
        spike_timeline = spike_timeline / n_trials

        # Calculate instantaneous firing rate
        self.ifr[cluster] = np.convolve(spike_timeline, gauss_win, mode='same')

        self.compute_z_score_from_ifr(cluster=cluster)

    def compute_z_score_from_ifr(self, cluster):
        """Calculate the z-score based on instantaneous firing rate"""
        self.z_score[cluster] = 0.0
        self.onset[cluster] = 0
        self.duration[cluster] = 0
        if len(self.ifr[cluster]) == 0:
            return
        baseline_ifr = self.ifr[cluster][200:1000]
        mean_baseline_ifr = np.mean(baseline_ifr)
        std_baseline_ifr = np.std(baseline_ifr)
        if std_baseline_ifr == 0:
            return
        onset = self.onset[cluster] + 1000
        onset = onset if onset > 1000 else 1000
        duration = self.duration[cluster] if self.duration[cluster] > 300 else 300
        stim_ifr = self.ifr[cluster][onset:onset+duration]
        mean_stim_ifr = np.mean(stim_ifr)
        self.z_score[cluster] = (mean_stim_ifr - mean_baseline_ifr) / std_baseline_ifr

    def compute_onset_duration(self, cluster, ifr_thr):
        """
        Calculate onset and duration from ifr & ifr_thr
        """
        # Locals
        t_stim_onset = 1000 # ms (when stimulus was presented, full window is 3 seconds (-1000 to 2000))
        t_min_onset = 80 # ms 
        t_max_onset = 1000 # ms
        self.onset[cluster] = 0
        self.duration[cluster] = 0
        self.good_lat[cluster] = False

        if len(self.ifr[cluster]) == 0:
            return 
        
        # Get ROI
        ifr_roi = self.ifr[cluster][t_stim_onset:]
        ifr_roi = ifr_roi[t_min_onset:t_max_onset]

        # Get onset (first time ifr > ifr_thr)
        above_ifr_thr = np.where(ifr_roi > ifr_thr)[0]
        if above_ifr_thr.size == 0:
            return 
        
        self.onset[cluster] = above_ifr_thr[0] + t_min_onset

        # Check if above_ifr_thr is continuous for at least 10 ms
        above_ifr_thr_arr = np.split(above_ifr_thr, np.where(np.diff(above_ifr_thr) > 50)[0] + 1)

        if len(above_ifr_thr_arr) > 1:
            above_ifr_thr = above_ifr_thr_arr[0]

        # Get duration of how long ifr > ifr_thr
        self.duration[cluster] = above_ifr_thr[-1] - above_ifr_thr[0]

        # Check if the latency is good
        self.good_lat[cluster] = self.duration[cluster] > 0 and \
                                 self.duration[cluster] < 800
        
    def compute_p_value(self, cluster):
        """
        Compute p-value for the response
        """
        self.p_value_sign[cluster] = 0
        self.p_value_thr[cluster] = 0
        self.p_test[cluster] = False
        self.median_post[cluster] = 0
        self.min_spk_test[cluster] = False
        self.rank[cluster] = 0

        if len(self.ifr[cluster]) == 0:
            return
        
    def set_keep(self, cluster, choice, prob=0.0):
        self.raster_class[cluster] = choice
        if choice == 'good':
            self.keep[cluster]   = (True,  prob)
            self.remove[cluster] = (False, prob)
            self.meh[cluster]    = (False, prob)
            self.rnet_score[cluster] = 1 - prob
        elif choice == 'bad':
            self.keep[cluster]   = (False, prob)
            self.remove[cluster] = (True,  prob)
            self.meh[cluster]    = (False, prob)
            self.rnet_score[cluster] = 1 - prob + 2
        elif choice == 'meh':
            self.keep[cluster]   = (False, prob)
            self.remove[cluster] = (False, prob)
            self.meh[cluster]    = (True,  prob)
            self.rnet_score[cluster] = 1 - prob + 1
        else:
            self.keep[cluster]   = (False, prob)
            self.remove[cluster] = (False, prob)
            self.meh[cluster]    = (False, prob)
            self.rnet_score[cluster] = 1 - prob + 2

    def set_rnet_score(self, cluster, score=0.0):
        self.rnet_score[cluster] = score

    def get_responses_dl(self, cluster='mu', raster_size=150):
        """
        Converts the raster into a n_trials x raster_size array for deep learning
        """
        dl_resol = 3000 / raster_size
        resp_dl = []
        if cluster == 'mu':
            raster = self.responses
        else:
            if len(cluster) > 4:
                cluster = int(cluster[-1])
            
            raster = np.empty_like(self.responses)
            for idx, resp in enumerate(self.responses):
                if type(resp) == np.ndarray:
                    resp = resp[self.cluster_labels[idx] == cluster]
                # else:
                #     resp = np.empty()
                raster[idx] = resp
                    

        for resp in raster:
            response = np.zeros(raster_size)
            if type(resp) == np.ndarray:
                for spike in resp:
                    spike_idx = int((spike+1000)/dl_resol) - 1
                    response[spike_idx] = 1
            elif resp == None:
                pass
            else:
                spike_idx = int((resp+1000)/dl_resol) - 1
                if spike_idx < raster_size and spike_idx >= 0:
                    response[spike_idx] = 1
            resp_dl.append(response)
        return resp_dl
    
    def get_responses(self, cluster='mu'):
        """
        Converts the raster into a n_trials x raster_size array for deep learning
        """
        if cluster == 'mu':
            raster = self.responses
        else:
            if len(cluster) > 4:
                cluster = int(cluster[-1])
            
            raster = np.empty_like(self.responses)
            for idx, resp in enumerate(raster):
                if type(resp) == np.ndarray:
                    resp = resp[self.cluster_labels[idx] == cluster]
                raster[idx] = resp

        return raster

class SingleNeuronResponses():
    """
    Holds information for displaying a raster plot with 
    response metrics for a single neuron to a stimulus shown multiple times.
    """
    def __init__(self, c, responses, gauss_win) -> None:
        self.c = c
        self.responses = responses # Contains spike trains for each time the stimulus was presented
        
        self.z_score = -1
        self.rank = -1     
        self.ifr = []
        self.raster_x_range = [-1000, 2000] # seconds
        self.tot_x = 3000 # 3 seconds
        self.dl_size = 150
        self.dl_resol = self.tot_x / self.dl_size
        self.tons         = -1
        self.onset        = -1
        self.duration     = -1
        self.good_lat     = False
        self.ifr_thr      = -1
        self.ifr_max      = -1
        self.p_value_sign = -1
        self.p_value_thr  = -1
        self.p_test       = False
        self.median_post  = -1
        self.min_spk_test = False

        self.keep = False
        self.remove = False
        self.meh = False

        self.compute_ifr(gauss_win=gauss_win)

    def compute_ifr(self, gauss_win):
        """Calculate the instantaneous firing rate of the responses"""
        self.ifr = []
        n_trials = len(self.responses)
        if n_trials == 0:
            return
        
        # Ignore items that are not numpy arrays
        spikes = []
        for response in self.responses:
            spikes.append(response)
        
        if len(spikes) == 0:
            return
        spikes = np.concatenate(spikes)

        # Histogram of spikes
        spike_timeline = np.histogram(spikes, bins=np.arange(-1000, 2000, 1))[0]
        spike_timeline = spike_timeline / n_trials

        # Calculate instantaneous firing rate
        self.ifr = np.convolve(spike_timeline, gauss_win, mode='same')

        self.compute_z_score_from_ifr()

    def compute_z_score_from_ifr(self):
        """Calculate the z-score based on instantaneous firing rate"""
        self.z_score = 0.0
        self.onset = 0
        self.duration = 0
        if len(self.ifr) == 0:
            return
        baseline_ifr = self.ifr[200:1000]
        mean_baseline_ifr = np.mean(baseline_ifr)
        std_baseline_ifr = np.std(baseline_ifr)
        if std_baseline_ifr == 0:
            return
        onset = self.onset + 1000
        onset = onset if onset > 1000 else 1000
        duration = self.duration if self.duration > 300 else 300
        stim_ifr = self.ifr[onset:onset+duration]
        mean_stim_ifr = np.mean(stim_ifr)
        self.z_score = (mean_stim_ifr - mean_baseline_ifr) / std_baseline_ifr

    def compute_onset_duration(self, ifr_thr):
        """
        Calculate onset and duration from ifr & ifr_thr
        """
        # Locals
        t_stim_onset = 1000 # ms (when stimulus was presented, full window is 3 seconds (-1000 to 2000))
        t_min_onset = 80 # ms 
        t_max_onset = 1000 # ms
        self.onset = 0
        self.duration = 0
        self.good_lat = False

        if len(self.ifr) == 0:
            return 
        
        # Get ROI
        ifr_roi = self.ifr[t_stim_onset:]
        ifr_roi = ifr_roi[t_min_onset:t_max_onset]

        # Get onset (first time ifr > ifr_thr)
        above_ifr_thr = np.where(ifr_roi > ifr_thr)[0]
        if above_ifr_thr.size == 0:
            return 
        
        self.onset = above_ifr_thr[0] + t_min_onset

        # Check if above_ifr_thr is continuous for at least 10 ms
        above_ifr_thr_arr = np.split(above_ifr_thr, np.where(np.diff(above_ifr_thr) > 50)[0] + 1)

        if len(above_ifr_thr_arr) > 1:
            above_ifr_thr = above_ifr_thr_arr[0]

        # Get duration of how long ifr > ifr_thr
        self.duration = above_ifr_thr[-1] - above_ifr_thr[0]

        # Check if the latency is good
        self.good_lat = self.duration > 0 and \
                                 self.duration < 800
        
    def compute_p_value(self, cluster):
        """
        Compute p-value for the response
        """
        self.p_value_sign = 0
        self.p_value_thr = 0
        self.p_test = False
        self.median_post = 0
        self.min_spk_test = False
        self.rank = 0

        if len(self.ifr) == 0:
            return
        
    def set_keep(self, choice):
        if choice == 'keep':
            self.keep = True
            self.remove = False
            self.meh = False
        elif choice == 'remove':
            self.keep = False
            self.remove = True
            self.meh = False
        elif choice == 'meh':
            self.keep = False
            self.remove = False
            self.meh = True
        else:
            self.keep = False
            self.remove = False
            self.meh = False

    def get_responses_dl(self):
        """
        Converts the raster into a n_trials x raster_size array for deep learning
        """
        resp_dl = []

        for resp in self.responses:
            response = np.zeros(self.dl_size)
            if type(resp) != np.ndarray:
                spike_idx = int((resp+1000)/self.dl_resol) - 1
                if spike_idx < self.dl_size and spike_idx >= 0:
                    response[spike_idx] = 1
            else:
                for spike in resp:
                    spike_idx = int((spike+1000)/self.dl_resol) - 1
                    response[spike_idx] = 1
            resp_dl.append(response)
        return resp_dl

class ResponseViewer():
    def __init__(self, responses_view, c, 
                 fs=30000, ifr_win_res_ms=1,
                 img_paths=None, channels=None, 
                 study_info=None,
                 rows=3, cols=4, sort_reverse=False) -> None:
        self.responses_view = responses_view
        self.study_info = study_info
        self.init_responses_layout()

        self.rasternet_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai/rasternet'))
        self.raster_size_px = 150
        self.fs = fs
        self.ifr_thr = 0
        self.ifr_max = {}
        self.ifr_win_res_ms = ifr_win_res_ms
        self.t_scale = 1
        self.responses_view = responses_view
        self.c = c
        self.rows = rows
        self.cols = cols
        self.sort_criteria = 'id'
        self.sort_reverse = sort_reverse
        self.asc = True
        self.b_make_stat_plots = False
        self.image_names = []
        self.stimulus_list = []
        self.rasters = None
        self.cluster_count = 6
        self.clus_colors = [pg.intColor(i, hues=self.cluster_count) for i in range(self.cluster_count)]
        self.clus_info_dict = {} # Clustering info for a channel
        # self.t = np.arange(0, 3, 1/self.fs, dtype=np.float32)
        self.t = np.arange(0, 150, 1)
        self.channels = channels
        self.channels_short = []
        # self.init_stimulus_list(img_paths=img_paths)
        self.init_ifr_params()
        self.load_rasternet()
        self.load_existing = True
        self.stored_labels = {}

    def init_ifr_params(self):
        self.alpha_gauss = 3.035
        self.sigma_gauss = 10
        self.sample_period_gauss = 1000 / self.fs # Since spikes are in ms

        half_width_gauss = self.alpha_gauss * self.sigma_gauss
        n_gauss_points = 2 * round(half_width_gauss / self.sample_period_gauss) + 1

        stdev = (n_gauss_points - 1) / (2 * self.alpha_gauss)

        self.gauss_window = gaussian(n_gauss_points, std=stdev)
        # Reduce resolution to 1 ms
        self.gauss_window = self.gauss_window[::int(self.ifr_win_res_ms / self.sample_period_gauss)]
        self.gauss_window = self.gauss_window / np.sum(self.gauss_window) * 1000 / self.ifr_win_res_ms  # Hz

        

        print(f'Gauss window size: {len(self.gauss_window)}')

    def read_image(self, file):
        return imageio.imread(file)

    def add_stimulus(self, id, img, lbl):
        """Add a stimulus to the list"""
        # img = self.read_image(img)
        self.stimulus_list.append(Stimulus(id=id, img=img, lbl=lbl))

    def update_stimulus_raster(self, img_name, responses):
        """Update the stimulus with the given label"""
        for stim in self.stimulus_list:
            if stim.lbl == img_name:
                for response in responses:
                    stim.add_response(response=response)
                break

    def init_stimulus_list(self, img_ids, img_paths=None):
        """Initialize the UI"""
        try:
            if len(self.stimulus_list) > 0:
                self.stimulus_list = []

            if img_paths is None:
                return
                # for i in range(self.rows*self.cols):
                #     self.stimulus_list.append(Stimulus())
            else:
                for id, img_path in zip(img_ids, img_paths):
                    img = self.read_image(img_path)
                    self.add_stimulus(id=id, img=img, lbl=os.path.basename(img_path))

            self.create_stimulus_list_ui()
        except:
            print(traceback.format_exc())
            self.c.log.emit(traceback.format_exc())

    def create_stimulus_raster_ui(self, stim, stim_idx, row, col, 
                                  lyt_info_row_span, lyt_raster_row_span, 
                                  lyt_avg_response_row_span, stim_row_span, stim_col_span):
        """Create the stimulus UI.

        Raises:
            Exception: If an error occurs during the create process, the exception is caught and logged.

        """
        try:
            # Create a layout for the stimulus
            lyt_stim = self.glw_responses.addLayout(border=(50, 0, 0), row=row * stim_row_span, 
                                                    col=col * stim_col_span,
                                                    rowspan=stim_row_span, colspan=stim_col_span)
            lyt_stim.setSpacing(0)

            # Add layout for stimulus information
            lyt_info = lyt_stim.addLayout(row=0, col=0, rowspan=lyt_info_row_span, colspan=stim_col_span)
            lyt_info.setSpacing(0)

            # Add stimulus image file name
            lyt_info.addItem(stim.get_img_lbl(), row=3, col=0, rowspan=1, colspan=stim_col_span)

            # Add a combobox for channel
            cmb_res_ch = QtWidgets.QComboBox()
            cmb_res_ch.setStyleSheet('QComboBox {font-size: 8pt;}')
            proxy = QtWidgets.QGraphicsProxyWidget()
            proxy.setWidget(cmb_res_ch)
            lyt_info.addItem(proxy, row=1, col=6, colspan=3)

            # Add a combobox for cluster
            cmb_res_clus = QtWidgets.QComboBox()
            cmb_res_clus.addItem('mu')
            cmb_res_clus.setStyleSheet('QComboBox {font-size: 8pt;}')
            proxy = QtWidgets.QGraphicsProxyWidget()
            proxy.setWidget(cmb_res_clus)
            lyt_info.addItem(proxy, row=2, col=6, colspan=3)

            # Add rank and z-score
            lyt_info.addItem(stim.get_rank_z_lbl(), row=0, col=0, rowspan=1, colspan=2)

            # Add RasterNet score
            lyt_info.addItem(stim.get_rnet_score_lbl(), row=2, col=0, rowspan=1, colspan=2)

            # Add stimulus image
            vb = lyt_info.addViewBox(row=0, col=3, rowspan=3, colspan=3, lockAspect=True, invertY=True)
            vb.addItem(stim.get_img_item())
            vb.autoRange()

            # Add checkbox for keep
            proxy_k = QtWidgets.QGraphicsProxyWidget()
            rb_keep = QtWidgets.QCheckBox('✔️')
            rb_keep.setObjectName('keep;' + str(stim_idx) + ';' + stim.lbl + ';' + str(stim.channel) + ';' + str(
                stim.cluster))
            if len(stim.ch_responses) > 0:
                rb_keep.setChecked(stim.ch_responses[stim.channel].keep[stim.cluster][0])
            rb_keep.stateChanged.connect(self.mark_raster)
            proxy_k.setWidget(rb_keep)
            lyt_info.addItem(proxy_k, row=0, col=6, rowspan=1, colspan=1)

            # Add checkbox for meh
            proxy_m = QtWidgets.QGraphicsProxyWidget()
            rb_meh = QtWidgets.QCheckBox('~')
            rb_meh.setObjectName('meh;' + str(stim_idx) + ';' + stim.lbl + ';' + str(stim.channel) + ';' + str(
                stim.cluster))
            if len(stim.ch_responses) > 0:
                rb_meh.setChecked(stim.ch_responses[stim.channel].meh[stim.cluster][0])
            rb_meh.stateChanged.connect(self.mark_raster)
            proxy_m.setWidget(rb_meh)
            lyt_info.addItem(proxy_m, row=0, col=7, rowspan=1, colspan=1)

            proxy_r = QtWidgets.QGraphicsProxyWidget()
            rb_remove = QtWidgets.QCheckBox('✖️')
            rb_remove.setObjectName('remove;' + str(stim_idx) + ';' + stim.lbl + ';' + str(stim.channel) + ';' + str(
                stim.cluster))
            if len(stim.ch_responses) > 0:
                rb_remove.setChecked(stim.ch_responses[stim.channel].remove[stim.cluster][0])
            rb_remove.stateChanged.connect(self.mark_raster)
            proxy_r.setWidget(rb_remove)
            lyt_info.addItem(proxy_r, row=0, col=8, rowspan=1, colspan=1)
            
            # Add responses count label
            lyt_info.addItem(stim.get_response_count_lbl(), row=1, col=0, rowspan=1, colspan=2)
            lyt_info.setSpacing(0)
            lyt_info.setContentsMargins(2, 2, 2, 0) # left, top, right, bottom
            lyt_info.setMinimumHeight(100)

            lyt_stim.nextRow()

            # TODO: Add raster plot
            lyt_raster = lyt_stim.addLayout(row=lyt_info_row_span, col=0, rowspan=lyt_raster_row_span, colspan=stim_col_span)
            # lyt_raster.setSpacing(0)
            lyt_raster.setContentsMargins(0, 0, 0, 0)
            lyt_raster.setSpacing(0)
            lyt_raster.setMinimumHeight(100)
            
            raster_plot = lyt_raster.addPlot(row=0, col=0)  # rowspan=len(stim.responses))

            self.plot_raster(raster_plot, stim)

            lyt_stim.nextRow()

            lyt_avg_response = lyt_stim.addLayout(row=lyt_info_row_span + lyt_raster_row_span, col=0,
                                                    rowspan=lyt_avg_response_row_span, colspan=stim_col_span)
            lyt_avg_response.setSpacing(0)
            lyt_avg_response.setContentsMargins(0, 0, 0, 0)
            lyt_avg_response.setMinimumHeight(90)
            plt_ifr = lyt_avg_response.addPlot()
            self.plot_ifr(plt_ifr, stim, raster_plot)

            # Ensure that lyt_stim does not shrink vertically
            lyt_stim.setMinimumHeight(lyt_stim.minimumHeight()*1)
            lyt_stim.setMaximumHeight(lyt_stim.minimumHeight()*2)

            # Keep width of raster plot constant & such that it fits the screen (screewn width / cols)
            lyt_stim.setMinimumWidth(270)
            lyt_stim.setMaximumWidth(270)
        except:
            print(traceback.format_exc())
            self.c.log.emit(traceback.format_exc())

    def plot_raster(self, raster_plot, stim):
        if len(stim.ch_responses) == 0:
            return
        
        channel = stim.channel
        responses = stim.ch_responses[channel].responses
        cluster_labels = stim.ch_responses[channel].cluster_labels
        n_responses = max(len(responses), 6)
        space_between = 20

        raster_plot.hideAxis('left')
        raster_plot.hideAxis('bottom')
        raster_plot.setRange(xRange=stim.raster_x_range, yRange=[0, n_responses * space_between])
        raster_plot.disableAutoRange()

        y_positions = np.arange(n_responses - 1, -1, -1) * space_between

        if stim.cluster == 'mu':
            for res_idx, (res, labels) in enumerate(zip(responses, cluster_labels)):
                if isinstance(res, np.ndarray):
                    y = np.full_like(res, y_positions[res_idx])
                    raster_plot.plot(x=res, y=y, pen=None, symbol=custom_symbol('|'),
                                    symbolPen=labels, skipFiniteCheck=True)
        else:
            if len(stim.cluster) > 4:
                cluster = int(stim.cluster[-1])
            else:
                cluster = int(stim.cluster)
            for res_idx, (res, labels) in enumerate(zip(responses, cluster_labels)):
                if isinstance(res, np.ndarray):
                    mask = labels == cluster
                    clus_responses = res[mask]
                    colors = np.array(self.clus_colors)[labels[mask].astype(int)]
                    y = np.full_like(clus_responses, y_positions[res_idx])
                    raster_plot.plot(x=clus_responses, y=y, pen=None, symbol=custom_symbol('|'),
                                    # symbolPen=colors, 
                                    symbolPen='k', 
                                    skipFiniteCheck=True)

        rgn = pg.LinearRegionItem([0.2 * self.t_scale, 0.7 * self.t_scale])
        raster_plot.addItem(rgn)

    def plot_ifr(self, plt_ifr, stim, raster_plot):
        if len(stim.ch_responses) == 0:
            return
        
        channel = stim.channel
        cluster = stim.cluster
        ch_response = stim.ch_responses[channel]
        
        ifr = ch_response.ifr[cluster]
        onset = ch_response.onset[cluster]
        duration = ch_response.duration[cluster]
        ifr_thr = ch_response.ifr_thr[cluster]
        ifr_max = self.ifr_max[channel][cluster]
        good_lat = ch_response.good_lat[cluster]
        
        plt_ifr.clear()
        plt_ifr.hideAxis('left')
        
        # Pre-calculate x values
        x = np.arange(-1000, -1000 + len(ifr))
        
        # Plot IFR
        plt_item = plt_ifr.plot(x, ifr, pen=pg.mkPen('r', width=2))
        
        # Add horizontal line for IFR threshold
        h_line_ifr_thr = pg.InfiniteLine(
            pos=ifr_thr, 
            angle=0, 
            pen=pg.mkPen('k', width=1, style=QtCore.Qt.DashLine),
            label='{value:0.1f} Hz', 
            labelOpts={'position': 0.1, 'color': (0,0,200), 'movable': True}
        )
        plt_ifr.addItem(h_line_ifr_thr)
        
        # Add vertical lines at 0 and 1000
        for pos in (0, 1000):
            v_line = pg.InfiniteLine(pos=pos, angle=90, pen=pg.mkPen('k', width=1, style=QtCore.Qt.DashLine))
            plt_ifr.addItem(v_line)
        
        # Add onset and duration region
        rgn = pg.LinearRegionItem([onset, onset + duration], brush=(0, 255, 255, 100), movable=False)
        plt_ifr.addItem(rgn)
        
        # Set title
        title = f'Onset: {onset:.1f} Duration: {duration:.1f}'
        title_style = {'size': '8pt'}
        if good_lat:
            title_style.update({'color': 'm', 'bold': True})
        plt_ifr.setTitle(title, **title_style)
        
        # Link x axis and set y range
        plt_ifr.setXLink(raster_plot)
        plt_ifr.setRange(yRange=[0, ifr_max])
        
        # Add onset line to raster plot
        v_line_onset = pg.InfiniteLine(pos=onset, angle=90, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
        raster_plot.addItem(v_line_onset)

    def update_stimulus_raster_ui(self, stim, lyt_stim, lyt_info_row_span, lyt_raster_row_span):
        """Update the stimulus UI."""
        try:
            # Cache frequently accessed data
            channel = stim.channel
            cluster = stim.cluster
            ch_response = stim.ch_responses[channel]

            # Get the layout for the stimulus information
            lyt_info = lyt_stim.layout.itemAt(0, 0)

            # Update UI elements
            self._update_labels(lyt_info, stim, ch_response, cluster)
            self._update_comboboxes(lyt_info, stim, ch_response)
            self._update_checkboxes(lyt_info, stim, ch_response, cluster)
            self._update_stimulus_image(lyt_info, stim)

            # Update plots
            self._update_plots(lyt_stim, lyt_info_row_span, lyt_raster_row_span, stim)

        except:
            print(traceback.format_exc())

    def _update_labels(self, lyt_info, stim, ch_response, cluster):
        # Update image file name label
        lyt_info.layout.itemAt(3, 0).setText(stim.get_short_lbl_txt())

        # Update rank and z-score label
        rank_z_lbl = lyt_info.layout.itemAt(0, 0)
        rank = ch_response.rank[cluster]
        z_score = ch_response.z_score[cluster]
        
        conditions = [
            ch_response.p_test[cluster],
            z_score > stim.z_thr,
            ch_response.min_spk_test[cluster],
            ch_response.good_lat[cluster]
        ]
        
        if all(conditions):# or (all(conditions[:-1]) and ch_response.good_lat[cluster]):
            rank_z_lbl.setText(f'R:{rank}* Z:{z_score:.2f}', size='8pt', bold=True, color='b')
        else:
            rank_z_lbl.setText(f'R:{rank} Z:{z_score:.2f}', size='8pt', bold=False, color='k')

        # Update response count label
        lyt_info.layout.itemAt(1, 0).setText(f'ID:{stim.id} REP:{len(ch_response.responses)}', size='8pt')

        # Update RasterNet score label
        # lyt_info.layout.itemAt(2, 0).setText(f'RNET:{ch_response.rnet_score[cluster]:.2f}', size='8pt')
        try:
            lyt_info.layout.itemAt(2, 0).setText(f"RNET:{ch_response.keep[cluster][1]*100:.0f}%", size='8pt')
        except:
            print(f'update labels: {stim.lbl} {stim.channel} {stim.cluster}')

    def _update_comboboxes(self, lyt_info, stim, ch_response):
        # Update channel combobox
        cmb_res_ch = lyt_info.layout.itemAt(1, 6).widget()
        cmb_res_ch.clear()
        cmb_res_ch.addItems(self.channels_short)
        cmb_res_ch.setCurrentText(stim.channel.split()[0])

        # Update cluster combobox
        cmb_res_clus = lyt_info.layout.itemAt(2, 6).widget()
        cmb_res_clus.clear()
        cmb_res_clus.addItems(['mu'] + ch_response.clusters)
        cmb_res_clus.setCurrentText(stim.cluster)

    def _update_checkboxes(self, lyt_info, stim, ch_response, cluster):
        for idx, attr in enumerate(['keep', 'meh', 'remove']):
            checkbox = lyt_info.layout.itemAt(0, 6 + idx).widget()
            checkbox.setObjectName(f'{attr};' + str(stim.id) + ';' + stim.lbl + ';' + str(stim.channel) + ';' + str(
                stim.cluster))
            try:
                checkbox.stateChanged.disconnect(self.mark_raster)
            except:
                pass
            checkbox.setChecked(stim.ch_responses[stim.channel].__getattribute__(attr)[stim.cluster][0])
            checkbox.stateChanged.connect(self.mark_raster)

    def _update_stimulus_image(self, lyt_info, stim):
        vb = lyt_info.layout.itemAt(0, 3)
        vb.clear()
        vb.addItem(stim.get_img_item())
        vb.autoRange()

    def _update_plots(self, lyt_stim, lyt_info_row_span, lyt_raster_row_span, stim):
        # Update raster plot
        lyt_raster = lyt_stim.layout.itemAt(lyt_info_row_span, 0)
        raster_plot = lyt_raster.layout.itemAt(0, 0)
        raster_plot.clear()
        self.plot_raster(raster_plot, stim)

        # Update average response plot
        lyt_avg_response = lyt_stim.layout.itemAt(lyt_info_row_span + lyt_raster_row_span, 0)
        plt_ifr = lyt_avg_response.layout.itemAt(0, 0)
        self.plot_ifr(plt_ifr, stim, raster_plot)

    def create_stimulus_list_ui(self):
        """Update the stimulus list UI.

        This method updates the user interface (UI) for the stimulus list. It iterates over the stimulus list and creates
        layout elements for each stimulus, including image labels, comboboxes, radio buttons, response plots, and average
        response plots. It also handles user interactions with the UI elements.

        Raises:
            Exception: If an error occurs during the update process, the exception is caught and logged.

        """
        try:
            stimulus_list_ui_start = time.time()
            # Define layout parameters
            stim_ui_config = config['stimulus_ui']
            lyt_info_row_span = stim_ui_config['lyt_info_row_span']
            lyt_raster_row_span = stim_ui_config['lyt_raster_row_span']
            lyt_avg_response_row_span = stim_ui_config['lyt_avg_response_row_span']
            stim_row_span = lyt_info_row_span + lyt_raster_row_span + lyt_avg_response_row_span
            stim_col_span = stim_ui_config['stim_col_span']

            self.c.progress.emit(0, f'Updating stimulus list UI ({self.rows * self.cols} stimuli)')

            # Reduce padding between rows and columns
            self.glw_responses.ci.layout.setSpacing(1)
            self.glw_responses.ci.layout.setContentsMargins(1, 1, 1, 1) # left, top, right, bottom
            self.glw_responses.ci.layout.setHorizontalSpacing(1)
            self.glw_responses.ci.layout.setVerticalSpacing(1)
            self.glw_responses.ci.layout.setRowStretchFactor(0, 1)
            self.glw_responses.ci.layout.setColumnStretchFactor(0, 1)

            stim_idx = 0
            progress = 0
            # Iterate over rows and columns
            for row in range(self.rows):
                for col in range(self.cols):
                    if stim_idx >= len(self.stimulus_list):
                        break
                    # Get the current stimulus
                    stim = self.stimulus_list[stim_idx]

                    if self.glw_responses.ci.layout.count() < stim_idx + 1:
                        # Create the stimulus raster UI
                        self.create_stimulus_raster_ui(stim, stim_idx, row, col, lyt_info_row_span, lyt_raster_row_span,
                                                        lyt_avg_response_row_span, stim_row_span, stim_col_span)
                    
                    stim_idx += 1
                    progress = int((stim_idx / (self.rows * self.cols)) * 100)
                    self.c.progress.emit(progress, None)

                self.glw_responses.nextRow()
            print(f"Creating stimulus list UI took: {time.time() - stimulus_list_ui_start:.2f} s")
            self.c.progress.emit(progress, f"Creating stimulus list UI took: {time.time() - stimulus_list_ui_start:.2f} s")
        except:
            print(traceback.format_exc())
            self.c.log.emit(traceback.format_exc())

    def update_stimulus_list_ui(self):
        """Update the stimulus list UI.

        This method updates the user interface (UI) for the stimulus list. It iterates over the stimulus list and creates
        layout elements for each stimulus, including image labels, comboboxes, radio buttons, response plots, and average
        response plots. It also handles user interactions with the UI elements.

        Raises:
            Exception: If an error occurs during the update process, the exception is caught and logged.

        """
        try:
            b_multi_threaded = False
            stimulus_list_ui_start = time.time()
            # Define layout parameters
            stim_ui_config = config['stimulus_ui']
            lyt_info_row_span = stim_ui_config['lyt_info_row_span']
            lyt_raster_row_span = stim_ui_config['lyt_raster_row_span']
            lyt_avg_response_row_span = stim_ui_config['lyt_avg_response_row_span']
            stim_row_span = lyt_info_row_span + lyt_raster_row_span + lyt_avg_response_row_span
            stim_col_span = stim_ui_config['stim_col_span']

            self.c.progress.emit(0, f'Updating stimulus list UI ({self.rows * self.cols} stimuli)')

            # Reduce padding between rows and columns
            self.glw_responses.ci.layout.setSpacing(1)
            self.glw_responses.ci.layout.setContentsMargins(1, 1, 1, 1) # left, top, right, bottom
            self.glw_responses.ci.layout.setHorizontalSpacing(1)
            self.glw_responses.ci.layout.setVerticalSpacing(1)
            self.glw_responses.ci.layout.setRowStretchFactor(0, 1)
            self.glw_responses.ci.layout.setColumnStretchFactor(0, 1)

            stim_idx = 0
            progress = 0
            
            if b_multi_threaded:
                if self.rows * self.cols == self.glw_responses.ci.layout.count():
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for stim in self.stimulus_list:
                            futures = {executor.submit(self.update_stimulus_raster_ui, stim, self.glw_responses.ci.layout.itemAt(stim_idx), 
                                            lyt_info_row_span, lyt_raster_row_span): stim_idx for stim_idx in range(len(self.stimulus_list))}
                            for future in concurrent.futures.as_completed(futures):
                                stim_idx = futures[future]
                                progress = int((stim_idx / (self.rows * self.cols)) * 100)
                                self.c.progress.emit(progress, None)
            else:
                # Iterate over rows and columns
                for row in range(self.rows):
                    for col in range(self.cols):
                        if stim_idx >= len(self.stimulus_list):
                            if self.glw_responses.ci.layout.count() > stim_idx:
                                for i in range(stim_idx, self.glw_responses.ci.layout.count()):
                                    self.glw_responses.ci.layout.itemAt(i).hide()
                            break
                        # Get the current stimulus
                        stim = self.stimulus_list[stim_idx]

                        if self.glw_responses.ci.layout.count() < stim_idx + 1:
                            # Create the stimulus raster UI
                            self.create_stimulus_raster_ui(stim, stim_idx, row, col, lyt_info_row_span, lyt_raster_row_span,
                                                            lyt_avg_response_row_span, stim_row_span, stim_col_span)
                        else:
                            # Update the stimulus raster UI
                            self.update_stimulus_raster_ui(stim, self.glw_responses.ci.layout.itemAt(stim_idx), 
                                                           lyt_info_row_span, lyt_raster_row_span)
                        
                        stim_idx += 1
                        progress = int((stim_idx / (self.rows * self.cols)) * 100)
                        self.c.progress.emit(progress, None)

                    self.glw_responses.nextRow()
            print(f"Stimulus list UI update time: {time.time() - stimulus_list_ui_start:.2f} s")
            self.c.progress.emit(progress, f"Stimulus list UI update time: {time.time() - stimulus_list_ui_start:.2f} s")
        except:
            print(traceback.format_exc())
            self.c.log.emit(traceback.format_exc())

    def sort_stimulus_list(self):
        """Sort stimuli based on sort criteria"""
        cmb_channel = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbchannels')
        if cmb_channel is not None:
            self.channel = cmb_channel.currentText()
        cmb_cluster = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbcluster')
        if cmb_cluster is not None:
            self.cluster = cmb_cluster.currentText()

        for stim in self.stimulus_list:
            stim.channel = self.channel
            stim.cluster = self.cluster
            stim.sort_criteria = self.sort_criteria
        self.stimulus_list.sort(reverse=not self.asc)
        if self.sort_criteria == 'rank':
            self.rnet_ranking = []
            self.metrics_ranking = []
            self.metrics_zscore = []
            self.raster_class = []
            self.responsive_stim_id = [410,330,532,776,120,492,72,496,124,951,533,35,166,273,585,726,864,490,130,574,16,69,542,664,3,141,159,180,381,919]
            self.responsive_rasters = [] # based on criterion
            rank = 1
            for stim in self.stimulus_list:
                if len(stim.ch_responses[self.channel].responses) < 6:
                    continue

                self.rnet_ranking.append(rank)
                self.metrics_ranking.append(stim.ch_responses[self.channel].rank[self.cluster])
                self.metrics_zscore.append(stim.ch_responses[self.channel].z_score[self.cluster])
                self.raster_class.append(stim.ch_responses[self.channel].raster_class[self.cluster])

                if len(self.responsive_stim_id) == 0:
                    if stim.ch_responses[self.channel].rank[self.cluster] <= 9:
                        self.responsive_rasters.append('responsive')
                    else:
                        self.responsive_rasters.append('non-responsive')
                else:
                    if stim.id in self.responsive_stim_id:
                        self.responsive_rasters.append('responsive')
                    else:
                        self.responsive_rasters.append('non-responsive')
                    rank += 1
            

        if hasattr(self, 'metrics_ranking') and \
           hasattr(self, 'rnet_ranking') and \
           self.b_make_stat_plots:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd

            # Plot metrics vs z-score scatter with color based on raster class and marker shape based on responsive rasters
            plt.figure()

            # Define color mapping for raster classes
            class_to_color = {
                'good': 'green',
                'bad': 'red',
                'meh': 'blue'
            }

            # Define marker shape mapping for responsive rasters
            responsive_to_marker = {
                'responsive': '*',
                'non-responsive': 'o'
            }

            data = pd.DataFrame({
                'Metrics Ranking': self.metrics_ranking,
                'Metrics Z-Score': self.metrics_zscore,
                'Class': self.raster_class,
                'Responsive': self.responsive_rasters
            })

            # Create the scatter plot
            plt.figure(figsize=(10, 6))
            scatter = sns.scatterplot(
                data=data,
                x='Metrics Ranking',
                y='Metrics Z-Score',
                hue='Class',
                style='Responsive',
                palette=class_to_color,
                markers=responsive_to_marker,
                s=100,  # Size of the markers
                alpha=0.6
            )

            # Add labels and title
            plt.xlabel('Response rank')
            plt.ylabel('z-score')
            plt.title('Response ranking vs z-score with RasterNet classes')

            # Add grid
            plt.grid(True)

            # Add legends
            handles, labels = scatter.get_legend_handles_labels()
            from matplotlib.lines import Line2D
            color_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=class_to_color[label], markersize=10) for label in class_to_color]
            color_legend = plt.legend(color_handles, class_to_color.keys(), title='RasterNet', loc='upper right', bbox_to_anchor=(1,0.85))
            plt.gca().add_artist(color_legend)
            plt.legend(handles[len(class_to_color)+2:], labels[len(class_to_color)+2:], title='Metrics-based', loc='upper right', bbox_to_anchor=(1,1))
            # plt.legend(loc='upper right')

            # Save the plot
            plt.savefig(f"ranking_vs_zscore_{self.study_folder.split('/')[-1]}_{self.channel}_{self.sel_cluster}.png", dpi=300, bbox_inches='tight')

            from scipy.stats import f_oneway

            data = pd.DataFrame({
                'Metrics Ranking': self.metrics_ranking,
                'Metrics Z-Score': self.metrics_zscore,
                'Class': self.raster_class
            })

            # Calculate summary statistics
            summary_stats = data.groupby('Class')['Metrics Z-Score'].agg(['mean', 'std', 'count']).reset_index()
            print("Summary Statistics:")
            print(summary_stats)

            # Perform ANOVA
            anova_result = f_oneway(
                data[data['Class'] == 'good']['Metrics Z-Score'],
                data[data['Class'] == 'bad']['Metrics Z-Score'],
                data[data['Class'] == 'meh']['Metrics Z-Score']
            )
            print("\nANOVA Result:")
            print(f"F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")

            # Visualize the data
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Class', y='Metrics Z-Score', data=data, palette=class_to_color)
            plt.title('ANOVA: Z-Score by RasterNet Class')
            plt.xlabel('RasterNet Class')
            plt.ylabel('Z-Score')
            plt.grid(True)

            if anova_result.pvalue < 0.001:
                p_value_text = "< 0.001"
            else:
                p_value_text = f"{anova_result.pvalue:.3f}"

            # Add ANOVA result as text annotation in the top-right corner
            anova_text = f"F-statistic: {anova_result.statistic:.2f}\np-value: {p_value_text}"
            # plt.gcf().text(0.02, 0.95, anova_text, fontsize=12, verticalalignment='top')
            plt.text(.8, 0.95, anova_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            # Save the plot
            plt.savefig(f"boxplot_zscore_{self.study_folder.split('/')[-1]}_{self.channel}_{self.sel_cluster}.png", dpi=300, bbox_inches='tight')
            plt.show()

            # # create violin plot
            # plt.figure(figsize=(10, 6))
            # sns.violinplot(x='Class', y='Metrics Z-Score', data=data, palette=class_to_color)
            # plt.title('ANOVA: Z-Score by RasterNet Class')
            # plt.xlabel('RasterNet Class')
            # plt.ylabel('Z-Score')
            # plt.grid(True)

            # plt.text(.8, 0.95, anova_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            # plt.savefig(f"violinplot_zscore_{self.study_folder.split('/')[-1]}_{self.channel}_{self.sel_cluster}.png", dpi=300, bbox_inches='tight')

            # # rank correlation heatmap
            
            # df = pd.DataFrame({'metrics': self.metrics_ranking, 'rasternet': self.rnet_ranking})
            # corr = df.corr()
            # sns.heatmap(corr, annot=True)
            # plt.figure()
            # plt.title('Metrics vs Rasternet ranking correlation')
            # plt.savefig(f'metrics_vs_rasternet_ranking_corr_{self.channel}_{self.sel_cluster}.png')

            # # Ranking distribution plot
            # plt.figure()
            # plt.hist(self.metrics_ranking, bins=20, alpha=0.5, label='Metrics')
            # plt.hist(self.rnet_ranking, bins=20, alpha=0.5, label='Rasternet')
            # plt.legend(loc='upper right')
            # plt.xlabel('Ranking')
            # plt.ylabel('Count')
            # plt.title('Metrics vs Rasternet ranking distribution')
            # plt.savefig(f'metrics_vs_rasternet_ranking_dist_{self.channel}_{self.sel_cluster}.png')

            # # Cumulative distribution plot
            # plt.figure()
            # plt.hist(self.metrics_ranking, bins=20, alpha=0.5, label='Metrics', cumulative=True, histtype='step')
            # plt.hist(self.rnet_ranking, bins=20, alpha=0.5, label='Rasternet', cumulative=True, histtype='step')
            # plt.legend(loc='upper left')
            # plt.xlabel('Ranking')
            # plt.ylabel('Count')
            # plt.title('Metrics vs Rasternet ranking cumulative distribution')
            # plt.savefig(f'metrics_vs_rasternet_ranking_cum_dist_{self.channel}_{self.sel_cluster}.png')

            # # cumulative rank plot
            # plt.figure()
            # plt.plot(np.cumsum(self.metrics_ranking), label='Metrics')
            # plt.plot(np.cumsum(self.rnet_ranking), label='Rasternet')
            # plt.legend(loc='upper left')
            # plt.xlabel('Stimulus')
            # plt.ylabel('Cumulative rank')
            # plt.title('Metrics vs Rasternet cumulative rank')
            # plt.savefig(f'metrics_vs_rasternet_cum_rank_{self.channel}_{self.sel_cluster}.png')

            # # rank difference plot
            # plt.figure()
            # plt.plot(np.abs(np.array(self.metrics_ranking) - np.array(self.rnet_ranking)))
            # plt.xlabel('Stimulus')
            # plt.ylabel('Rank difference')
            # plt.title('Metrics vs Rasternet rank difference')
            # plt.savefig(f'metrics_vs_rasternet_rank_diff_{self.channel}_{self.sel_cluster}.png')

            # # rank difference histogram
            # plt.figure()
            # plt.hist(np.abs(np.array(self.metrics_ranking) - np.array(self.rnet_ranking)), bins=20)
            # plt.xlabel('Rank difference')
            # plt.ylabel('Count')
            # plt.title('Metrics vs Rasternet rank difference histogram')
            # plt.savefig(f'metrics_vs_rasternet_rank_diff_hist_{self.channel}_{self.sel_cluster}.png')

            # # NDCG plot
            # from sklearn.metrics import ndcg_score
            # ndcg = ndcg_score([self.rnet_ranking], [self.metrics_ranking])
            # plt.plot(ndcg)
            # plt.xlabel('Stimulus')
            # plt.ylabel('NDCG')
            # plt.title('Metrics vs Rasternet NDCG')
            # plt.savefig(f'metrics_vs_rasternet_ndcg_{self.channel}_{self.sel_cluster}.png')

    def mark_raster(self):
        chk = self.glw_responses.sender()
        obj_name = chk.objectName()
        stim_idx = int(obj_name.split(';')[-4])
        # find the stimulus with id == stim_idx
        for stim in self.stimulus_list:
            if stim.id == stim_idx:
                break
        if 'keep' in obj_name:
            stim.ch_responses[stim.channel].keep[stim.cluster] = (chk.isChecked(), 1.0)
        elif 'remove' in obj_name:
            stim.ch_responses[stim.channel].remove[stim.cluster] = (chk.isChecked(), 1.0)
        elif 'meh' in obj_name:
            stim.ch_responses[stim.channel].meh[stim.cluster] = (chk.isChecked(), 1.0)
        # print(obj_name)
        # print(state)
        # self.c.mark_raster.emit(obj_name, chk.isChecked())

    def calculate_ifr_thr(self, ch, clusters):
        """
        Loop through each stimuli, calculate ifr, then get mean and std of ifr
        """
        try:
            for cluster in clusters:
                ifr_max = 0
                baseline_ifr_list = []
                for stim in self.stimulus_list:
                    baseline_ifr_list.append(stim.ch_responses[ch].ifr[cluster][100:1000])
                    if len(stim.ch_responses[ch].ifr[cluster]) == 0:
                        continue
                    ifr_max = max(ifr_max, max(stim.ch_responses[ch].ifr[cluster]))
                baseline_ifr_list = np.concatenate(baseline_ifr_list)
                ifr_thr = np.mean(baseline_ifr_list) + 3 * np.std(baseline_ifr_list)
                for stim in self.stimulus_list:
                    stim.ch_responses[ch].ifr_thr[cluster] = ifr_thr
                    stim.ch_responses[ch].ifr_max[cluster] = ifr_max
                    stim.ch_responses[ch].compute_onset_duration(cluster=cluster, ifr_thr=ifr_thr)
                    stim.ch_responses[ch].compute_p_value(cluster=cluster)
                    # if hasattr(self, 'rasternet'):
                    #     prob, raster_class = self.get_rasternet_prediction(stim, ch, cluster)
                    #     stim.ch_responses[ch].set_keep(cluster=cluster, choice=raster_class, prob=prob)
        except:
            print(f'Error calculating ifr_thr {ch}: {traceback.format_exc()}')
            self.c.log.emit(traceback.format_exc())
            
    def calculate_ifr_max(self):
        if self.sel_cluster in self.ifr_max[self.sel_channel]:
            return  # already calculated
        ifr_max = 0
        for stim in self.stimulus_list:
            ifr_max = max(ifr_max, max(stim.ch_responses[self.sel_channel].ifr[self.sel_cluster]))

        self.ifr_max[self.sel_channel][self.sel_cluster] = ifr_max

    #region Responses display

    def init_responses_layout(self):
        if not hasattr(self, 'responses_layout'):
            self.responses_widget = QtWidgets.QWidget()
            self.lyt_responses = QtWidgets.QVBoxLayout()
            self.responses_widget.setLayout(self.lyt_responses)

            # Controls 
            self.responses_params_layout = QtWidgets.QHBoxLayout()
            self.lyt_responses.addLayout(self.responses_params_layout)
            for param in config['responses_cmb_params']:
                vLyt = QtWidgets.QVBoxLayout()
                vLyt.addWidget(QtWidgets.QLabel(param))
                cmb = QtWidgets.QComboBox(self.responses_widget)
                cmb.setObjectName(f'cmb{param}')
                cmb.addItems(config['responses_cmb_params'][param])
                cmb.currentIndexChanged.connect(self.update_responses)
                vLyt.addWidget(cmb)
                self.responses_params_layout.addLayout(vLyt)

            self.btn_save_rasters = QtWidgets.QPushButton(self.responses_widget)
            self.btn_save_rasters.setText('Save Rasters')
            self.btn_save_rasters.clicked.connect(self.save_raster_selections)
            self.responses_params_layout.addWidget(self.btn_save_rasters)

            self.btn_save_img = QtWidgets.QPushButton(self.responses_widget)
            self.btn_save_img.setText('Save Img')
            self.btn_save_img.clicked.connect(self.save_raster_image)
            self.responses_params_layout.addWidget(self.btn_save_img)

            self.btn_load_existing = QtWidgets.QCheckBox(self.responses_widget)
            self.btn_load_existing.setText('Load existing')
            self.btn_load_existing.clicked.connect(self.load_existing_rankings)
            self.btn_load_existing.setChecked(True)
            self.responses_params_layout.addWidget(self.btn_load_existing)

            self.add_rasternet_controls(self.responses_widget, self.responses_params_layout)
            
            self.glw_responses = pg.GraphicsLayoutWidget()
            self.glw_responses.setContentsMargins(0, 0, 0, 0)
            self.glw_responses.ci.layout.setSpacing(0)
            self.glw_responses.ci.geometryChanged.connect(self.update_response_plots_size)
            # self.responses_view.addWidget(self.responses_layout)
            self.responses_scroll = QtWidgets.QScrollArea()
            self.responses_scroll.setWidgetResizable(True)
            self.responses_scroll.setWidget(self.glw_responses)
            self.responses_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.responses_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            # self.responses_scroll.verticalScrollBar().valueChanged.connect(self.scrolled)
            self.lyt_responses.addWidget(self.responses_scroll)
            self.responses_view.addWidget(self.responses_widget)

    def scrolled(self, value):
        if value == self.responses_scroll.verticalScrollBar().maximum():
            self.update_stimulus_list_ui(at_bottom=True)
        # elif value == self.responses_scroll.verticalScrollBar().minimum():
            # self.update_stimulus_list_ui()

    def add_rasternet_controls(self, widget, layout):

        # Spinbox for number of epochs
        self.spb_epochs = QtWidgets.QSpinBox(widget)
        self.spb_epochs.setObjectName('spbepochs')
        self.spb_epochs.setRange(1, 1000)
        self.spb_epochs.setValue(10)

        vLyt = QtWidgets.QVBoxLayout()
        vLyt.addWidget(QtWidgets.QLabel('Epochs'))
        vLyt.addWidget(self.spb_epochs)
        layout.addLayout(vLyt)

        # Spinbox for batch size
        self.spb_batch_size = QtWidgets.QSpinBox(widget)
        self.spb_batch_size.setObjectName('spbbatchsize')
        self.spb_batch_size.setRange(1, 1000)
        self.spb_batch_size.setValue(32)

        vLyt = QtWidgets.QVBoxLayout()
        vLyt.addWidget(QtWidgets.QLabel('Batch Size'))
        vLyt.addWidget(self.spb_batch_size)
        layout.addLayout(vLyt)

        self.btn_train = QtWidgets.QPushButton(widget)
        self.btn_train.setText('Train')
        self.btn_train.clicked.connect(self.train_rasternet)
        layout.addWidget(self.btn_train)

       

    def save_raster_selections(self):
        bundle = self.sel_channel.split()[0][:-2]
        # for bun in self.bundles.keys():
        #     if bun in self.sel_channel:
        #         bundle = bun

        total = {}
        errors = {}
        accuracy = {}   
        correct = {}

        ch_folder_path = os.path.join(self.rasternet_folder, 'data', 
                                      os.path.basename(self.study_folder), 
                                      bundle, self.sel_channel, self.sel_cluster)
        if os.path.exists(ch_folder_path):
            shutil.rmtree(ch_folder_path)
        os.makedirs(ch_folder_path, exist_ok=True)

        good_path = os.path.join(ch_folder_path, 'good')
        os.makedirs(good_path, exist_ok=True)
        bad_path = os.path.join(ch_folder_path, 'bad')
        os.makedirs(bad_path, exist_ok=True)
        meh_path = os.path.join(ch_folder_path, 'meh')
        os.makedirs(meh_path, exist_ok=True)

        try:
            for stim in self.stimulus_list:
                save_file = stim.lbl+';'+str(stim.channel)+';'+str(stim.cluster)
                responses = np.array(stim.ch_responses[self.sel_channel].get_responses())
                n_trials = responses.shape[0]

                # check if rasternet inference was performed on this raster
                if stim.cluster in stim.ch_responses[stim.channel].initial_rank:
                    initial_rank = stim.ch_responses[stim.channel].initial_rank[stim.cluster][n_trials]
                    final_rank = 'error'
                    if n_trials not in total:
                        total[n_trials] = {'good': 0, 'bad': 0, 'meh': 0}
                        errors[n_trials] = {'good': 0, 'bad': 0, 'meh': 0}
                        accuracy[n_trials] = {'good': 0, 'bad': 0, 'meh': 0}
                        correct[n_trials] = {'good': 0, 'bad': 0, 'meh': 0}

                    if stim.ch_responses[self.sel_channel].keep[stim.cluster][0]: 
                        final_rank = 'good'
                        save_folder = os.path.join(good_path, str(n_trials))
                        total[n_trials]['good'] += 1
                        if initial_rank != 'good':
                            errors[n_trials]['good'] +=1

                    elif stim.ch_responses[self.sel_channel].meh[stim.cluster][0]:
                        final_rank = 'meh'
                        save_folder = os.path.join(meh_path, str(n_trials))
                        total[n_trials]['meh'] += 1
                        if initial_rank != 'meh':
                            errors[n_trials]['meh'] +=1

                    elif stim.ch_responses[self.sel_channel].remove[stim.cluster][0]:
                        final_rank = 'bad'
                        save_folder = os.path.join(bad_path, str(n_trials))
                        total[n_trials]['bad'] += 1
                        if initial_rank != 'bad':
                            errors[n_trials]['bad'] +=1

                    else:
                        save_folder = os.path.join(bad_path, str(n_trials))

                    # save raster file at folder specified
                    print(f'stim {stim.id} initial rank: {initial_rank}, final rank: {final_rank}')
                    os.makedirs(save_folder, exist_ok=True)
                    np.save(os.path.join(save_folder, save_file), responses)
                
                
                else:
                    # if rasternet was never used on this raster:
                    if stim.ch_responses[self.sel_channel].keep[stim.cluster][0]: 
                        rank = 'good'
                        save_folder = os.path.join(good_path, str(n_trials))

                    elif stim.ch_responses[self.sel_channel].meh[stim.cluster][0]:
                        rank = 'meh'
                        save_folder = os.path.join(meh_path, str(n_trials))

                    elif stim.ch_responses[self.sel_channel].remove[stim.cluster][0]:
                        rank = 'bad'
                        save_folder = os.path.join(bad_path, str(n_trials))

                    else:
                        save_folder = os.path.join(bad_path, str(n_trials))

                    # save raster file at folder specified
                    print(f'stim {stim.id} rank: {rank}')
                    os.makedirs(save_folder, exist_ok=True)
                    np.save(os.path.join(save_folder, save_file), responses)
            
            # Calculate live accuracy for this specific channel
            tme = time.localtime()
            entries_list = []
            for model in total.keys():
                for rank in total[model].keys():
                    if total[model][rank] != 0:
                        accuracy[model][rank] = 100 * (1 - (errors[model][rank]/total[model][rank]))
                    correct[model][rank] = total[model][rank] - errors[model][rank]
                    self.c.log.emit(f'Live accuracy of {model} trial model on true {rank} data during this session: ({correct}/{total[model][rank]}), {accuracy[model][rank]:.2f}%')

                # Generate dataframe with live accuracy data
                for rank in total[model].keys():
                    entry = {}
                    entry['Study'] = os.path.basename(self.study_folder)
                    entry['Region'] = bundle
                    entry['Channel'] = self.sel_channel
                    entry['Cluster'] = self.sel_cluster
                    entry['Model'] = model
                    entry['Rank'] = rank
                    entry['Accuracy'] = f'{accuracy[model][rank]:.2f}% ({correct[model][rank]}/{total[model][rank]})'
                    entry['Timestamp'] = f'{tme[1]:02d}/{tme[2]:02d}/{tme[0]} at {tme[3]:02d}:{tme[4]:02d}:{tme[5]:02d}'
                    entries_list.append(entry)
                
            accuracy_df = pd.DataFrame(entries_list)
                
            # Save live accuracy data to excel file
            filename = 'Live_Accuracy_Record.xlsx'
            filepath = os.path.join(self.rasternet_folder, filename)
            if os.path.exists(filepath):
                old_file = pd.read_excel(filepath)
                start_row = old_file.shape[0] + 1
                with pd.ExcelWriter(filepath, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                    accuracy_df.to_excel(writer, sheet_name="Sheet1", header=False, index=False, startrow=start_row)
            else:
                with pd.ExcelWriter(filepath, mode="w", engine="openpyxl") as writer:
                    accuracy_df.to_excel(writer, sheet_name="Sheet1", header=True, index=False)
            
            self.c.log.emit(f'Live accuracy data saved to {filename}.')
            print(f'Live accuracy data saved to {filename}.')
            # stim.ch_responses[stim.channel].initial_rank[stim.cluster][n_trials]
            
            # clear initial rankings table for next time
            for stim in self.stimulus_list:
                for channel in stim.ch_responses:
                    stim.ch_responses[channel].initial_rank = {}

        except:
            print(f'Error saving raster selection: {traceback.format_exc()}')
            # self.c.log.emit(traceback.format_exc())


    def save_raster_image(self):
        """
        Saves the raster image to a file.
        """
        
        # import matplotlib.pyplot as plt
        
        # # Save stim images in a 4x3 subplots
        # fig, axs = plt.subplots(3, 4)
        # ctr_limit = 12
        # ctr = 0
        # for i, stim in enumerate(self.stimulus_list):
        #     if ctr >= ctr_limit:
        #         break
        #     if stim.remove:
        #         continue
        #     row = ctr // 4
        #     col = ctr % 4
        #     img = stim.stim_img
        #     # plot rgb image
        #     axs[row, col].imshow(img)
        #     axs[row, col].axis('off')
        #     ctr += 1

        # # remove all spacing between subplots
        # plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        # fig.savefig('rasters_keep.png')
        # plt.close(fig)
        
        # save_path = QtWidgets.QFileDialog.getSaveFileName(self.responses_widget, 'Save Raster Image', '', 'PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)')
        
        # img = pg.makeQImage(self.responses_layout.scene())
        # img.save(save_path[0])

        # # Save with publication quality
        # img.save('raster_image2.png', quality=100)

        exporter = pg.exporters.ImageExporter(self.glw_responses.scene())
        exporter.export(f"{self.study_folder.split('/')[-1]}_{self.channel}_{self.cluster}_rasters.png")

        # Save as svg
        # exporter = pg.exporters.SVGExporter(self.responses_layout.scene())
        # exporter.export('raster_image4.svg')

    def load_existing_rankings(self):
        chk = self.glw_responses.sender()
        self.load_existing = chk.isChecked()

    def update_responsess(self):
        cmb = self.responses_widget.sender()
        param = cmb.objectName()
        value = cmb.currentText()
        if value == 'None' or value == '':
            return

        if param == 'cmbchannels':
            self.sel_channel = value
            clusters = np.unique(self.clus_info_dict[self.sel_channel]['cluster_labels'])
            cmb_cluster = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbcluster')
            try:
                cmb_cluster.currentIndexChanged.disconnect(self.update_responses)
            except:
                pass
            cmb_cluster.clear()
            cmb_cluster.addItem('mu')
            cmb_cluster.addItems(clusters.astype(str))
            cmb_cluster.currentIndexChanged.connect(self.update_responses)
            self.sel_cluster = 'mu'

        elif param == 'cmbcluster':
            self.sel_cluster = value

        elif param == 'cmbshow':
            if value == 'all':
                self.rows = len(self.stimulus_list) // self.cols
            else:
                stim_count = int(value)
                self.rows = stim_count // self.cols
        elif param == 'cmbsorting':
            self.sort_criteria = value
        elif param == 'cmbsortorder':
            if value == 'asc':
                self.asc = True
            else:
                self.asc = False

        self.sort_stimulus_list()
        self.update_stimulus_list_ui()

    def update_responses(self):
        """
        Update the responses based on the selected channel and cluster
        Uses loaded ranking table
        """
        self.c.log.emit('Updating Responses')
        cmb = self.responses_widget.sender()
        param = cmb.objectName()
        value = cmb.currentText()
        if value == 'None' or value == '':
            return

        try:
            update_responses_start = time.time()
            raster_size = 150
            if param == 'cmbchannels' or param == 'cmbcluster':
                if param == 'cmbchannels':
                    self.sel_channel = value
                    ch_idx = 'chan'+self.sel_channel.split('_')[-1]
                    clusters = self.rasters['rasters'][ch_idx].keys()
                    # remove details
                    clusters = [clus for clus in clusters if clus != 'details']

                    cmb_cluster = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbcluster')
                    try:
                        cmb_cluster.currentIndexChanged.disconnect(self.update_responses)
                    except:
                        pass
                    cmb_cluster.clear()
                    cmb_cluster.addItems(clusters)
                    cmb_cluster.currentIndexChanged.connect(self.update_responses)
                    self.sel_cluster = 'mu'
                    self.cluster_count = len(clusters)
                    self.clus_colors = [pg.intColor(i, hues=self.cluster_count) for i in range(self.cluster_count)]
                    
                elif param == 'cmbcluster':
                    self.sel_cluster = value
                    ch_idx = 'chan'+self.sel_channel.split('_')[-1]
                    clusters = self.rasters['rasters'][ch_idx].keys()
                    # remove details
                    clusters = [clus for clus in clusters if clus != 'details']

                if hasattr(self, 'rasters'):
                    self.t = np.arange(0, raster_size, 1)
                    self.t_scale = 1000 # One second is 1000 pixels/points
                    # self.rows = int(np.ceil(len(self.stimulus_list) / self.cols))
                else:
                    return
                
                # Get the responses for the selected channel and cluster
                responses_table = self.ranking_table[(self.ranking_table['channel'] == ch_idx) & 
                                                     (self.ranking_table['class'] == self.sel_cluster)]

                # Check if responses table is empty
                if responses_table.empty:
                    self.c.log.emit(f'No responses found for {self.sel_channel} {self.sel_cluster}')
                    return

                # Set ranking based on order in the responses table
                rank = pd.Series(range(len(responses_table)), name='ranking')
                responses_table.set_index(rank, inplace=True)
                
                # Parallelize the update of stimuli
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     futures = {executor.submit(self.update_stimuli, stim, responses_table, ch_idx): stim for stim in self.stimulus_list}
                #     completed = 0
                #     for future in concurrent.futures.as_completed(futures):
                #         completed += 1
                #         self.c.progress.emit(int((completed / len(self.stimulus_list)) * 100), None)

                self.stored_labels = {}
                self.update_single_neuron_metrics(clusters, self.sel_cluster, ch_idx, self.chann_dict[ch_idx])
                # self.calculate_ifr_thr()
                # self.calculate_ifr_max()
            elif param == 'cmbshow':
                if value == 'all':
                    self.rows = len(self.stimulus_list) // self.cols
                else:
                    stim_count = int(value)
                    self.rows = stim_count // self.cols

            elif param == 'cmbsorting':
                self.sort_criteria = value
            elif param == 'cmbsortorder':
                if value == 'asc':
                    self.asc = True
                else:
                    self.asc = False


            print(f'{len(self.stimulus_list)} Responses updated in {time.time()-update_responses_start:.2f} seconds.')
        except:
            print(traceback.format_exc())

        self.calculate_ifr_max()
        self.sort_stimulus_list()
        self.update_stimulus_list_ui()

    # def update_stimuli(self, stim, responses_table, ch_idx=0):
    #     img_idx = self.image_names.index(stim.lbl)
    #     if img_idx+1 not in responses_table['stim_number'].values:
    #         return

    #     stim.channel = self.sel_channel
    #     if self.sel_cluster == 'mu' or self.sel_cluster == 'details':
    #         stim.cluster = 'mu'
    #     else:
    #         stim.cluster = int(self.sel_cluster[-1]) - 1
    #     stim.keep    = False
    #     stim.meh     = False
    #     stim.remove  = False
    #     stim_r_idx        = (self.ranking_table['stim_number'] == img_idx+1) & \
    #                         (self.ranking_table['channel'] == ch_idx) & \
    #                         (self.ranking_table['class'] == self.sel_cluster)
    #     rank              = responses_table[responses_table['stim_number'] == img_idx+1].index[0]
    #     ifr_idx           = self.ranking_table[stim_r_idx].index[0]
    #     stim.rank         = rank + 1
    #     stim.ifr          = self.ifr['ifrmat'][ifr_idx]
    #     stim.z_score      = self.ranking_table[stim_r_idx]['zscore'].values[0]
    #     stim.tons         = self.ranking_table[stim_r_idx]['tons'].values[0]
    #     stim.onset        = self.ranking_table[stim_r_idx]['onset'].values[0]
    #     stim.duration     = self.ranking_table[stim_r_idx]['dura'].values[0]
    #     stim.good_lat     = self.ranking_table[stim_r_idx]['good_lat'].values[0]
    #     stim.ifr_thr      = self.ranking_table[stim_r_idx]['IFR_thr'].values[0]
    #     stim.p_value_sign = self.ranking_table[stim_r_idx]['p_value_sign'].values[0]
    #     stim.p_test       = self.ranking_table[stim_r_idx]['p_test'].values[0]
    #     stim.median_post  = self.ranking_table[stim_r_idx]['median_post'].values[0]
    #     stim.min_spk_test = self.ranking_table[stim_r_idx]['min_spk_test'].values[0]
        
    #     stim.responses = self.rasters['rasters'][ch_idx]['mu']['stim'][img_idx]
    #     # array to store the cluster labels same size as responses. labels are class1, class2, etc
    #     stim.cluster_labels = []
    #     for i, response in enumerate(stim.responses):
    #         clus_lbls = np.zeros_like(response, dtype=int)
    #         for clus in self.rasters['rasters'][ch_idx].keys():
    #             if clus == 'mu' or clus == 'details':
    #                 continue
    #             clus_responses = self.rasters['rasters'][ch_idx][clus]['stim'][img_idx]
    #             clus_idx = np.where(np.isin(response, clus_responses[i]))[0]
    #             clus_lbls[clus_idx] = str(int(clus[-1]) - 1)
    #         stim.cluster_labels.append(clus_lbls)

    #     if hasattr(self, 'rasternet'):
    #         # Convert to tensor
    #         resp_tensor = torch.tensor(stim.get_responses_dl()).float()
    #         prob, raster_class = self.rasternet.predict_raster_class(resp_tensor)
            
    #         # Update live tally of stim rankings
    #         self.rankings[stim.id] = raster_class
    #         # Update stim properties
    #         if raster_class == 'good':
    #             stim.keep = True
    #             stim.meh = False
    #             stim.remove = False
    #         elif raster_class == 'meh':
    #             stim.meh = True
    #             stim.keep = False
    #             stim.remove = False
    #         else:
    #             stim.remove = True
    #             stim.keep = False
    #             stim.meh = False

    def update_response_plots_size(self):
        self.glw_responses.setFixedHeight(int(self.glw_responses.ci.contentsRect().height()))
    
    def load_study_info(self, study_info):
        self.image_names = study_info.image_names
        self.study_folder = study_info.study_folder  
        self.pics_order_1d = study_info.pics_order_1d  
        self.ranking_table = study_info.ranking_table
        self.pics_onset = study_info.pics_onset_1d
        self.ifr = study_info.ifr
        # self.stimulus = study_info.stimulus_1d
        used_pics_ids = study_info.used_pics

        imgs_dir = os.path.join(self.study_folder, 'pics_used')
        if not os.path.exists(imgs_dir):
            print('No pics folder found.')
            self.c.log.emit('No pics folder found. Wont be able to load grapes.') 
        
        img_paths = []
        used_pics_ids = used_pics_ids - 1 # Matlab index starts from 1, so subtract 1
        self.used_pics_names = np.array(self.image_names)[used_pics_ids]

        for img in self.used_pics_names:
            img_path = os.path.join(imgs_dir, img)
            if os.path.exists(img_path):
                img_paths.append(img_path)
            else:
                # Check one folder up in custom_pics folder
                img_path = os.path.join(os.path.dirname(imgs_dir), 'custom_pics', img)
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    self.c.log.emit(f'Image {img} does not exist')

        self.init_stimulus_list(img_ids=used_pics_ids+1, img_paths=img_paths)
        self.img_paths = img_paths

    def update_responses_cmb_channels(self, channels):
        """
        Updates the combobox in responses tab.
        """
        self.channels = channels
        self.channels_short = [ch.split()[0] for ch in self.channels]
        if self.channels is not None:
            for channel in self.channels:
                self.ifr_max[channel] = {}
        cmb_channels = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbchannels')
        try:
            cmb_channels.currentIndexChanged.disconnect(self.update_responses)
        except:
            pass
        cmb_channels.clear()
        cmb_channels.addItems(self.channels)
        cmb_channels.currentIndexChanged.connect(self.update_responses)

        # cmb_stimulus = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbstimulus')
        # cmb_stimulus.clear()

        cmb_cluster = self.responses_widget.findChild(QtWidgets.QComboBox, 'cmbcluster')
        try:
            cmb_cluster.currentIndexChanged.disconnect(self.update_responses)
        except:
            pass
        cmb_cluster.clear()


        # # Get unique stimuli from the ranking table
        # stimuli = self.ranking_table['stim_number'].unique()
        # # Convert to string
        # stimuli = [self.image_names[stim-1] for stim in stimuli]
        # cmb_stimulus.addItems(stimuli)

        # Get unique clusters from the ranking table   
        if hasattr(self, 'cluster_labels'):     
            cmb_cluster.addItems(np.unique(self.cluster_labels))
            cmb_cluster.currentIndexChanged.connect(self.update_responses)
    
    def compute_response_metrics_threaded(self, clus_info_dict, micro_ch_lbls):
        """
        Compute the response metrics for each channel in the clus_info_dict.
        """
        self.clus_info_dict = clus_info_dict
        self.update_responses_cmb_channels(micro_ch_lbls)
        compute_response_metrics_thread = threading.Thread(target=self.compute_response_metrics, args=(clus_info_dict, micro_ch_lbls))
        compute_response_metrics_thread.start()

    def compute_response_metrics(self, clus_info_dict, micro_ch_lbls):        

        # Get photo diode spike times for each instance
        photo_ch = next((k for k in self.clus_info_dict if "Photo" in k), None)
        all_stimulus_onset_list = self.clus_info_dict[photo_ch]['spikes']
        # all_stimulus_onset_list = self.pics_onset.astype(int)

        self.c.progress.emit(0, f'Computing responses for {len(micro_ch_lbls)} channels')
        
        for ch in micro_ch_lbls:
            clusters = ['mu']
            clusters.extend(np.unique(self.clus_info_dict[ch]['cluster_labels']).astype(str).tolist())
            channel_spikes_ts = np.array(self.clus_info_dict[ch]['spikes'])
            channel_clus_lbls = np.array(self.clus_info_dict[ch]['cluster_labels'])

            for stim in self.stimulus_list:
                stim_idx = self.image_names.index(stim.lbl)
                stim.keep    = False
                stim.meh     = False
                stim.remove  = False
                
                # Get all instances of the image being shown
                stimulus_onset_idx_list = [i for i, x in enumerate(self.pics_order_1d[:len(all_stimulus_onset_list)]) if x == stim_idx+1]
                stimulus_onset_list = [all_stimulus_onset_list[i] for i in stimulus_onset_idx_list]
                responses = []
                cluster_labels = []
                
                spike_idxs = []
                spike_ts_list = []
                cluster_labels = []
                for stimulus_onset in stimulus_onset_list:
                    # Get spikes 1 sec before and 2 secs after the stimulus onset
                    start_time = stimulus_onset - 1 * self.fs
                    end_time = stimulus_onset + 2 * self.fs
                    
                    spike_idxs = np.where((channel_spikes_ts >= start_time) & (channel_spikes_ts <= end_time))[0]
                    if spike_idxs.size < 12:
                        continue
                    spike_ts_list = channel_spikes_ts[spike_idxs] - stimulus_onset
                    spike_ts_list = spike_ts_list / 30 # Convert to ms
                    
                    # # -1 to 2 sec, spike_ts at 0
                    # spike_ts_list = spike_ts_list - stimulus_onset
                    # # Convert to ms
                    # spike_ts_list = spike_ts_list / (self.fs / 1000)

                    # response = np.zeros((3 * self.fs))
                    # response[spikes] = 1
                    responses.append(spike_ts_list)
                    if channel_clus_lbls.size > 0:
                        cluster_labels.append(channel_clus_lbls[spike_idxs])

                stim.ch_responses[ch] = ChannelResponses(c=self.c, channel=ch, responses=responses, 
                                                         clusters=clusters, cluster_labels=cluster_labels,
                                                         gauss_win=self.gauss_window)
            
            self.calculate_ifr_thr(ch=ch, clusters=clusters)
            # self.sort_stimulus_list()
            self.c.progress.emit(int((micro_ch_lbls.index(ch) / len(micro_ch_lbls)) * 100), f'{micro_ch_lbls.index(ch)+1}/{len(micro_ch_lbls)} channels done')

        self.c.progress.emit(100, f'Completed computing response metrics for {len(micro_ch_lbls)} channels')

    def load_response_metrics_threaded(self, clus_info_dict):
        """
        Load the response metrics from a the ranking table.
        """
        self.clus_info_dict = clus_info_dict
        micro_ch_lbls = list(clus_info_dict.keys())
        self.update_responses_cmb_channels(micro_ch_lbls)
        self.channel_list = self.ranking_table['channel'].unique()
        self.chann_dict = {}
        for ch in self.channel_list:
            ch_id = ch.split('chan')[-1]
            for ch_lbl in micro_ch_lbls:
                if ch_id in ch_lbl:
                    self.chann_dict[ch] = ch_lbl
                    break
        # load_response_metrics_thread = threading.Thread(target=self.load_response_metrics, args=(clus_info_dict,))
        # load_response_metrics_thread.start()

    def load_response_metrics(self, clus_info_dict, chann_dict, channel_list):
        """
        Load the response metrics from a the ranking table created using matlab code.
        """
        multi_thread = False
        load_response_metrics_start = time.time()
        self.c.progress.emit(1, f'Loading response metrics for {len(clus_info_dict)} channels')    
        
        if multi_thread:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.load_response_metrics_channel, ch, chann_dict[ch]): ch for ch in channel_list}
                completed_count = 0
                ch_done_time = time.time()
                for future in concurrent.futures.as_completed(futures):
                    completed_count += 1
                    remaining_time = (time.time() - ch_done_time) / completed_count * (len(channel_list) - completed_count)
                    hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(remaining_time)
                    self.c.progress.emit(int((completed_count / len(channel_list)) * 100), f'{completed_count}/{len(channel_list)} channels done. Remaining time: {hrs} hrs {mins} mins {secs} secs')
                    
        else:
            for ch_idx, ch in enumerate(channel_list):
                self.load_response_metrics_channel(ch, chann_dict[ch])
                self.c.progress.emit(int((ch_idx / len(channel_list)) * 100), f'{ch_idx+1}/{len(channel_list)} channels done')
                remaining_time = (time.time() - load_response_metrics_start) / (ch_idx + 1) * (len(channel_list) - ch_idx - 1)
                hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(remaining_time)
                self.c.progress.emit(int((ch_idx / len(channel_list)) * 100), f'{ch_idx+1}/{len(channel_list)} channels done. Remaining time: {hrs} hrs {mins} mins {secs} secs')

        hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(time.time() - load_response_metrics_start)
        print(f'Loaded response metrics in {hrs:.0f} hrs {mins:.0f} mins {secs:.0f} secs')
        self.c.progress.emit(100, f'Loaded response metrics for {len(clus_info_dict)} channels in {hrs:.0f} hrs {mins:.0f} mins {secs:.0f} secs')

    def load_response_metrics_channel(self, ch, ch_lbl):
        """
        Load the response metrics for a single channel from matlab code created table.
        """
        clusters = self.ranking_table[self.ranking_table['channel'] == ch]['class'].unique()
        multi_thread = True
        if multi_thread:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.update_single_neuron_metrics, clusters, cluster, ch, ch_lbl): cluster for cluster in clusters}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    self.c.progress.emit(int((completed / len(self.stimulus_list)) * 100), None)
        else:
            completed = 0
            for cluster in clusters:
                self.update_single_neuron_metrics(clusters, cluster, ch, ch_lbl)
                completed += 1
                self.c.progress.emit(int((completed / len(self.stimulus_list)) * 100), None)

    def update_single_neuron_metrics(self, clusters, cluster, ch, ch_lbl):
        """
        Update metrics for a single neuron.
        """
        multi_thread = False                      

        if multi_thread:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.update_stimulus_metrics, 
                                           stim, ch, ch_lbl, clusters): 
                                           stim for stim in self.stimulus_list}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    self.c.progress.emit(int((completed / len(self.stimulus_list)) * 100), None)
        else:
            completed = 0
            for stim in self.stimulus_list:
                self.update_stimulus_metrics(stim, ch, ch_lbl, clusters)
                completed += 1
                self.c.progress.emit(int((completed / len(self.stimulus_list)) * 100), None)

    def update_stimulus_metrics(self, stim, ch, ch_lbl, clusters):
        """
        Update the stimulus metrics for a single channel and cluster.
        uses matlab ranking table.
        """
        try:
            stim_idx = self.image_names.index(stim.lbl)
            if ch_lbl not in stim.ch_responses.keys():
                responses = self.rasters['rasters'][ch]['mu']['stim'][stim_idx]
                
                cluster_labels = []
                for i, response in enumerate(responses):
                    clus_lbls = np.zeros_like(response, dtype=int)
                    if not isinstance(response, np.ndarray):
                        cluster_labels.append(clus_lbls)
                        continue
                    
                    for clus in clusters:
                        if clus == 'mu':
                            continue
                        clus_id = clus[-1]
                        clus_responses = self.rasters['rasters'][ch][clus]['stim'][stim_idx]
                        if clus_responses.size != responses.size or \
                            not isinstance(clus_responses[i], np.ndarray):
                            continue
                        clus_idx = np.where(np.isin(response, clus_responses[i]))[0]
                        clus_lbls[clus_idx] = str(clus_id)
                    cluster_labels.append(clus_lbls) 

                stim.ch_responses[ch_lbl] = ChannelResponses(c=self.c, channel=ch, 
                                                                clusters=clusters,
                                                                cluster_labels=cluster_labels,
                                                                responses=responses)
                for cluster in clusters:
                    # Get the responses for the selected channel and cluster
                    responses_table = self.ranking_table[(self.ranking_table['channel'] == ch) & 
                                                         (self.ranking_table['class'] == cluster)]

                    # Check if responses table is empty
                    if responses_table.empty:
                        self.c.log.emit(f'No responses found for {ch} {cluster}')
                        return

                    # Set ranking based on order in the responses table
                    rank = pd.Series(range(len(responses_table)), name='ranking')
                    responses_table.set_index(rank, inplace=True)  
                
                    stim.ch_responses[ch_lbl].keep[cluster]   = (False, 0.0)
                    stim.ch_responses[ch_lbl].meh[cluster]    = (False, 0.0)
                    stim.ch_responses[ch_lbl].remove[cluster] = (False, 0.0)
                    if stim_idx+1 not in responses_table['stim_number'].values:
                        return

                    stim_r_idx        = (self.ranking_table['stim_number'] == stim_idx+1) & \
                                        (self.ranking_table['channel'] == ch) & \
                                        (self.ranking_table['class'] == cluster)
                    rank              = responses_table[responses_table['stim_number'] == stim_idx+1].index[0]
                    ifr_idx           = self.ranking_table[stim_r_idx].index[0]
                    stim.ch_responses[ch_lbl].rank[cluster]         = rank + 1
                    stim.ch_responses[ch_lbl].ifr[cluster]          = self.ifr['ifrmat'][ifr_idx]
                    
                    stim.ch_responses[ch_lbl].z_score[cluster]      = self.ranking_table[stim_r_idx]['zscore'].values[0]
                    stim.ch_responses[ch_lbl].tons[cluster]         = self.ranking_table[stim_r_idx]['tons'].values[0]
                    stim.ch_responses[ch_lbl].onset[cluster]        = self.ranking_table[stim_r_idx]['onset'].values[0]
                    stim.ch_responses[ch_lbl].duration[cluster]     = self.ranking_table[stim_r_idx]['dura'].values[0]
                    stim.ch_responses[ch_lbl].good_lat[cluster]     = self.ranking_table[stim_r_idx]['good_lat'].values[0]
                    stim.ch_responses[ch_lbl].ifr_thr[cluster]      = self.ranking_table[stim_r_idx]['IFR_thr'].values[0]
                    stim.ch_responses[ch_lbl].p_value_sign[cluster] = self.ranking_table[stim_r_idx]['p_value_sign'].values[0]
                    stim.ch_responses[ch_lbl].p_test[cluster]       = self.ranking_table[stim_r_idx]['p_test'].values[0]
                    stim.ch_responses[ch_lbl].median_post[cluster]  = self.ranking_table[stim_r_idx]['median_post'].values[0]
                    stim.ch_responses[ch_lbl].min_spk_test[cluster] = self.ranking_table[stim_r_idx]['min_spk_test'].values[0]

                    # if cluster in stim.ch_responses[ch_lbl].ifr_max.keys():
                    #     stim.ch_responses[ch_lbl].ifr_max[cluster] = max(max(stim.ch_responses[ch_lbl].ifr[cluster]), 
                    #                                                         stim.ch_responses[ch_lbl].ifr_max[cluster])
                    # else:
                    #     stim.ch_responses[ch_lbl].ifr_max[cluster] = max(self.ifr['ifrmat'][ifr_idx])

                    if self.load_existing:
                        try:
                            # get existing rank, if it exists
                            bFound, prob, raster_class = self.get_saved_rnet_class(stim, ch_lbl, cluster, responses.size)
                            if not bFound and hasattr(self, 'rasternet'):
                                # raster wasn't found in saved data folder, run rasternet inference
                                prob, raster_class = self.get_rasternet_prediction(stim, ch_lbl, cluster)
                                stim.ch_responses[ch_lbl].set_keep(cluster=cluster, choice=raster_class, prob=prob)
                            elif bFound:
                                # raster was found in saved data folder, update responses using this saved data
                                stim.ch_responses[ch_lbl].set_keep(cluster=cluster, choice=raster_class, prob=prob)

                        except:
                            # an error has occured loading saved data, attempt to use rasternet instead
                            print(traceback.print_exc())

                    else:
                        # load_existing set to false, run rasternet inference regardless of what's in data folder
                        if hasattr(self, 'rasternet'):
                            prob, raster_class = self.get_rasternet_prediction(stim, ch_lbl, cluster)
                            stim.ch_responses[ch_lbl].set_keep(cluster=cluster, choice=raster_class, prob=prob)
        except:
            print(f'Error updating stimulus metrics {ch_lbl}')
            # self.c.log.emit(traceback.format_exc())
    
    def get_saved_rnet_class(self, stim, ch_lbl, cluster, resp_size):
        # Locals
        bFound = False

        file_name = f'{stim.lbl};{ch_lbl};{cluster}.npy'

        # else:
        run = os.path.basename(self.study_folder)
        bundle = ch_lbl.split(' ')[0][:-2]

        root_dir = os.path.join(self.rasternet_folder,'data',run,bundle,ch_lbl,cluster)
        if os.path.exists(root_dir):
            rnet_classes = os.listdir(root_dir)
            for rnet_class in rnet_classes:
                raster_file = os.path.join(root_dir, rnet_class, str(resp_size), file_name)
                if os.path.exists(raster_file):
                    bFound = True

                    return bFound, 1.0, rnet_class
            
        return bFound, 0.0, 'bad'
            
        
        
        

    #endregion Responses display

    #region RasterNet training
                
    def train_rasternet(self):
        if not hasattr(self, 'rasternet'):
            self.rasternet = {}
        data_folder   = os.path.join(self.rasternet_folder,'data')
        models_folder = os.path.join(self.rasternet_folder, 'models')
        
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        raster_dataset = RasterDataset(root_dir=data_folder)
        for key in raster_dataset.data.keys():

            model_path = os.path.join(models_folder, f'rasternet_{key}_{raster_dataset.raster_size}.pt')
        
            if os.path.exists(model_path):
                # Show warning that model already exists
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText(f'Model for {key} trials already exists. Do you want to overwrite?')
                msg.setWindowTitle('Model Exists')
                msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                retval = msg.exec_()
                if retval == QtWidgets.QMessageBox.No:
                    continue

            if raster_dataset.valid_data[key]:
                train_start = time.time()
                raster_dataset.current_n_trials = key
                train_data, test_data = torch.utils.data.random_split(raster_dataset, 
                                                [int(0.8 * len(raster_dataset)), 
                                                len(raster_dataset) - int(0.8 * len(raster_dataset))])
                self.rasternet[key] = RasterNet(self.c, n_trials=key, raster_size=self.raster_size_px)
                self.rasternet[key].train_model(train_data=train_data, 
                                        num_epochs=self.spb_epochs.value(), 
                                        batch_size=self.spb_batch_size.value())
        
                torch.save(self.rasternet[key].model.state_dict(), model_path)
                correct, total = self.rasternet[key].test_model(test_data)
                print(f'rasternet_{key} trained in {time.time()-train_start:.2f} secs. Accuracy: {100 * correct / total:.2f}% ({correct}/{total})')
                self.c.log.emit(f'rasternet_{key} trained in {time.time()-train_start:.2f} secs. Accuracy: {100 * correct / total:.2f}% ({correct}/{total})')

    def load_rasternet(self):
        if not hasattr(self, 'rasternet'):
            self.rasternet = {}
        models_folder = os.path.join(self.rasternet_folder, 'models')
        for model in os.listdir(models_folder):
            model_name = model.split('.')[0]
            n_trials = int(model_name.split('_')[1])
            try:
                raster_size = int(model_name.split('_')[2])
            except:
                raster_size = 150

            if raster_size == self.raster_size_px:
                self.rasternet[f'{n_trials}_{raster_size}'] = RasterNet(self.c, n_trials=n_trials, raster_size=raster_size)
                self.rasternet[f'{n_trials}_{raster_size}'].model.load_state_dict(torch.load(os.path.join(models_folder, model)))

    def get_rasternet_prediction(self, stim, ch_lbl, cluster):
        # Convert to tensor
        raster_img = torch.tensor(stim.ch_responses[ch_lbl].get_responses_dl(cluster=cluster, raster_size=self.raster_size_px)).float()
        key = f'{raster_img.shape[0]}_{raster_img.shape[1]}'
        if key not in self.rasternet.keys():
            if raster_img.shape[0] > 6 and raster_img.shape[0] < 9:
                raster_img = raster_img[:6, :]
                key = f'6_{self.raster_size_px}'
            elif raster_img.shape[0] > 9 and raster_img.shape[0] < 13:
                raster_img = raster_img[:9, :]
                key = f'9_{self.raster_size_px}'
            elif raster_img.shape[0] > 13 and raster_img.shape[0] < 16: # for 14 trials
                raster_img = raster_img[:13, :]
                key = f'13_{self.raster_size_px}'
            elif raster_img.shape[0] > 15 and raster_img.shape[0] < 20: # for 16 trials
                raster_img = raster_img[:15, :]
                key = f'15_{self.raster_size_px}'
            elif raster_img.shape[0] < 6 and raster_img.shape[0] > 3:
                raster_img = raster_img[:3, :]
                key = f'3_{self.raster_size_px}'
            else:
                print(f'Invalid rasternet input shape: {raster_img.shape}.')
            prob, raster_class = self.rasternet[key].predict_raster_class(raster_img)
        else:
            prob, raster_class = self.rasternet[key].predict_raster_class(raster_img)

        # Update live tally of stim rankings
        if cluster not in stim.ch_responses[ch_lbl].initial_rank:
            stim.ch_responses[ch_lbl].initial_rank[cluster] = {}

        stim.ch_responses[ch_lbl].initial_rank[cluster][raster_img.shape[0]] = raster_class


        return prob, raster_class
    #endregion RasterNet training