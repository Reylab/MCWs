# Author: Sunil Mathew
# Date: 10 Jan 2024
# Clustering module, easy switching between different clustering algorithms
# multithreaded execution for processing multiple channels simultaneously
import os
import copy
import json
import concurrent.futures

import traceback
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from spclustering import SPC, plot_temperature_plot
from isosplit6 import isosplit6

import pyqtgraph as pg
import threading
import time

from qtpy import QtWidgets, QtCore
from usercontrols.toggle_switch import Switch
from core.config import config

from core.utils import get_elapsed_time, get_time_hrs_mins_secs_ms
from dsp.spike_detection import detect_spikes_in_segments, extract_spikes, amp_detect_spikes_in_segments, detect_photo_spikes
from dsp.spike_features import extract_features

class ClusteringHistory():
    def __init__(self) -> None:
        self.undo_stack = []
        self.redo_stack = []
        self.cluster_labels = None

    def add_selection(self, cluster, spikes):
        """
        Adds the selected spikes to the selection history.
        """
        self.undo_stack.append({'selection': (cluster, spikes)})

    def add_merge(self, merged_clusters):
        """
        Adds the merged clusters to the merge history.
        """
        self.undo_stack.append({'merge': merged_clusters})

    def undo(self):
        """
        Undoes the last action.
        """
        if len(self.undo_stack) > 0:
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            return action

    def redo(self):
        """
        Redoes the last action.
        """
        if len(self.redo_stack) > 0:
            return self.redo_stack.pop()
        
class Clustering():
    def __init__(self, filtering, clustering_view, c) -> None:
        self.filtering = filtering
        self.clustering_view = clustering_view
        self.c = c
        self.curr_clus_idx = -1
        self.cluster_count = 6
        self.clus_spike_count = 50
        self.clus_prob_thr = 0.8
        self.clus_ch = None
        self.clus_algorithm = 'GMM'
        self.clus_info_dict = {} # Clustering results are stored for each channel
        self.b_drawing_roi = False
        self.b_mouse_moved = False
        self.do_sorting = False
        self.edit_lock = threading.Lock()
        self.spk_detect_lock = threading.Lock()
        self.history = ClusteringHistory()
        self.mouse_loc = None
        # Create a selection win
        self.selection_win = pg.PlotDataItem()
        # Create a crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=1, style=pg.QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', width=1, style=pg.QtCore.Qt.DashLine))
        # Create a list of colors for each cluster
        self.clus_colors = [pg.intColor(i, hues=self.cluster_count) for i in range(self.cluster_count)]
        self.merge_clusters = []
        self.selected_spikes = []
        self.init_clustering_layout()

    def clear_clus_info_dict(self, ch_lbls, n_feats=3):
        """
        Clears the clustering info dictionary.
        """
        self.clus_info_dict = {}

        for ch_name in ch_lbls:
            self.clus_info_dict[ch_name] = {'spikes': [], 'spike_waveforms': [], 'threshold': [],
                                            'features': [], 'cluster_labels': [], 
                                            'cluster_probs': []}

    def init_clus_info_dict(self, ch_lbls, n_feats=3):
        """
        Initializes the clustering info dictionary, to append clustering results for each channel.
        """
        # Create a spike waveform
        rise_1 = np.linspace(10, 12, 4)
        fall_1 = np.linspace(12, 9, 4)
        rise_2 = np.linspace(9, 12, 4)
        fall_2 = np.linspace(12, -64, 8)
        rise_3 = np.linspace(-64, 20, 36)
        fall_3 = np.linspace(20, 7, 8)

        # Concatenate the sections to create the spike
        spike = np.concatenate([rise_1, fall_1, rise_2, fall_2, rise_3, fall_3])

        # Create a list of spike waveforms with added random noise
        spike_waveforms = [spike + np.random.randn(64) * 10 for _ in range(200)]
        # Split spike waveforms to have equal number of spikes in each cluster
        cluster_labels = np.random.randint(0, self.cluster_count, len(spike_waveforms))

        self.clus_info_dict = {}
        for ch_name in ch_lbls:
            self.clus_info_dict[ch_name] = {'spikes': np.array([], dtype=int), 'spike_waveforms': spike_waveforms, 
                                            'features': np.empty((0,n_feats), int), 'cluster_labels': cluster_labels, 
                                            'cluster_probs': np.empty((0, self.cluster_count), float)}

    #region Undo/Redo
    
    def add_undo_redo_buttons(self):
        """
        Adds undo and redo buttons to the clustering layout.
        """
        if not hasattr(self, 'btn_undo'):
            self.btn_undo = QtWidgets.QPushButton('undo')
            self.btn_undo.clicked.connect(self.undo)
            self.clustering_params_layout.addWidget(self.btn_undo)

        if not hasattr(self, 'btn_redo'):
            self.btn_redo = QtWidgets.QPushButton('redo')
            self.btn_redo.clicked.connect(self.redo)
            self.clustering_params_layout.addWidget(self.btn_redo)

    def undo(self):
        """
        Undo the last action.
        """
        last_action = self.history.undo()
        
        if last_action is None:
            return

        for key, val in last_action.items():
            if key == 'selection':
                cluster, spikes = val
                for spike in spikes:
                    spike.setPen(pg.mkPen('grey', width=1))
                self.selected_spikes = []

            elif key == 'merge':
                self.clus_info_dict[self.clus_ch] = self.history.cluster_labels
                self.plot_clusters(self.clus_ch)

    def redo(self):
        """
        Redo the last action.
        """
        last_action = self.history.redo()
        if last_action is None:
            return

        for key, val in last_action.items():
            if key == 'selection':
                cluster, spikes = val
                for spike in spikes:
                    spike.setPen(pg.mkPen('magenta', width=1.5))
                self.selected_spikes = spikes

            elif key == 'merge':
                self.clus_info_dict = self.history.clus_info_dict
                self.plot_clusters(self.clus_ch)

    #endregion Undo/Redo
    
    #region Clustering UI

    def create_clusters_ui(self):
        self.pw_clus_list = []
        self.spike_curves = {}
        self.clus_lbls = {}
        self.clus_chks = {}
        self.clus_edit_toggles = {}

        for clus_idx in range(self.cluster_count):
            self.spike_curves[clus_idx] = []

        self.cols = 2 if self.cluster_count <= 8 else 3
        self.rows = np.ceil(self.cluster_count / self.cols).astype(int)
        self.row_span = 24
        self.col_span = 16
        # Clear the layout
        if self.clus_algorithm == 'SPC':
            self.plot_temperature_plot(self.metadata)
            row_start = 1            
        else:
            self.glw_clustering.clear()
            row_start = 0

        clus_idx = 0
        for row in range(row_start, self.rows + row_start):
            for col in range(self.cols):
                self.create_cluster_ui(clus_idx, row, col, self.row_span, self.col_span) 
                clus_idx += 1               
                
            self.glw_clustering.nextRow()

    def create_cluster_ui(self, clus_idx, row, col, row_span, col_span):
        lyt_clus = self.glw_clustering.addLayout(border=(50,0,0), row=row*row_span, 
                                                col=col*col_span, rowspan=row_span, 
                                                colspan=col_span)
        lyt_clus.setSpacing(0)
        lyt_clus.setMinimumHeight(200)

        lyt_info = lyt_clus.addLayout(row=0, col=0, rowspan=1, colspan=col_span)
        lyt_info.setSpacing(0)
        lyt_info.setContentsMargins(1,1,1,1)
        lyt_info.setMaximumHeight(20)

        clus_lbl = pg.LabelItem(f'Cluster {row * self.cols + col} 50/1000')
        clus_lbl.setObjectName(f'lbl_clus_{clus_idx}')
        self.clus_lbls[clus_idx] = clus_lbl

        # Add cluster label
        lyt_info.addItem(clus_lbl, row=0, col=0, rowspan=1, colspan=11)

        switch_control = Switch()
        switch_control.setFixedSize(35, 18)
        switch_control.setObjectName(f'switch_{clus_idx}')
        switch_control.clicked.connect(self.start_clus_edit)
        proxy_switch = QtWidgets.QGraphicsProxyWidget()
        proxy_switch.setWidget(switch_control)
        lyt_info.addItem(proxy_switch, row=0, col=11, rowspan=1, colspan=1)
        self.clus_edit_toggles[clus_idx] = switch_control

        # Add spacing
        lyt_info.addItem(pg.LabelItem(''), row=0, col=12, rowspan=1, colspan=1)

        # Add checkbox for keep
        proxy_k = QtWidgets.QGraphicsProxyWidget()
        rb_keep = QtWidgets.QCheckBox('keep')
        rb_keep.setMaximumWidth(60)
        rb_keep.setObjectName('keep_' + str(clus_idx))
        rb_keep.setChecked(False)
        rb_keep.toolTip = 'Keep this cluster'
        rb_keep.stateChanged.connect(self.clus_edit_chk_toggled)
        proxy_k.setWidget(rb_keep)
        lyt_info.addItem(proxy_k, row=0, col=13,
                         rowspan=1, colspan=1)

        # Add checkbox for meh
        proxy_m = QtWidgets.QGraphicsProxyWidget()
        rb_merge = QtWidgets.QCheckBox('merge')
        rb_merge.setMaximumWidth(60)
        rb_merge.setObjectName('merge_' + str(clus_idx))
        rb_merge.setChecked(False)
        rb_merge.toolTip = 'Merge with another cluster'
        rb_merge.stateChanged.connect(self.clus_edit_chk_toggled)
        proxy_m.setWidget(rb_merge)
        lyt_info.addItem(proxy_m, row=0, col=14)

        proxy_r = QtWidgets.QGraphicsProxyWidget()
        rb_remove = QtWidgets.QCheckBox('remove')
        rb_remove.setMaximumWidth(60)
        rb_remove.setObjectName('remove_' + str(clus_idx))
        rb_remove.setChecked(False)
        rb_remove.toolTip = 'Remove this cluster'
        rb_remove.stateChanged.connect(self.clus_edit_chk_toggled)
        proxy_r.setWidget(rb_remove)
        lyt_info.addItem(proxy_r, row=0, col=15)

        self.clus_chks[clus_idx] = {'keep': rb_keep, 'merge': rb_merge, 'remove': rb_remove}

        lyt_clus.nextRow()
        
        pw_clus = lyt_clus.addPlot(row=1, col=0, 
                                   rowspan=row_span-1, 
                                   colspan=col_span)
        pw_clus.setLabel('left', 'Amplitude (uV)')
        pw_clus.setLabel('bottom', 'Time')
        pw_clus.showGrid(x=True, y=True)
        pw_clus.vb.setObjectName(f'vb_clus_{clus_idx}')

        self.pw_clus_list.append(pw_clus)
        
        # Create a triphasic sEEG spike waveform
        # spike = np.concatenate([np.linspace(0, -1, 10), 
        #                         np.linspace(-1, 0.5, 10), 
        #                         np.linspace(0.5, 0, 10)])
        
    def clus_clicked(self, evt):
        """
        Selects the highlighted spikes.
        """
        if self.curr_clus_idx == -1:
            return
        vb = self.pw_clus_list[self.curr_clus_idx].vb
        pos = evt.scenePos()
        if vb.sceneBoundingRect().contains(pos) and self.mouse_loc is not None:
            with self.edit_lock:
                # Highlight spke curves that are within the roi & unhighlight the rest
                for curve in self.spike_curves[self.curr_clus_idx]:
                    if self.is_within_roi(curve_x=curve.getData()[0], 
                                          curve_y=curve.getData()[1],
                                          roi_pos=self.mouse_loc, roi_size_x=4):
                        curve.setPen(pg.mkPen('magenta', width=1.5))
                        self.selected_spikes.append(curve)
                    else:
                        curve.setPen(pg.mkPen('grey', width=1))

                if len(self.selected_spikes) > 0:
                    self.history.add_selection(self.curr_clus_idx, self.selected_spikes)
                    self.add_undo_redo_buttons()

    def clus_mouseMoved(self, pos):
        if self.curr_clus_idx == -1:
            return
        
        vb = self.pw_clus_list[self.curr_clus_idx].vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = int(mouse_point.x())
            y = int(mouse_point.y())
            # Draw square crop window with mouse cursor at the center (white dotted lines)
            # Get scale of the viewbox
            width = vb.screenGeometry().width()
            height = vb.screenGeometry().height()
            range_x = vb.viewRange()[0][1] - vb.viewRange()[0][0]
            range_y = vb.viewRange()[1][1] - vb.viewRange()[1][0]
            # Set the size of the roi to be be a square with width 25% of viewbox height
            roi_size = range_y * 0.20
            x_size = range_x/width * roi_size
            y_size = range_y/height * roi_size
            
            x1 = x - x_size/2
            x2 = x + x_size/2
            y1 = y - y_size/2
            y2 = y + y_size/2

            if x1 < vb.viewRange()[0][0] or x2 > vb.viewRange()[0][1] or \
                y1 < vb.viewRange()[1][0] or y2 > vb.viewRange()[1][1]:
                if self.b_drawing_roi:
                    vb.removeItem(self.selection_win)
                    self.b_drawing_roi = False
                return
            
            self.selection_win.setData([x1, x1, x2, x2, x1], 
                                        [y1, y2, y2, y1, y1], 
                                        pen=pg.mkPen('b', width=1, style=QtCore.Qt.DotLine))
            vb.addItem(self.selection_win, ignoreBounds=True)
            self.b_drawing_roi = True
            with self.edit_lock:
                # Highlight spike curves that are within the roi & unhighlight the rest
                for curve in self.spike_curves[self.curr_clus_idx]:
                    # Get the RGB values of the curve's color
                    curve_color = curve.opts['pen'].color()
    
                    # Compare the RGB values instead of the color name
                    if curve_color == pg.mkColor('magenta'):
                        continue
                    if self.is_within_roi(curve_x=curve.getData()[0], 
                                          curve_y=curve.getData()[1],
                                          roi_pos=mouse_point, 
                                          roi_size_x=x_size, roi_size_y=y_size):
                        curve.setPen(pg.mkPen('b', width=2))
                    else:
                        curve.setPen(pg.mkPen('grey', width=1))

                self.mouse_loc = mouse_point

    def is_within_roi(self, curve_x, curve_y, roi_pos, roi_size_x, roi_size_y=None):
        """
        Checks if the curve is within the roi.
        """
        roi_x = roi_pos.x()
        roi_y = roi_pos.y()

        if roi_size_y is None:
            roi_size_y = roi_size_x
        
        # Check if any part of the curve is within the roi
        for idx, x in enumerate(curve_x):
            y = curve_y[idx]
            if roi_x - roi_size_x/2 < x < roi_x + roi_size_x/2 and \
               roi_y - roi_size_y/2 < y < roi_y + roi_size_y/2:
                return True

        return False

    def update_selection(self, roi):
        """
        Updates the selection.
        """
        print(roi.pos(), roi.size())
        clus_idx = int(roi.objectName().split('_')[1])
        # Get the selected spikes
        clus_spikes_idxs = self.get_clus_spike_idxs(clus_idx)
        # Get the selected waveforms
        clus_waveforms = [self.clus_info_dict[self.clus_ch]['spike_waveforms'][idx] for idx in clus_spikes_idxs]

        # Use roi to get the selected spikes
        selected_spikes = []

    def start_clus_edit(self):
        """
        Starts the cluster editing process with toggle switch.
        """
        toggle_switch = self.clustering_widget.sender()
        clus_idx = int(toggle_switch.objectName().split('_')[-1])

        if toggle_switch.isChecked():
            self.curr_clus_idx = clus_idx
            # Uncheck the other toggle switches
            for idx in range(self.cluster_count):
                if idx != clus_idx:
                    switch = self.clus_edit_toggles[idx]
                    switch.setChecked(False)
        else:
            self.curr_clus_idx = -1

    def clus_edit_chk_toggled(self):
        """
        Starts the cluster editing process.
        """
        sender = self.clustering_widget.sender()
        if sender.isChecked():
            curr_action = sender.objectName().split('_')[0]
            clus_idx = int(sender.objectName().split('_')[-1])

            # Uncheck the other checkboxes
            for action in ['keep', 'merge', 'remove']:
                if action != curr_action:
                    rb = self.clus_chks[clus_idx][action]
                    rb.setChecked(False)


        # Add an update button to clustering_params_layout
        if not hasattr(self, 'btn_update_clusters'):
            self.btn_update_clusters = QtWidgets.QPushButton('Update Clusters')
            self.btn_update_clusters.clicked.connect(self.update_clusters)
            self.clustering_params_layout.addWidget(self.btn_update_clusters)

    def update_clusters(self):
        """
        Updates the clusters based on the user input.
        """
        self.cluster_labels = self.clus_info_dict[self.clus_ch]['cluster_labels']


        for clus_idx in range(self.cluster_count):
            for action in ['keep', 'merge', 'remove']:
                rb = self.clus_chks[clus_idx][action]
                if rb.isChecked():
                    if action == 'keep':
                        continue
                    elif action == 'merge':
                        self.merge_clusters.append(clus_idx)
                    elif action == 'remove':
                        self.cluster_labels[self.cluster_labels == clus_idx] = -1

        # Merge the clusters
        if len(self.merge_clusters) > 1:
            self.history.clus_info_dict_prev = self.clus_info_dict.copy()
            for clus_idx in self.merge_clusters:
                # Merge the clusters
                self.cluster_labels[self.cluster_labels == clus_idx] = self.merge_clusters[0]

            # Add the merged clusters to the history
            self.history.add_merge(self.merge_clusters)
            # Update the clusters
            self.clus_info_dict[self.clus_ch]['cluster_labels'] = self.cluster_labels

            self.plot_clusters(self.clus_ch)
            self.merge_clusters = []

        # Clear the checkboxes
        for clus_idx in range(self.cluster_count):
            for action in ['keep', 'merge', 'remove']:
                rb = self.clus_chks[clus_idx][action]
                rb.setChecked(False)

        self.add_undo_redo_buttons()
            
    def init_clustering_layout(self):
        """
        Initializes the clustering layout.
        """
        if not hasattr(self, 'clustering_layout'):
            self.clustering_layout = QtWidgets.QVBoxLayout()
            self.clustering_widget = QtWidgets.QWidget()
            self.clustering_widget.setLayout(self.clustering_layout)
            self.clustering_view.addWidget(self.clustering_widget)

            # Add clustering parameters
            self.clustering_params_layout = QtWidgets.QHBoxLayout()

            self.init_clus_cmb()

            self.init_clus_spb()            

            self.clustering_layout.addLayout(self.clustering_params_layout)
            self.glw_clustering = pg.GraphicsLayoutWidget()
            self.glw_clustering.ci.layout.setSpacing(0)

            self.clustering_scroll = QtWidgets.QScrollArea()
            self.clustering_scroll.setWidget(self.glw_clustering)
            self.clustering_scroll.setWidgetResizable(True)
            self.clustering_layout.addWidget(self.clustering_scroll)
            
            self.init_clus_info_dict(['ch1'])
            self.create_clusters_ui()
            self.plot_clusters('ch1')

    def init_clus_cmb(self):
        for param in config['clustering_cmb_params']:
            vLyt = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel(param)
            vLyt.addWidget(lbl)
            cmb = QtWidgets.QComboBox(self.clustering_widget)
            cmb.setObjectName(f'cmb_{param}')
            cmb.addItems(config['clustering_cmb_params'][param])
            cmb.currentIndexChanged.connect(self.update_clustering_params)
            vLyt.addWidget(cmb)
            self.clustering_params_layout.addLayout(vLyt)

    def init_clus_spb(self):
        for param in config['clustering_spb_params']:
            # Spinbox for clustering parameters, use the config values to figure out if doublespinbox or spinbox
            if isinstance(config['clustering_spb_params'][param][0], float):                       
                spb = QtWidgets.QDoubleSpinBox(self.clustering_widget)               
            else:
                spb = QtWidgets.QSpinBox(self.clustering_widget)

            spb.setObjectName(f'spb_{param}')
            spb.setRange(config['clustering_spb_params'][param][1], config['clustering_spb_params'][param][2])
            spb.setValue(config['clustering_spb_params'][param][0])
            spb.setSingleStep(config['clustering_spb_params'][param][3])
            spb.valueChanged.connect(self.update_clustering_params)
            lbl = QtWidgets.QLabel(param)

            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(lbl)
            vLyt.addWidget(spb)
            self.clustering_params_layout.addLayout(vLyt)

    #endregion Clustering UI

    # Delta (0-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), Gamma (30-100 Hz)

    def cluster_spikes(self, features, spikes=None):
        """
        Clusters the spikes.
        """
        # Locals.
        template_matching = False
        n_spikes = 1000
        cluster_labels = []
        cluster_probs = []
        
        # If there's over n_spikes spikes, randomly select n_spikes spikes for clustering, use templates for rest
        if template_matching and \
           features.shape[0] > n_spikes:
            idxs = np.random.choice(features.shape[0], n_spikes, replace=False)
            idxs_others = ~np.isin(np.arange(features.shape[0]), idxs)
            idxs_others = np.where(idxs_others)[0]
            feats = features[idxs]
            feats_others = features[idxs_others]
        else:
            feats = features
            feats_others = None

        if self.clus_algorithm == "GMM":
            self.alg = GaussianMixture(n_components=self.cluster_count, random_state=0)
            cluster_labels = self.alg.fit_predict(feats)
            cluster_probs = self.alg.predict_proba(feats)
        elif self.clus_algorithm == "BGMM":
            self.alg = BayesianGaussianMixture(n_components=self.cluster_count, random_state=0)
            cluster_labels = self.alg.fit_predict(feats)
            cluster_probs = self.alg.predict_proba(feats)
        elif self.clus_algorithm == "KMEANS":
            self.alg = KMeans(n_clusters=self.cluster_count, random_state=0)
            cluster_labels = self.alg.fit_predict(feats)
        elif self.clus_algorithm == "SPECTRAL":
            self.alg = SpectralClustering(affinity='precomputed', n_clusters=self.cluster_count, random_state=0)
            cluster_labels = self.alg.fit_predict(feats)
        elif self.clus_algorithm == "ISOSPLIT6":
            cluster_labels = isosplit6(feats)
            # Subtract 1 as isosplit6 labels start from 1
            cluster_labels -= 1
            self.cluster_count = len(np.unique(cluster_labels))
            self.clus_colors = [pg.intColor(i, hues=self.cluster_count) for i in range(self.cluster_count)]
        elif self.clus_algorithm == "SPC":
            self.alg = SPC()
            cluster_labels, self.metadata = self.alg.fit(feats, min_clus=150, return_metadata=True)
            # plot_temperature_plot(self.metadata)
            # self.plot_temperature_plot(metadata)
                
        # Template matching
        if feats_others is not None:
            clus_lbls_others, clus_probs_others = self.do_template_matching(feats_others, feats, cluster_labels, cluster_probs)

            cluster_labels = np.append(cluster_labels, clus_lbls_others, axis=0)
            cluster_probs = np.append(cluster_probs, clus_probs_others, axis=0)
            # reordering to match features order
            cluster_labels = cluster_labels[np.argsort(np.concatenate([idxs, idxs_others]))]
            cluster_probs = cluster_probs[np.argsort(np.concatenate([idxs, idxs_others]))]

        # plot results
        # self.plot_gmm_results(features, cluster_labels, gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture Model')
        metrics = {}
        if self.clus_algorithm == "GMM" or self.clus_algorithm == "BGMM":
            # self.plot_clustering_results(spikes, np.array(feats), cluster_labels, self.alg)
            metrics = self.evaluate_gmm_clustering(np.array(feats), cluster_labels, self.alg)

        return cluster_labels.astype(int).tolist(), cluster_probs.astype(float).tolist(), metrics
    
    def evaluate_gmm_clustering(self, X, labels, gmm):
        """
        Evaluate GMM clustering using various metrics.
        
        Parameters:
        X (array-like): The input data
        labels (array-like): The cluster labels
        gmm (GaussianMixture): The fitted GMM model
        
        Returns:
        dict: A dictionary containing the computed metrics
        """
        metrics = {}
        
        # Silhouette Score
        metrics['silhouette'] = silhouette_score(X, labels)
        
        # Calinski-Harabasz Index
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        
        # Davies-Bouldin Index
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        
        # BIC
        metrics['bic'] = gmm.bic(X)
        
        # AIC
        metrics['aic'] = gmm.aic(X)
        
        # Log-likelihood
        metrics['log_likelihood'] = gmm.score(X) * X.shape[0]

        # print metrics
        print(f'Evaluation metrics for {self.clus_algorithm}:')
        print(f'Silhouette Score: {metrics["silhouette"]}. Range: [-1, 1]')
        print(f'Calinski-Harabasz Index: {metrics["calinski_harabasz"]}. Higher is better.')
        print(f'Davies-Bouldin Index: {metrics["davies_bouldin"]}. Lower is better.')
        print(f'BIC: {metrics["bic"]}. Lower is better.')
        print(f'AIC: {metrics["aic"]}. Lower is better.')
        print(f'Log-likelihood: {metrics["log_likelihood"]}. Higher is better.')

        self.c.log.emit(f'Evaluation metrics for {self.clus_algorithm}:')
        self.c.log.emit(f'Silhouette Score: {metrics["silhouette"]}. Range: [-1, 1]')
        self.c.log.emit(f'Calinski-Harabasz Index: {metrics["calinski_harabasz"]}. Higher is better.')
        self.c.log.emit(f'Davies-Bouldin Index: {metrics["davies_bouldin"]}. Lower is better.')
        self.c.log.emit(f'BIC: {metrics["bic"]}. Lower is better.')
        self.c.log.emit(f'AIC: {metrics["aic"]}. Lower is better.')
        self.c.log.emit(f'Log-likelihood: {metrics["log_likelihood"]}. Higher is better.')
        
        return metrics

    def isi_violations(self, spike_times, isi_threshold=0.002, min_isi=0.001):
        """
        Calculate ISI (Inter-Spike Interval) violations.
        
        Parameters:
        spike_times (array-like): Array of spike times in seconds
        isi_threshold (float): Threshold for ISI violations (default: 2ms)
        min_isi (float): Minimum allowed ISI (default: 1ms)
        
        Returns:
        float: Proportion of ISI violations
        """
        isi = np.diff(np.sort(spike_times))
        violations = np.sum((isi < isi_threshold) & (isi > min_isi))
        return violations / len(spike_times)

    def isolation_distance(self, X, labels, cluster_id):
        """
        Calculate Isolation Distance for a specific cluster.
        
        Parameters:
        X (array-like): The input data
        labels (array-like): The cluster labels
        cluster_id: The ID of the cluster to evaluate
        
        Returns:
        float: Isolation Distance
        """
        cluster_points = X[labels == cluster_id]
        other_points = X[labels != cluster_id]
        
        n_cluster = len(cluster_points)
        
        if len(other_points) < n_cluster:
            return np.inf
        
        distances = np.sum((other_points[:, np.newaxis, :] - cluster_points[np.newaxis, :, :]) ** 2, axis=2)
        nth_distance = np.partition(distances, n_cluster, axis=0)[n_cluster-1]
        
        return np.mean(nth_distance)
    
    def plot_clustering_results(self, spike_times, X, labels, gmm, n_components=2):
        """
        Create various plots to visualize GMM clustering results.
        
        Parameters:
        X (array-like): The input data
        labels (array-like): The cluster labels
        gmm (GaussianMixture): The fitted GMM model
        n_components (int): Number of components to use for PCA (default: 2)
        """
        # Reduce dimensionality for visualization if needed
        if X.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X

        # Set up the plots
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        
        # 1. Scatter plot of data points
        axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        axs[0, 0].set_title('Clustered Data Points')
        axs[0, 0].set_xlabel('PC1')
        axs[0, 0].set_ylabel('PC2')

        # 2. Contour plot of GMM
        x = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
        y = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
        # If we used PCA, we need to transform the grid points back to the original space
        if X.shape[1] > n_components:
            XX_original = pca.inverse_transform(XX)
            Z = -gmm.score_samples(XX_original)
        else:
            Z = -gmm.score_samples(XX)
        Z = Z.reshape(X_grid.shape)
        axs[0, 1].contourf(X_grid, Y_grid, Z, levels=50, cmap='viridis')
        axs[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c='white', alpha=0.5, s=10)
        axs[0, 1].set_title('GMM Contour Plot')
        axs[0, 1].set_xlabel('PC1')
        axs[0, 1].set_ylabel('PC2')

        # 3. Silhouette plot
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, labels)
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            axs[1, 0].barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1)
            axs[1, 0].text(-0.03, (y_lower + y_upper) / 2, str(i))
            y_lower += len(cluster_silhouette_vals)
        axs[1, 0].set_title('Silhouette Plot')
        axs[1, 0].set_xlabel('Silhouette coefficient values')
        axs[1, 0].set_ylabel('Cluster labels')
        axs[1, 0].axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")

        # 4. Inter-Spike Interval (ISI) histogram
        if spike_times is not None:  # Assuming the 3rd column contains spike times
            spike_times /= 30000  # Convert to seconds
            isi = np.diff(np.sort(spike_times))
            axs[1, 1].hist(isi, bins=50, density=True)
            axs[1, 1].set_title('Inter-Spike Interval Histogram')
            axs[1, 1].set_xlabel('ISI (s)')
            axs[1, 1].set_ylabel('Density')
        else:
            axs[1, 1].text(0.5, 0.5, 'Insufficient dimensions for ISI plot', 
                        ha='center', va='center')

        plt.tight_layout()
        plt.show()
    
    def do_template_matching(self, feats_others, feats, cluster_labels, cluster_probs):
        """
        Performs template matching.
        """
        temp_match_start_time = time.time()
        clus_lbls_others = np.array([], dtype=int)
        clus_probs_others = np.empty((0, self.cluster_count), float)
        # Generate templates for each cluster
        clus_templates_dict = {}
        for cluster in range(self.cluster_count):
            clus_feats = feats[cluster_labels == cluster]
            clus_mean = np.mean(clus_feats, axis=0)
            clus_cov = np.cov(clus_feats.T)
            if np.isnan(clus_cov).any() or \
               np.linalg.matrix_rank(clus_cov) < clus_cov.shape[0]:
                continue
            clus_templates_dict[cluster] = (clus_mean, clus_cov)
        
        # Template matching
        for feat in feats_others:
            max_prob = 0
            max_cluster = -1
            clus_prob = np.zeros((1, self.cluster_count), dtype=float)
            for cluster, (clus_mean, clus_cov) in clus_templates_dict.items():
                prob = np.exp(-0.5 * np.dot(np.dot((feat - clus_mean), np.linalg.inv(clus_cov)), (feat - clus_mean).T))
                clus_prob[0, cluster] = prob
                if prob > max_prob:
                    max_prob = prob
                    max_cluster = cluster
            clus_lbls_others = np.append(clus_lbls_others, max_cluster)
            clus_probs_others = np.append(clus_probs_others, clus_prob, axis=0)

        hrs, mins, secs, ms = get_elapsed_time(start_time=temp_match_start_time)
        print(f'Template matching took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

        return clus_lbls_others, clus_probs_others
    
    def plot_temperature_plot(self, metadata):
        """
        Function to plot temperature map using optional output from fit methods.
        """
        self.temperatures = metadata['temperatures']
        self.glw_clustering.clear()
        lyt_spc_temp = self.glw_clustering.addLayout(border=(50,0,0),  row=0, 
                                                 col=0, rowspan=self.row_span, 
                                                 colspan=self.cols*self.col_span)
        lyt_spc_temp.setSpacing(0)
        lyt_spc_temp.setContentsMargins(1,1,1,1)
        lyt_spc_temp.setMinimumHeight(250)
        self.pw_spc_temp = lyt_spc_temp.addPlot()
        self.pw_clus_list.append(self.pw_spc_temp)
        
        # Set log scale
        self.pw_spc_temp.setLogMode(y=True)
        num_lines = metadata['sizes'].T.shape[0]  # Number of lines
        for i, size in enumerate(metadata['sizes'].T):
            color = pg.hsvColor(i / num_lines)  # Generate a unique color for this line
            self.pw_spc_temp.plot(metadata['temperatures'], size, 
                                  pen=pg.mkPen(color=color, width=2))
        self.pw_spc_temp.setLabel('left', 'Cluster Sizes')
        self.pw_spc_temp.setLabel('bottom', 'Temperatures')

        if metadata['method'] == 'WC3':
            self.pw_spc_temp.addLine(x=metadata['temperatures'][metadata['method_info']['elbow']], pen=pg.mkPen('k', width=1, style=pg.QtCore.Qt.DashLine))
            self.pw_spc_temp.plot(metadata['temperatures'][metadata['method_info']['peaks_temp']],
                    metadata['sizes'][metadata['method_info']['peaks_temp'], metadata['method_info']['peaks_cl']], pen=None, symbol='x', symbolPen='k', symbolSize=10)
        self.cluster_count = metadata['sizes'].shape[1]
        self.clus_colors = [pg.intColor(i, hues=self.cluster_count) for i in range(self.cluster_count)]
        for c, info in metadata['clusters_info'].items():
            self.pw_spc_temp.plot([metadata['temperatures'][info['itemp']]], 
                                  [metadata['sizes'][info['itemp'], info['index']]], 
                                  pen=None, symbol='o', symbolSize=9, symbolPen=None, 
                                  symbolBrush=pg.mkBrush(color=pg.intColor(c, hues=self.cluster_count)))
            
        
        # Crosshair
        self.pw_spc_temp.addItem(self.vLine, ignoreBounds=True)
        self.pw_spc_temp.addItem(self.hLine, ignoreBounds=True)
        self.vb_spc_temp = self.pw_spc_temp.getViewBox()

        # label at top right of viewbox
        self.lbl_temp = pg.LabelItem(justify='right')
        self.pw_spc_temp.addItem(self.lbl_temp)

        self.pw_spc_temp.scene().sigMouseMoved.connect(self.spc_mouseMoved)
        self.pw_spc_temp.scene().sigMouseClicked.connect(self.spc_mouseClicked)

    def spc_mouseMoved(self, evt):
        pos = evt
        if self.pw_spc_temp.sceneBoundingRect().contains(pos):
            mouse_loc = self.vb_spc_temp.mapSceneToView(pos)
            index = mouse_loc.x()
            if index > self.temperatures[0] and index < self.temperatures[-1]:
                self.lbl_temp.setText("<span style='font-size: 12pt'>temp=%0.2f, <span style='color: black'>size=%0.1f</span>" % (mouse_loc.x(), mouse_loc.y()))
                self.vLine.setPos(mouse_loc.x())
                self.hLine.setPos(mouse_loc.y())

    def spc_mouseClicked(self, evt):
        # Set temperature to the clicked point
        pos = evt
        if self.pw_spc_temp.sceneBoundingRect().contains(pos):
            mouse_loc = self.vb_spc_temp.mapSceneToView(pos)
            index = mouse_loc.x()
            if index > self.temperatures[0] and index < self.temperatures[-1]:
                self.update_clustering_params(index)
    
    def plot_gmm_results(self, X, Y_, means, covariances, index, title):
        """
        Plots the results of the Gaussian Mixture Model.
        """
        splot = self.pw_clus_list[index]
        splot.clear()
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                    'darkorange'])
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            splot.setData(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi

    def get_clus_spike_idxs(self, cluster):
        if self.clus_algorithm == "GMM" or self.clus_algorithm == "BGMM":
            # Get the spike indices for the current cluster with probability > 0.5
            # clus_spike_idxs = np.where((self.cluster_labels == cluster) & (self.cluster_probs[:, cluster] > self.clus_prob_thr))[0].astype(int)       
            # clus_spike_idxs = np.where(self.cluster_labels == cluster)[0].astype(int) 
            clus_spike_idxs = [spike_idx for spike_idx, prob in enumerate(self.cluster_probs) if prob[cluster] > self.clus_prob_thr]
        else:
            # clus_spike_idxs = np.where(self.cluster_labels == cluster)[0].astype(int)     
            clus_spike_idxs = [spike_idx for spike_idx, lbl in enumerate(self.cluster_labels) if lbl == cluster]
        return clus_spike_idxs
    
    def plot_clusters(self, channel):
        """
        Plots the clustered spikes.

        Parameters:
        - channel: The channel for which to plot the clustered spikes.

        Returns:
        None
        """
        spike_waveforms = self.clus_info_dict[channel]['spike_waveforms']
        self.cluster_labels = self.clus_info_dict[channel]['cluster_labels']
        self.cluster_probs = self.clus_info_dict[channel]['cluster_probs']
        clus_spike_dict = {}
        # Plot the clustered spikes
        for cluster in range(self.cluster_count):    
            clus_spike_dict[cluster] = self.get_clus_spike_idxs(cluster)

        self.create_clusters_ui()
        tot_spikes = len(spike_waveforms)
        # Sort the clusters by the number of spikes
        # clus_spike_dict = dict(sorted(clus_spike_dict.items(), key=lambda item: len(item[1]), reverse=True))

        plt_idx = 0
        for clus, spikes in clus_spike_dict.items():
            pw_clus = self.pw_clus_list[plt_idx]
            plt_idx += 1
            pw_clus.clear()
            self.spike_curves[clus] = []
            clus_waveforms = []
            tot_spike_count = len(spikes)
            if len(spikes) == 0:  
                self.clus_lbls[clus].setText(f'Cluster {clus} (0 spikes)')
                continue
            elif len(spikes) > self.clus_spike_count:
                # clus_spike_idxs = np.random.choice(clus_spike_idxs, self.clus_spike_count, replace=False)
                spikes = spikes[:self.clus_spike_count]
            for idx in spikes:
                waveform = spike_waveforms[idx]
                spike_curve = pw_clus.plot(waveform, pen=pg.mkPen('grey', width=1))
                spike_curve.curve.setClickable(False)
                self.spike_curves[clus].append(spike_curve)
                clus_waveforms.append(waveform)
            # plot avg waveform
            avg_waveform = np.mean(clus_waveforms, axis=0)
            std_waveform = np.std(clus_waveforms, axis=0)
            pw_clus.plot(avg_waveform, pen=pg.mkPen(self.clus_colors[clus], width=2))
            pw_clus.plot(avg_waveform + std_waveform, pen=pg.mkPen(self.clus_colors[clus], width=1.5, style=QtCore.Qt.DotLine))
            pw_clus.plot(avg_waveform - std_waveform, pen=pg.mkPen(self.clus_colors[clus], width=1.5, style=QtCore.Qt.DotLine))
            self.clus_lbls[clus].setText(f'Cluster {clus} ({tot_spike_count/tot_spikes*100:.0f}%) ({len(spikes)}/{tot_spike_count} spikes)')
            self.c.log.emit(f'{channel} cluster {clus} ({np.sum(self.cluster_labels == clus)} spikes)')
            pw_clus.scene().sigMouseClicked.connect(self.clus_clicked)
            pw_clus.scene().sigMouseMoved.connect(self.clus_mouseMoved)

            

        # Plot the clustered spikes on the signal viewer
        # Mark spikes on filtered curve with different colors for each cluster
        # spike_plot = self.pw_spike_plots[0]
        # spike_plot.clear()
        # for cluster in range(self.cluster_count):
        #     clus_spike_idxs = np.where(self.cluster_labels == cluster)[0].astype(int)
            # spike_plot.setData(x=self.t[spikes[clus]], y=filtered_data[spikes[clus]], pen=None, symbol='o', symbolPen=self.clus_colors[cluster], symbolSize=5, symbolBrush=None)

    def update_clustering_params(self, cmb_index):
        """
        Updates the clustering parameters.
        """
        value = None
        try:
            sender = self.clustering_widget.sender()
            param = sender.objectName()
            if param.startswith('cmb'):
                value = sender.currentText()
            else:
                value = sender.value()

            if value == None or value == '':
                return
            
            if self.clus_ch is not None and self.clus_ch != '':
                self.history.cluster_labels = copy.deepcopy(self.clus_info_dict[self.clus_ch]['cluster_labels'])

            if param == 'cmb_clusters':
                self.clus_colors = [pg.intColor(i, hues=self.cluster_count) for i in range(self.cluster_count)]
                # self.init_clustering_layout()
                # self.plot_clusters()
            elif param == 'cmb_algorithm':
                self.clus_algorithm = value
                if not hasattr(self,'features'):
                    # self.features = extract_features(self.clus_info_dict[self.clus_ch]['spike_waveforms'])
                    self.features = np.array(self.clus_info_dict[self.clus_ch]['spike_waveforms'])
                self.cluster_labels, self.cluster_probs, metrics = self.cluster_spikes(self.features)
                self.clus_info_dict[self.clus_ch]['cluster_labels'] = self.cluster_labels
                self.clus_info_dict[self.clus_ch]['cluster_probs'] = self.cluster_probs
                self.clus_info_dict[self.clus_ch]['metrics'] = metrics
            elif param == 'cmb_channels':
                self.clus_ch = value
                # self.features = extract_features(self.clus_info_dict[value]['spike_waveforms'])
                self.features = np.array(self.clus_info_dict[self.clus_ch]['spike_waveforms'])
                self.cluster_labels  = self.clus_info_dict[value]['cluster_labels']
                self.cluster_probs   = self.clus_info_dict[value]['cluster_probs']
            elif param == 'spb_pca_components':
                self.pca_n_components = value
                # self.features = extract_features(self.clus_info_dict[self.clus_ch]['spike_waveforms'])
                self.features = np.array(self.clus_info_dict[self.clus_ch]['spike_waveforms'])
                self.spikes = np.array(self.clus_info_dict[self.clus_ch]['spikes'], dtype=np.float32)
                self.cluster_labels, self.cluster_probs, metrics = self.cluster_spikes(self.features)
                self.clus_info_dict[self.clus_ch]['cluster_labels'] = self.cluster_labels
                self.clus_info_dict[self.clus_ch]['cluster_probs'] = self.cluster_probs
                self.clus_info_dict[self.clus_ch]['metrics'] = metrics
            elif param == 'spb_cluster_count':
                self.cluster_count = value
                self.features = self.clus_info_dict[self.clus_ch]['features']
                self.spikes = np.array(self.clus_info_dict[self.clus_ch]['spikes'], dtype=np.float32)
                self.cluster_labels, self.cluster_probs, metrics = self.cluster_spikes(self.features)
                self.clus_info_dict[self.clus_ch]['cluster_labels'] = self.cluster_labels
                self.clus_info_dict[self.clus_ch]['cluster_probs'] = self.cluster_probs
                self.clus_info_dict[self.clus_ch]['metrics'] = metrics
            elif param == 'spb_clus_spike_count':
                self.clus_spike_count = value
            elif param == 'spb_clus_prob_thr':
                self.clus_prob_thr = value
            
            self.plot_clusters(channel=self.clus_ch)
        except:
            print(traceback.format_exc())
            print(value)

    def update_clustering_cmb_channels(self, ch_lbls):
        """
        Updates the combobox in clustering tab.
        """
        cmb_channels = self.clustering_widget.findChild(QtWidgets.QComboBox, 'cmb_channels')
        try:
            cmb_channels.currentIndexChanged.disconnect(self.update_clustering_params)
        except:
            pass
        cmb_channels.clear()
        for ch_idx, ch_name in enumerate(ch_lbls):
            cmb_channels.addItem(ch_name)

        cmb_channels.currentIndexChanged.connect(self.update_clustering_params)

    def process_nsx_data(self, raw_data, ch_lbls, bundle=None):
        """
        Processes the raw data in all channels and returns
        spikes, spike_waveforms, which cluster they belong to.

        Args:
            raw_data: raw data
            ch_lbls: channel labels

        """
        b_multi_threaded = False
        num_threads = None # os.cpu_count() * 2

        results = []

        # Initialize a counters
        completed_tasks = 0
        total_tasks = len(raw_data)

        process_start_time = time.time()

        if b_multi_threaded:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_ch_raw_data, data, ch_lbls[ch_idx]): ch_idx for ch_idx, data in enumerate(raw_data)}
                for future in concurrent.futures.as_completed(futures):
                    ch_idx = futures[future]
                    try:
                        result = future.result()
                        results.append(future)
                    except Exception as exc:
                        print(f'Channel {ch_idx} generated an exception: {exc}')
                    else:
                        # Increment the counter when a task is completed
                        completed_tasks += 1
                        progress_percentage = (completed_tasks / total_tasks) * 100
                        print(f'Channel {ch_idx} is processed. Progress: {progress_percentage:.2f}%')
                        
                        if completed_tasks > 4:
                            hrs, mins, secs, ms = get_elapsed_time(start_time=process_start_time)
                            # Remaining time = elapsed time / completed tasks * remaining tasks
                            remaining_time = (hrs * 3600 + mins * 60 + secs) / completed_tasks * (total_tasks - completed_tasks)
                            hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
                            # print(f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                            self.c.progress.emit(int(progress_percentage), f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')
                        else:
                            self.c.progress.emit(int(progress_percentage), None)

            for result in results:
                try:
                    ch_name, spikes, spike_waveforms, thr, features, cluster_labels, cluster_probs = result.result()
                    self.update_clus_info_dict(ch_name, spikes, spike_waveforms, thr, features, cluster_labels, cluster_probs)
                except:
                    print(f'Error processing results from channel {ch_name}')
                    print(traceback.format_exc())

        else:
            for ch_idx, data in enumerate(raw_data):
                try:
                    ch_name, spikes, spike_waveforms, thr, features, cluster_labels, cluster_probs = self.process_ch_raw_data(data, ch_lbls[ch_idx])
                    self.update_clus_info_dict(ch_name, spikes, spike_waveforms, thr, features, cluster_labels, cluster_probs)

                    completed_tasks += 1
                    progress_percentage = (completed_tasks / total_tasks) * 100
                    hrs, mins, secs, ms = get_elapsed_time(start_time=process_start_time)
                    # Remaining time = elapsed time / completed tasks * remaining tasks
                    remaining_time = (hrs * 3600 + mins * 60 + secs) / completed_tasks * (total_tasks - completed_tasks)
                    hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
                    # print(f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                    self.c.progress.emit(int(progress_percentage), f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')
                except:
                    print(f'Error processing channel {ch_name}')
                    print(traceback.format_exc())

        # if bundle is not None:
        #     self.remove_collisions_bundle(ch_lbls=ch_lbls)

    def update_clus_info_dict(self, ch_name, spikes, spike_waveforms, 
                              thr, features, cluster_labels, cluster_probs):
        self.clus_info_dict[ch_name]['spikes'].extend(spikes)
        self.clus_info_dict[ch_name]['spike_waveforms'].extend(spike_waveforms)
        self.clus_info_dict[ch_name]['threshold'].extend(thr)
        self.clus_info_dict[ch_name]['features'].extend(features)
        self.clus_info_dict[ch_name]['cluster_labels'].extend(cluster_labels)
        self.clus_info_dict[ch_name]['cluster_probs'].extend(cluster_probs)

    def write_clus_info_dict_to_file(self):
        """
        Writes the results to a file.
        """
        write_clus_info_dict_start_time = time.time()
        if os.path.exists(self.nsx_path):
            save_folder = os.path.dirname(self.nsx_path)
        else:
            save_folder = os.path.dirname(__file__)
            print(f'NSX file not found. Saving to {save_folder}.')
        with open(f'{save_folder}/clus_info_dict.json', 'w') as f:
            json.dump(self.clus_info_dict, f, indent=4)

        hrs, mins, secs, ms = get_elapsed_time(start_time=write_clus_info_dict_start_time)
        print(f'Writing clus_info_dict to file took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

    def load_clus_info_dict_from_file(self, clus_info_dict_path):
        """
        Loads the results from a file.
        """
        bOK = False
        clus_info_dict_path = os.path.join(os.path.dirname(clus_info_dict_path), 'clus_info_dict.json')
        if os.path.exists(clus_info_dict_path):
            with open(clus_info_dict_path, 'r') as f:
                self.clus_info_dict = json.load(f)
            bOK = True
        
        return bOK
    
    def load_clus_info_dict_from_times_mat(self, study_folder, nsx_info):
        """
        Loads the results from a file.
        """
        self.clus_info_dict = {}
        self.clus_algorithm = 'WC3'
        for ch in nsx_info['NSx']:
            ch_name = ch['output_name']
            self.clus_info_dict[ch_name] = {'spikes': [], 'spike_waveforms': [], 'threshold': [], 'features': [], 'cluster_labels': [], 'cluster_probs': []}
            times_mat_file = os.path.join(study_folder, f'times_{ch_name}.mat')
            if os.path.exists(times_mat_file):
                try:
                    ch_times = loadmat(times_mat_file)
                    self.clus_info_dict[ch_name]['spikes'].extend(ch_times['cluster_class'][:,1].tolist())
                    self.clus_info_dict[ch_name]['spike_waveforms'].extend(ch_times['spikes'].tolist())
                    # self.clus_info_dict[ch_name]['threshold'].extend(thr)
                    self.clus_info_dict[ch_name]['features'].extend(ch_times['inspk'].tolist())
                    self.clus_info_dict[ch_name]['cluster_labels'].extend(ch_times['cluster_class'][:,0].astype(int).tolist())
                    # subtract 1 from cluster labels (MATLAB) to make them 0-based
                    self.clus_info_dict[ch_name]['cluster_labels'] = [lbl - 1 for lbl in self.clus_info_dict[ch_name]['cluster_labels']]
                    # self.clus_info_dict[ch_name]['cluster_probs'].extend(cluster_probs)
                except:
                    print(traceback.format_exc())

    def remove_collisions_bundle(self, ch_lbls):
        """
        Removes collisions in the bundle.
        """
        remove_collisions_bundle_start_time = time.time()
        b_multi_threaded = True
        bundle_min_artifacts = 6 # atleast 6 channels should have a spike in t_win ms
        t_win = 0.5 * 30 # 0.5 ms in samples
        bundle_spikes = []
        bundle_spikes_ch_lbls = []
        for ch_idx, ch_name in enumerate(ch_lbls):
            bundle_spikes.extend(self.clus_info_dict[ch_name]['spikes'])
            bundle_spikes_ch_lbls.extend(ch_idx * np.ones(self.clus_info_dict[ch_name]['spikes'].shape[0], dtype=int))

        bundle_spikes_sorted_idxs = np.argsort(bundle_spikes)
        bundle_spikes_sorted = np.array(bundle_spikes)[bundle_spikes_sorted_idxs]
        bundle_spikes_ch_lbls_sorted = np.array(bundle_spikes_ch_lbls)[bundle_spikes_sorted_idxs]

        # Remove collisions (spikes within t_win ms)
        artifact_idxs = []
        b_artifact_arr = np.zeros_like(bundle_spikes_sorted, dtype=bool)

        if b_multi_threaded:
            num_splits = os.cpu_count()
            split_idxs = np.arange(0, len(bundle_spikes_sorted), len(bundle_spikes_sorted)//num_splits)
            split_idxs[-1] = len(bundle_spikes_sorted)
            futures = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_splits) as executor:
                for i in range(num_splits-1):
                    buffer_spks = 1
                    if split_idxs[i+1] + buffer_spks > len(bundle_spikes_sorted):
                        buffer_spks = 0
                    else:
                        while bundle_spikes_sorted[split_idxs[i+1]+buffer_spks] < bundle_spikes_sorted[split_idxs[i+1]] + t_win:
                            buffer_spks += 1
                    futures[executor.submit(self.detect_artifacts, bundle_spikes_sorted[split_idxs[i]:split_idxs[i+1]+buffer_spks], t_win, bundle_min_artifacts)] = i
                # futures = {executor.submit(self.detect_artifacts, bundle_spikes_sorted[split_idxs[i]:split_idxs[i+1]], t_win, bundle_min_artifacts): i for i in range(num_splits)}
                for future in concurrent.futures.as_completed(futures):
                    split_idx = futures[future]
                    try:
                        artifact_idxs_split = future.result()
                        artifact_idxs.extend(artifact_idxs_split)
                    except:
                        print(traceback.format_exc())
        else:
            artifact_idxs = self.detect_artifacts(bundle_spikes_sorted, t_win, bundle_min_artifacts)
        
        artifact_idxs = np.unique(artifact_idxs)
        

        # try:
        #     if len(artifact_idxs) > 0:
        #         b_artifact_arr[np.array(artifact_idxs, dtype=int)] = True
        #         for ch_idx, ch_name in enumerate(ch_lbls):
        #             ch_spikes = bundle_spikes_sorted[bundle_spikes_ch_lbls_sorted == ch_idx & ~b_artifact_arr]
        #             self.clus_info_dict[ch_name]['spikes'] = ch_spikes
        #             # Label artifacts as -1
        #             self.clus_info_dict[ch_name]['cluster_labels'][b_artifact_arr[bundle_spikes_ch_lbls_sorted == ch_idx]] = -1
        # except:
        #     print(f'{ch_name}:{traceback.format_exc()}')

        print(f'Removing {len(artifact_idxs)} collisions in the bundle {ch_lbls[0].split(" ")[0]}.')
        art_perc = len(artifact_idxs)/len(bundle_spikes_sorted) * 100
        self.c.log.emit(f'Removing {len(artifact_idxs)}/{len(bundle_spikes_sorted)}({art_perc:.2f}%) collisions in the bundle {ch_lbls[0].split(" ")[0]}.')
        
        hrs, mins, secs, ms = get_elapsed_time(start_time=remove_collisions_bundle_start_time)
        print(f'Removing collisions in the bundle {ch_lbls[0].split(" ")[0]} {len(artifact_idxs)}/{len(bundle_spikes_sorted)}({art_perc:.2f}%) took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

    def detect_artifacts(self, spikes, t_win, min_artifacts):
        """
        Detects artifacts in the spikes.
        """
        artifact_idxs = []
        for spike in spikes:
            spike_idxs_t_win = np.where((spikes >= spike) & (spikes <= spike + t_win))[0]
            if len(spike_idxs_t_win) >= min_artifacts:
                artifact_idxs.extend(spike_idxs_t_win)
        return artifact_idxs

    def process_bundle_raw_data(self, file_path, raw_data, bundles, ch_lbls):
        """
        Processes the raw data in a bundle
        """
        try:
            self.nsx_path = file_path
            # Get only bundles that are micros
            bundles = {bundle: chs for bundle, chs in bundles.items() if bundle.startswith('m')}
            for bundle, chs in bundles.items(): 
                bundle_start_time = time.time()

                raw_ch_idx = [ch_lbls.index(ch) for ch in chs]
                bundle_raw_data = raw_data[raw_ch_idx]
                self.process_nsx_data(bundle_raw_data, chs, bundle)
                hrs, mins, secs, ms = get_elapsed_time(start_time=bundle_start_time)
                print(f'Processing bundle {bundle} took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                # self.c.log.emit(f'Processing bundle {bundle} took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
            # Process the rest of the channels
            rest_chs = [ch for ch in ch_lbls if not ch.startswith('m')]
            rest_ch_idx = [ch_lbls.index(ch) for ch in rest_chs]
            rest_raw_data = raw_data[rest_ch_idx]
            self.process_nsx_data(rest_raw_data, rest_chs)
            # Write the results to a file
            self.write_clus_info_dict_to_file()
        except :
            print(traceback.format_exc())  

    def process_raw_data(self, file_path, raw_data, ch_lbls):
        """
        Processes the raw data in all channels and returns
        spikes, spike_waveforms, which cluster they belong to.

        Args:
            raw_data: raw data
            ch_lbls: channel labels

        """
        try:
            self.nsx_path = file_path
            process_start_time = time.time()
            self.process_nsx_data(raw_data, ch_lbls)
            hrs, mins, secs, ms = get_elapsed_time(start_time=process_start_time)
            print(f'Processing {len(ch_lbls)} channels took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
            self.c.log.emit(f'Processing {len(ch_lbls)} channels took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

            # self.c.log.emit(f'Processing all channels took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
            # Write the results to a file
            self.write_clus_info_dict_to_file()
        except :
            print(traceback.format_exc())

    def compute_photo_diode_error(self, spike_idxs):
        """
        Computes the photo diod error.
        """
        try:
            if len(spike_idxs) > 0:
                photo_diff = np.diff(spike_idxs)
                photo_diff -= 15000 # 500ms is the expected delay
                nnz = np.where((photo_diff > 15) & (photo_diff < 30000))[0]
                print(f'Photodiode error: {len(nnz)}/{len(photo_diff)} {len(nnz)/len(photo_diff)*100:.2f}%')
                self.c.log.emit(f'Photodiode error: {len(nnz)}/{len(photo_diff)} {len(nnz)/len(photo_diff)*100:.2f}%')
        except:
            print(traceback.format_exc())

    def process_ch_raw_data(self, raw_data, ch_name):
        """
        Processes the raw data in a channel and returns
        spikes, spike_waveforms, which cluster they belong to.

        Args:
            raw_data: raw data
            ch_idx: channel index

        Returns:
            ch_name: channel name
            spikes: detected spikes
            spike_waveforms: extracted spike waveforms
            features: extracted features
            cluster_labels: cluster labels
            cluster_probs: cluster probabilities

        """
        spike_idxs = []
        spike_waveforms = []
        thr = []
        features = []
        cluster_labels = []
        cluster_probs = []
        try:
            start_time = time.time()
            filter_start_time = time.time()
            
            if ch_name.lower().find('photo') != -1:
                spike_idxs, thr = detect_spikes_in_segments(data=raw_data, photo=True)
                self.compute_photo_diode_error(spike_idxs)

                return ch_name, spike_idxs, spike_waveforms, thr, features, cluster_labels, cluster_probs
            elif ch_name.lower().find('mic') != -1 or ch_name.lower().find('parallel') != -1:
                return ch_name, spike_idxs, spike_waveforms, thr, features, cluster_labels, cluster_probs
            else:
                # if raw_data.size >= 0: # To temporarily skip clustering
                #     return ch_name, spikes, spike_waveforms, features, cluster_labels, cluster_probs
                spike_detection_start_time = time.time()

                # xf_sort = self.filtering.spk_detection_filter(data=raw_data, channel=ch_name, filt_order=2)
                # xf_detect = self.filtering.spk_detection_filter(data=raw_data, channel=ch_name, filt_order=4)
                

                # spike_waveforms, spike_idxs, thr = amp_detect_spikes_in_segments(xf=xf_sort, xf_detect=xf_detect)
                # with self.spk_detect_lock:
                filtered_data = self.filtering.spk_detection_filter(data=raw_data, channel=ch_name)
                hrs, mins, secs, ms = get_elapsed_time(start_time=filter_start_time)
                print(f'Filtering {ch_name} took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                spike_idxs, thr = detect_spikes_in_segments(data=filtered_data)
                thr = [thr]
                
                hrs, mins, secs, ms = get_elapsed_time(start_time=spike_detection_start_time)
                print(f'{ch_name} Spike detection ({len(spike_idxs)} spikes, thr:{thr}) took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                # self.c.log.emit(f'{ch_name} Spike detection ({len(spike_idxs)} spikes) took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

                spike_extraction_start_time = time.time()
                spike_waveforms = extract_spikes(data=filtered_data, spikes=spike_idxs)
                hrs, mins, secs, ms = get_elapsed_time(start_time=spike_extraction_start_time)
                print(f'{ch_name} Spike extraction took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                # self.c.log.emit(f'{ch_name} Spike extraction took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

                if len(spike_idxs) < 12:
                    print(f'Skipping clustering ({len(spike_idxs)} spikes detected in {ch_name}).')
                    self.c.log.emit(f'Skipping clustering ({len(spike_idxs)} spikes detected in {ch_name}).')
                    return ch_name, spike_idxs, spike_waveforms, thr, features, cluster_labels, cluster_probs

                feature_extraction_start_time = time.time()
                features = extract_features(spike_waveforms)
                hrs, mins, secs, ms = get_elapsed_time(start_time=feature_extraction_start_time)
                print(f'{ch_name} Feature extraction took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                # self.c.log.emit(f'{ch_name} Feature extraction took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')

                clustering_start_time = time.time()
                # cluster_labels, cluster_probs = self.cluster_spikes(features)
                cluster_labels, cluster_probs, metrics = self.cluster_spikes(np.array(spike_waveforms))
                hrs, mins, secs, ms = get_elapsed_time(start_time=clustering_start_time)
                print(f'{ch_name} Clustering took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                # self.c.log.emit(f'{ch_name} Clustering took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                
                hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
                print(f'{ch_name}({len(spike_idxs)} spikes) processed in {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                self.c.log.emit(f'{ch_name}({len(spike_idxs)} spikes) processed in {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
        except:
            print(f'{ch_name}({len(spike_idxs)}):{traceback.format_exc()}')
            self.c.log.emit(f'{ch_name}({len(spike_idxs)}):{traceback.format_exc()}')
            return ch_name, spike_idxs, spike_waveforms, thr, features, cluster_labels, cluster_probs

        return ch_name, spike_idxs, spike_waveforms, thr, features, cluster_labels, cluster_probs