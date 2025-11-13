# Author: Sunil Mathew
# Date: 29 November 2023
# Orchestrator class for the sEEG analysis suite. 
# This class is responsible for managing the signal processing 
# and visualization for the sEEG data acquired during the RSVPSCR study.

import os

import numpy as np
from datetime import datetime

from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtWidgets import QApplication, QWidget
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtGui import QPalette

from pyqtgraph.dockarea.Dock import Dock

import sys
import traceback
sys.path.append(os.path.dirname(__file__))
from core.seeg_3d import SEEG_3D
from dsp.filtering import Filtering
from core.signal_viewer import SignalViewer
from dsp.power_spectrum import PowerSpectrum
from dsp.clustering import Clustering
from tasks.study_info import StudyInfo
from acquisition.ripple_acquisition import RippleAcq
from nsx.process_ripple_nsx import RippleNsx
from nsx.process_blackrock_nsx import BlackrockNsx

from core.responses import ResponseViewer
from core.concepts import ConceptsStudio
from tasks.tasks import Tasks

# import spikeinterface.sorters as ss

class Communicate(QObject):
    log = Signal(str)
    show_stim_img = Signal()
    img_list = Signal(list)
    img = Signal(int, object, str)
    trellis = Signal(bool)
    record = Signal(bool)
    recording_status = Signal(bool)
    progress = Signal(int, str)
    # self.c.analyze.emit(self.seq_imgs, self.seq_img_paths, self.subscr_idx, self.subscr_seq_idx)
    analyze = Signal(list, list, int, int)
    plot_clusters = Signal(bool)
    start_task = Signal(bool)
    start_task_acq = Signal(bool, int, int)
    task_win = Signal(bool)
    task_clus_param = Signal(str, object)
    mark_raster = Signal(int)
    nsx_read = Signal(bool)    
    nsx_processed = Signal(bool)    
    channel_waveform_clicked = Signal(int)
    sig_view_param = Signal(str, object)
    sig_view_filt_param = Signal(str, object)
    sig_view_nav = Signal(str)
    sig_view_go_to = Signal(datetime)
    sig_view_acq = Signal(bool)
    sig_view_acq_channels = Signal(dict)
    sig_view_acq_channels_selection = Signal(dict)
    related_concepts_single_done = Signal()
    related_concepts_done = Signal()
    exp_config = Signal(str)

class Orchestrator:
    def __init__(self, sig_view=None, ui=True) -> None:
        np.random.seed(123)
        self.has_ui = ui
        self.sig_view = sig_view
        self.c = Communicate()
        self.asc = True
        self.init_signal_params()
        self.init_ui_layout()
        self.init_comm()
        self.init_ripple()
        self.init_ripple_nsx()

    #region Communication

    def init_comm(self):
        """
        Use signals and slots to update the log to prevent the GUI from freezing
        """
        # Add a signal to update the log
        self.c.log.connect(self.update_log)
        self.c.show_stim_img.connect(self.show_stim_img)
        self.c.img_list.connect(self.update_subscreening_img_list)
        self.c.img.connect(self.update_subscreening_img)
        self.c.trellis.connect(self.update_trellis_status)
        self.c.record.connect(self.start_or_stop_recording)
        self.c.recording_status.connect(self.update_recording_status)
        self.c.analyze.connect(self.analyze_subscreening_seq)
        self.c.plot_clusters.connect(self.plot_clusters)
        self.c.start_task.connect(self.start_task)
        self.c.start_task_acq.connect(self.start_stop_task_acquisition)
        self.c.task_win.connect(self.tasks.stop_task)
        self.c.task_clus_param.connect(self.update_task_clus_param)
        self.c.mark_raster.connect(self.responses_list.mark_raster)
        self.c.nsx_read.connect(self.post_read_nsx)
        self.c.nsx_processed.connect(self.post_clustering_nsx)
        self.c.channel_waveform_clicked.connect(self.channel_waveform_clicked)
        self.c.sig_view_param.connect(self.update_sig_view_param)
        self.c.sig_view_filt_param.connect(self.update_sig_view_filt_param)
        self.c.sig_view_nav.connect(self.nav_sig_view)
        self.c.sig_view_go_to.connect(self.go_to_time)
        self.c.sig_view_acq.connect(self.start_stop_acquisition)
        self.c.sig_view_acq_channels.connect(self.signal_viewer.update_channel_list)
        self.c.sig_view_acq_channels_selection.connect(self.update_channel_selection)
        self.c.related_concepts_single_done.connect(self.update_related_concepts_single)
        self.c.related_concepts_done.connect(self.update_related_concepts)
        self.c.exp_config.connect(self.update_exp_config)

    def show_stim_img(self):
        self.tasks.show_stim_img()

    def update_exp_config(self, config):
        self.concepts_studio.update_exp_config(config)

    def update_related_concepts_single(self):
        self.concepts_studio.update_related_concepts(b_llm=True)

    def update_related_concepts(self):
        self.concepts_studio.post_process_related_concepts()

    def start_task(self, state):
        bOK = state
        if not bOK:
            return
        if not self.ripple_acq.b_trellis:
            bOK = self.ripple_acq.init_ripple()
        if bOK:
            self.tasks.start_rsvp_task()
        else:
            # pop up a message box
            trellis_msg_box = QtWidgets.QMessageBox()
            trellis_msg_box.setIcon(QtWidgets.QMessageBox.Information)
            trellis_msg_box.setText("Trellis is not running. Please start Trellis and try again.")
            trellis_msg_box.setWindowTitle("No pictures found")
            trellis_msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            # msgBox.buttonClicked.connect(msgButtonClick)

            returnValue = trellis_msg_box.exec()
            if returnValue == QtWidgets.QMessageBox.Ok:
                print('OK clicked')

    def update_trellis_status(self, state):
        self.tasks.b_trellis = state

    def start_stop_acquisition(self, state):
        bOK = False
        if state:
            bOK = self.ripple_acq.start_acq()
            if bOK:
                self.signal_viewer.update_acq_btn(acq=True)
        else:
            bOK = self.ripple_acq.stop_acq()
            if bOK:
                self.signal_viewer.update_acq_btn(acq=False)

        self.b_start_acq = self.ripple_acq.b_start_acq

    def start_stop_task_acquisition(self, state, subscr_id, seq_id):
        bOK = False
        if state:
            bOK = self.ripple_acq.start_task_acq(subscr_id=subscr_id, seq_id=seq_id)
            if bOK:
                self.signal_viewer.update_acq_btn(acq=True)
        else:
            bOK = self.ripple_acq.stop_task_acq()
            if bOK:
                self.signal_viewer.update_acq_btn(acq=False)

        self.b_start_acq = self.ripple_acq.b_start_acq

    def start_or_stop_recording(self, state):
        if state:
            self.ripple_acq.start_recording()
        else:
            self.ripple_acq.stop_recording()

    def update_recording_status(self, state):

        self.tasks.update_recording_status(state)

    def update_subscreening_img_list(self, img_list=None):
        try:
            if img_list is None:
                return

            self.tasks.subscr_img_list = img_list
            self.c.log.emit(f'Starting RSVPSCR with {len(self.tasks.subscr_img_list)} images')
        except:
            print(traceback.format_exc())

    def update_subscreening_img(self, id=None, img=None, img_path=None):
        try:
            if img is None:
                return
            
            # Add stimulus to responses display
            self.responses_list.add_stimulus(id=id, img=img, lbl=os.path.basename(img_path))
        except:
            print(traceback.format_exc())

    def nav_sig_view(self, direction='forward'):
        if self.b_start_acq:
            # self.ripple_acq.navigate(direction=direction)
            pass
        elif self.b_nsx_read:
            self.ripple_nsx.navigate(direction=direction)

    def go_to_time(self, date_time):
        if self.b_start_acq:
            # self.ripple_acq.go_to_time(h, m, s)
            pass
        else:
            self.ripple_nsx.go_to_time(date_time)

    def update_sig_view_param(self, param, value):
        setattr(self.signal_viewer, param, value)
        if param == 'channel':
            if self.b_start_acq:
                self.ripple_acq.change_channel(channel=value)
            elif self.b_nsx_read:
                self.ripple_nsx.draw_nsx_channels()
        else:
            if self.b_start_acq:
                setattr(self.ripple_acq, param, value)
                self.ripple_acq.change_channel(channel=self.ripple_acq.channel)
            elif self.b_nsx_read:
                setattr(self.ripple_nsx, param, value)
                self.ripple_nsx.draw_nsx_channels()

    def update_task_clus_param(self, param, value):
        setattr(self.clustering, param, value)
        setattr(self.ripple_nsx, param, value)

    def update_sig_view_filt_param(self, param, value):
        setattr(self.filtering, param, value)
        if self.b_start_acq:
            self.ripple_acq.filtering = self.filtering
            self.ripple_acq.change_channel(channel=self.ripple_acq.channel)
        elif self.b_nsx_read:
            self.ripple_nsx.filtering = self.filtering
            self.ripple_nsx.draw_nsx_channels()

    def channel_waveform_clicked(self, plt_idx=None):
        if self.b_start_acq:
            self.ripple_acq.channel_waveform_clicked(plt_idx=plt_idx)
        elif self.b_nsx_read:
            self.ripple_nsx.channel_waveform_clicked(plt_idx=plt_idx)

    def update_channel_selection(self, channel_dict):
        if hasattr(self, 'ripple_acq'):
            self.ripple_acq.update_channel_selection(channel_dict=channel_dict)
        if hasattr(self, 'ripple_nsx'):
            self.ripple_nsx.update_channel_selection(channel_dict=channel_dict)
        

    def update_log(self, message):
        """
         Add a message to the task log. This is called when an error occurs during the execution of a task
         
         Args:
         	 message: The message to add
        """
        self.tasks.txtTaskLog.append(message)

    #endregion Communication
    
    def init_signal_params(self):
        self.channel = 'micros'
        self.b_start_acq = False
        self.b_nsx_read = False
        self.raw = True
        self.filtered = True
        self.spikes = True
        self.psd = True
        self.photo = True
        self.pics = True
        self.responses = True
        self.region = True
        self.length = 3 # in seconds
        self.num_electrodes = 4
        self.num_time_points = 5000
        self.raw_y = 1000 # Range of the analog signal in uV
        self.filt_y = 60 # Range of the filtered signal in uV

        self.entities_visualized = {}

        # Spike detection parameters
        self.spikes_dict = {}
        self.threshold = 3.0

        # Feature extraction parameters
        self.pca_n_components = 4

        self.filtering = Filtering()

    def init_ripple(self):
        self.ripple_acq = RippleAcq(c=self.c, 
                                    filtering=self.filtering,
                                    clustering=self.clustering,
                                    power_spectrum=self.power_spectrum,
                                    signal_viewer=self.signal_viewer)
        
    def init_ripple_nsx(self):
        self.ripple_nsx = RippleNsx(c = self.c, 
                                    filtering=self.filtering,
                                    clustering=self.clustering, 
                                    power_spectrum=self.power_spectrum,
                                    signal_viewer=self.signal_viewer)

    def init_ui_layout(self):
        """
        Initializes the layout of the signal viewer.
        """
        self.chs_view = Dock(name='Channels', closable=True)
        self.chs_view.sigClosed.connect(self.dock_closed_event_handler)
        self.concepts_view = Dock(name='Concepts Studio', closable=True)
        self.concepts_view.sigClosed.connect(self.dock_closed_event_handler)
        self.psd_view = Dock(name='Power spectrum', closable=True)
        self.psd_view.sigClosed.connect(self.dock_closed_event_handler)
        self.responses_view = Dock(name='Responses', closable=True)
        self.responses_view.sigClosed.connect(self.dock_closed_event_handler)
        self.seeg_3d_view = Dock(name='3D viewer', closable=True)
        self.seeg_3d_view.sigClosed.connect(self.dock_closed_event_handler)
        self.clustering_view = Dock(name='Clustering', closable=True)
        self.clustering_view.sigClosed.connect(self.dock_closed_event_handler)
        self.tasks_view = Dock(name='Tasks', closable=True)
        self.tasks_view.sigClosed.connect(self.dock_closed_event_handler)

        self.sig_view.addDock(self.chs_view, 'left')
        self.sig_view.addDock(self.responses_view, 'right')
        self.sig_view.addDock(self.concepts_view, 'below', self.chs_view)
        self.sig_view.addDock(self.psd_view, 'above', self.responses_view)
        self.sig_view.addDock(self.seeg_3d_view, 'above', self.psd_view)
        self.sig_view.addDock(self.clustering_view, 'above', self.seeg_3d_view)
        self.sig_view.addDock(self.tasks_view, 'bottom', self.clustering_view)

        self.sig_view_dock_status_dict = {}

        self.init_3d_layout()
        self.init_chs_layout()
        self.init_concepts_layout()
        self.init_power_spectrum_layout()
        self.init_responses_layout()
        self.init_clustering_layout()
        self.init_tasks_layout()

        self.sig_view_state = self.sig_view.saveState()

    def restore_ui(self):
        self.sig_view.restoreState(self.sig_view_state, missing='create')
        for dock, status in self.sig_view_dock_status_dict.items():
            if dock == 'Channels' and not status:
                self.sig_view.addDock(self.chs_view, 'left')
            elif dock == 'Responses' and not status:
                self.sig_view.addDock(self.responses_view, 'right')
            elif dock == 'Concepts Studio' and not status:
                self.sig_view.addDock(self.concepts_view, 'below', self.chs_view)
            elif dock == 'Power spectrum' and not status:
                self.sig_view.addDock(self.psd_view, 'above', self.responses_view)
            elif dock == '3D viewer' and not status:
                self.sig_view.addDock(self.seeg_3d_view, 'bottom', self.responses_view)
            elif dock == 'Clustering' and not status:
                self.sig_view.addDock(self.clustering_view, 'above', self.seeg_3d_view)
            elif dock == 'Tasks' and not status:
                self.sig_view.addDock(self.tasks_view, 'bottom', self.clustering_view)


    def dock_closed_event_handler(self, dock):
        self.sig_view_dock_status_dict[dock.name()] = False

    def clear_layout(self):
        """Clears the layout of the signal viewer."""
        for pw in self.pw_list:
            pw.clear()
        for vb in self.filtered_vb_list:
            vb.clear()
        self.vb_pics.clear()
        self.chs_layout.clear()

        self.chs_layout.ci.layout.setSpacing(0)
        self.chs_layout.ci.layout.setContentsMargins(0, 0, 0, 0)
            
    def init_chs_layout(self):
        """Initializes the signal viewer with a plot and separate curves for each channel."""

        self.signal_viewer = SignalViewer(c=self.c, chs_view=self.chs_view, seeg_3d=self.seeg_3d)
        
    #region Offline sEEG study playback.

    def get_ch_spikes(self, ch_data):
        spikes = np.where(ch_data > self.threshold)
        spike_waveforms = []

        for t in spikes[0]:
            waveform_start = max(0, t - self.waveform_length // 2)
            waveform_end = waveform_start + self.waveform_length
            waveform = ch_data[waveform_start:waveform_end]
            spike_waveforms.append(waveform)

    def get_spikes(self, filtered_data):
        # Thresholding
        spikes = np.where(self.raw_data > self.threshold)
        
        # Extract and store spike waveforms
        spike_waveforms = []
        for electrode, time_points in enumerate(spikes):
            for t in time_points:
                waveform_start = max(0, t - self.waveform_length // 2)
                waveform_end = waveform_start + self.waveform_length
                waveform = filtered_data[electrode, waveform_start:waveform_end]
                spike_waveforms.append(waveform)

        spike_waveforms = np.array(spike_waveforms)

        # Reshape spike_waveforms to be 2D (n_samples x n_features)
        spike_waveforms = spike_waveforms.reshape(-1, self.waveform_length)

        return spike_waveforms

    def preprocess_data(self):
        # Bandpass filter
        filtered_data = []
        for electrode in self.raw_data:
            filtered_signal = self.bandpass_filter(electrode)
            filtered_data.append(filtered_signal)
        filtered_data = np.array(filtered_data)
        
        spike_waveforms = self.get_spikes(filtered_data)
        
        return filtered_data, spike_waveforms

    #endregion Offline sEEG study playback.
            
    #region Spike detection & sorting
            
    def init_clustering_layout(self):
        """
        Initializes the clustering layout.
        """
        self.clustering = Clustering(filtering=self.filtering, clustering_view=self.clustering_view, c=self.c)

    def plot_clusters(self, b_plot=True):
        """
         Plot clusters. This is called when the user presses the Plot Clusters button
         
         Args:
         	 clusters: 
        """
        try:
            if b_plot:
                # self.dsp.process_ripple_nsx_file_thread.join()
                self.orchestrator.plot_clusters()
        except:
            self.c.log.emit(traceback.format_exc())
            
    #endregion Spike detection

    #region Tasks

    def init_tasks_layout(self):
        """
        Initializes the tasks layout.
        """
        self.tasks = Tasks(c=self.c, tasks_view=self.tasks_view)

    #endregion Tasks

    #region Read EDF+ file
            
    def read_edf_file(self):
        pass
    
    #endregion Read EDF+ file
            
    #region Read NSx file (Ripple and Blackrock)
    
    def read_ripple_nsx_file(self, filename, analog_only=False, segment_only=False, 
                             event_only=False, skip=0, 
                             max_segment=100, max_analog=90000):
        """
        Reads a Ripple NSx file and stores the entities in a list.

        Args:
            filename: path to the NSx file
            analog_only: if True, only analog entities will be stored
            segment_only: if True, only segment entities will be stored
            event_only: if True, only event entities will be stored
            skip: number of entities to skip
            max_segment: maximum number of segments to read
            max_analog: maximum number of analog entities to read
        """
        self.study_info = StudyInfo(nsx_file_path=filename, c=self.c)
        if hasattr(self.study_info, 'notches_info'):
            self.filtering.notches_info = self.study_info.notches_info
        if len(self.study_info.image_names) > 0:
            self.responses_list.load_study_info(study_info=self.study_info)
            self.responses_list.rasters = self.study_info.rasters
        
        self.ripple_nsx.read_nsx_file(filename, analog_only=analog_only, segment_only=segment_only,
                                      event_only=event_only, skip=skip, max_segment=max_segment, 
                                      max_analog=max_analog)
    
    def post_read_nsx(self, state):
        if self.ripple_nsx.b_nsx_read:
            self.b_nsx_read = True
            self.ripple_acq.ripple_map.create_ripple_map_nsx(self.ripple_nsx.entities)
            self.signal_viewer.update_channel_list(self.ripple_acq.ripple_map.channel_dict)
        
    def post_clustering_nsx(self, state):
        """
        Updates the cluster info.
        """
        if state:
            if self.ripple_nsx.legacy:
                self.clustering.update_clustering_cmb_channels(ch_lbls=self.ripple_nsx.micro_ch_labels)
                self.clustering.load_clus_info_dict_from_times_mat(study_folder=self.study_info.study_folder, 
                                                                   nsx_info=self.study_info.nsx)
                self.responses_list.load_response_metrics_threaded(clus_info_dict=self.clustering.clus_info_dict)
            else:
                self.clustering.update_clustering_cmb_channels(ch_lbls=self.ripple_nsx.micro_ch_labels)
                # Start computing response metrics like ifr, z_score, etc.
                self.responses_list.compute_response_metrics_threaded(clus_info_dict=self.ripple_nsx.clustering.clus_info_dict,
                                                         micro_ch_lbls=self.ripple_nsx.micro_ch_labels)
            

    def read_brk_ns_file(self, filename):
        self.brk_nsx = BlackrockNsx()
        self.brk_nsx.read_brk_ns_file(filename)

    #endregion Read NSx file (Ripple and Blackrock)
    
    #region Plotting signals (analog, segment, event)

    #region Power spectrum

    def init_power_spectrum_layout(self):
        """
        Initializes the power spectrum plots.
        """
        self.power_spectrum = PowerSpectrum(psd_view=self.psd_view, c=self.c)

    #endregion Power spectrum

    #endregion Plotting signals (analog, segment, event)

    #region Analysis
        
    def analyze_subscreening_seq(self, seq_imgs=None, 
                                       seq_img_paths=None, 
                                       subscr_idx=None, 
                                       subscr_seq_idx=None):
        
        if seq_imgs is None:
            # Align the images to spikes in the photo diode channel.
            spike_ts = self.detect_photo_spikes()

            # Check if number of spikes is equal to number of images
            if len(spike_ts) != len(self.seq_imgs):
                self.c.log.emit(f'Number of spikes ({len(spike_ts)}) is not equal to number of images ({len(self.seq_imgs)})')
                return
            

            # Add the image and signals 100ms before and after the spike to each response
            id = 0
            for img, img_path, ts in zip(self.seq_imgs, self.seq_img_paths, spike_ts):
                id += 1
                img_filename = os.path.basename(img_path)
                self.responses_list.add_stimulus(id=id, img=img, lbl=img_filename)

                # Get 100ms before and after spike from each channel
                for ch_idx, ch in enumerate(self.disp_chs):
                    ch_data = self.seq_info.data[ch_idx, :]
                    ch_data = ch_data[int((ts - 0.1) * self.fs_clk) : int((ts + 0.1) * self.fs_clk)]
                    self.responses_list.add_response(ch_data, lbl=ch.label)

    def detect_photo_spikes(self):
        """
        Detects spikes in the photo diode channel.
        """
        if not self.photo:
            return

        # Get the photo diode channel
        photo_diode = self.seq_info.data[-1, :]
        # Get the threshold
        threshold = np.mean(photo_diode) + 3 * np.std(photo_diode)
        # Detect spikes
        spikes = np.where(photo_diode > threshold)[0]
        # Get the time stamps of the spikes
        spike_ts = spikes / self.fs_clk

        return spike_ts

    #endregion Analysis

    #region Responses

    def init_responses_layout(self):
        self.responses_list = ResponseViewer(responses_view=self.responses_view, c=self.c, rows=7, cols=5)

    #endregion Responses

    #region 3D viewer
        
    def init_3d_layout(self):

        self.seeg_3d = SEEG_3D(chs_3d_view=self.seeg_3d_view, c=self.c)
        
    #endregion 3D viewer

    #region Concepts Studio

    def init_concepts_layout(self):
        self.concepts_studio = ConceptsStudio(c=self.c, concepts_view=self.concepts_view)

    def load_concepts(self, file_path):
        """
        Load concepts from the internet.
        """
        self.concepts_studio.load_concepts(file_path)

    #endregion Concepts Studio