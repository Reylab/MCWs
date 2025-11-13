import os
import shutil
import stat
import traceback
import numpy as np
import json_tricks as jsont
import h5py
from collections import defaultdict

import subprocess
import sys
import threading
import datetime
import time
import re

sys.path.append(os.path.dirname(__file__))

from qtpy import QtCore
from core.config import config
from core.utils import get_elapsed_time, get_time_hrs_mins_secs_ms
from core.sequence_info import SequenceInfo
from ripple_map_file import RippleMapFile
from dsp.spike_detection import detect_spikes, detect_photo_spikes

try:
    import xipppy as xp
except:
    print('xipppy is not installed')
    xp = None

BLUE_LED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets/icons/blue-led-on.png'))
GREEN_LED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets/icons/green-led-on.png'))

class SequenceData():
    def __init__(self, acq_chs, subscr_id, seq_id) -> None:
        self.data = [] # Each item is a tuple of (data, timestamp)
        self.events = [] # For daq events
        self.acq_chs = acq_chs
        self.subscr_id = subscr_id
        self.seq_id = seq_id
        self.seq_data_folder = os.path.join(config['seq_data_folder'], config['subject_id'])
        if not os.path.exists(self.seq_data_folder):
            os.makedirs(self.seq_data_folder)
        else:
            shutil.rmtree(self.seq_data_folder)
        for elec_info in self.acq_chs:
            # Create folders for each electrode
            elec_folder = os.path.join(self.seq_data_folder, elec_info.label)
            if not os.path.exists(elec_folder):
                os.makedirs(elec_folder)
    def append(self, data):
        """
        Appends the data to the sequence data.
        """
        self.data.append(data)

    def append_event(self, event):
        """
        Appends the event to the sequence data.
        """
        self.events.append(event)

    def process_sequence_data(self, save=True):
        """
        Saves the sequence data to a file for each electrode using HDF5 format.
        """
        if len(self.data) == 0:
            return

        ch_data_dict = defaultdict(lambda: {'data': [], 'timestamp': []})

        prev_timestamp = None
        for acq_id, (data, timestamp) in enumerate(self.data):
            if acq_id > 0:
                expected_timestamp = prev_timestamp + samples_read_per_channel
                if timestamp != expected_timestamp:
                    print(f'Warning: Expected timestamp {expected_timestamp} but received {timestamp} with {len(data)} samples')

            samples_read_per_channel = int(len(data) / len(self.acq_chs))

            for ch_idx, elec_info in enumerate(self.acq_chs):
                ch_start_idx = ch_idx * samples_read_per_channel
                ch_end_idx = ch_start_idx + samples_read_per_channel
                ch_data = data[ch_start_idx : ch_end_idx]
                
                ch_data_dict[elec_info.label]['data'].extend(ch_data)
                ch_data_dict[elec_info.label]['timestamp'].extend(timestamp)

            prev_timestamp = timestamp

        self.raw_data = []
        self.ch_lbls = []
        for ch_lbl, ch_data in ch_data_dict.items():
            dir_path = os.path.join(self.seq_data_folder, ch_lbl)
            os.makedirs(dir_path, exist_ok=True)
            # file_path = os.path.join(dir_path, f'{self.subscr_id}_{self.seq_id}_{ch_lbl}.h5')
            file_path = os.path.join(dir_path, f'{self.subscr_id}_{self.seq_id}_{ch_lbl}.npy')
            file_path_ts = os.path.join(dir_path, f'{self.subscr_id}_{self.seq_id}_{ch_lbl}_ts.npy')

            # Convert lists to numpy arrays
            data_array = np.array(ch_data['data'])
            timestamp_array = np.array(ch_data['timestamp'])

            if save:
                # Save as numpy array
                np.save(file_path, data_array)
                np.save(file_path_ts, timestamp_array)

            # Save as HDF5
            # with h5py.File(file_path, 'w') as f:
            #     f.create_dataset('data', data=data_array, compression='gzip', compression_opts=9)
            #     f.create_dataset('timestamp', data=timestamp_array, compression='gzip', compression_opts=9)
            self.raw_data.append(data_array)
            self.ch_lbls.append(ch_lbl)
        self.clear()


    def clear(self):
        self.data = []

class RippleAcq():
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(RippleAcq, cls).__new__(cls)
        return cls.instance

    def __init__(self, c, filtering, 
                 clustering,
                 power_spectrum,
                 signal_viewer) -> None:
        self.c = c
        self.subscr_id = 0
        self.seq_id = 0
        self.init_acq_params()
        self.init_ripple()
        self.filtering = filtering
        self.clustering = clustering
        self.power_spectrum = power_spectrum
        self.signal_viewer = signal_viewer

    #region Online Data acquisition

    def init_acq_params(self):
        self.th_stop_evt = threading.Event()
        self.channel = None
        self.enable_ripple = False
        self.b_seq_ended = False
        self.b_clr_buffer = True
        self.fs_clk = 30000 # samples/sec
        self.b_trellis = False
        self.b_summit = False
        self.b_start_acq = False
        self.b_start_task_acq = False
        self.b_update_plot_params = True
        self.b_clr_trellis_data_folder = False
        self.rec_file_id = 1
        self.rec_length_secs = 3600 # in seconds
        self.vis_buffer_secs = 0.5 # seconds
        self.vis_buffer_npoints = int(self.vis_buffer_secs * self.fs_clk)
        self.draw_delay_ms = 200 # ms
        self.b_scroll = True
        self.n_points = self.fs_clk * 60 * 5 # 5 mins
        self.daq_events_list = []
        self.disp_chs = []
        self.analysis_chs = []
        self.acq_chs = [] # Will be disp_chs + analysis_chs
        self.vis_buffer_t = None # Time vector for the buffer
        self.vis_buffer_data = None

    def check_trellis(self):
        """
        Checks if Trellis is running on the system
        """
        if os.name == 'nt':
            username = os.environ['USERNAME']
            self.trellis_data_path = f'C:\\Users\\{username}\\Trellis\\dataFiles'
            cmd = 'WMIC PROCESS get Caption,Commandline,Processid'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            for line in proc.stdout:
                if 'Trellis.exe' in str(line):
                    self.b_trellis = True
                    break
            if not self.b_trellis:
                self.c.log.emit('Trellis is not running on this PC, are you running Trellis elsewhere? If no, start Trellis.')
                return
        else:
            username = os.environ['USER']
            self.trellis_data_path = f'/home/{username}/Trellis/dataFiles'

        return self.b_trellis

    def init_ripple(self, map_file=None):
        if map_file is not None:
            self.ripple_map = RippleMapFile(filename=map_file)
            self.update_enabled_status()
            self.c.sig_view_acq_channels.emit(self.ripple_map.channel_dict)
        elif not hasattr(self, 'ripple_map'):
            self.ripple_map = RippleMapFile(filename=config['ripple_map'])
            self.update_enabled_status()
            self.c.sig_view_acq_channels.emit(self.ripple_map.channel_dict)
        else:
            # self.update_enabled_status()
            self.c.sig_view_acq_channels.emit(self.ripple_map.channel_dict)
        
        try:
            self.check_trellis()
            if xp is None:
                self.b_summit = False
                self.c.log.emit('xipppy is not installed. Cannot acquire data from Ripple NIP.')
                return
            with xp.xipppy_open(use_tcp=True):
                summit_elapsed = datetime.timedelta(seconds=xp.time()/30000)
                days = summit_elapsed.days
                hrs, rem = divmod(summit_elapsed.seconds, 3600)
                mins, secs = divmod(rem, 60)
                print(f'Status: Summit has been up for {days:.0f} days, {hrs:.0f} hrs and {mins:.0f} mins')
                self.c.log.emit(f'Status: Summit has been up for {days:.0f} days, {hrs:.0f} hrs and {mins:.0f} mins')

                if not os.path.exists(self.trellis_data_path):
                    os.makedirs(self.trellis_data_path)
                else:
                    if self.b_clr_trellis_data_folder:
                        for file in os.listdir(self.trellis_data_path):
                            os.remove(os.path.join(self.trellis_data_path, file))

                file_base_name = os.path.join(self.trellis_data_path, config['subject_id'])
                self.ripple_buffer_timestamp = xp.time()
                # if self.enable_ripple:
                #     self.enable_mapped_channels()
                self.init_signal_plot_timer()
                # try:
                #     trial_desc = None
                #     xp.add_operator(int(config['trellis_oper']))
                #     self.b_trellis = True
                #     self.c.trellis.emit(self.b_trellis)
                #     # try:
                #     #     trial_desc = xp.trial(status='stopped', 
                #     #                           file_name_base=file_base_name, 
                #     #                           #auto_stop_time=self.rec_length_secs, 
                #     #                           auto_incr=False)
                #     #     # trial_desc = xp.trial()
                #     #     self.trellis_data_path = os.path.dirname(trial_desc[1])
                #     #     print(trial_desc)
                #     # except:
                #     #     print(traceback.format_exc())
                #     #     self.c.log.emit('Failed to start trial, see if Trellis is running and setting for recording are ok.')
                #     #     return
                # except:
                #     print(traceback.format_exc())
                #     self.c.log.emit('Failed to add operator to Ripple NIP, see config to see if right IP is provided or Trellis is running.')
                #     self.b_trellis = False
                #     self.c.trellis.emit(self.b_trellis)
                #     return
                # self.micros = xp.list_elec(fe_type='micro',max_elecs=1024).tolist()
                # self.analogs = xp.list_elec(fe_type='analog').tolist()
                
        except:
            print(traceback.format_exc())
            self.b_summit = False
            self.b_trellis = False
            self.c.trellis.emit(self.b_trellis)
            return
        
    def enable_mapped_channels(self):
        if hasattr(self, 'ripple_map'):
            for port, bundles in self.ripple_map.channel_dict.items():
                if port == 'A' or port == 'B':
                    stream_ty = 'hi-res'
                elif port == 'C' or port == 'D':
                    stream_ty = 'spk'  
                else:
                    continue            
                for bundle, chs in bundles[1].items():
                    for ch, elec in chs[1].items():
                        try:
                            with xp.xipppy_open(use_tcp=True):
                                xp.signal_set(elec=elec[1].id, stream_ty=stream_ty, val=elec[0])
                                # self.c.log.emit(f'Enabled {elec[1].label} for streaming')
                        except:
                            # print(traceback.format_exc())
                            if elec[0]:
                                self.c.log.emit(f'Failed to enable {elec[1].label} for streaming')
                            else:
                                self.c.log.emit(f'Failed to disable {elec[1].label} for streaming')

    def update_enabled_status(self):
        if hasattr(self, 'ripple_map'):
            start_time = time.time()
            for port, bundles in self.ripple_map.channel_dict.items():
                if port == 'A' or port == 'B':
                    stream_ty = 'hi-res'
                elif port == 'C' or port == 'D':
                    stream_ty = 'raw'  
                else:
                    continue 
                for bundle, chs in bundles[1].items():
                    for ch, elec in chs[1].items():
                        try:
                            with xp.xipppy_open(use_tcp=True):
                                elec[1].enabled = xp.signal(elec=elec[1].id, stream_ty=stream_ty) # 1-indexed list of electrodes
                                self.ripple_map.channel_dict[port][1][bundle][1][ch] = (elec[1].enabled, elec[1])
                        except xp.exception.XippPyException:
                            print(f'Failed to get enabled status for {elec[1].label}. Port {port} might not be connected.')
                        finally:
                            break
            hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
            print(f'Ripple update_enabled_status took {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
        
    def update_channel_selection(self, channel_dict):
        """
        Updates the selected channels for data acquisition.
        """
        self.ripple_map.update_channel_dict(channel_dict=channel_dict)
        if self.b_trellis and self.enable_ripple:
            self.enable_mapped_channels()
        
        # TBD: Checking channel enable status with Ripple takes time, so do it in a separate thread
        # for now, just update the channel_dict
        # self.update_enabled_status()
        self.signal_viewer.update_ch_lbls(self.ripple_map.channel_dict)
        self.b_update_plot_params = True

    def init_signal_plot_timer(self):
        self.signal_plot_timer = QtCore.QTimer()
        self.signal_plot_timer.timeout.connect(self.plot_seq_data_pg)

    def start_acq(self):
        self.b_start_acq = True
        self.init_ripple()
        self.update_signal_params()
        if self.b_trellis:
            self.start_buffering_seq_data_thread()
            self.signal_plot_timer.start(self.draw_delay_ms)
        else:
            self.b_start_acq = False

        return self.b_start_acq

    def get_rec_file_id(self):
        """
        Gets the next recording file id to be used based 
        on the last modified file in the folder.

        returns:
            rec_file_id: int
        """
        rec_file_id = 1
        if os.path.exists(self.trellis_data_path):
            rec_files = os.listdir(self.trellis_data_path)
            if len(rec_files) > 0:
                # Get the last modified file in the folder
                rec_files = [os.path.join(self.trellis_data_path, f) 
                             for f in rec_files 
                             if os.path.isfile(os.path.join(self.trellis_data_path, f))]
                rec_files.sort(key=lambda x: os.path.getmtime(x))
                if len(rec_files) > 0:
                    last_file = rec_files[-1]
                    last_file_base = os.path.basename(last_file)
                    last_file_base = last_file_base.split('.')[0]
                    last_file_base = last_file_base.split('_')[-1]
                    rec_file_id = int(last_file_base) + 1
            else:
                rec_file_id = 1

        return rec_file_id

    #region Recording

    def start_recording(self):
        """
        Starts recording signals from Ripple NIP. Due to a bug in xipppy,
        there'll be an exception thrown while starting the recording with auto increment disabled.
        """
        trial_desc = None

        if not self.b_trellis:
            print("Trellis is not running/connected. So cannot start recording")
            return trial_desc
        
        try:
            with xp.xipppy_open(use_tcp=True):
                print(f'Status: Time elapsed after NIP start up: {xp.time()/30000/60:.2f} mins')
                self.c.log.emit(f'Status: Time elapsed after NIP start up: {xp.time()/30000/60:.2f} mins')
                trellis_oper_ip = int(config['trellis_oper'])
                xp.add_operator(trellis_oper_ip)

                self.rec_file_id = self.get_rec_file_id()
                file_name_base = f'{config["subject_id"]}_{self.rec_file_id}'
                self.rec_file = os.path.join(self.trellis_data_path, file_name_base)
                
                try:
                    # Returns: the resulting state of: status, file_name_base, auto_stop_time, auto_incr, incr_num
                    
                    trial_desc = xp.trial(oper=trellis_oper_ip, status='recording', 
                                        #   auto_stop_time=self.rec_length,
                                          file_name_base=self.rec_file
                                         )
                except:
                    print(traceback.format_exc())
                    self.c.log.emit('Failed to start recording signals, Trying again.')
                    trial_desc = xp.trial(oper=trellis_oper_ip, status='recording', 
                                        #   auto_stop_time=self.rec_length,
                                          file_name_base=self.rec_file
                                         )
                
                self.ripple_buffer_timestamp = xp.time()
                print(trial_desc)
                self.c.log.emit(f'Status: {trial_desc[0]} at {trial_desc[1]}.ns*')
                if trial_desc is not None and trial_desc[0] == 'recording':
                    self.c.recording_status.emit(True)
                    # if not hasattr(self, 'rec_start_timer'):
                    #     self.rec_start_timer = QtCore.QTimer()
                    #     self.rec_start_timer.timeout.connect(self.stop_start_recording)

                    # # restart timer
                    # self.rec_start_timer.stop()
                    # self.rec_start_timer.start((self.rec_length_secs + 1) * 1000) # start a new recording after this one ends
                    self.rec_file_id += 1

                else:
                    self.c.recording_status.emit(False)
                    self.c.log.emit(f'Failed to start recording signals')
            
        except:
            print(traceback.format_exc())

        return trial_desc   

    def stop_start_recording(self):
        self.stop_recording()
        self.start_recording() 

    def stop_recording(self):
        """
        This function stops the recording of signals from Ripple NIP. Due to a bug in xipppy, 
        there'll be an exception thrown while stopping the recording with auto increment disabled.
        """
        trial_desc = None
        try:
            with xp.xipppy_open(use_tcp=True):
                print(f'Status: Time elapsed after NIP start up: {xp.time()/30000/60:.2f} mins')
                # self.lblStatus.setText(f'Status: Time elapsed after NIP start up: {xp.time()/30000/60:.2f} mins')
                trellis_oper_ip = int(config['trellis_oper'])
                xp.add_operator(trellis_oper_ip)
                try:
                    # This just gets the current status of the recording
                    # trial_desc = xp.trial(oper=trellis_oper_ip, status=None, file_name_base=None, auto_stop_time=None, auto_incr=None, incr_num=None)
                    trial_desc = xp.trial()
                    if trial_desc[0] == 'recording':
                        trial_desc = xp.trial(oper=trellis_oper_ip, status='stopped')
                except:
                    print(traceback.format_exc())
                    trial_desc = xp.trial(oper=trellis_oper_ip, status='stopped')
                print(trial_desc)
                if trial_desc is not None:
                    self.c.log.emit(f'Status: {trial_desc[0]} at {trial_desc[1]}.ns*')
                    if trial_desc[0] == 'stopped':
                        self.c.recording_status.emit(False)
                else:
                    self.c.log.emit('Failed to stop recording signals')

        except:
            print(traceback.format_exc())

        return trial_desc
    
    def stop_recording_timer(self):
        """
        This function stops the recording of signals from Ripple NIP. Due to a bug in xipppy, 
        there'll be an exception thrown while stopping the recording with auto increment disabled.
        """
        trial_desc = None
        try:
            with xp.xipppy_open(use_tcp=True):
                print(f'Status: Time elapsed after NIP start up: {xp.time()/30000/60:.2f} mins')
                # self.lblStatus.setText(f'Status: Time elapsed after NIP start up: {xp.time()/30000/60:.2f} mins')
                trellis_oper_ip = int(config['trellis_oper'])
                xp.add_operator(trellis_oper_ip)
                try:
                    if hasattr(self, 'rec_start_timer') and self.rec_start_timer is not None:
                        self.rec_start_timer.stop()
                        # This just gets the current status of the recording
                        trial_desc = xp.trial(oper=trellis_oper_ip, status=None, file_name_base=None, auto_stop_time=None, auto_incr=None, incr_num=None)
                        if trial_desc[0] == 'recording':
                            trial_desc = xp.trial(oper=trellis_oper_ip, status='stopped')
                except:
                    print(traceback.format_exc())
                    trial_desc = xp.trial(oper=trellis_oper_ip, status='stopped')
                print(trial_desc)
                if trial_desc is not None:
                    self.c.log.emit(f'Status: {trial_desc[0]} at {trial_desc[1]}.ns*')
                    if trial_desc[0] == 'stopped':
                        self.c.recording.emit(False)
                else:
                    self.c.log.emit('Failed to stop recording signals')

        except:
            print(traceback.format_exc())

        return trial_desc

    #endregion Recording

    def start_buffering_seq_data_thread(self):
        if hasattr(self, 'buff_thread') and self.buff_thread is not None:
            self.buff_thread.join()
            self.buff_thread = None

        self.buff_thread = threading.Thread(target=self.start_buffering_data)
        self.buff_thread.start()

    def request_data(self):
        with xp.xipppy_open(use_tcp=True):
            data = xp.cont_raw(npoints=self.vis_buffer_npoints, elecs=self.elec_ids, start_timestamp=self.req_ts)
            events = xp.digin()
            # events is a list of SegmentEventPacket class containing digital event data. 
            # Fields are ‘timestamp’ ‘reason’, ‘sma1’, ‘sma2’, ‘sma3’, ‘sma4’, and ‘parallel’. 
            # The fields ‘sma1’ through ‘sma4’ and ‘parallel’ hold unsigned 16-bit integer values for the corresponding input. 
            # The ‘reason’ field is a bit mask showing which input triggered the digital event. 
            # 1 corresponds to the parallel port. 2, 4, 8, 16 refer to sma inputs 1, 2, 3, and 4 respectively. 
            # 32 refers to a digital output marker created when using the ‘digout’ command.
            daq_event = [event for event in events[1] if event.reason == 1]
            if len(daq_event) > 0:
                # print(f'DAQ event at {daq_event[0].timestamp} is {daq_event[0].parallel}')
                self.daq_events_list.extend(daq_event)
                self.seq_data.append_event(daq_event)
            if self.b_start_task_acq:
                self.seq_data.append(data)
            # print(f'Acquired {len(data[0])} samples at {data[1]}')
            if data[1] != self.req_ts:
                print(f'Warning: Requested timestamp {self.req_ts} but received {data[1]} with {len(data[0])} samples')
                self.req_ts = xp.time()
                if self.b_start_acq:
                    self.request_data()

            data_received = data[0].tolist()
            
            if not self.b_seq_started:
                self.timestamp_zero = data[1]
                self.b_seq_started = True

            samples_read_per_channel = int(len(data_received)/len(self.acq_chs))
            if data[1] != 1:
                self.req_ts = data[1] + samples_read_per_channel
            
            if self.sample_idx + samples_read_per_channel >= self.vis_buffer_samples:
                samples_to_write = self.vis_buffer_samples - self.sample_idx - 1
                self.b_seq_ended = True
                self.b_seq_started = False
            else:
                samples_to_write = samples_read_per_channel

            ch_idx = 0
            for ch in self.acq_chs:
                ch_start_idx = ch_idx * samples_read_per_channel
                if self.b_scroll:
                    self.vis_buffer_data[ch_idx, 0 : self.vis_buffer_samples - samples_to_write] = self.vis_buffer_data[ch_idx, samples_to_write : self.vis_buffer_samples]
                    self.vis_buffer_data[ch_idx, self.vis_buffer_samples - samples_to_write : self.vis_buffer_samples] = data_received[ch_start_idx : ch_start_idx + samples_to_write]
                else:
                    self.vis_buffer_data[ch_idx, self.sample_idx : self.sample_idx + samples_to_write] = data_received[ch_start_idx : ch_start_idx + samples_to_write]
                ch_idx += 1                    

            self.sample_idx += samples_to_write

            if self.b_seq_ended:
                # print('Sequence ended')
                time.sleep(self.vis_buffer_secs)
                if self.b_clr_buffer:
                    self.b_seq_ended = False
                    self.sample_idx = 0
                    self.b_seq_started = False
            else:
                time.sleep(self.vis_buffer_secs)

    def start_buffering_data(self):
        try:
            with xp.xipppy_open(use_tcp=True):
                
                self.timestamp_zero = 0
                self.req_ts = xp.time()
                self.sample_idx = 0
                self.b_seq_started = False
                self.b_seq_ended = False
                self.seq_data = SequenceData(acq_chs=self.acq_chs, subscr_id=self.subscr_id, seq_id=self.seq_id)
                
                while not self.b_seq_ended and self.b_start_acq:
                    self.request_data()
        except:
            print(traceback.format_exc())
            # self.lblStatus.setText('Failed to buffer data')
            # self.conn.send(b'Failed to buffer data')

    def update_signal_params(self):        
        if hasattr(self, 'ripple_map'):
            self.analysis_chs = []
            self.disp_chs = []
            self.exp_chs = [] # Ripple explorer channels (photo diode, microphone, etc.)
            self.elec_ids = []
            y_ticks = []

            for port, bundles in self.ripple_map.channel_dict.items():
                for bundle, electrodes in bundles[1].items():
                    for label, electrode in electrodes[1].items():
                        elec = self.ripple_map.channel_dict[port][1][bundle][1][label]
                        if not elec[1].enabled:
                            continue
                        if port == 'C' or port == 'D':
                            self.analysis_chs.append(elec[1])
                        if elec[0] and len(self.disp_chs) < self.signal_viewer.num_chs_to_plot:
                            self.disp_chs.append(elec[1])
                            y_ticks.append(f'{elec[1].label}')
                        if port == 'AIO' or port == 'DIO':
                            self.exp_chs.append(elec[1])

            self.signal_viewer.set_ch_lbls_y_ticks(y_ticks)
            self.signal_viewer.pw_photo.setXRange(0, self.signal_viewer.length, padding=0)
            # analysis + disp channels
            self.acq_chs = list(set(self.analysis_chs + self.disp_chs))
            self.elec_ids = [elec_info.id for elec_info in self.acq_chs]

            for elec in self.exp_chs:
                if elec.label.lower().find('parallel_dig') != -1 and elec.port == 'DIO':
                    self.disp_chs.append(elec)
                    self.acq_chs.append(elec)
                    self.elec_ids.append(elec.id)
                    break

            for elec in self.exp_chs:
                if elec.label.lower().find('photo_analog') != -1 and elec.port == 'AIO':
                    self.disp_chs.append(elec)
                    self.acq_chs.append(elec)
                    self.elec_ids.append(elec.id)
                    break

            self.vis_buffer_t = np.arange(0, self.signal_viewer.length, 1/self.fs_clk, dtype=np.float32)
            self.vis_buffer_samples = int(self.signal_viewer.length * self.fs_clk)
            self.vis_buffer_data = np.zeros((len(self.acq_chs), self.vis_buffer_samples), dtype=np.float32)

    def plot_seq_data_pg(self):
        """Plots the data acquired continously from the Ripple NIP."""
        if self.b_update_plot_params:
            self.update_signal_params()
            
            self.b_update_plot_params = False

        filtered_data = None
        for curve_id, ch in enumerate(self.disp_chs[:self.signal_viewer.num_chs_to_plot-2]):
            try:
                ch_idx = self.acq_chs.index(ch)
            except:
                # print(f'Channel {ch.label} not found in acq_chs')
                continue
            data = self.vis_buffer_data[ch_idx, :]
            filtered_data = self.filtering.custom_bp_filter(data=data, channel=ch.id)
            
            if self.signal_viewer.raw:
                curve = self.signal_viewer.ch_curves[curve_id]
                curve.setData(x=self.vis_buffer_t, y=data)

            if self.signal_viewer.filt:
                curve = self.signal_viewer.ch_filt_curves[curve_id]
                
                curve.setData(x=self.vis_buffer_t, y=filtered_data)
                # self.signal_viewer.filtered_vb_list[ch_idx].addItem(curve)

            if self.signal_viewer.spikes:
                spikes, thr = detect_spikes(data=filtered_data)
                # # Draw spikes on filtered signal
                # curve = self.ch_plot_filt_spikes[curve_id]
                # curve.setData(x=self.t[spikes], y=filtered_data[spikes])
                # self.filtered_vb_list[curve_id].addItem(curve)

                # Draw spikes on raw signal
                spike_plot = self.signal_viewer.ch_spikes[curve_id]
                spike_plot.setData(x=self.vis_buffer_t[spikes], y=data[spikes])

            if self.signal_viewer.ph_diode:
                ch = self.disp_chs[-1]
                ch_idx = self.acq_chs.index(ch)
                data = self.vis_buffer_data[ch_idx, :]
                curve = self.signal_viewer.ch_curves[-1]
                curve.setData(x=self.vis_buffer_t, y=data)

                photo_spikes, thr = detect_photo_spikes(data=data)
                # Draw photo spikes on raw signal
                spike_plot = self.signal_viewer.ch_spikes[-2]
                spike_plot.setData(x=self.vis_buffer_t[photo_spikes], y=data[photo_spikes])
        
        events = []
        timestamps = []
        for daq_event in self.daq_events_list:
            events.append(daq_event.parallel)
            timestamps.append((daq_event.timestamp-self.timestamp_zero)/self.fs_clk)

        self.signal_viewer.draw_events(events=events, timestamps=timestamps)
            
    def sequence_started(self):
        try:
            # self.conn.send(b'Started sequence')
            self.start_buffering_seq_data_thread();
        except:
            print(traceback.format_exc())
            self.c.log.emit('Failed to start sequence')
            # self.conn.send(b'Failed to start sequence')

    def sequence_ended(self):
        try:
            # self.conn.send(b'Ended sequence')
            self.b_seq_ended = True
        except:
            print(traceback.format_exc())
            self.c.log.emit('Failed to start signal analysis')
            # self.conn.send(b'Failed to start analysis')

    def start_analysis(self):
        try:
            with xp.xipppy_open(use_tcp=True):
                micro_channels = xp.list_elec(fe_type='micro')
                analog_channels = xp.list_elec(fe_type='analog')
                self.c.log.emit(f'Status: Analyzing {micro_channels}')
                (timestamp, data) = xp.cont_raw(npoints=5000, elecs=micro_channels, start_timestamp=0)
                self.c.log.emit(f'Read {len(data)} from Summit buffer at {timestamp}')
                self.conn.send(b'Started signal analysis')
        except:
            print(traceback.format_exc())
            self.c.log.emit('Failed to start signal analysis')
            # self.conn.send(b'Failed to start analysis')

    def end_analysis(self):
        pass
        # self.conn.send(b'Completed online analysis')

    def end_experiment(self):
        self.stop_recording()
        # self.conn.send(b'Completed online analysis')

    def channel_waveform_clicked(self, plt_idx):
        self.plot_power_spectrum(plt_idx=plt_idx)

    def change_channel(self, channel):
        self.channel = channel
        self.b_update_plot_params = True
        # TBD

    #endregion Online Data acquisition

    #region Power spectrum

    def plot_power_spectrum(self, plt_idx):
        """
        Plots the power spectrum of a selected channel. 
        There are two plots: one with most of the frequency range(10kHz) and one with 0 to 300 Hz.
        """
        fs = 30000 # This should be infered from the channel info, would be different for macros
        fft_length = 2**np.ceil(np.log2(fs*2))/fs
        nfft = int(fft_length * fs)

        ch = self.disp_chs[plt_idx]
        ch_label = ch.label
        acq_ch_idx = self.acq_chs.index(ch)
        data = self.vis_buffer_data[acq_ch_idx, :]
        filtered_data = self.filtering.custom_bp_filter(data=data)

        # Spike clustering, sorting
        # other_params = ss.get_default_sorter_params('tridesclous')
        # other_params['detect_threshold'] = 6
        # sorting_TDC = ss.run_sorter(sorter_name='tridesclous', recording=filtered_data, 
        #                             output_folder='tridesclous_output', verbose=True, **other_params)
        # print(sorting_TDC)
        
        self.power_spectrum.plot_power_spectrum_nsx(ch_label=ch_label,
                                                    data=data,
                                                    filtered_data=filtered_data)

    #endregion Power spectrum

    #region Data acquisition
       
    def update_rec_length(self, index=None):
        try:
            if index is None:
                return

            if hasattr(self, 'dsp'):
                cmb_txt = self.cmbRecLength.currentText()
                if cmb_txt.endswith('mins'):
                    x_factor = 60
                elif cmb_txt.endswith('hr'):
                    x_factor = 3600
                # strip the text to get the number with floating point
                cmb_txt = re.sub('[^0-9.]', '', cmb_txt)
                self.orchestrator.ripple_acq.rec_length = int(float(cmb_txt) * x_factor) # convert to seconds
        except:
            print(traceback.format_exc())

    # def start_recording(self):
    #     """
    #      Start recording data. This is called when the user presses the Record button
    #     """
    #     # Start recording
    #     # self.btnAcqRecord.setText('Stop Recording')
    #     # self.btnAcqRecord.clicked.disconnect(self.start_recording)
    #     self.orchestrator.start_recording()
    #     # self.btnAcqRecord.clicked.connect(self.stop_recording)

    # def stop_recording(self):
    #     """
    #      Stop recording data. This is called when the user presses the Stop Recording button
    #     """
    #     # Stop recording
    #     # self.btnAcqRecord.setText('Start Recording')
    #     # self.btnAcqRecord.clicked.disconnect(self.stop_recording)
    #     self.stop_recording()
        
    #     # self.btnAcqRecord.clicked.connect(self.start_recording)

    def stop_recording_task(self, state=None):
        """
         Stop recording data. This is called when the task ends or aborted by user.
        """
        if state:
            self.stop_recording()

    #endregion Data acquisition

    #region Data Analysis

    def stop_acq(self):
        self.b_start_acq = False
        self.signal_plot_timer.stop()

        return True
    
    def start_task_acq(self, subscr_id, seq_id):
        """
        Start saving sequence data.
        """
        self.subscr_id = subscr_id
        self.seq_id = seq_id
        bOK = False
        self.b_start_task_acq = True
        if not self.b_start_acq:
            self.start_acq()

        bOK = True
        return bOK

    def stop_task_acq(self):
        """
        Save the sequence data and stop the acquisition.
        """
        self.b_start_task_acq = False
        self.b_start_acq = False

        if hasattr(self, 'seq_data'):
            time.sleep(0.01)
            self.seq_data.process_sequence_data()
            if self.subscr_id == 0 and self.seq_id == 0:
                self.clustering.clear_clus_info_dict(ch_lbls=self.seq_data.ch_lbls)
            self.clustering.process_nsx_data(self.seq_data.raw_data, self.seq_data.ch_lbls)

        return True

    #endregion Data Analysis