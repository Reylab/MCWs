# Author: Sunil Mathew
# Date: 29 November 2023
# Signal processing and visualization for sEEG data acquired during RSVPSCR study.

import os
import re

import traceback
import numpy as np

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

import pyqtgraph as pg

import sys
import imageio
import datetime


sys.path.append(os.path.dirname(__file__))

from qtpy import QtGui
from core.config import config
from tasks.study_info import StudyInfo
from usercontrols.checkable_treecombobox import CheckableTreeComboBox

LINES_ONOFF = 13
BLANK_ON = 11
LINES_FLIP_BLANK = 103
LINES_FLIP_PIC = 22
TRIAL_ON = 26
DATA_SIGNATURE_ON = 64
DATA_SIGNATURE_OFF = 128

class SignalViewer():
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SignalViewer, cls).__new__(cls)
        return cls.instance

    def __init__(self, c, chs_view=None, seeg_3d=None) -> None:
        self.chs_view = chs_view
        self.c = c
        
        self.raw = True
        self.filt = True
        self.spikes = True
        self.psd = True
        self.ph_diode = True
        self.pics = True
        self.responses = True
        self.region = True
        self.b_use_multiple_pws = True
        self.num_chs_to_plot = 10
        self.fs = 30000
        self.length = 3 # in seconds
        self.num_electrodes = 4
        self.num_time_points = 5000
        self.raw_y = 1000 # Range of the analog signal in uV
        self.filt_y = 70 # Range of the filtered signal in uV

        self.study_info = StudyInfo(c=self.c)
        self.seeg_3d = seeg_3d

        # Spike detection parameters
        self.spikes_dict = {}
        self.threshold = 3.0

        # Feature extraction parameters
        self.pca_n_components = 4

        self.init_chs_layout()

    def clear_layout(self):
        """Clears the layout of the signal viewer."""
        for pw in self.pw_list:
            pw.clear()
        for vb in self.filtered_vb_list:
            vb.clear()
        if hasattr(self, 'pw_daq'):
            self.pw_daq.clear()
        if hasattr(self, 'pw_photo'):
            self.pw_photo.clear()
        if hasattr(self, 'vb_pics'):
            self.vb_pics.clear()

        self.glw_chs.clear()

        self.glw_chs.ci.layout.setSpacing(0)
        self.glw_chs.ci.layout.setContentsMargins(0, 0, 0, 0)

        self.init_plot_items()

    def init_plot_items(self):
        """Initializes the plot items for the signal viewer."""
        self.pw_list = []
        self.pw_plots = []
        self.ch_curves = [] # raw data curve
        self.ch_spikes = [] # spikes 
        self.ch_filt_curves = [] # filtered data curve
        self.ch_filt_spikes = [] # filtered data spikes
        self.filtered_vb_list = []
        self.region_list = [] # 500ms region around the stimulus onset
        self.thr_list = [] # threshold line for spikes

    def init_chs_layout(self):
        """
        Initializes the signal viewer layout without separate pw for each channel.
        """
        if not hasattr(self, 'chs_layout'):
            self.chs_layout = QtWidgets.QVBoxLayout()
            self.chs_widget = QtWidgets.QWidget()
            self.chs_widget.setLayout(self.chs_layout)
            self.chs_view.addWidget(self.chs_widget)
            # self.pw_chs = self.chs_layout.addPlot()
            

        # Signal viewer controls layout
        self.chs_params_layout = QtWidgets.QHBoxLayout()
        self.init_sig_view_cmb()
        self.init_sig_view_spb()
        self.init_sig_view_chkboxes()
        self.init_sig_view_nav_btns()
        self.init_time_input()
        # self.init_sig_view_data_acq_btn()
        self.chs_layout.addLayout(self.chs_params_layout)
        self.glw_chs = pg.GraphicsLayoutWidget()
        self.glw_chs.ci.layout.setSpacing(0)

        self.chs_scroll = QtWidgets.QScrollArea()
        self.chs_scroll.setWidget(self.glw_chs)
        self.chs_scroll.setWidgetResizable(True)
        self.chs_layout.addWidget(self.chs_scroll)

        self.chs_layout.setStretchFactor(self.chs_params_layout, 0)
        self.chs_layout.setStretchFactor(self.chs_scroll, 1)

        self.t = np.arange(0, self.length, 1/self.fs, dtype=np.float32)

        self.init_plot_items()
        self.init_pw()

    def init_pw(self):
        """
        Initializes the signal viewer layout plot widgets.
        """
        self.clear_layout()
        if self.b_use_multiple_pws:
            self.init_chs_multiple_pws()
        else:
            self.init_chs_single_pw()

    def init_chs_single_pw(self):
        """
        Initializes the signal viewer layout with a single pw for all channels.
        """
        pw = self.glw_chs.addPlot()
        vb_filtered = pg.ViewBox()
        vb_filtered.setXLink(pw)
        # vb_filtered.setYRange(-self.filt_y, self.filt_y-40)
        vb_filtered.setYRange(0, self.num_chs_to_plot*500)
        pw.scene().addItem(vb_filtered)
        pw.setLabel('left', f'raw', units='uV')
        # pw.setLabel('bottom', 'Time', units='s')
        pw.setXRange(0, self.length)
        # pw.setYRange(-self.raw_y, self.raw_y)
        pw.setYRange(0, self.num_chs_to_plot*500)
        pw.showAxis('right')
        pw.getAxis('right').linkToView(vb_filtered)
        pw.getAxis('right').setLabel('Filtered', color='r', units='uV')
        pw.vb.sigResized.connect(self.update_views)
        pw.hideAxis('bottom')

        # linear region
        region = pg.LinearRegionItem(values=[0, 0.5], orientation=pg.LinearRegionItem.Vertical, brush=pg.mkBrush(color=(0, 0, 255, 10)))
        pw.addItem(region)
        
        for ch in range(self.num_chs_to_plot):
            # plot = self.pw_chs.plot(name=f'ch_{ch}')
            curve = pg.PlotCurveItem(pen=({'color': (ch, self.num_chs_to_plot*1.3), 'width': 1}), skipFiniteCheck=True)
            pw.addItem(curve)
            curve.setPos(0,ch*500)
            curve.setClickable(True, width=30)
            curve.sigClicked.connect(self.curve_clicked)
            self.ch_curves.append(curve)

            # spikes
            spikes_curve = pg.PlotCurveItem(pen=None, symbol='o', symbolPen='g', symbolSize=5, symbolBrush=None, skipFiniteCheck=True)
            self.ch_spikes.append(spikes_curve)

            filtered_curve = pg.PlotCurveItem(pen=({'color': (ch, self.num_chs_to_plot*1.3), 'width': 1}), skipFiniteCheck=True)
            vb_filtered.addItem(filtered_curve)
            filtered_curve.setPos(0,ch*500)
            self.ch_filt_curves.append(filtered_curve)

            filt_spikes = pg.ScatterPlotItem(pen=pg.mkPen(color='b', width=1), brush=None, size=5)
            vb_filtered.addItem(filt_spikes)
            filt_spikes.setPos(0,ch*500)
            self.ch_filt_spikes.append(filt_spikes)
            
        # DAQ
        self.glw_chs.nextRow()
        self.pw_daq = self.glw_chs.addPlot()

        # photo diode
        self.glw_chs.nextRow()
        self.pw_photo = self.glw_chs.addPlot()

        # Add pics vb in photo diode plot
        self.vb_pics = pg.ViewBox(lockAspect=True, invertY=True)
        self.vb_pics.setXLink(self.pw_photo)
        # self.vb_pics.setYLink(self.pw_photo)
        self.pw_photo.scene().addItem(self.vb_pics)

        
        self.pw_photo.setLabel('left', 'Photo', units='mV')
        self.pw_photo.setLabel('bottom', 'Time', units='s')
        self.pw_photo.setXRange(0, self.length)
        self.pw_photo.setYRange(-5000, 5000)
        self.pw_photo.showAxis('bottom')
        vb_photo_filtered = pg.ViewBox()
        vb_photo_filtered.setXLink(self.pw_photo)
        vb_photo_filtered.setYRange(-self.raw_y, self.raw_y)

        pw.setXLink(self.pw_photo)

        self.pw_list = [pw, self.pw_daq, self.pw_photo] # 90%, 5%, 5%
        self.filtered_vb_list = [vb_filtered, vb_photo_filtered]

        self.glw_chs.ci.layout.setRowStretchFactor(0, 90)
        self.glw_chs.ci.layout.setRowStretchFactor(1, 5)
        self.glw_chs.ci.layout.setRowStretchFactor(2, 5)

    def init_chs_multiple_pws(self):
        for ch in range(self.num_chs_to_plot):
            # plot = self.pw_chs.plot(name=f'ch_{ch}')
            pw = self.glw_chs.addPlot()
            self.pw_list.append(pw)

            curve = pg.PlotCurveItem(pen=({'color': 'b', 'width': 1}), skipFiniteCheck=True)
            pw.addItem(curve)
            curve.setClickable(True, width=30)
            curve.sigClicked.connect(self.curve_clicked)
            self.ch_curves.append(curve)

            vb_filtered = pg.ViewBox()
            self.filtered_vb_list.append(vb_filtered)
            filtered_curve = pg.PlotCurveItem(pen=pg.mkPen(color='r', width=1), skipFiniteCheck=True)
            self.ch_filt_curves.append(filtered_curve)
            filt_spikes = pg.ScatterPlotItem(pen=pg.mkPen(color='b', width=1), brush=None, size=5)
            self.ch_filt_spikes.append(filt_spikes)
            vb_filtered.addItem(filtered_curve)
            vb_filtered.addItem(filt_spikes)
            vb_filtered.setXLink(pw)
            vb_filtered.setYRange(-self.filt_y, self.filt_y-40)
            pw.scene().addItem(vb_filtered)
            leftaxis = pw.getAxis('left')
            leftaxis.setStyle(textFillLimits=[(0, 0.8)])
            leftaxis.setWidth(40)
            leftaxis.setLabel(f'Ch {ch}', units='uV', angle=45)
            # pw.setLabel('bottom', 'Time', units='s')
            pw.setXRange(0, self.length)
            pw.setYRange(-self.raw_y, self.raw_y)
            pw.showAxis('right')
            rightaxis = pw.getAxis('right')
            rightaxis.linkToView(vb_filtered)
            rightaxis.setLabel('Filtered', color='r', units='uV')
            rightaxis.setWidth(40)
            pw.vb.sigResized.connect(self.update_views)
            pw.hideAxis('bottom')

            # spikes
            spikes_plot = pw.plot(pen=None, symbol='o', symbolPen='g', symbolSize=5, symbolBrush=None)
            spikes_plot.setSkipFiniteCheck(True)
            self.ch_spikes.append(spikes_plot)

            # linear region
            region = pg.LinearRegionItem(values=[0, 0.5], orientation=pg.LinearRegionItem.Vertical, brush=pg.mkBrush(color=(0, 0, 255, 10)))
            self.region_list.append(region)
            pw.addItem(region)

            self.glw_chs.nextRow()

        # DAQ
        self.pw_daq = self.pw_list[-2]
        self.pw_daq.setLabel('left', 'DAQ', units='mV')
        self.pw_daq.setXRange(0, self.length)
        self.pw_daq.setYRange(-5000, 5000)
        self.pw_daq.enableAutoRange('x', False)

        # photo diode
        self.pw_photo = self.pw_list[-1]

        # Add pics vb in photo diode plot
        self.vb_pics = pg.ViewBox(lockAspect=True, invertY=True)
        self.vb_pics.setXLink(self.pw_photo)
        # self.vb_pics.setYLink(self.pw_photo)
        self.pw_photo.scene().addItem(self.vb_pics)

        
        self.pw_photo.setLabel('left', 'Photo', units='mV')
        self.pw_photo.setLabel('bottom', 'Time', units='s')
        self.pw_photo.setXRange(0, self.length)
        self.pw_photo.setYRange(-5000, 5000)
        self.pw_photo.showAxis('bottom')
        self.pw_photo.enableAutoRange('x', False)
        self.filtered_vb_list[-1].setXLink(self.pw_photo)
        self.filtered_vb_list[-1].setYRange(-self.raw_y, self.raw_y)

        for pw in self.pw_list[:-1]:
            # link the x axis of the photo diode plot with the other plots
            pw.setXLink(self.pw_photo)

    def init_sig_view_chkboxes(self):
        """
         Initialize check boxes for acquisition signals and raster data. This is called by init_data
        """
        params_list = list(config['sEEG']['chk_params'].keys())
        n_chks_per_vlyt = 2
        n_vlyts = np.ceil(len(params_list)/n_chks_per_vlyt).astype(int)
        for i in range(0, n_vlyts):
            # Add two checkboxes in one vertical layout
            vLyt = QtWidgets.QVBoxLayout()
            vLyt.setObjectName(f'vLytAcq{i}')
            vLyt.setContentsMargins(0, 0, 0, 0)
            vLyt.setSpacing(0)

            for j in range(n_chks_per_vlyt):
                param_idx = i*n_chks_per_vlyt+j
                if param_idx >= len(params_list):
                    break
                item = params_list[param_idx] 
                check_state = config['sEEG']['chk_params'][item].lower() == 'true'
                chk = QtWidgets.QCheckBox()
                chk.setObjectName(f'chkAcq{item}')
                chk.setText(item)
                chk.setChecked(check_state)
                chk.stateChanged.connect(self.update_sig_view_chk_params)

                vLyt.addWidget(chk)
            self.chs_params_layout.addLayout(vLyt)

    def update_sig_view_chk_params(self, state=None):
        """
         Update acquisition check box parameters. This is called when the user checks/unchecks a check box
         
         Args:
         	 state: The state of the check box
        """
        try:
            chk = self.chs_widget.sender()
            param = chk.objectName().split('chkAcq')[1]
            value = chk.isChecked()
            setattr(self, param, value)
            self.toggle_params(param=param)
        except:
            print(traceback.format_exc())

    def init_sig_view_cmb(self):
        """
         Initialize acquisition filters for the data frame. This is called by init_data
        """
        # Add the acq filters to the QtComboBox.
        for param in config['sEEG']['cmb_params']:
            # Add the combobox to a vertical layout with a spacer on top
            vLyt = QtWidgets.QVBoxLayout()
            vLyt.setObjectName(f'vLytAcq{param}')
            vLyt.setContentsMargins(0, 0, 0, 0)
            vLyt.setSpacing(0)
            vLyt.addStretch()

            # Add a label with the column name
            lbl = QtWidgets.QLabel()
            lbl.setObjectName(f'lblAcq{param}')
            lbl.setText(param)
            lbl.setAlignment(Qt.AlignCenter)
            vLyt.addWidget(lbl)

            if param == 'channel':
                # cmb = CheckableComboBox(self.chs_widget)
                # cmb = QtWidgets.QComboBox(self.chs_widget)
                # cmb.setMinimumContentsLength(15)
                # cmb.addItems(config['sEEG']['cmb_params'][param])
                cmbMont = QtWidgets.QComboBox(self.chs_widget)
                cmbMont.addItems(['uni', 'diff', 'half', 'laplace'])
                cmbMont.setObjectName('cmb_sig_mont')
                cmbMont.currentIndexChanged.connect(self.update_sig_view_cmb_params)
                vLyt.addWidget(cmbMont)
                cmb = CheckableTreeComboBox(self.c, self.chs_widget)
            else:
                cmb = QtWidgets.QComboBox(self.chs_widget)
                cmb.addItems(config['sEEG']['cmb_params'][param])
            cmb.setObjectName(f'cmb_sig_{param}')
            
            cmb.currentIndexChanged.connect(self.update_sig_view_cmb_params)
            # Add the combobox
            vLyt.addWidget(cmb)
            # vLyt.addStretch()
            self.chs_params_layout.addLayout(vLyt)

    def init_sig_view_nav_btns(self):
        """
         Initialize buttons for navigating signals. This is called by init_data
        """
        
        btnFwd = QtWidgets.QPushButton()
        btnFwd.setObjectName(f'btnForward')
        btnFwd.setText('>')
        btnFwd.clicked.connect(self.forward)
        btnFwd.setMaximumWidth(30)

        btnBwd = QtWidgets.QPushButton()
        btnBwd.setObjectName(f'btnBackward')
        btnBwd.setText('<')
        btnBwd.clicked.connect(self.backward)
        btnBwd.setMaximumWidth(30)

        self.chs_params_layout.addWidget(btnBwd)
        self.chs_params_layout.addWidget(btnFwd)

    def init_sig_view_data_acq_btn(self):
        """
         Initialize buttons for data acquisition. This is called by init_data
        """
        self.btnAcqStart = QtWidgets.QPushButton()
        self.btnAcqStart.setObjectName(f'btnAcqStart')
        self.btnAcqStart.setText('Start Acq')
        self.btnAcqStart.clicked.connect(self.start_stop_acq)
        # self.chs_params_layout.addWidget(self.btnAcqStart)

    def init_sig_view_spb(self):
        """
            Initialize spinboxes for the signal viewer. This is called by init_data
        """
        spb_params = config['sEEG']['spb_params']
        for param in spb_params:
            if param == 'ch_count':
                self.num_chs_to_plot = spb_params[param][0]
            # Spinbox for clustering parameters, use the config values to figure out if doublespinbox or spinbox
            if isinstance(spb_params[param][0], float):                       
                spb = QtWidgets.QDoubleSpinBox(self.chs_widget)               
            else:
                spb = QtWidgets.QSpinBox(self.chs_widget)

            spb.setObjectName(f'spb_sig_{param}')
            spb.setRange(spb_params[param][1], spb_params[param][2])
            spb.setValue(spb_params[param][0])
            spb.setSingleStep(spb_params[param][3])
            spb.valueChanged.connect(self.start_update_sig_view_spb_params_timer)
            lbl = QtWidgets.QLabel(param)

            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(lbl)
            vLyt.addWidget(spb)
            self.chs_params_layout.addLayout(vLyt)

    def init_time_input(self):
        """
        Lets user input time in HH:MM:SS format.
        """
        self.datetime_input = QtWidgets.QDateTimeEdit()
        self.datetime_input.setDisplayFormat('MM/dd/yyyy HH:mm:ss')
        self.datetime_input.setDateTime(QtCore.QDateTime.currentDateTime())
        self.datetime_input.setCalendarPopup(True)
        self.datetime_input.dateTimeChanged.connect(self.go_to_time_timer_start)
        self.datetime_input.setMaximumWidth(150)
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.addWidget(self.datetime_input)
        self.init_sig_view_data_acq_btn()
        vLyt.addWidget(self.btnAcqStart)
        self.chs_params_layout.addLayout(vLyt)

    def go_to_time_timer_start(self):
        """
        Start the timer to update the time in the signal viewer.
        """
        if not hasattr(self, 'goto_timer'):
            self.goto_timer = QtCore.QTimer()
            self.goto_timer.timeout.connect(self.go_to_time)
        else:
            self.goto_timer.stop()
        
        self.goto_timer.start(6000)

    def go_to_time(self):
        """
        Update time in the signal viewer.
        """
        try:
            self.goto_timer.stop()
            selected_date_time = self.datetime_input.dateTime().toPyDateTime()
            # Get current time zone
            tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            selected_date_time = selected_date_time.replace(tzinfo=tz)
            self.c.sig_view_go_to.emit(selected_date_time)
        except:
            self.c.log.emit('Invalid time format. Please enter time in HH:MM:SS format.')
    
    def set_ch_lbls_y_ticks(self, y_ticks):
        """
        Set the y-ticks for the channel labels.
        """
        if self.b_use_multiple_pws:
            pass
            # for i, pw in enumerate(self.pw_list):
            #     pw.setLabel('left', f'Ch {y_ticks[i]}', units='uV')
        else:
            self.pw_list[0].getAxis('left').setTicks([[(idx*500, y_ticks[idx]) for idx in range(self.num_chs_to_plot)]])

    def update_ch_lbls(self, port_bundles):
        """
        Update the channel labels in the signal viewer.
        """
        enabled_chs = []
        if self.b_use_multiple_pws:
            for port, bundles in port_bundles.items():
                if port == 'A' or port == 'B':
                    stream_ty = 'hi-res'
                elif port == 'C' or port == 'D':
                    stream_ty = 'spk'  
                else:
                    continue            
                for bundle, chs in bundles[1].items():
                    for ch, elec in chs[1].items():
                        if elec[0]:
                            enabled_chs.append(elec[1].label)
                
            for i, pw in enumerate(self.pw_list):
                if i < len(enabled_chs):
                    # horizontal labels
                    # pw.setLabel('left', f'{enabled_chs[i]}', units='uV', angle=45)
                    leftaxis = pw.getAxis('left')
                    leftaxis.setStyle(textFillLimits=[(0, 0.8)])
                    leftaxis.setWidth(40)
                    leftaxis.setLabel(f'{enabled_chs[i]}', units='uV', angle=0)
                else:
                    break
        else:
            pass
            # self.pw_list[0].getAxis('left').setTicks([[(idx*500, port_bundles[idx]) for idx in range(self.num_chs_to_plot)])

    def forward(self):
        """
         Move forward in the data frame. This is called when the user clicks the forward button
        """
        self.c.sig_view_nav.emit('forward')

    def backward(self):
        """
         Move backward in the data frame. This is called when the user clicks the backward button
        """
        self.c.sig_view_nav.emit('back')

    def update_sig_view_cmb_params(self, index=None):
        """
         Filter acquisition data. This is called when the user selects a filter from the combobox
         
         Args:
         	 index: Index of data to
        """
        try:
            cmb = self.chs_widget.sender()
            param = cmb.objectName().split('cmb_sig_')[1]
            cmb_txt = cmb.currentText()
            if param != 'channel':
                cmb_txt = re.sub('[^0-9]', '', cmb_txt)
                cmb_txt = int(cmb_txt)

            self.c.sig_view_param.emit(param, cmb_txt)
            
        except:
            print(traceback.format_exc())

    def start_update_sig_view_spb_params_timer(self):
        """
        Start the timer to update the spinbox parameters.
        """
        self.spb = self.chs_widget.sender()
        if not hasattr(self, 'spb_timer'):
            self.spb_timer = QtCore.QTimer()
            self.spb_timer.timeout.connect(self.update_sig_view_spb_params)
        else:
            self.spb_timer.stop()
        
        self.spb_timer.start(1000)

    def update_sig_view_spb_params(self):
        """
         Update acquisition data. This is called when the user changes the value of a spinbox
         
         Args:
         	 value: The value of the spinbox
        """
        try:
            self.spb_timer.stop()
            
            param = self.spb.objectName().split('spb_sig_')[1]
            value = self.spb.value()
            if param == 'lowcut' or \
               param == 'highcut':
                self.c.sig_view_filt_param.emit(param, value)
            elif param == 'ch_count':
                self.num_chs_to_plot = value
                self.init_pw()
                self.c.sig_view_filt_param.emit(param, value)
            elif param == 'filt_y':
                self.filt_y = value
                self.update_filt_y_range()
            elif param == 'raw_y':
                self.raw_y = value
                self.update_raw_y_range()
            else:
                self.c.sig_view_param.emit(param, value)
        except:
            print(traceback.format_exc())

    def update_x_range(self):
        """
        Update the x-range of the signal viewer.
        """
        for pw in self.pw_list:
            pw.setXRange(0, self.length)

    def update_raw_y_range(self):
        """
        Update the y-range of the raw signal.
        """
        for pw in self.pw_list:
            pw.setYRange(-self.raw_y, self.raw_y)

    def start_filt_y_range_timer(self):
        """
        Start the timer to update the y-range of the filtered signal.
        """
        if not hasattr(self, 'filt_y_timer'):
            self.filt_y_timer = QtCore.QTimer()
            self.filt_y_timer.timeout.connect(self.update_filt_y_range)
        else:
            self.filt_y_timer.stop()
        
        self.filt_y_timer.start(2000)

    def update_filt_y_range(self):
        """
        Update the y-range of the filtered signal.
        """
        for vb in self.filtered_vb_list:
            vb.setYRange(-self.filt_y, self.filt_y-20)

    def start_init_pw_timer(self):
        """
        Start the timer to initialize the signal viewer.
        """
        if not hasattr(self, 'init_pw_timer'):
            self.init_pw_timer = QtCore.QTimer()
            self.init_pw_timer.timeout.connect(self.init_pw_timer_elapsed)
        else:
            self.init_pw_timer.stop()
        
        self.init_pw_timer.start(2000)

    def init_pw_timer_elapsed(self):
        """
        Initialize the signal viewer.
        """
        self.init_pw_timer.stop()
        self.init_pw()
        self.c.sig_view_param.emit('ch_count', self.num_chs_to_plot)

    def update_views(self):
        for pw, vb in zip(self.pw_list, self.filtered_vb_list):
            vb.setGeometry(pw.vb.sceneBoundingRect())
            vb.linkedViewChanged(pw.vb, vb.XAxis)

        if hasattr(self, 'vb_pics'):
            self.vb_pics.setGeometry(self.pw_photo.vb.sceneBoundingRect())
            self.vb_pics.linkedViewChanged(self.pw_photo.vb, self.vb_pics.XAxis)

    def update_channel_list(self, port_bundles):
        cmb = self.chs_widget.findChild(CheckableTreeComboBox, f'cmb_sig_channel')
        try:
            cmb.currentIndexChanged.disconnect(self.update_sig_view_cmb_params)
        except:
            print(traceback.format_exc())
        cmb.clear()
        cmb.addItems(port_bundles)
        self.update_ch_lbls(port_bundles)
        # cmb.addItems(['All'] + ['micros'] + ['macros'] + ['test_chs'] + bundles + channel_list)
        # # Set current text first electrode label
        # cmb.setCurrentText(channel_list[0])
        cmb.currentIndexChanged.connect(self.update_sig_view_cmb_params)

    def curve_clicked(self, curve):
        """Pyqtgraph curve clicked event handler. 
           Update power spectrum plots when a curve is clicked.
           
           Args:
            curve: curve that was clicked
        """
        # unhighlight previous curve
        for i, plot_item in enumerate(self.ch_curves):
            plot_item.setPen(({'color': 'b', 'width': 1}))
        # highlight curve
        plt_idx = self.ch_curves.index(curve)
        plot_item = self.ch_curves[plt_idx]
        plot_item.setPen(({'color': 'b', 'width': 1}))

        self.c.channel_waveform_clicked.emit(plt_idx)

        # if self.b_start_acq:
        #     self.plot_power_spectrum()
        # else:
        #     self.plot_power_spectrum_nsx()

    #region Stimulus images
            
    def plot_pics(self, time_arr=None, spikes=None):
        """
        Plots the stimulus images synchronized with photo detector spikes.
        """
        if spikes is None:
            spikes = self.photo_spikes
        else:
            self.photo_spikes = spikes

        if time_arr is None:
            time_arr = self.time_arr
        else:
            self.time_arr = time_arr

        self.vb_pics.clear()
        if hasattr(self.study_info, 'pics_onset') and len(spikes) > 0:
            first_spike_ms = time_arr[spikes[0]] * 1000
            
            spike_img_idx_list = []
            first_spike_found = False
            count = 0

            for i, block in enumerate(self.study_info.pics_onset):
                for j, seq in enumerate(block.T): 
                    if count >= len(spikes):
                            break
                    for k, onset in enumerate(seq):
                        if abs(onset - first_spike_ms) <= 50:
                            first_spike_found = True
                        if first_spike_found and count < len(spikes):
                            spike_img_idx_list.append(self.study_info.pics_order[i]['order_pic'].T[j][k])
                            count += 1
                        if count >= len(spikes):
                            break

            if len(spike_img_idx_list) == 0:
                self.c.log.emit(f'Could not find first spike({first_spike_ms}) in pics_onset({self.study_info.pics_onset[0][0][0]}).')
                return
        else:
            self.c.log.emit('No photo diode spikes found.')
            return
        
        tr = QtGui.QTransform()
        scale = 0.002
        tr.scale(scale, scale)
        imgs_dir = os.path.join(self.study_info.study_folder, 'pics_now')
        if not os.path.exists(imgs_dir):
            imgs_dir = os.path.join(self.study_info.study_folder, 'custom_pics')
        if not os.path.exists(imgs_dir):
            imgs_dir = os.path.join(self.study_info.study_folder, 'pics_used')
        if not os.path.exists(imgs_dir):
            self.c.log.emit('No pics folder found. Checked folders: pics_now, custom_pics, pics_used.')
            return
        
        for img_idx, spike in zip(spike_img_idx_list, spikes):
            if img_idx >= len(self.study_info.image_names):
                break
            img_name = self.study_info.image_names[img_idx]
            img_path = os.path.join(imgs_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.study_info.study_folder, 'custom_pics', img_name)
            if os.path.exists(img_path):
                img = imageio.imread(img_path)
                img = pg.ImageItem(img)
                img.setTransform(tr)
                # Find where in self.t the image should be plotted
                spike_s = time_arr[spike]
                spike_t_idx = np.where(time_arr >= spike_s)[0][0]
                
                xpos = time_arr[spike_t_idx]
                self.vb_pics.addItem(img, ignoreBounds=True)      
                img.setPos(xpos-img.width()*scale/2, img.width()*scale)  
            else:
                self.c.log.emit(f'Image {img_path} not found.')
            
    #endregion Stimulus images

    #region Audio transcriptions

    def plot_audio_transcription_pw(self, plot, 
                                 transcription):
        """
        Plots the audio transcription on the signal viewer.
        """
        # check if transcription is empty
        if len(transcription) == 0:
            return
        for i, sentence in enumerate(transcription):
            for word in sentence['words']:
                if 'word' in word and 'start' in word and 'end' in word:
                    start_time = word['start']
                    end_time = word['end']
                    spkr_color = int(word['speaker'][-2:])
                    spkr_color = f'#{spkr_color:06x}'
                    spkr_color = '#FFFFFF' if spkr_color == '#000000' else spkr_color
                    text = pg.TextItem(html=f'<div style="text-align: center"><span style="color:{spkr_color};">{word["word"]}</span></div>', 
                                       anchor=(-0.3,0.5), angle=0, border='w', fill=(0, 0, 255, 255))
                    plot.addItem(text)
                    text.setPos(start_time, 0)

    def plot_audio_transcription(self, transcription):
        """
        Plots the audio transcription on the signal viewer.
        """
        plot = self.pw_list[0]
        # check if transcription is empty
        if len(transcription) == 0:
            return
        for i, sentence in enumerate(transcription):
            for word in sentence['words']:
                if 'word' in word and 'start' in word and 'end' in word:
                    start_time = word['start']
                    end_time = word['end']
                    spkr_color = int(word['speaker'][-2:])
                    spkr_color = f'#{spkr_color:06x}'
                    spkr_color = '#FFFFFF' if spkr_color == '#000000' else spkr_color
                    text = pg.TextItem(html=f'<div style="text-align: center"><span style="color:{spkr_color};">{word["word"]}</span></div>', 
                                       anchor=(-0.3,0.5), angle=0, border='w', fill=(0, 0, 255, 255))
                    plot.addItem(text)
                    text.setPos(start_time, 0)

    #endregion Audio transcriptions

    #region UI callbacks from the main window checkboxes, comboboxes.
    
    def toggle_params(self, param):
        """
        Toggles different items on the signal viewer via checkboxes on the UI.
        """
        if param == 'ph_diode':
            self.toggle_photo()
        elif param == 'raw':
            self.toggle_raw()
        elif param == 'filt':
            self.toggle_filtered()
        elif param == 'spikes':
            self.toggle_spikes()        
        elif param == 'pics':
            self.toggle_pics()
        elif param == 'remove_collisions':
            self.toggle_collisions()
        elif param == 'region':
            self.toggle_region()

    def toggle_region(self):
        """
        Toggles the region around the stimulus onset.
        """
        if self.region:
            for pw, region in zip(self.pw_list, self.region_list):
                pw.addItem(region)
        else:
            for pw, region in zip(self.pw_list, self.region_list):
                pw.removeItem(region)

    def toggle_collisions(self):
        self.change_channel(channel=self.channel)

    def toggle_photo(self, glw=None):
        
        if self.ph_diode:
            self.glw_chs.addItem(self.pw_daq, row=self.num_chs_to_plot-2, col=0)
            self.glw_chs.addItem(self.pw_photo, row=self.num_chs_to_plot-1, col=0)
        else:
            self.glw_chs.removeItem(self.pw_photo)
            self.glw_chs.removeItem(self.pw_daq)

    def toggle_raw(self):
        if self.raw:
            for pw, curve in zip(self.pw_list, self.ch_curves):
                pw.addItem(curve)
        else:
            for pw, curve in zip(self.pw_list, self.ch_curves):
                pw.removeItem(curve)

    def toggle_spikes(self):
        if self.spikes:
            if self.raw:
                for pw, curve in zip(self.pw_list, self.ch_spikes):
                    pw.addItem(curve)
            if self.filt:
                for vb, curve in zip(self.filtered_vb_list, self.ch_filt_spikes):
                    vb.addItem(curve)
        else:
            for pw, curve in zip(self.pw_list, self.ch_spikes):
                pw.removeItem(curve)
            for vb, curve in zip(self.filtered_vb_list, self.ch_filt_spikes):
                vb.removeItem(curve)

    def toggle_filtered(self):
        if self.filt:
            for vb, curve in zip(self.filtered_vb_list, self.ch_filt_curves):
                vb.addItem(curve)
        else:
            for vb, curve in zip(self.filtered_vb_list, self.ch_filt_curves):
                vb.removeItem(curve)

    def toggle_pics(self):
        if self.pics:
            self.plot_pics()
        else:
            self.vb_pics.clear()

    def start_stop_acq(self):
        """
         Start acquisition of data and store in self. data Args : None Returns :
        """
        try:
            if self.btnAcqStart.text() == 'Start Acq':
                self.c.sig_view_acq.emit(True)
            else:
                self.c.sig_view_acq.emit(False)
        except:
            print(traceback.format_exc())

    def update_acq_btn(self, acq):
        """
        Updates the acquisition button text.
        """
        if acq:
            self.btnAcqStart.setText('Stop Acq')
        else:
            self.btnAcqStart.setText('Start Acq')


    #endregion UI callbacks from the main window checkboxes, comboboxes.
    
    def show_bundle_vol(self, channel):
        try:
            if not channel.startswith('m'):
                return
            bundle = channel.split(' ')[0][:-2]
            hemi   = f'{bundle.lower()[1]}h'
            side   = 'Left' if hemi == 'lh' else 'Right'
            part   = bundle.lower()[2:]
            ctx    = f'{hemi}-{part}'

            # Capitalize first letter of part
            part = part[0].upper() + part[1:]
            part = f'{side}-{part}'

            # Find the label corresponding to the bundle
            bundle = None
            for lbl in self.seeg_3d.label_names:
                if ctx in lbl or part in lbl:
                    bundle = lbl
                    break

            if bundle is None:
                self.c.log.emit(f'No label found for {channel}')
                return
            
            self.seeg_3d.show_bundle_vol(bundle=bundle)
        except:
            print(traceback.print_exc())

    def draw_events_pw(self, events, timestamps, pw=None):
        if pw is None:
            pw = self.pw_list[-2]
        pw.clear()
        # plot the values as text
        for timestamp, value in zip(timestamps, events):
            if value == LINES_FLIP_BLANK:
                # text = pg.TextItem(html=f'<div style="text-align: center"><span style="color: #000;">LINES_FLIP_BLANK</span></br><span style="color: #000;">{str(value)}</span></div>',
                #                       anchor=(-0.3,0.5), angle=0, border='w', fill=(255, 0, 100, 100))
                text = pg.TextItem(
                    html=f'<div style="text-align: center"><span style="color: #000;">LINES_FLIP_BLANK</span></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 255, 0, 50),
                )  # yellow
                text.setPos(timestamp, 0)
            elif value == LINES_FLIP_PIC:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 0, 0, 50),
                )  # red
                text.setPos(timestamp, value)
            elif value == LINES_ONOFF:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="r",
                    fill=(0, 255, 0, 50),
                )
                text.setPos(timestamp, 0)
            elif value == BLANK_ON:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 255, 255, 50),
                )
                text.setPos(timestamp, 0)
            elif value == DATA_SIGNATURE_ON:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(0, 255, 255, 50),
                )
                text.setPos(timestamp, 0)
            elif value == DATA_SIGNATURE_OFF:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 255, 0, 50),
                )
                text.setPos(timestamp, value)
            else:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(0, 0, 255, 100),
                )
                text.setPos(timestamp, 0)

            pw.addItem(text)

    def draw_events(self, events, timestamps):
        self.pw_daq.clear()
        # plot the values as text
        for timestamp, value in zip(timestamps, events):
            if value == LINES_FLIP_BLANK:
                # text = pg.TextItem(html=f'<div style="text-align: center"><span style="color: #000;">LINES_FLIP_BLANK</span></br><span style="color: #000;">{str(value)}</span></div>',
                #                       anchor=(-0.3,0.5), angle=0, border='w', fill=(255, 0, 100, 100))
                text = pg.TextItem(
                    html=f'<div style="text-align: center"><span style="color: #000;">LINES_FLIP_BLANK</span></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 255, 0, 50),
                )  # yellow
                text.setPos(timestamp, 0)
            elif value == LINES_FLIP_PIC:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 0, 0, 50),
                )  # red
                text.setPos(timestamp, value)
            elif value == LINES_ONOFF:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="r",
                    fill=(0, 255, 0, 50),
                )
                text.setPos(timestamp, 0)
            elif value == BLANK_ON:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 255, 255, 50),
                )
                text.setPos(timestamp, 0)
            elif value == DATA_SIGNATURE_ON:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(0, 255, 255, 50),
                )
                text.setPos(timestamp, 0)
            elif value == DATA_SIGNATURE_OFF:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"></br><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(255, 255, 0, 50),
                )
                text.setPos(timestamp, value)
            else:
                text = pg.TextItem(
                    html=f'<div style="text-align: center"><span style="color: #000;">{str(value)}</span></div>',
                    anchor=(-0.3, 0.5),
                    angle=0,
                    border="w",
                    fill=(0, 0, 255, 100),
                )
                text.setPos(timestamp, 0)

            self.pw_daq.addItem(text)