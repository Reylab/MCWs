# Author: Sunil Mathew
# Date: 29 November 2023
# Class to conduct RSVPSCR study.

import datetime
import random
import time
import imageio
import numpy as np

# from psychopy import visual, core
# from psychopy.hardware import joystick
# import psychopy
import traceback
import threading
import os
import socket
import screeninfo
# from apscheduler.schedulers.background import BackgroundScheduler

from qtpy import QtGui, QtWidgets
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt, QMetaObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from usercontrols.frameless import RSVPWorker, FramelessWindow
from pyqtgraph.Qt.QtCore import QTimer

# from pynput.keyboard import Key, Listener
from daq.daq_mc import DAQ_MC
from core.config import config

BLUE_LED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets/icons/blue-led-on.png")
)
GREEN_LED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets/icons/green-led-on.png")
)

RESET = 0
LINES_ONOFF = 13
BLANK_ON = 11
LINES_FLIP_BLANK = 103
LINES_FLIP_PIC = 22
PIC_ON_OFF = np.array([[1, 4, 16], [2, 8, 32]])
TRIAL_ON = 26
DATA_SIGNATURE_ON = 64
DATA_SIGNATURE_OFF = 128

class SubscrConfig:
    def __init__(self, img_paths, block_params, exp_params, language, trial_config) -> None:
        self.img_paths = img_paths
        self.block_params = block_params
        self.exp_params = exp_params
        self.language = language
        self.trial_config = trial_config


class Tasks:  # client
    def __init__(self, c, tasks_view) -> None:
        self.c = c
        self.tasks_view = tasks_view
        self.config = config
        self.seq_no = 0
        self.b_trellis = False
        self.b_abort = False
        self.b_exp_paused = False
        self.b_auto_advance = True
        self.b_flip_lines_color = False
        self.do_sorting = False # For clustering
        self.th_stop_conn_evt = threading.Event()
        self.daq = DAQ_MC()
        
        self.rsvpscr_timer = QTimer()
        self.rsvpscr_timer.timeout.connect(self.run_screening)
        # self.rsvpscr_scheduler = BackgroundScheduler()
        # self.rsvpscr_scheduler.add_job(self.run_screening, "interval", seconds=0.5)

        self.init_tasks()
        # self.rsvp_worker = RSVPWorker(c=c, config=config, monitor=self.monitor)

    def read_image(self, file):
        return imageio.imread(file)

    # region Tasks

    def analyze_subscreening_seq(
        self, seq_imgs=None, seq_img_paths=None, subscr_idx=None, subscr_seq_idx=None
    ):
        try:
            if (
                seq_imgs is None
                or seq_img_paths is None
                or subscr_idx is None
                or subscr_seq_idx is None
            ):
                return

            self.orchestrator.analyze_subscreening_seq(
                seq_imgs=seq_imgs,
                seq_img_paths=seq_img_paths,
                subscr_idx=subscr_idx,
                subscr_seq_idx=subscr_seq_idx,
            )
        except:
            print(traceback.format_exc())

    def update_task_progress(self, value=None, msg=""):
        try:
            if value is None:
                return

            self.pbTask.setValue(value)
            self.lbl_pb.setText(msg)
        except:
            print(traceback.format_exc())

    def init_tasks(self):
        self.init_task_ui()
        self.init_task_controls()
        self.load_task_config()
        self.update_task_params()  # update task params with the current values in the comboboxes
        # self.connect_to_analysis_machine_thread() # try to connect to analysis machine

    def init_task_ui(self):

        # Experiment controls layout
        self.hLytTasksControls = QtWidgets.QHBoxLayout()

        # Experiment block design, progressbar
        self.tblBlockDesign = QtWidgets.QTableWidget()

        self.vLytTaskConfig = QtWidgets.QVBoxLayout()
        self.vLytTaskConfig.addWidget(self.tblBlockDesign)
        # self.vLytTaskConfig.addWidget(self.pbTask)

        # Logging
        self.txtTaskLog = QtWidgets.QTextEdit()
        self.txtTaskLog.setReadOnly(True)

        # Task images
        self.vLytTaskImgsMain = QtWidgets.QVBoxLayout()
        self.hLytTaskImgsControls = QtWidgets.QHBoxLayout()
        self.vLytTaskImgs = QtWidgets.QVBoxLayout()
        self.vLytTaskImgsMain.addLayout(self.hLytTaskImgsControls)
        self.vLytTaskImgsMain.addLayout(self.vLytTaskImgs)

        self.hLytBDLogsImgs = QtWidgets.QHBoxLayout()
        self.hLytBDLogsImgs.addLayout(self.vLytTaskConfig)
        self.hLytBDLogsImgs.addWidget(self.txtTaskLog)
        self.hLytBDLogsImgs.addLayout(self.vLytTaskImgsMain)

        self.hLytBDLogsImgs.setStretchFactor(self.vLytTaskConfig, 1)
        self.hLytBDLogsImgs.setStretchFactor(self.txtTaskLog, 2)
        self.hLytBDLogsImgs.setStretchFactor(self.vLytTaskImgsMain, 2)

        self.tasks_widget = QtWidgets.QWidget()
        vLytTasks = QtWidgets.QVBoxLayout()
        vLytTasks.addLayout(self.hLytTasksControls)
        vLytTasks.addLayout(self.hLytBDLogsImgs)
        self.tasks_widget.setLayout(vLytTasks)

        self.tasks_view.addWidget(self.tasks_widget)

        # self.taskControlsWidget = QWidget()
        # self.taskControlsWidget.setLayout(self.hLytTasksControls)

        # self.hSplitterMain.addWidget(self.taskControlsWidget)
        # self.hSplitterMain.setStretchFactor(2, 1)

        # self.taskWidget = QWidget()
        # self.taskWidget.setLayout(self.hLytBDLogsImgs)

        # self.hSplitterMain.addWidget(self.taskWidget)
        # self.hSplitterMain.setStretchFactor(3, 1)

    def connect_to_analysis_machine_thread(self):
        chk = self.tasks_widget.findChild(
            QtWidgets.QCheckBox, f"chkTaskremote_analysis"
        )
        if chk.isChecked():
            self.conn_thread = threading.Thread(target=self.connect_to_analysis_machine)
            self.conn_thread.start()

    def connect_to_analysis_machine(self):
        self.socket = "b"
        attempts = 0
        if not hasattr(self, "task"):
            return

        if not hasattr(self.task, "analysis_machine"):
            return

        while self.socket is None and not self.th_stop_conn_evt.is_set():
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.analysis_machine, 12345))
                self.socket.send(b"initialization")
                self.socket = self.socket

                # Activate start task button
                btn = self.tasks_widget.findChild(QtWidgets.QPushButton, "btnTaskStart")
                btn.setEnabled(True)
                break

            except:
                self.socket = None
                attempts += 1
                error_msg = traceback.format_exc()
                print(error_msg)
                self.c.log.emit(
                    f"Attempt {attempts} to connect to {self.analysis_machine} failed. Retrying in 5 seconds..."
                )
                self.c.log.emit(error_msg)

                # Disable start task button if enabled

                btn = self.tasks_widget.findChild(QtWidgets.QPushButton, "btnTaskStart")
                if btn.isEnabled():
                    btn.setEnabled(False)
                if attempts > 10:
                    self.c.log.emit(f"Could not connect to {self.analysis_machine}.")
                    break
                time.sleep(5)

    def init_task_controls(self):
        self.init_task_cmb()
        self.init_task_chkboxes()
        self.init_task_buttons()
        self.init_task_block_design()
        self.init_task_imgs()

    def init_task_imgs(self):
        # Add a button to browse and select folder with images, and a label to display the selected folder
        btn = QtWidgets.QPushButton()
        btn.setObjectName("btnTaskImgs")
        btn.setText("Browse")
        btn.clicked.connect(self.browse_task_imgs)

        # Add the button to a vertical layout with a spacer on top
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.setObjectName("vLytTaskImgsBtn")
        vLyt.setContentsMargins(0, 0, 0, 0)
        vLyt.setSpacing(0)
        vLyt.addStretch()
        vLyt.addWidget(btn)
        vLyt.addStretch()
        self.hLytTaskImgsControls.addLayout(vLyt)

        # Add a combo box to select image category/concept
        cmb = QtWidgets.QComboBox()
        cmb.setObjectName("cmbTaskImgsConcept")
        cmb.addItems(["All"])
        cmb.currentIndexChanged.connect(self.filter_task_imgs)
        self.hLytTaskImgsControls.addWidget(cmb)

        # Add a label to display the selected folder
        lbl = QtWidgets.QLabel()
        lbl.setObjectName("lblTaskImgsFolder")
        lbl.setText("No folder selected")
        lbl.setAlignment(Qt.AlignCenter)
        self.hLytTaskImgsControls.addWidget(lbl)

        # Add a horizontal spacer
        spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.hLytTaskImgsControls.addItem(spacer)

        # create a figure
        self.fig_tasks = plt.figure()
        self.fig_tasks.patch.set_facecolor("lightgrey")
        self.can_tasks = FigureCanvas(self.fig_tasks)
        self.fig_tasks.subplots_adjust(
            wspace=0.02, hspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98
        )

        self.vLytTaskImgs.addWidget(self.can_tasks)

    def filter_task_imgs(self, index=None):
        try:
            if index is not None:
                concept = self.tasks_widget.findChild(
                    QtWidgets.QComboBox, "cmbTaskImgsConcept"
                ).currentText()
                # images that contain the selected concept
                if concept == "All":
                    images = self.img_paths[1:50]
                else:
                    images = [
                        image
                        for image in self.img_paths
                        if concept in os.path.basename(image).split("~")
                    ]
                self.display_task_imgs_helper(images=images)
        except:
            print(traceback.format_exc())

    def browse_task_imgs(self):
        # Browse and select folder with images
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.tasks_widget, "Select folder with images", os.path.dirname(__file__)
        )
        if folder_path:
            # Set folder path in status bar
            self.c.log.emit(f"Selected imgs folder: {folder_path}")

            # Display folder path in the label
            lbl = self.tasks_widget.findChild(QtWidgets.QLabel, "lblTaskImgsFolder")
            lbl.setText(folder_path)

            # Display images in the label
            self.display_task_imgs_thread(folder_path=folder_path)

    def display_task_imgs_thread(self, folder_path=None):
        try:
            imgs_thread = threading.Thread(
                target=self.display_task_imgs, args=(folder_path,)
            )
            imgs_thread.start()
        except:
            print(traceback.format_exc())

    def display_task_imgs(self, folder_path=None):
        try:
            if folder_path is None:
                return

            # Get images
            self.img_paths = self.get_task_imgs(folder_path=folder_path)

            # Get concepts
            self.concepts = self.get_concept_names(images=self.img_paths)
            # Update combobox
            cmb = self.tasks_widget.findChild(QtWidgets.QComboBox, "cmbTaskImgsConcept")
            cmb.clear()
            concepts = [item for sublist in self.concepts for item in sublist]
            cmb.addItems(["All"] + list(set(concepts)))
            cmb.adjustSize()

            # Display images
            self.display_task_imgs_helper(images=self.img_paths[1:20])
        except:
            print(traceback.format_exc())

    def get_concept_names(self, images=None):
        try:
            if images is None:
                return

            # Get concept names
            concepts = []
            for image in images:
                if "~" in image:
                    concepts.append(os.path.basename(image).split("~")[:-1])

            return concepts
        except:
            print(traceback.format_exc())

    def get_task_imgs(self, folder_path=None):
        try:
            if folder_path is None:
                return

            # Walk through the folder and get all images
            images = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        images.append(os.path.join(root, file))

            return images
        except:
            print(traceback.format_exc())

    def display_task_imgs_helper(self, images=None):
        try:
            if images is None:
                return

            # Divide images into rows and columns. 4 times more columns than rows
            rows = int(0.5 * np.sqrt(len(images)))
            if rows == 0:
                rows = 1
            cols = len(images) // rows

            self.fig_tasks.clear()

            # Display images using matplotlib
            for i in range(rows):
                for j in range(cols):
                    ax = self.fig_tasks.add_subplot(rows, cols, i * cols + j + 1)
                    ax.imshow(plt.imread(images[i * cols + j]))
                    ax.axis("off")

            self.can_tasks.draw()

        except:
            print(traceback.format_exc())

    def init_task_block_design(self):
        # Add a table for block design

        self.tblBlockDesign.setAlternatingRowColors(True)
        self.tblBlockDesign.resizeColumnsToContents()
        self.tblBlockDesign.resizeRowsToContents()
        self.tblBlockDesign.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.tblBlockDesign.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

    def init_task_chkboxes(self):
        self.task_chk_list = [
            #"remote_analysis",
            #"acq_network",
            "auto_resp",
            #"use_only_main_pics",
            #"templates_required",
            #"is_online",
            #"remove_pics",
            "do_sorting",
            "legacy",
        ]

        for chkitem in self.task_chk_list:
            # Add a check box for show/hide of vLytAcqSig
            chk = QtWidgets.QCheckBox(self.tasks_widget)
            chk.setObjectName(f"chkTask{chkitem}")
            chk.setText(chkitem)
            chk.setChecked(False)
            chk.stateChanged.connect(self.task_chk_toggled)

            # Add the check box to a vertical layout with a spacer on top
            vLyt = QtWidgets.QVBoxLayout()
            vLyt.setObjectName(f"vLytTask{chkitem}")
            vLyt.setContentsMargins(0, 0, 0, 0)
            vLyt.setSpacing(0)
            vLyt.addStretch()
            vLyt.addWidget(chk)
            vLyt.addStretch()
            self.hLytTasksControls.addLayout(vLyt)

    def task_chk_toggled(self, state=None):
        try:
            chk = self.tasks_widget.sender()
            param = chk.objectName().replace("chkTask", "")
            setattr(self, param, chk.isChecked())
            self.c.task_clus_param.emit(param, chk.isChecked())
        except:
            print(traceback.format_exc())

    def toggle_conn_thread(self):
        chk_remote_analysis = self.tasks_widget.findChild(
            QtWidgets.QCheckBox, "chkTaskremote_analysis"
        )
        if chk_remote_analysis.isChecked():
            self.th_stop_conn_evt.clear()
            if hasattr(self, "conn_thread") and not self.conn_thread.is_alive():
                self.conn_thread = threading.Thread(
                    target=self.connect_to_analysis_machine
                )
                self.conn_thread.start()
        else:
            self.th_stop_conn_evt.set()

    def update_task_params(self, state=None):
        try:
            # Update analysis machine
            self.analysis_machine = self.tasks_widget.findChild(
                QtWidgets.QComboBox, "cmbTaskAnalysisMachine"
            ).currentText()
            self.monitor = self.tasks_widget.findChild(
                QtWidgets.QComboBox, "cmbMonitor"
            ).currentText()
            self.language = self.tasks_widget.findChild(
                QtWidgets.QComboBox, "cmbLanguage"
            ).currentText()
            self.trial_config = self.tasks_widget.findChild(
                QtWidgets.QComboBox, "cmbTrialConfig"
            ).currentText()
            self.load_task_config()

        except:
            print(traceback.format_exc())

    def load_task_config(self):
        # Locals
        seq_length = 0
        num_seqs = 0

        trial_config = self.tasks_widget.findChild(
            QtWidgets.QComboBox, "cmbTrialConfig"
        ).currentText()
        self.c.exp_config.emit(trial_config)
        self.block_params = config["trial_config"][trial_config]["block_params"]
        self.exp_params = config["trial_config"][trial_config]["exp_params"]
        # self.update_task_chkboxes(chk_params=config['trial_config'][trial_config]['chk_params'])
        self.tblBlockDesign.setColumnCount(len(self.block_params["NPICS"]))
        self.tblBlockDesign.setHorizontalHeaderLabels(
            [f"T {i+1}" for i in range(len(self.block_params["NPICS"]))]
        )
        self.tblBlockDesign.setRowCount(0)  # clear table
        self.tblBlockDesign.setRowCount(len(self.block_params) + 1)

        for i, block in enumerate(self.block_params):
            self.tblBlockDesign.setVerticalHeaderItem(
                i, QtWidgets.QTableWidgetItem(str(block))
            )
            for j, task in enumerate(self.block_params[block]):
                item = QtWidgets.QTableWidgetItem(str(task))
                item.setTextAlignment(Qt.AlignCenter)
                self.tblBlockDesign.setItem(i, j, item)

        seq_length = int(self.exp_params["MAX_SCR_TIME"][0] / self.exp_params["ISI"][0])
        self.tblBlockDesign.setVerticalHeaderItem(
            len(self.block_params), QtWidgets.QTableWidgetItem("SEQ_LEN")
        )
        item = QtWidgets.QTableWidgetItem(str(seq_length))
        item.setTextAlignment(Qt.AlignCenter)
        self.tblBlockDesign.setItem(len(self.block_params), 0, item)

        self.tblBlockDesign.resizeRowsToContents()
        self.tblBlockDesign.resizeColumnsToContents()
        self.tblBlockDesign.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

    def update_task_chkboxes(self, chk_params=None):
        for item in self.task_chk_list:
            chk = self.tasks_widget.findChild(QtWidgets.QCheckBox, f"chkTask{item}")
            if (item in chk_params) and ("True" == chk_params[item]):
                chk.setChecked(True)
            else:
                chk.setChecked(False)

    def init_task_cmb(self):
        # Add a combobox to select analysis machine.
        cmb = QtWidgets.QComboBox()
        cmb.setObjectName("cmbTaskAnalysisMachine")
        cmb.addItems(config["analysis_machines"])
        cmb.currentIndexChanged.connect(self.update_task_params)

        # Add the combobox to a vertical layout with a spacer on top
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.setObjectName("vLytTaskAnalysisMachine")
        vLyt.setContentsMargins(0, 0, 0, 0)
        vLyt.setSpacing(0)
        vLyt.addStretch()

        # Add the combobox
        vLyt.addWidget(cmb)
        vLyt.addStretch()
        self.hLytTasksControls.addLayout(vLyt)

        # Add a combobox to select task
        cmb = QtWidgets.QComboBox()
        cmb.setObjectName("cmbTrialConfig")
        cmb.addItems(list(config["trial_config"].keys()))
        cmb.currentIndexChanged.connect(self.update_task_params)

        # Add the combobox to a vertical layout with a spacer on top
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.setObjectName("vLytSubTask")
        vLyt.setContentsMargins(0, 0, 0, 0)
        vLyt.setSpacing(0)
        vLyt.addStretch()

        # Add the combobox
        vLyt.addWidget(cmb)
        vLyt.addStretch()
        self.hLytTasksControls.addLayout(vLyt)

        # Add a combobox to select language (English, Spanish, French)
        cmb = QtWidgets.QComboBox()
        cmb.setObjectName("cmbLanguage")
        cmb.addItems(config["languages"])
        cmb.currentIndexChanged.connect(self.update_task_params)
        self.language = cmb.currentText()

        # Add the combobox to a vertical layout with a spacer on top
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.setObjectName("vLytLanguage")
        vLyt.setContentsMargins(0, 0, 0, 0)
        vLyt.setSpacing(0)
        vLyt.addStretch()

        # Add the combobox
        vLyt.addWidget(cmb)
        vLyt.addStretch()
        self.hLytTasksControls.addLayout(vLyt)

        # Add a combobox to select a monitor
        cmb = QtWidgets.QComboBox()
        cmb.setObjectName("cmbMonitor")
        monitors = [f"Monitor {i}" for i in range(len(screeninfo.get_monitors()))]
        cmb.addItems(monitors)
        cmb.currentIndexChanged.connect(self.update_task_params)

        # Add the combobox to a vertical layout with a spacer on top
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.setObjectName("vLytMonitor")
        vLyt.setContentsMargins(0, 0, 0, 0)
        vLyt.setSpacing(0)
        vLyt.addStretch()

        # Add the combobox
        vLyt.addWidget(cmb)
        vLyt.addStretch()
        self.hLytTasksControls.addLayout(vLyt)

    def init_task_buttons(self):
        # Add a button to start a task
        self.btnStartTask = QtWidgets.QPushButton()
        self.btnStartTask.setObjectName("btnTaskStart")
        self.btnStartTask.setText("Start Task")
        self.btnStartTask.clicked.connect(self.start_task)

        # Add the button to a vertical layout with a spacer on top
        vLyt = QtWidgets.QVBoxLayout()
        vLyt.setObjectName("vLytTaskStart")
        vLyt.setContentsMargins(0, 0, 0, 0)
        vLyt.setSpacing(0)
        vLyt.addStretch()
        vLyt.addWidget(self.btnStartTask)
        vLyt.addStretch()
        self.hLytTasksControls.addLayout(vLyt)

        # Add an empty label for recording indicator to hLytAcqControls
        self.lblRecordIndicator = QtWidgets.QLabel()
        self.lblRecordIndicator.setText("")
        self.lblRecordIndicator.setAlignment(Qt.AlignCenter)
        # self.lblRecordIndicator.setGeometry(QtCore.QRect(10, 20, 21, 21))
        self.lblRecordIndicator.setFixedSize(21, 21)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.lblRecordIndicator.setFont(font)
        self.lblRecordIndicator.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lblRecordIndicator.setText("")
        self.lblRecordIndicator.setPixmap(QtGui.QPixmap(BLUE_LED_PATH))
        self.lblRecordIndicator.setScaledContents(True)
        self.hLytTasksControls.addWidget(self.lblRecordIndicator)

    def update_recording_status(self, state=None):
        try:
            if state:
                self.lblRecordIndicator.setPixmap(
                    QtGui.QPixmap("../assets/icons/red-led-on.png")
                )
                self.lblRecordIndicator.setToolTip("Recording is on")
            else:
                self.lblRecordIndicator.setPixmap(
                    QtGui.QPixmap("../assets/icons/blue-led-on.png")
                )
                self.lblRecordIndicator.setToolTip("Recording is off")
        except:
            print(traceback.format_exc())

    def start_task(self):
        """
        Asks orchestrator to start the task if trellis & pictures, other params are set.
        """
        try:
            self.c.start_task.emit(True)
        except:
            print(traceback.format_exc())

    def start_rsvp_task(self):
        """
        Starts RSVP task.
        This function is called by orchestrator.py
        """
        try:
            if not hasattr(self, "img_paths"):
                # pop up a message box
                msgBox = QtWidgets.QMessageBox()
                msgBox.setIcon(QtWidgets.QMessageBox.Information)
                msgBox.setText(
                    "No pictures to start RSVP task. Browse pictures using the button on right bottom."
                )
                msgBox.setWindowTitle("No pictures found")
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
                # msgBox.buttonClicked.connect(msgButtonClick)

                returnValue = msgBox.exec()
                if returnValue == QtWidgets.QMessageBox.Ok:
                    print("OK clicked")
                self.c.log.emit(
                    "No pictures to start RSVP task. Browse pictures using the button on right bottom."
                )
                return
            
            self.update_task_params()

            self.btnStartTask.setText("Stop Task")
            self.btnStartTask.clicked.disconnect(self.start_task)
            self.btnStartTask.clicked.connect(self.stop_task)

            self.c.record.emit(True)
            
            # self.orchestrator.responses_view.raiseDock()
            self.start_experiment()
        except:
            print(traceback.format_exc())

    def stop_task(self, stop=True):
        """
        Ends the task and resets the button text.
        """
        try:
            self.btnStartTask.setText("Start Task")
            try:
                self.btnStartTask.clicked.disconnect(self.stop_task)
            except:
                pass
            self.btnStartTask.clicked.connect(self.start_task)

            self.abort_experiment()
        except:
            print(traceback.format_exc())

    # endregion Tasks

    # region Psychopy

    def display_photo_detect_sq(self, color="white"):
        # Draw a white square left bottom of the screen
        photo_detect_sq = visual.Rect(
            self.win,
            width=self.config["flicker_sq_size"][0],
            height=self.config["flicker_sq_size"][1],
            pos=(-self.screen_size[0] // 2, -self.screen_size[1] // 2),
            fillColor=color,
        )
        photo_detect_sq.draw()

    def display_rect_bar(self, pos, color="yellow"):
        # Draw a rectangle bar on top/bottom of the image
        rect_bar = visual.Rect(
            self.win,
            width=self.config["rect_bar_size"][0],
            height=self.config["rect_bar_size"][1],
            pos=pos,
            fillColor=color,
        )

        rect_bar.draw()

    def display_stim_img(self, img):
        img_stim = visual.ImageStim(self.win, image=img, pos=(0, 0))
        img_stim.draw()

        pos_above = (0, img_stim.size[0] // 2 + 20)
        color_above = "yellow" if random.random() > 0.6 else "red"
        self.display_rect_bar(pos_above, color=color_above)

        pos_below = (0, -img_stim.size[0] // 2 - 20)
        color_below = "yellow" if random.random() > 0.6 else "red"
        self.display_rect_bar(pos_below, color=color_below)

        # Draw photo detection square
        self.display_photo_detect_sq(color="white")
        self.win.flip()

        img_stim = visual.ImageStim(self.win, image=img, pos=(0, 0))
        img_stim.draw()

        self.display_rect_bar(pos_above, color=color_above)
        self.display_rect_bar(pos_below, color=color_below)

        self.display_photo_detect_sq(color="black")
        self.win.flip()

        core.wait(0.5)

    def display_stim_text(self, text, duration=0.5):

        text_stim = visual.TextStim(self.win, text=text, pos=(0, 0))
        text_stim.draw()
        self.win.flip()
        core.wait(duration)

    def start_experiment_psy(self):
        try:
            self.screen_size = [self.monitor.width, self.monitor.height]
            if not hasattr(self, "win"):
                self.win = visual.Window(
                    size=self.screen_size, fullscr=True, units="pix", color="black"
                )
            # self.win.mouseVisible = False
            self.display_stim_text(self.config["subscr_msgs"]["begin"][self.language])
            self.handle_keyboard_events()
            self.c.log.emit("Experiment started at {}".format(datetime.datetime.now()))

            # Start recording sEEG
            self.start_recording()

            trial_n_pics_list = self.block_params["NPICS"]
            trial = 0

            while (
                trial < len(trial_n_pics_list) and not self.b_abort
            ):  # Subscreening loop
                if trial > 0:
                    self.c.log.emit(f"Displaying analysis results from trial {trial}")

                if trial_n_pics_list[trial] > len(self.img_paths):
                    self.c.log.emit(
                        f"Subscreening {trial+1} skipped due to insufficient images. Check config file."
                    )
                    continue

                self.c.log.emit(
                    f"Subscreening {trial+1} started at {datetime.datetime.now()}"
                )

                # Pick random images for the trial
                imgs = random.sample(self.img_paths, trial_n_pics_list[trial])

                # Divide the images based on block params
                n_imgs_seq = (
                    self.exp_params["MAX_SCR_TIME"][0] / self.exp_params["ISI"][0]
                )
                self.c.img_list.emit(imgs)

                img_idx = 0
                while (
                    img_idx < len(imgs) and not self.b_abort
                ):  # Runs one subscreening.
                    if n_imgs_seq % len(imgs) == img_idx:
                        self.start_stop_sequence()
                        self.display_stim_text(
                            text=self.config["subscr_msgs"]["continue"][self.language]
                        )
                        self.handle_keyboard_events()  # wait for user's key press

                    self.display_stim_img(imgs[img_idx])
                    self.c.img.emit(imgs[img_idx])

                self.c.log.emit(
                    f"Subscreening {trial+1} ended at {datetime.datetime.now()}. Starting analysis"
                )

                # Start analysis
                self.start_analysis()

                self.display_stim_text(
                    text=self.config["subscr_msgs"]["short_break"][self.language],
                    duration=5,
                )
                self.display_stim_text(
                    text=self.config["subscr_msgs"]["continue"][self.language]
                )
                self.handle_keyboard_events()  # wait for user's key press

            if self.abort:
                self.c.log.emit("Experiment aborted by user")

            self.display_stim_text(
                self.config["subscr_msgs"]["end"][self.language], duration=5
            )

            self.end_experiment()
        except:
            traceback.print_exc()
            self.end_experiment()

    def handle_keyboard_events(self):
        while True:
            response = psychopy.event.getKeys()
            if response:
                if "escape" in response:
                    print("Experiment aborted by user")
                    self.end_experiment()
                    break
                else:
                    time.sleep(0.5)
                    break

    def start_stop_sequence(self):
        if not self.seq_no == 0:
            if hasattr(self, "socket"):
                # End the current sequence.
                self.socket.send(b"sequence_ended")
                data = self.socket.recv(1024)
            self.c.log.emit(
                f"{data.decode()} {self.seq_no} at {datetime.datetime.now()}"
            )

        self.seq_no += 1

        if hasattr(self, "socket"):
            # Start a new sequence
            self.socket.send(b"sequence_started")
            data = self.socket.recv(1024)
        self.c.log.emit(f"{data.decode()} {self.seq_no} at {datetime.datetime.now()}")

    def start_recording(self):
        if hasattr(self, "socket"):
            self.socket.send(b"start_recording")
            data = self.socket.recv(1024)
            self.c.log.emit(f"{data.decode()} at {datetime.datetime.now()}")

            self.socket.send(b"load_config")
            self.socket.send(f"{self.trial_config}")
            data = self.socket.recv(1024)
            self.c.log.emit(f"{data.decode()} at {datetime.datetime.now()}")

    def start_analysis(self):
        if hasattr(self, "socket"):
            self.socket.send(b"start_analysis")
            data = self.socket.recv(1024)
            self.c.log.emit(f"{data.decode()} at {datetime.datetime.now()}")

    def stop_recording(self):
        if hasattr(self, "socket"):
            self.socket.send(b"stop_recording")
            data = self.socket.recv(1024)
            self.c.log.emit(f"{data.decode()} at {datetime.datetime.now()}")

    def end_experiment_psy(self):
        core.wait(5)
        self.win.close()

        self.c.log.emit("Experiment ended at {}".format(datetime.datetime.now()))
        if hasattr(self, "socket"):
            self.socket.send(b"end_analysis")
            data = self.socket.recv(1024)
            self.c.log.emit(f"{data.decode()} at {datetime.datetime.now()}")

    # endregion Psychopy

    # region RL frameless window

    # region Keyboard event handling

    def on_press(self, key):
        print("{0} pressed".format(key))

    def on_release(self, key):
        print("{0} release".format(key))
        if key == Key.esc:
            self.close_exp_win()
            # QMetaObject.invokeMethod(self.rsvp_worker, 'close_exp_win', Qt.QueuedConnection)

        return False

    def wait_for_key_press(self):
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
        # self.b_exp_paused = True
        # while self.b_exp_paused and not self.b_abort:
        #     time.sleep(0.1)

    # endregion Keyboard event handling

    def show_exp_win(self):
        desktop = QApplication.desktop()
        if desktop.screenCount() > 1:
            monitorId = int(self.monitor[-1])
            rect = desktop.screenGeometry(
                monitorId
            )  # get the geometry of the second monitor
            if not hasattr(self, "framelessWindow") or self.framelessWindow is None:
                self.framelessWindow = FramelessWindow(c=self.c)
            self.framelessWindow.move(rect.left(), rect.top())
            self.framelessWindow.setWindowFlags(
                Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            )
            self.framelessWindow.showFullScreen()
        else:
            self.framelessWindow = FramelessWindow(c=self.c)
            self.framelessWindow.show()

    def close_exp_win(self):
        if hasattr(self, "framelessWindow"):
            self.framelessWindow.close()
            self.framelessWindow = None

    def init_subscreening_config(self):
        self.exp_img_idx = 0
        self.exp_total_imgs = np.sum(
            np.multiply(
                np.array(self.block_params["NPICS"]),
                np.array(self.block_params["NREP"]),
            )
        )

        self.n_subscreenings = len(self.block_params["NPICS"])
        self.subscr_idx = 0

        self.n_imgs_seq = int(
            self.exp_params["MAX_SCR_TIME"][0] / self.exp_params["ISI"][0]
        )

    def start_experiment(self):
        # Locals
        bOK = False

        self.exp_start_time = time.time()

        # Initialize frameless window where stimuli will be shown to patient
        self.show_exp_win()
        self.framelessWindow.show_text(
            self.config["subscr_msgs"]["begin"][self.language]
        )
        # self.rsvp_worker.start()
        QApplication.processEvents()

        # Progress bar value
        self.progress = 0
        self.c.progress.emit(self.progress, None)

        self.init_subscreening_config()

        # Initialize the first trial
        bOK = self.init_subscreening()

        if bOK:
            # Initialize the first sequence
            self.init_sequence()

    def init_subscreening(self):
        # Locals
        bOK = False
        if self.subscr_idx > 0:
            self.c.log.emit(f"Displaying analysis results from trial {self.subscr_idx}")

        if self.block_params["NPICS"][self.subscr_idx] >= len(self.img_paths):
            self.c.log.emit(
                f"Subscreening {self.subscr_idx+1} skipped due to insufficient images. Check config file."
            )
            return

        self.c.log.emit(
            f'Subscreening {self.subscr_idx+1} started at {datetime.datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")}'
        )

        # Pick random images for the trial
        self.subscr_img_paths = random.sample(
            self.img_paths, self.block_params["NPICS"][self.subscr_idx]
        )

        # trial img paths repetition list
        self.subscr_img_paths_rep = []

        # Make a copy of self.trial_img_paths
        img_paths_copy = self.subscr_img_paths.copy()

        # Step 2: Repeat n times
        for _ in range(self.block_params["NREP"][self.subscr_idx]):

            # Shuffle the copy
            np.random.shuffle(img_paths_copy)

            # Append the shuffled copy to the result list
            self.subscr_img_paths_rep.append(img_paths_copy)

        # Step 3: Flatten the result list
        self.subscr_img_paths_rep = [
            img_path for sublist in self.subscr_img_paths_rep for img_path in sublist
        ]

        self.c.img_list.emit(self.subscr_img_paths)

        self.n_seqs = (
            len(self.subscr_img_paths)
            * self.block_params["NREP"][self.subscr_idx]
            // self.n_imgs_seq
        )
        self.subscr_seq_idx = 0

        bOK = True
        return bOK

    def init_sequence(self):
        """
        This method initializes the sequence by setting the sequence image paths and images.
        """
        seq_img_start_idx = self.subscr_seq_idx * self.n_imgs_seq
        self.seq_img_paths = self.subscr_img_paths_rep[
            seq_img_start_idx : seq_img_start_idx + self.n_imgs_seq
        ]
        self.seq_imgs = [self.read_image(img) for img in self.seq_img_paths]
        self.framelessWindow.show_text(
            txt=self.config["subscr_msgs"]["continue"][self.language]
        )
        # QMetaObject.invokeMethod(self.rsvp_worker, 'show_msg', 
        #                          Qt.QueuedConnection, 
        #                          Qt.Q_ARG(str, self.config["subscr_msgs"]["continue"][self.language]))

        self.seq_img_idx = 0

        # Start the sequence if auto advance is enabled, otherwise wait for user's key press
        if self.b_auto_advance:
            self.continue_key_press_handler()
        else:
            self.c.log.emit("Waiting for patient to resume screening.")

    def continue_key_press_handler(self):
        """
        This method is called when user presses a key to continue the screening.
        This starts a sequence where about 60 images are shown half a second apart.
        """
        self.c.start_task_acq.emit(True, self.subscr_idx, self.subscr_seq_idx)
        if self.subscr_seq_idx == 0 and self.seq_img_idx == 0:
            # Wait 6 seconds
            time.sleep(6)

            self.daq.send_info(DATA_SIGNATURE_ON)
            time.sleep(0.05)
            self.daq.send_info(DATA_SIGNATURE_OFF)
            time.sleep(0.45)
            self.daq.send_info(DATA_SIGNATURE_ON)
            time.sleep(0.05)
            self.daq.send_info(DATA_SIGNATURE_OFF)
            time.sleep(0.45)
            self.daq.send_info(DATA_SIGNATURE_ON)
            time.sleep(0.05)
            self.daq.send_info(DATA_SIGNATURE_OFF)
        else:
            self.daq.send_info(TRIAL_ON)
            time.sleep(0.15)
            self.daq.send_info(BLANK_ON)
            time.sleep(0.5)
            self.flip_lines_color()
            self.daq.send_info(LINES_ONOFF)
            if np.random.rand() > 0.5:
                self.daq.send_info(LINES_FLIP_BLANK)

        self.rsvpscr_timer.start(
            int(self.exp_params["ISI"][0] * 1000)
        )  # Half a second in ms
        # if self.rsvpscr_scheduler.running:
        #     self.rsvpscr_scheduler.resume()
        # else:
        #     self.rsvpscr_scheduler.start()

    def run_screening(self):
        """
        This method is called by a timer to show images in a sequence. It also updates trial and sequence indices.
        """
        if self.seq_img_idx >= len(self.seq_imgs):  # End of sequence check
            self.c.start_task_acq.emit(False, self.subscr_idx, self.subscr_seq_idx)
            self.rsvpscr_timer.stop()
            # self.rsvpscr_scheduler.pause()


            # Log the end of the sequence
            hrs, mins, secs = self.get_elapsed_time()
            hrs_rem, mins_rem, secs_rem = self.get_remaining_time()
            self.c.log.emit(
                f"Subscreening {self.subscr_idx+1}/{self.n_subscreenings} | Seq {self.subscr_seq_idx+1}/{self.n_seqs} | " +
                f"Elapsed {mins:02.0f} mins {secs:02.0f} secs | Remaining: {mins_rem:02.0f} mins {secs_rem:02.0f} secs"
            )

            # Initialize analysis of sequence
            self.c.analyze.emit(
                self.seq_imgs, self.seq_img_paths, self.subscr_idx, self.subscr_seq_idx
            )

            # Update trial and sequence indices
            self.subscr_seq_idx += 1
            if self.subscr_seq_idx >= self.n_seqs:
                self.subscr_idx += 1
                if self.subscr_idx >= self.n_subscreenings:
                    # self.rsvpscr_scheduler.shutdown()
                    self.end_experiment()
                    return
                else:
                    self.init_subscreening()

            if self.b_abort:  # Check if user aborted the experiment
                self.c.log.emit("Experiment aborted by user.")
                self.end_experiment()
                return

            # Initialize the next sequence
            self.init_sequence()

        else:
            # self.show_stim_img()
            # self.c.show_stim_img.emit(self.seq_imgs[self.seq_img_idx])
            self.c.show_stim_img.emit()
            self.seq_img_idx += 1
            self.exp_img_idx += 1

        self.progress = int(self.exp_img_idx / self.exp_total_imgs * 100)
        self.c.progress.emit(self.progress, None)

    def abort_experiment(self):
        if not hasattr(self, "rsvpscr_timer"):
            return
        self.b_abort = True

        if self.rsvpscr_timer.isActive():
            self.c.log.emit("Experiment aborted. Will end after the current sequence.")
            pass
        else:
            self.end_experiment()
        # if self.rsvpscr_scheduler.running:
        #     self.c.log.emit("Experiment aborted. Will end after the current sequence.")
        #     pass
        # else:
        #     self.end_experiment()

    def get_elapsed_time(self, start_time=None):
        if start_time is None:
            start_time = self.exp_start_time
        elapsed_time = time.time() - start_time
        hrs, secs = divmod(elapsed_time, 3600)
        mins, secs = divmod(secs, 60)
        return hrs, mins, secs

    def get_remaining_time(self, start_time=None):
        if start_time is None:
            start_time = self.exp_start_time
        elapsed_time = time.time() - start_time
        remaining_time = (
            elapsed_time / self.exp_img_idx * (self.exp_total_imgs - self.exp_img_idx)
        )
        hrs, secs = divmod(remaining_time, 3600)
        mins, secs = divmod(secs, 60)
        return hrs, mins, secs
    
    def flip_lines_color(self):
        self.b_flip_lines_color = not self.b_flip_lines_color
        if self.b_flip_lines_color:
            self.framelessWindow.change_bar_color(
                self.framelessWindow.topBar, [1, 1, 0]
            )
            self.framelessWindow.change_bar_color(
                self.framelessWindow.bottomBar, [1, 0, 0]
            )
        else:
            self.framelessWindow.change_bar_color(
                self.framelessWindow.topBar, [1, 0, 0]
            )
            self.framelessWindow.change_bar_color(
                self.framelessWindow.bottomBar, [1, 1, 0]
            )
        self.daq.send_info(LINES_FLIP_PIC)

    def show_stim_img(self, img=None):
        """
        This method is called by a timer to show images in a sequence.

        Args:
            img ([type], optional): img to show on the frameless task window. Defaults to None.
        """

        if img is None:
            img = self.seq_imgs[self.seq_img_idx]

        # QMetaObject.invokeMethod(self.rsvp_worker, 'show_stim_img',
        #                             Qt.QueuedConnection, 
        #                             Qt.Q_ARG(np.ndarray, img))
        if hasattr(self, "framelessWindow"):
            self.framelessWindow.textItem.setHtml("")  # Clear text item
            self.framelessWindow.show_photo_detect_square()
            self.framelessWindow.show_image(img)

            # Send a pulse from the DAQ to the sEEG system
            self.daq.send_info(PIC_ON_OFF[self.seq_img_idx % 2, int(np.ceil(2 * self.subscr_seq_idx/ self.n_seqs))])

            if np.random.rand() > 0.8:
                self.flip_lines_color()

        self.c.img.emit(self.exp_img_idx, img, self.seq_img_paths[self.seq_img_idx])

    def end_experiment(self):
        # QMetaObject.invokeMethod(self.rsvp_worker, 'close_exp_win', 
        #                          Qt.QueuedConnection)
        self.close_exp_win()
        self.c.progress.emit(100, None)

        hrs, mins, secs = self.get_elapsed_time()
        self.c.log.emit(
            (
                "Subscreening {}/{} | Seq {}/{} | "
                "Elapsed {:02.0f} mins {:02.0f} secs"
            ).format(
                self.subscr_idx + 1,
                self.n_subscreenings,
                self.subscr_seq_idx,
                self.n_seqs,
                mins,
                secs,
            )
        )

        self.c.log.emit(
            "Experiment ended at {}".format(
                datetime.datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
            )
        )
        self.c.record.emit(False)

    # endregion RL frameless window

if __name__ == "__main__":
    n_seqs = 18
    seq_img_idx = 10
    subscr_seq_idx = 1
    print(PIC_ON_OFF[seq_img_idx % 2, int(np.ceil(2 * subscr_seq_idx/ n_seqs))])