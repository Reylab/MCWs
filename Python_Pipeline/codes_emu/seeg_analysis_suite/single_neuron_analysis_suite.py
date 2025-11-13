# Author: Sunil Mathew
# Date: 10 November 2023
# sEEG analysis suite
# sphinx-build -M html docs/source/ docs/build/ for updating the documentation
# remember to add docstrings to all functions and classes for auto documentation

import pyqtgraph as pg
import pyvista as pv

import traceback
import os
import sys

app_path = os.path.join(os.path.dirname(__file__))
sys.path.append(app_path)

os.environ["QT_API"] = "pyqt5"
os.environ["OMP_NUM_THREADS"] = '6'
os.environ["USE_SYSTEM_VTK"] = 'OFF'

from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtWidgets import QApplication, QWidget
from qtpy.QtGui import QPalette
from pyvistaqt import MainWindow
from core.orchestrator import Orchestrator

class SNAAS(MainWindow):

    #region Initialization

    def __init__(self):
        """
         Initialize the window. This is called by __init__ () and can be overridden
        """
        super().__init__()
        ui_file = os.path.join(app_path, 'eeg_suite.ui')
        uic.loadUi(ui_file, self)
        
        self.init_ui()
        self.init_orchestrator()
        # self.load_mat_data()

    def resizeEvent(self, event):
        """
         Called when the window is resized. This is the default implementation of QWidget. resizeEvent
         
         Args:
         	 event: The resize event to
        """
        try:
            QtWidgets.QMainWindow.resizeEvent(self, event)

        except:
            print("resizeEvent")
            print(traceback.format_exc())

    def init_ui(self):
        """
         Initialize the UI. This is called by __init__ () and can be overridden
        """
        widget = QWidget()
        widget.setLayout(self.vLytMain)
        self.setCentralWidget(widget)
        
        self.tabData.setLayout(self.vLytData)

        self.init_file_menu()
        self.init_3d_viewer()
        self.init_dock_area()
        
        self.tabTasks.setLayout(self.vLytTasksMain)

        self.statusBar().showMessage('Ready')
        self.pbTask = QtWidgets.QProgressBar()
        self.statusBar().addPermanentWidget(self.pbTask)

    def init_dock_area(self):
        """
            Initialize signal viewer. This is called by init_ui
        """
        palette = QApplication.palette()
        # Change background color signal viewer based on whether dark mode is on or off.
        if palette.color(QPalette.Window).value() < 128:
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
        else:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
        self.dock_area = pg.dockarea.DockArea()
        self.vLytTasksMain.addWidget(self.dock_area)

    def restore_ui(self):
        self.orchestrator.restore_ui()

    #endregion Initialization

    #region File menu

    def init_file_menu(self):
        """
         Initialize menu for file dialog and add actions to menubar. This is called by init_ui
        """
        # Add File menu
        self.mnuFile = self.menuBar().addMenu('File')

        self.mnuFile.addAction('Open concepts file', self.open_concepts_dialog, 'Ctrl+O')
        self.mnuFile.addAction('Open ripple map file', self.open_ripple_map_dialog, 'Ctrl+O')
        self.mnuFile.addAction('Open .ns* file', self.open_ns_file_dialog, 'Ctrl+O')
        self.mnuFile.addAction('Open CT/MRI', self.open_ct_mri_dialog, 'Ctrl+O')
        self.mnuFile.addAction('Exit', self.close, 'Ctrl+Q')

        self.edit_menu = self.menuBar().addMenu('Edit')
        self.edit_menu.addAction('Restore UI', self.restore_ui, 'Ctrl+R')

    def open_concepts_dialog(self):
        """
        Open concepts file dialog
        """
        # Select file with concepts
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select concepts file', os.path.dirname(__file__), 'Concepts file (*.txt)')[0]
        # Load data from file_path if file_path is not empty
        if file_path:
            self.orchestrator.load_concepts(file_path=file_path)

    def open_ripple_map_dialog(self):
        """
        Open ripple map file dialog
        """
        # Select folder with CT/MRI files
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select ripple map file', os.path.dirname(__file__), 'Ripple map (*.map)')[0]
        # Load data from file_path if file_path is not empty
        if file_path:
            self.orchestrator.ripple_acq.init_ripple(map_file=file_path)
            self.orchestrator.c.log.emit(f'Loaded ripple map file: {file_path}')
            # self.orchestrator.update_channel_list()

            # cmbChannels = self.findChild(CheckableTreeComboBox, 'cmb_sig_channel')
            # cmbChannels.clear()
            # # cmbChannels.addItems([bundle for bundle in self.orchestrator.ripple_acq.ripple_map.bundles])
            # # cmbChannels.addItems([electrode.label for electrode in self.orchestrator.ripple_acq.ripple_map.electrodes])
            # cmbChannels.addItems({'Channels': self.orchestrator.ripple_acq.ripple_map.port_bundles})

    def open_ct_mri_dialog(self):
        """
        Open CT/MRI file dialog
        """
        # Select folder with CT/MRI files
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder with CT/MRI files', os.path.dirname(__file__))
        # Load data from file_path if file_path is not empty
        if folder_path:
            self.load_img_data(folder_path=folder_path)


    def open_ns_file_dialog(self):
        """
        Open. ns file dialog and load data from file_path if file_path is not empty
        """
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open .ns file', os.path.dirname(__file__))[0]
        # Load ns file if file_path is not empty
        if file_path:
            self.load_ns_file(file_path=file_path)

    #endregion File menus

    #region Load NsX file

    def load_ns_file(self, file_path=None):
        """
         Loads a. ns * file and displays the data. This is a convenience method for loading and displaying the data from a. ns * file
         
         Args:
         	 file_path: Path to the. ns *
         Loads a .ns* file and displays the data
        """
        # This method is called by the simulation.
        if file_path.endswith('.ns6'):
            self.orchestrator.read_brk_ns_file(filename=file_path)
            return
        
        self.orchestrator.read_ripple_nsx_file(filename=file_path)
        self.b_nsx_read = self.orchestrator.ripple_nsx.b_nsx_read

        imgs_dir = os.path.join(os.path.dirname(file_path), 'pics_now')
        if not os.path.exists(imgs_dir):
            imgs_dir = os.path.join(os.path.dirname(file_path), 'pics_used')
        if not os.path.exists(imgs_dir):
            print('No pics folder found.')
            self.orchestrator.c.log.emit('No pics folder found.')
        if os.path.exists(imgs_dir):
            self.orchestrator.tasks.display_task_imgs_thread(folder_path=imgs_dir)
    
    #endregion Load NsX file

    #region Simulation, spike sorting, and clustering

    def init_orchestrator(self):
        """
         Initialize DSP and add it to the plot widget.
        """
        try:
            self.orchestrator = Orchestrator(sig_view=self.dock_area)
            self.orchestrator.c.progress.connect(self.update_progress)
        except:
            print(traceback.format_exc())

    #endregion Simulation, spike sorting, and clustering

    # region Main window events

    def update_progress(self, value=None, msg=''):
        try:
            if value is None:
                return

            self.pbTask.setValue(value)
            self.statusBar().showMessage(msg)
        except:
            print(traceback.format_exc())

    def keyPressEvent (self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == QtCore.Qt.Key_F1:
            print('Help')
        elif key == QtCore.Qt.Key_F5:
            print('Refresh')
        elif key == QtCore.Qt.Key_Left:
            self.orchestrator.ripple_nsx.back()
        elif key == QtCore.Qt.Key_Up:
            print('Up')
        elif key == QtCore.Qt.Key_T:
            self.orchestrator.ripple_nsx.forward()
        elif key == QtCore.Qt.Key_Right:
            self.orchestrator.ripple_nsx.forward()
        elif key == QtCore.Qt.Key_Down:
            print('Down')
        elif key == QtCore.Qt.Key_Escape:
            self.orchestrator.tasks.abort_experiment()
        elif key == QtCore.Qt.Key_C:
            self.orchestrator.tasks.continue_key_press_handler()

    def closeEvent(self, event):
        """
         Close the window and close pyvista plotter
         
         Args:
         	 event: 
        """
        try:
            if hasattr(self, 'plotter_3d'):
                self.plotter_3d.close()
        except:
            print(traceback.format_exc())

    #endregion Main window events

    #region 3D Image data

    def init_3d_viewer(self):
        self.tabImgData.setLayout(self.vLytImgData3D)

        # # Add a pyvista widget to display 3D image data
        # self.plotter_3d = QtInteractor()
        # self.vLytImgData3D.addWidget(self.plotter_3d.interactor)


    def load_img_data(self, folder_path=None):
        if folder_path is None:
            return
        
        # use pyvista to load CT series data
        self.img_data = pv.DICOMReader(folder_path).read()

        self.img_data_mesh = self.img_data.contour(method='marching_cubes')

        self.plotter_3d.add_mesh(self.img_data_mesh, color='tan')

    #endregion 3D Image data

if __name__ == "__main__":
    app = QApplication([])
    window = SNAAS()
    window.showMaximized()
    logo_path = os.path.abspath(os.path.join(app_path, 'rey_lab_logo.png'))
    if os.path.exists(logo_path):
        app.setWindowIcon(QtGui.QIcon(logo_path))
    window.setWindowTitle('Single Neuron Activity Analysis Suite')
    app.exec_()



