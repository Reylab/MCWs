"""
Author: Sunil Mathew
Date: 05 December 2023

This class is for displaying sEEG data in 3D. The 3D viewer is based on mne-python's fsaverage data. 
The patient's CT/MRI can be coregistered to use the 3D model and associated segmentations of different
parts of the brain. The sEEG macro electrodes can be visualized in the 3D space.

"""

import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QApplication 
from qtpy import QtWidgets, QtCore

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from usercontrols.checkable_combobox import CheckableComboBox


class SEEG_3D:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SEEG_3D, cls).__new__(cls)
        return cls.instance

    def __init__(self, c=None, chs_3d_view=None):
        self.c = c
        if chs_3d_view is not None:
            self.chs_3d_view = chs_3d_view
            self.init_3d_layout()

    #region 3D viewer
        
    def init_3d_layout(self):
        # paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
        # which is in MNI space
        misc_path = mne.datasets.misc.data_path()
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = sample_path / "subjects"

        # use mne-python's fsaverage data
        fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed
        
        # Let's load some sEEG data with channel locations and make epochs.

        raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")

        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id, detrend=1, baseline=None)
        epochs = epochs["Response"][0]  # just process one epoch of data for speed

        
        # Let use the Talairach transform computed in the Freesurfer recon-all
        # to apply the Freesurfer surface RAS ('mri') to MNI ('mni_tal') transform.

        montage = epochs.get_montage()

        # first we need a head to mri transform since the data is stored in "head"
        # coordinates, let's load the mri to head transform and invert it
        this_subject_dir = misc_path / "seeg"
        head_mri_t = mne.coreg.estimate_head_mri_t("sample_seeg", this_subject_dir)
        # apply the transform to our montage
        montage.apply_trans(head_mri_t)

        # now let's load our Talairach transform and apply it
        mri_mni_t = mne.read_talxfm("sample_seeg", misc_path / "seeg")
        montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)

        # for fsaverage, "mri" and "mni_tal" are equivalent and, since
        # we want to plot in fsaverage "mri" space, we need use an identity
        # transform to equate these coordinate frames
        montage.apply_trans(mne.transforms.Transform(fro="mni_tal", to="mri", trans=np.eye(4)))

        epochs.set_montage(montage)

        
        # Let's check to make sure everything is aligned.
        #
        # .. note::
        #    The most rostral electrode in the temporal lobe is outside the
        #    fsaverage template self.brain. This is not ideal but it is the best that
        #    the linear Talairach transform can accomplish. A more complex
        #    transform is necessary for more accurate warping, see
        #    :ref:`tut-ieeg-localize`.

        # compute the transform to head for plotting
        trans = mne.channels.compute_native_head_t(montage)
        # note that this is the same as:
        # ``mne.transforms.invert_transform(
        #      mne.transforms.combine_transforms(head_mri_t, mri_mni_t))``

        view_kwargs = dict(azimuth=105, elevation=100, focalpoint=(0, 0, -15))
        self.brain = mne.viz.Brain(
            "fsaverage",
            subjects_dir=subjects_dir,
            cortex="low_contrast",
            alpha=0.1,
            background="white",
            show=False,
        )
        self.brain.add_sensors(epochs.info, trans=trans)
        # self.brain.add_head(alpha=0.25, color="tan")
        # self.brain.show_view(distance=400, **view_kwargs)

        fname_aseg = subjects_dir / "sample" / "mri" / "aparc+aseg.mgz"
        self.label_names = mne.get_volume_labels_from_aseg(fname_aseg)
        self.active_labels = [
            label
            for label in self.label_names  # just plot the first 7 labels
            if "unknown" not in label.lower() and "ctx" in label.lower() 
        ]
        self.active_labels = self.active_labels[1:5]
        legend_kwargs = dict(bcolor=None)
        self.brain.add_volume_labels(aseg="aparc+aseg", labels=self.active_labels, legend=legend_kwargs)

        # Make the legend larger in size so many labels can be seen
        # self.brain._renderer.plotter.legend.SetHeight(0.5)

        self.chs_3d_view.addWidget(self.brain._renderer.plotter.interactor)
        self.chs_3d_view.addWidget(self.get_label_chart())

    def get_label_chart(self):
        # Add a checkbox for each label in vertical layout and create a widget
        self.labels_widget = QtWidgets.QWidget()
        vLyt = QtWidgets.QVBoxLayout()
        self.labels_widget.setLayout(vLyt)
        
        # Add combo boxes for Left, Right, Cortex, CC and Others
        hLytCombo        = QtWidgets.QHBoxLayout()
        self.cmbLeft     = CheckableComboBox()
        self.cmbRight    = CheckableComboBox()
        self.cmbCortexLH = CheckableComboBox()
        self.cmbCortexRH = CheckableComboBox()
        self.cmbCC       = CheckableComboBox()
        self.cmbOthers   = CheckableComboBox()
        hLytCombo.addWidget(self.cmbLeft)
        hLytCombo.addWidget(self.cmbRight)
        hLytCombo.addWidget(self.cmbCortexLH)
        hLytCombo.addWidget(self.cmbCortexRH)
        hLytCombo.addWidget(self.cmbCC)
        hLytCombo.addWidget(self.cmbOthers)

        # clear button
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_labels)
        hLytCombo.addWidget(self.clear_button)
        vLyt.addLayout(hLytCombo)

        self.cmbLeft.view().pressed.connect(self.show_hide_labels)
        self.cmbRight.view().pressed.connect(self.show_hide_labels)
        self.cmbCortexLH.view().pressed.connect(self.show_hide_labels)
        self.cmbCortexRH.view().pressed.connect(self.show_hide_labels)
        self.cmbCC.view().pressed.connect(self.show_hide_labels)
        self.cmbOthers.view().pressed.connect(self.show_hide_labels)

        self.label_checkboxes = {}
        ctr = 0
        for label in self.label_names:
            if "unknown" not in label.lower():
                cmb = None
                if label.lower().startswith("left"):
                    cmb = self.cmbLeft
                elif label.lower().startswith("right"):
                    cmb = self.cmbRight
                elif label.lower().startswith("ctx-lh"):
                    cmb = self.cmbCortexLH
                elif label.lower().startswith("ctx-rh"):
                    cmb = self.cmbCortexRH
                elif label.lower().startswith("cc"):
                    cmb = self.cmbCC
                else:
                    cmb = self.cmbOthers

                cmb.addItem(label)
                
                ctr += 1

        return self.labels_widget

    def clear_labels(self):
        for cmb in [self.cmbLeft, self.cmbRight, self.cmbCortexLH, self.cmbCortexRH, self.cmbCC, self.cmbOthers]:
            for i in range(cmb.model().rowCount()):
                cmb.model().item(i).setCheckState(QtCore.Qt.Unchecked)
        self.active_labels = []
        self.brain.remove_volume_labels()
        self.brain.remove_sensors()
        self.brain._renderer.plotter.update()
    
    def show_hide_labels(self, index):

        # Get combobox from QModelIndex
        cmb = index.model().parent()

        # getting the item
        item = cmb.model().itemFromIndex(index)

        # checking if item is checked
        if item.checkState() == QtCore.Qt.Checked:

            # making it unchecked
            item.setCheckState(QtCore.Qt.Unchecked)

            self.active_labels.remove(item.text())

            self.brain.remove_volume_labels()
            self.brain.remove_sensors()

        # if not checked
        else:
            # making the item checked
            item.setCheckState(QtCore.Qt.Checked)

            self.active_labels.append(item.text())


        legend_kwargs = dict(bcolor=None)
        self.brain.add_volume_labels(aseg="aparc+aseg", labels=self.active_labels, legend=legend_kwargs)
        self.brain._renderer.plotter.update()

    def show_bundle_vol(self, bundle):
        legend_kwargs = dict(bcolor=None)
        # Remove any existing labels
        self.brain.remove_volume_labels()
        if hasattr(self, 'brain') and bundle is not None:
            self.brain.add_volume_labels(aseg="aparc+aseg", labels=[bundle], legend=legend_kwargs)
            self.brain._renderer.plotter.update()

    def update_patient_mri_sensors(self, mri_path):
        pass
        
    #endregion 3D viewer
    
class SEEG_3D_MainWin(QtWidgets.QMainWindow):
    def __init__(self):
        super(SEEG_3D_MainWin, self).__init__()
        self.initUI()

    def initUI(self):
        sig_view = pg.dockarea.DockArea()
        seeg_3d_view = Dock(name='3D viewer', closable=False)
        seeg_3d = SEEG_3D(chs_3d_view=seeg_3d_view, c=None)
        sig_view.addDock(seeg_3d_view, 'left')

        self.setCentralWidget(sig_view)

if __name__ == "__main__":

    app = QApplication([])
    window = SEEG_3D_MainWin()
    window.showMaximized()
    window.setWindowTitle('3d viewer')
    app.exec_()