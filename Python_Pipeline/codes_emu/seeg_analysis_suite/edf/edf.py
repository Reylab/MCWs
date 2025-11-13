import re
import mne
mne.set_config('MNE_BROWSER_USE_OPENGL', 'true')
mne.viz.set_browser_backend('qt')
import pyqtgraph as pg
# pg.setConfigOptions(useOpenGL=True)
from qtpy.QtWidgets import QApplication

class EDFReader(object):
    def __init__(self, file_path, inc_buns=None, inc_chs=None):
        self.file_path = file_path
        if file_path.endswith('.edf'):
            self.raw = mne.io.read_raw_edf(file_path, preload=True, infer_types=True)
        elif file_path.endswith('.ns*'):
            self.raw = mne.io.read_raw_nsx(file_path, preload=True)
        elif file_path.endswith('.nf*'):
            self.raw = mne.io.read_raw_artemis123
        
        self.sfreq = self.raw.info['sfreq']
        self.n_channels = self.raw.info['nchan']
        self.channel_names = self.raw.info['ch_names']
        self.data = self.raw.get_data()
        self.n_samples = self.data.shape[1]
        self.n_seconds = self.n_samples / self.sfreq
        if inc_chs is not None:
            inc_chs = self.extract_channels(inc_buns, inc_chs)
            self.raw.pick_channels(inc_chs)

    def get_info(self):
        return self.raw.info

    def get_data(self):
        return self.data
    
    def get_scaled_data(self):
        return self.data / 1e-4

    def get_channel_names(self):
        return self.channel_names

    def get_n_channels(self):
        return self.n_channels

    def get_n_samples(self):
        return self.n_samples

    def get_n_seconds(self):
        return self.n_seconds

    def get_sfreq(self):
        return self.sfreq

    def get_raw(self):
        return self.raw

    def get_file_path(self):
        return self.file_path
    
    def extract_channels(self, bundle_names, wanted_chs):
        ch_list = []
        if bundle_names is not None:
            for bundle_name in bundle_names:
                pattern = r'POL {}\d+'.format(bundle_name)
                ch_list.extend([channel for channel in self.channel_names if re.match(pattern, channel)])

        
        for wanted_ch in wanted_chs:
            for ch in self.channel_names:
                if wanted_ch in ch:
                    ch_list.append(ch)
                    break
        return ch_list
    
    def plot(self):
        data = self.get_data()
        # Create a plot window with ability to plot multiple channels
        plot = pg.plot()
        plot.setWindowTitle('EDF Plot')
        plot.setLabel('left', 'Amplitude')
        plot.setLabel('bottom', 'Time', 's')
        plot.showGrid(x=True, y=True)
        plot.enableAutoRange()
        # plot.setYRange(-10, 4*10)
        # plot.setXRange(0, self.n_seconds*self.sfreq)
        ch_ticks = [(i*0.005, self.channel_names[i]) for i in range(50)]
        for i in range(50):
            curve = pg.PlotCurveItem(pen=({'color': 'b', 'width': .1}), skipFiniteCheck=True)
            curve.setData(data[i, :])
            curve.setPos(0, i*0.005)
            plot.addItem(curve)
        # Set channel label

        plot.getPlotItem().getAxis('left').setTicks([ch_ticks])


    

if __name__ == '__main__':
    app = QApplication([])
    # edf_path = 'E:\Epilepsy\MCW-FH_010\EMU-027-subj-MCW-FH_010_task-gaps\EMU-027_subj-MCW-FH_010_task-gaps_run-05_RIP.nf3'
    edf_path = 'E:\Epilepsy\MCW-FH_010\Pt10 EDFs\Pt10-Seizure1-EDF.edf'
    # edf_path = 'C:\\Users\\sunil\\Trellis\\dataFiles\\test_4.edf'
    # inc_buns = ['LW', 'LO', 'LA', 'LB', 'LC', 'LE', 'LF']
    pt10_sz1 = ['LW5', 'LW6', 'LW7', 'LW8', 'LW9', 'LW10', 'LW11', 
                      'LO1', 'L02', 'LO6', 'LO7', 'LO8', 'LO9', 'LO10', 'LO11',
                      'LA1', 'LA2', 'LA3', 'LA4', 'LA5', 'LA6', 'LA7', 'LA8',
                      'LB1', 'LB2', 'LB3', 'LB4', 'LB5', 'LB6', 'LB7', 'LB8',
                      'LC7', 'LC8', 'LC9',
                      'LE2', 'LE3', 'LE4', 'LE5', 'LE6', 'LE7', 
                      'LF1', 'LF2', 'LF3', 'LF4', 'LF5', 'LF6', 'LF7']
    pt15_sz13 = ['RA1', 'RA2', 'RA3', 'RB1', 'RB2', 'RB3', 'RC1', 'RC2', 'RC3', 'RC4', 'RD1', 'RD2']
    edf = EDFReader(file_path=edf_path, inc_chs=pt15_sz13)
    print(edf.get_info())
    # print(edf.get_data())
    # print(edf.get_channel_names())
    # print(edf.get_n_channels())
    # print(edf.get_n_samples())
    # print(edf.get_n_seconds())
    # print(edf.get_sfreq())
    # print(edf.get_raw())
    # print(edf.get_file_path())
    
    edf.raw.plot(
        # start=600,
        # duration=10,
        # scalings=dict(eeg=1e-3, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1),
        use_opengl=True,
    )

    # edf.plot()

    app.exec_()