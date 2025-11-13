import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
import numpy as np
import configparser
import pathlib
from os import path
import sys
sys.path.append(path.join(pathlib.Path(__file__).parent.resolve(),'..','useful_functions'))
from read_nsx_file import read_nsx_file
from scipy.signal import filtfilt
from scipy.signal import ellip


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('nsx_cmp.ui', self)
        self.plot_info = pg.LabelItem(justify='right')
        self.pg_area.addItem(self.plot_info)
        self.plotwidget = self.pg_area.addPlot(row=1, col=0)
        self.proxy = pg.SignalProxy(self.plotwidget.scene().sigMouseMoved, rateLimit=5, slot=self.mouseMoved)
        self.datapaths = [self.data1path, self.data2path]
        self.seachdata1.clicked.connect(lambda: self.openfile(0))
        self.seachdata2.clicked.connect(lambda: self.openfile(1))
        self.export_bt.clicked.connect(self.save_sreenshot)
       

        self.scale1.editingFinished.connect(self.update_plot)
        self.scale2.editingFinished.connect(self.update_plot)

        self.forder.editingFinished.connect(lambda: self._load_dat(reset_lims = False))
        self.fmin.editingFinished.connect(lambda: self._load_dat(reset_lims = False))
        self.fmax.editingFinished.connect(lambda: self._load_dat(reset_lims = False))

        
        self.rm_s1_sl.sliderReleased.connect(lambda: self.slider2spin(self.rm_s1_sl, self.rm_s1))
        self.rm_s2_sl.sliderReleased.connect(lambda: self.slider2spin(self.rm_s2_sl, self.rm_s2))
        self.rm_e1_sl.sliderReleased.connect(lambda: self.slider2spin(self.rm_e1_sl, self.rm_e1))
        self.rm_e2_sl.sliderReleased.connect(lambda: self.slider2spin(self.rm_e2_sl, self.rm_e2))


        self.rm_s1.editingFinished.connect(self.update_plot)
        self.rm_s2.editingFinished.connect(self.update_plot)
        self.rm_e1.editingFinished.connect(self.update_plot)
        self.rm_e2.editingFinished.connect(self.update_plot)
        self.dc1.editingFinished.connect(self.update_plot)
        self.dc2.editingFinished.connect(self.update_plot)

        self.srate.valueChanged.connect(self._load_dat)

        self.load_button.clicked.connect(self.load_config)
        self.save_button.clicked.connect(self.save_config)
        self.data = [[],[]]
        self.pg_area.setBackground('w')
        self.plotwidget.showGrid(x=True, y=True)

    def config_gui(self, reset_lims = [0,1]):
        if len(self.data[0])==0 and len(self.data[1])==0:
            self.plotwidget.plot(clear=True)
            return

        if len(self.data[0])>0:
            self.rm_s1_sl.setMaximum(len(self.data[0]))
            self.rm_e1_sl.setMaximum(len(self.data[0]))
            self.rm_e1.setMaximum(len(self.data[0]))
            self.rm_s1.setMaximum(len(self.data[0]))
            if 0 in reset_lims:
                self.rm_e1_sl.setValue(len(self.data[0]))
                self.rm_s1_sl.setValue(1)
                self.rm_e1.setValue(len(self.data[0]))
                self.rm_s1.setValue(1)
        
        if len(self.data[1])>0:
            self.rm_s2_sl.setMaximum(len(self.data[1]))
            self.rm_e2_sl.setMaximum(len(self.data[1]))
            self.rm_e2.setMaximum(len(self.data[1]))
            self.rm_s2.setMaximum(len(self.data[1]))
            if 1 in reset_lims:
                self.rm_e2_sl.setValue(len(self.data[1]))
                self.rm_s2_sl.setValue(1)
                self.rm_e2.setValue(len(self.data[1]))
                self.rm_s2.setValue(1)

    def openfile(self, n):
        file = QtWidgets.QFileDialog.getOpenFileName(None, "Select data "+ str(n+1),
                                               '', "NCx Files (*.NC*)")[0]
        if len(file)==0:
            return
        self.datapaths[n].setText(file)
        self._load_dat(load_only=[n], reset_lims = [n])

    def _load_dat(self, unused=None ,load_only=[0, 1], reset_lims = [0,1]):
        for n in load_only:
            if len(self.datapaths[n].text())==0:
                continue
            [self.data[n], metadata] = read_nsx_file(self.datapaths[n].text())
            sr = metadata['sr']
            filter_order = self.forder.value()
            user_srate = self.srate.value()
            fmax = self.fmax.value()
            fmin = self.fmin.value()
            if fmax < (metadata['sr']/2) and fmin>0:
                [b_d, a_d] = ellip(filter_order, 0.1, 40, np.array([fmin, fmax]) * 2 / sr, btype='bandpass')
            elif fmax < metadata['sr']/2:
                [b_d, a_d] = ellip(filter_order, 0.1, 40,  fmax * 2 / sr, btype='lowpass')
            elif fmin>0:
                [b_d, a_d] = ellip(filter_order, 0.1, 40,  fmin * 2 / sr, btype='highpass') 
            else:
                b_d = None
            
            if b_d is not None:
                self.data[n] = filtfilt(b_d, a_d, self.data[n])
            if  user_srate > metadata['sr']:
                assert 'choose a smaller sampling rate.'
            if user_srate < metadata['sr']:
                subsampling = int( metadata['sr']/user_srate)
                self.data[n] = self.data[n][0:-1:subsampling]
        
            self.config_gui(reset_lims=[])
        self.update_plot()

    def update_plot(self):
        self.plotwidget.plot(clear=True)
        self.plotwidget.setLabels(bottom='time (s)', left = 'uV')
        user_srate = self.srate.value()
        if len(self.data[0])>0:
            self.plotwidget.plot(x=np.arange(1,self.rm_e1.value()-self.rm_s1.value()+2)/user_srate,y=self.scale1.value()* 
                (self.data[0][self.rm_s1.value()-1:self.rm_e1.value()]+
                    self.dc1.value())).setPen(color=(0,170,255), width=1)
        if len(self.data[1])>0:
            self.plotwidget.plot(x=np.arange(1,self.rm_e2.value()-self.rm_s2.value()+2)/user_srate,y=self.scale2.value()* 
                (self.data[1][self.rm_s2.value()-1:self.rm_e2.value()]+
                    self.dc2.value())).setPen(color=(255,170,0), width=1)

    def save_config(self):
        file = QtWidgets.QFileDialog.getSaveFileName(None, "Create config file",
                                               '', "config files (*.ini)")[0]
        if len(file)==0:
            return
        config = configparser.ConfigParser()
        config['CONFIG'] = {
            'data1path':self.data1path.text(),
            'data2path':self.data2path.text(),
            'srate': str(self.srate.value()),
            'scale1': str(self.scale1.value()),
            'scale2': str(self.scale2.value()),
            'dc1': str(self.dc1.value()),
            'dc2': str(self.dc2.value()),
            'rm_s1': str(self.rm_s1.value()),
            'rm_s2': str(self.rm_s2.value()),
            'rm_e1': str(self.rm_e1.value()),
            'rm_e2': str(self.rm_e2.value()),
            'filter_order': str(self.forder.value()),
            'fmax': str(self.fmax.value()),
            'fmin': str(self.fmin.value())}


        with open(file, 'w') as configfile:
            config.write(configfile)


    def load_config(self):
        file = QtWidgets.QFileDialog.getOpenFileName(None, "Select config file",
                                               '', "config files (*.ini)")[0]
        if len(file)==0:
            return
        config = configparser.ConfigParser()
        config.read(file)
       
        self.data1path.setText(config['CONFIG']['data1path'])
        self.data2path.setText(config['CONFIG']['data2path'])
        self.srate.setValue(int(config['CONFIG']['srate']))
        

        self.scale1.setValue(config['CONFIG'].getfloat('scale1'))
        self.scale2.setValue(config['CONFIG'].getfloat('scale2'))
        self.dc1.setValue(config['CONFIG'].getfloat('dc1'))
        self.dc2.setValue(config['CONFIG'].getfloat('dc2'))
        
        self.rm_s1.setValue(int(config['CONFIG']['rm_s1']))
        self.rm_s2.setValue(int(config['CONFIG']['rm_s2']))
        self.rm_e1.setValue(int(config['CONFIG']['rm_e1']))
        self.rm_e2.setValue(int(config['CONFIG']['rm_e2']))

        self.forder.setValue(int(config['CONFIG']['filter_order']))
        self.fmax.setValue(config['CONFIG'].getfloat('fmax'))
        self.fmin.setValue(config['CONFIG'].getfloat('fmin'))

        self._load_dat(reset_lims = False)

        self.rm_s1_sl.setValue(int(config['CONFIG']['rm_s1']))
        self.rm_s2_sl.setValue(int(config['CONFIG']['rm_s2']))
        self.rm_e1_sl.setValue(int(config['CONFIG']['rm_e1']))
        self.rm_e2_sl.setValue(int(config['CONFIG']['rm_e2']))

    def save_sreenshot(self):
        file = QtWidgets.QFileDialog.getSaveFileName(None, "Create png file",
                                               '', "png files (*.png)")[0]
        if len(file)==0:
            return
        #screen = QtWidgets.QApplication.primaryScreen()
        #screenshot = screen.grabWindow(0)
        #screenshot.save(file, 'png')
        self.grab().save(file)

    def slider2spin(self, slider, spin):
        spin.setValue(slider.value())
        self.update_plot()

    def mouseMoved(self,evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        if self.plotwidget.sceneBoundingRect().contains(pos):
            mousePoint = self.plotwidget.vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()
            text = "<span style='font-size: 12pt' style='color: black'>t=%0.2fs , samples=%d, y=%0.2fuV" % (x,int(x*self.srate.value()),y)
            self.plot_info.setText(text)



def main():
    app = QtWidgets.QApplication([])
    main = MainWindow()
    main.show()
    app.exec_()


if __name__ == '__main__':
    main()



#         % Button pushed function: saveconfig
#         function saveconfigButtonPushed(app, event)
#             [file,path] = uiputfile('*.mat');
#             allconfig = struct;
            
            
#             allconfig.data1path = app.data1path.Value;
#             allconfig.data2path = app.data2path.Value;
            
#             allconfig.srate = app.srate.Value;
            
#             allconfig.dc1 = app.dc1.Value;
#             allconfig.dc2 = app.dc2.Value;
            
#             allconfig.scale1 = app.scale1.Value;
#             allconfig.scale2 = app.scale2.Value;
            
            
#             allconfig.rm_s1 = app.rm_s1.Value;
#             allconfig.rm_s2 = app.rm_s2.Value;
            
            
#             allconfig.rm_e1 = app.rm_e1.Value;
#             allconfig.rm_e2 = app.rm_e2.Value;            
#             save([path, file], 'allconfig');
#         end

#         % Button pushed function: loadconfigButton
#         function loadconfigButtonPushed(app, event)
#             %lo inverso a save data
#             [file,path] = uigetfile('*.mat');
#             load([path, file], 'allconfig');
            
#             app.data1path.Value = allconfig.data1path;
#             app.data2path.Value = allconfig.data2path;
            
#             app.srate.Value = allconfig.srate;
            
#             app.dc1.Value = allconfig.dc1;
#             app.dc2.Value = allconfig.dc2;
            
#             app.scale1.Value = allconfig.scale1;
#             app.scale2.Value = allconfig.scale2;
            
            
#             app.rm_s1.Value = allconfig.rm_s1;
#             app.rm_s2.Value=allconfig.rm_s2;
            
#             app.rm_e1.Value = allconfig.em_e1;
#             app.rm_e2.Value=allconfig.rm_e2;
            
#         end