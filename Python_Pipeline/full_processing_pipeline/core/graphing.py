import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from core.ui import MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import pyqtgraph as pg
import time

class graphing():
    def __init__(self,f,pxx_db,pxx_thr_db,used_notches,pxx_filtered_db,f_filtered,pxx_nofilter_db,
                 x_raw,x_filtered,sr,samples_timeplot,title,outputpath,extra_title,log_info,thr,conf,unit):
        self.f = f
        self.pxx_db = pxx_db
        self.pxx_thr_db = pxx_thr_db
        self.used_notches = used_notches
        self.pxx_filtered_db = pxx_filtered_db
        self.f_filtered = f_filtered
        self.pxx_nofilter_db = pxx_nofilter_db
        self.x_raw = x_raw
        self.x_filtered = x_filtered
        self.sr = sr
        self.samples_timeplot = samples_timeplot
        self.title = title
        self.outputpath = outputpath
        self.extra_title = extra_title
        self.log_info = log_info
        self.thr = thr
        self.conf = conf
        self.unit = unit
    
    def plot_notches(self):
        self.main_window = MainWindow(self.conf,self.unit)
        
        #plt = pg.plot(self.f[0:300],self.pxx_db[0:300],pen=1)
        #viewbox = plt.getViewBox()
        #viewbox.setBorder(pg.mkPen())
        #plt.setDefaultPadding(padding=0)
        #plt.plot(self.f[0:300],self.pxx_thr_db[0:300],pen=2)
        #for notch in self.used_notches:
        #    plt.addLine(x=notch)
        #plt.plot(self.f_filtered.ravel()[0:300],self.pxx_filtered_db.ravel()[0:300],pen=3)
        #plt.plot(self.f[0:300],self.pxx_nofilter_db.ravel()[0:300],pen=4)
        #plt.setXRange(0,300)
        #plt.plotItem.setTitle(title=self.title)
        #plt.layout

        #plot 300hz graph
        #self.main_window.plot_widget_300hz.plot(self.f[0:np.where(self.f>=300)[0][0]],self.pxx_db[0:np.where(self.f>=300)[0][0]],pen=1)

        self.main_window.Title.setText(self.title)
        self.main_window.subtitle.setText(self.extra_title)

        self.main_window.plot_widget_300hz.plot(self.f[0:np.where(self.f>=self.conf['freqs_lim_zoom'][1])[0][0]],self.pxx_thr_db[0:np.where(self.f>=self.conf['freqs_lim_zoom'][1])[0][0]],pen=pg.mkPen('m'))
        self.main_window.plot_widget_300hz.plot(self.f_filtered.ravel()[0:np.where(self.f>=self.conf['freqs_lim_zoom'][1])[0][0]],self.pxx_filtered_db.ravel()[0:np.where(self.f>=self.conf['freqs_lim_zoom'][1])[0][0]],pen=pg.mkPen('r'))
        self.main_window.plot_widget_300hz.plot(self.f[0:np.where(self.f>=self.conf['freqs_lim_zoom'][1])[0][0]],self.pxx_nofilter_db.ravel()[0:np.where(self.f>=self.conf['freqs_lim_zoom'][1])[0][0]],pen=pg.mkPen('b'))
        self.main_window.plot_widget_300hz.setYRange(-30,50)
        dottedline = pg.mkPen('k',width = .5,style = Qt.DotLine)
        if self.used_notches.size:
            for notch in self.used_notches:
                self.main_window.plot_widget_300hz.addLine(x=notch,pen=dottedline)
                self.main_window.plot_widget_3000hz.addLine(x=notch,pen=dottedline)


        #plot 3000hz graph
        self.main_window.plot_widget_3000hz.plot(self.f[0:np.where(self.f>=self.conf['freqs_lim'][1])[0][0]],self.pxx_thr_db[0:np.where(self.f>=self.conf['freqs_lim'][1])[0][0]],pen=pg.mkPen('m'))
        self.main_window.plot_widget_3000hz.plot(self.f_filtered.ravel()[0:np.where(self.f>=self.conf['freqs_lim'][1])[0][0]],self.pxx_filtered_db.ravel()[0:np.where(self.f>=self.conf['freqs_lim'][1])[0][0]],pen=pg.mkPen('r'))
        self.main_window.plot_widget_3000hz.plot(self.f[0:np.where(self.f>=self.conf['freqs_lim'][1])[0][0]],self.pxx_nofilter_db.ravel()[0:np.where(self.f>=self.conf['freqs_lim'][1])[0][0]],pen=pg.mkPen('b'))
        self.main_window.plot_widget_3000hz.setYRange(-50,50)
        dashedline = pg.mkPen('g',width = .5, style = Qt.DashLine)        
        self.main_window.plot_widget_3000hz.addLine(x=self.conf['freqs_fit'][1],pen=dashedline)
        self.main_window.plot_widget_3000hz.addLine(x=self.conf['freqs_fit'][2],pen=dashedline)

        #plot raw
        time = np.array(range(0,self.samples_timeplot+1))/self.sr
        self.main_window.plot_widget_raw.plot(time,self.x_raw[-self.samples_timeplot-1:len(self.x_raw)],pen=pg.mkPen('b'))

        #plot filtered
        self.main_window.plot_widget_filtered.plot(time,self.x_filtered.ravel()[-self.samples_timeplot-1:len(self.x_raw)],pen=pg.mkPen('r'))
        if self.thr != None:
            self.main_window.plot_widget_filtered.addLine(y=self.thr,pen=pg.mkPen('k'))
            self.main_window.plot_widget_filtered.addLine(y=-self.thr,pen=pg.mkPen('k'))

        #log text info
        self.main_window.text_label.setText(self.log_info)

        #export
        pixmap = QPixmap(self.main_window.size())
        self.main_window.render(pixmap)
        pixmap.save(self.outputpath+'.png')
        self.main_window.destroy()

