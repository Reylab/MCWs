import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self,conf,unit):
        super().__init__()
        self.conf = conf
        loadUi('layout.ui',self)
        self.plot_widget_3000hz = pg.PlotWidget()
        self.plot_widget_300hz = pg.PlotWidget()
        self.plot_widget_raw = pg.PlotWidget()
        self.plot_widget_filtered = pg.PlotWidget()
        self.plot_widget_300hz.disableAutoRange()
        self.plot_widget_3000hz.disableAutoRange()
        self.plot_widget_300hz.setDefaultPadding(padding= 0)
        self.plot_widget_3000hz.setDefaultPadding(padding= 0)
        self.plot_widget_raw.setDefaultPadding(padding= 0)
        self.plot_widget_filtered.setDefaultPadding(padding= 0)
        self.plot_widget_300hz.setBackground('w')
        self.plot_widget_3000hz.setBackground('w')
        self.plot_widget_raw.setBackground('w')
        self.plot_widget_filtered.setBackground('w')
        self.vb_300hz = self.plot_widget_300hz.getViewBox()
        self.vb_3000hz = self.plot_widget_3000hz.getViewBox()
        self.vb_raw = self.plot_widget_raw.getViewBox()
        self.vb_filtered = self.plot_widget_filtered.getViewBox()
        self.plot_widget_3000hz.setLabel(axis='bottom',text = 'Frequency (Hz)')
        self.plot_widget_300hz.setLabel(axis='bottom',text = 'Frequency (Hz)')
        self.plot_widget_raw.setLabel(axis='bottom',text = 'Time (sec)')
        self.plot_widget_filtered.setLabel(axis='bottom',text = 'Time (sec)')
        self.plot_widget_300hz.setLabel(axis='left',text = 'Power Spectrum (dB/Hz)')
        self.plot_widget_3000hz.setLabel(axis='left',text = 'Power Spectrum (dB/Hz)')
        self.plot_widget_raw.setLabel(axis='left',text = f'Raw ({unit})')
        self.plot_widget_filtered.setLabel(axis='left',text = f'Filtered ({unit})')
        self.plot_widget_300hz.setXRange(self.conf['freqs_lim_zoom'][0],self.conf['freqs_lim_zoom'][1])
        self.plot_widget_3000hz.setXRange(self.conf['freqs_lim'][0],self.conf['freqs_lim'][1])
        self.plot_widget_filtered.setXRange(0,60)
        self.plot_widget_raw.setXRange(0,60)
        self.horizontalLayout_3.replaceWidget(self.plot300hz,self.plot_widget_300hz)
        self.horizontalLayout_3.replaceWidget(self.plot3000hz,self.plot_widget_3000hz)
        self.verticalLayout.replaceWidget(self.plotraw,self.plot_widget_raw)
        self.verticalLayout.replaceWidget(self.plotfiltered,self.plot_widget_filtered)
        self.text_label = pg.LabelItem("placeholder")
        self.text_label.setParentItem(self.plot_widget_3000hz.graphicsItem())
        self.text_label.anchor(itemPos=(0.5,0.1),parentPos=(0.5,0.1))

        #self.app = QApplication(sys.argv)
        #self.main_window = MainWindow()
        #self.main_window.show()
#if __name__ == '__main__':
#    main_window = MainWindow()
#    app.exec_()


