import sys
import time
import PyQt5
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot,QObject,QThreadPool,QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QProgressBar,QMainWindow
import traceback
from core.graphing import graphing
class WorkerSignals(QObject):
    def __init__(self,*args,**kwargs):
        finished = pyqtSignal()
        error = pyqtSignal(tuple)
        result = pyqtSignal(object)
        progress = pyqtSignal(int)

class MainWindow(QMainWindow):
    def __init__(self,results,*args,**kwargs):
        super(MainWindow,self).__init__()
        self.results =results
        self.counter = 0 
        
        self.s = kwargs['s']
        if self.s == 'plot_bundles':
            self.par = kwargs['par']
            self.nsx_file = kwargs['nsx_file']
            self.notchfilter = kwargs['notchfilter']
        layout = QVBoxLayout()
        self.l = QLabel("start")
        b = QPushButton("Save Images")
        b.pressed.connect(self.thread_start)
        layout.addWidget(b)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.threadpool=QThreadPool()
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def progress_fn(self,n):
        print("%d%% done" %n)

    def execute_this_fn(self,progress_callback):
        print(f"progress_callback = {progress_callback}")
        for n in range(0,5):
            time.sleep(1)
            print(f"n={n}")
            progress_callback(int(n*100/4))
        return "Done"
    
    def print_output(self,s):
        print(s)

    def thread_complete(self):
        print("Thread Complete")

    def thread_start(self):
        if self.s == 'plot_bundles':
            for result in self.results:
                worker = WorkerObject(self.function,s=self.s,par = self.par,nsx_file = self.nsx_file,notchfilter = self.notchfilter)
                self.threadpool.start(worker)
        else:
            for result in self.results:
                graph = graphing(result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9]
                                ,result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17],result[18])
                worker = WorkerObject(graph.plot_notches,s=self.s)
                self.threadpool.start(worker)

    def recurring_timer(self):
        self.counter +=1
        self.l.setText('Counter:%d'%self.counter)
        

class WorkerObject(QtCore.QRunnable):
    def __init__(self,function,parent=None,**kwargs):
        QtCore.QRunnable.__init__(self)
        self.kwargs = kwargs
        self.function = function
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.s = self.kwargs['s']
            if self.s == 'plot_bundles':
                self.par = self.kwargs['par']
                self.nsx_file = self.kwargs['nsx_file']
                self.notchfilter = self.kwargs['notchfilter']
                self.function(self.par,self.nsx_file,self.notchfilter)
            else:
                result=self.function()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype,value,traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class ProgressCallback(QObject):
    progressChanged = pyqtSignal(int)

    def __call__(self,value):
        self.progressChanged.emit(value)


#class initObject(QtCore.QObject):
#    def __init__(self):
#        QtCore.QObject.__init__(self)
#    @pyqtSlot
#    def afunc(self):
#        self.emit(SIGNAL(""))
#class endObject(Qtcore.QObject):
#    def __init__(self):
#        QtCore.QObject.__init__(self)
#    @pyqtSlot
#    def afunc(self):
#        self.emit()
#

        
        

            

            

        

