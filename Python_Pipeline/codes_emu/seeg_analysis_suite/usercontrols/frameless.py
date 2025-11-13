from qtpy.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout
from qtpy.QtCore import Qt, QTimer, QThread

from pyvistaqt import QtInteractor, MainWindow
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
import numpy as np
import sys
import time
import threading

class FramelessWindow(QWidget):
    def __init__(self, parent=None, c=None):
        super(FramelessWindow, self).__init__(parent)
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black;")

        self.c = c

        self.graphWidget = pg.GraphicsLayoutWidget()
        self.graphWidget.setBackground(None)
        self.viewBox = self.graphWidget.addViewBox()
        self.viewBox.setAspectLocked(True)
        self.viewBox.setRange(xRange=[-100, 100], yRange=[-100, 100])
        # Disable panning and zooming
        self.viewBox.disableAutoRange()
        self.viewBox.setMouseEnabled(x=False, y=False)

        self.imageItem = self.create_image_item()
        self.viewBox.addItem(self.imageItem)

        # Photo detect square
        self.photo_detect_square = self.create_photo_detect_square(x=-1920/2, y=1080/2-160, w=160, h=160)
        self.viewBox.addItem(self.photo_detect_square)
        self.photo_sq_timespan = 100  # 100ms
        self.photo_sq_timer = QTimer()
        self.photo_sq_timer.timeout.connect(self.hide_photo_detect_square)

        self.bar_h = 5

        # yellow bar at top
        self.topBar = self.create_bar(x=-80, y=-120-self.bar_h)
        self.bottomBar = self.create_bar(x=-80, y=120)
        
        self.viewBox.addItem(self.topBar)
        self.viewBox.addItem(self.bottomBar)

        self.textItem = self.create_text_item()
        self.viewBox.addItem(self.textItem)

        # randomly set the color of the top bar to red or yellow
        if np.random.rand() > 0.5:
            top_color = [1, 0, 0]
            bottom_color = [1, 1, 0]
        else:
            top_color = [1, 1, 0]
            bottom_color = [1, 0, 0]

        self.change_bar_color(self.topBar, top_color)
        self.change_bar_color(self.bottomBar, bottom_color)

        layout = QVBoxLayout()
        layout.addWidget(self.graphWidget)
        self.setLayout(layout)

    def create_image_item(self, x=-80, y=-80, w=160, h=160):
        imageItem = pg.ImageItem(np.random.rand(h, w))
        imageItem.setRect(pg.QtCore.QRectF(x, y, w, h))
        imageItem.setPxMode(True)
        
        return imageItem
    
    def create_photo_detect_square(self, x, y, w, h, color=[255, 255, 255]):
        imageItem = pg.ImageItem(np.ones((h, w, 3))*np.array(color))
        imageItem.setRect(pg.QtCore.QRectF(x, y, w, h))
        imageItem.setPxMode(True)

        return imageItem
    
    def create_text_item(self):
        textItem = pg.TextItem('', anchor=(0.5, 0))
        textItem.setFont(pg.QtGui.QFont('Arial', 20))
        # textItem.setPos(0, 0)
        # textItem.setHtml('<div style="text-align: center"><span style="color: #FFF; font-size: 20pt;">Hello World</span></div>')
        return textItem

    def show_image(self, image):
        self.imageItem.setImage(image)
        # self.imageItem.setImage(image.transpose(1, 0, 2))

    def change_bar_color(self, bar, color):
        bar.setImage(np.ones((5, 160, 3))*np.array(color))

    def create_bar(self, x=-80, y=100, h=5, w=160, color=[1, 1, 0]):
        bar = pg.ImageItem(np.ones((h, w, 3))*np.array(color))
        bar.setRect(pg.QtCore.QRectF(x, y, w, h))
        bar.setPxMode(True)
        # bar.setZValue(10)
        return bar

    def show_text(self, txt):  
        # Clear the image item and bars
        self.imageItem.setImage(np.zeros((160, 160)))
        self.change_bar_color(self.topBar, [0, 0, 0])
        self.change_bar_color(self.bottomBar, [0, 0, 0])
        self.textItem.setHtml('<div style="text-align: center"><span style="color: #FFF; font-size: 20pt;">{}</span></div>'.format(txt))

    def show_photo_detect_square(self):
        self.photo_detect_square.setImage(np.ones((160, 160, 3))*np.array([255, 255, 255]))
        
        self.photo_sq_timer.start(self.photo_sq_timespan)

    def hide_photo_detect_square(self):
        self.photo_detect_square.setImage(np.zeros((160, 160, 3)))
        self.photo_sq_timer.stop()

    def closeEvent(self, event):
        # release second monitor
        # self.destroy()
        self.c.task_win.emit(True)

class RSVPWorker(QThread):
    def __init__(self, c, config, monitor):
        super(RSVPWorker, self).__init__()
        self.c = c
        self.config = config
        self.monitor = monitor

    def run(self):
        self.show_exp_win()
        self.show_msg(
            self.config["subscr_msgs"]["begin"][self.language]
        )

    def show_msg(self, msg):
        self.framelessWindow.show_text(msg)

    def close_exp_win(self):
        if hasattr(self, "framelessWindow"):
            self.framelessWindow.close()
            self.framelessWindow = None

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

    def show_stim_img(self, img):
        """
        This method is called by a timer to show images in a sequence.

        Args:
            img (np.ndarray): The image to show.
        """

        if hasattr(self, "framelessWindow"):
            self.framelessWindow.textItem.setHtml("")  # Clear text item
            self.framelessWindow.show_photo_detect_square()
            self.framelessWindow.show_image(img)

            if np.random.rand() > 0.8:
                self.flip_lines_color()

class MainApp(MainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setGeometry(0, 0, 400, 400)

        self.button = QPushButton('Show Frameless Window', self)
        self.button.setFixedSize(200, 30)
        self.button.clicked.connect(self.showFramelessWindowOnSecondMonitor)

        self.closeButton = QPushButton('Close Frameless Window', self)
        self.closeButton.setFixedSize(200, 30)
        self.closeButton.clicked.connect(self.closeFramelessWindow)
        self.closeButton.move(0, 30)  # move the close button below the show button

        self.btnShowTxt = QPushButton('Show Text', self)
        self.btnShowTxt.setFixedSize(200, 30)
        self.btnShowTxt.clicked.connect(self.show_text)
        self.btnShowTxt.move(0, 60)

        self.btnShowImg = QPushButton('Show Image', self)
        self.btnShowImg.setFixedSize(200, 30)
        self.btnShowImg.clicked.connect(self.show_image)
        self.btnShowImg.move(0, 90)

        self.btnStartTask = QPushButton('Start Task', self)
        self.btnStartTask.clicked.connect(self.start_task_thread)
        self.btnStartTask.move(0, 120)

        self.b_stop = False

    def start_task_thread(self):
        if hasattr(self, 'task_th') and self.task_th.is_alive():
            self.task_th.join()
        self.b_stop = False
        self.showFramelessWindowOnSecondMonitor()
        self.task_th = threading.Thread(target=self.start_task)
        self.task_th.start()
    
    def start_task(self):
        self.btnStartTask.setText('Stop Task')  
        self.btnStartTask.clicked.disconnect(self.start_task_thread)
        self.btnStartTask.clicked.connect(self.stop_task)
        # self.showFramelessWindowOnSecondMonitor()
        self.show_text()
        while not self.b_stop:
            self.show_image()
            QApplication.processEvents()
            time.sleep(0.5)

    def stop_task(self):
        self.b_stop = True
        self.btnStartTask.setText('Start Task')  
        self.btnStartTask.clicked.disconnect(self.stop_task)
        self.btnStartTask.clicked.connect(self.start_task_thread)
        self.closeFramelessWindow()

    def show_image(self):
        if hasattr(self, 'framelessWindow'):
            # Clear text item
            self.framelessWindow.textItem.setHtml('')
            self.framelessWindow.show_image(np.random.rand(160, 160))
            if np.random.rand() > 0.5:
                self.framelessWindow.change_bar_color(self.framelessWindow.topBar, [1, 1, 0])
                self.framelessWindow.change_bar_color(self.framelessWindow.bottomBar, [1, 0, 0])
            else:
                self.framelessWindow.change_bar_color(self.framelessWindow.topBar, [1, 0, 0])
                self.framelessWindow.change_bar_color(self.framelessWindow.bottomBar, [1, 1, 0])

            self.framelessWindow.show_photo_detect_square()

    def show_text(self):
        if hasattr(self, 'framelessWindow'):
            self.framelessWindow.show_text('Ready to start?')

    def showFramelessWindowOnSecondMonitor(self):
        desktop = QApplication.desktop()
        
        if not hasattr(self, 'framelessWindow') or self.framelessWindow is None:
            self.framelessWindow = FramelessWindow()
        if desktop.screenCount() > 1:   
            rect = desktop.screenGeometry(2)  # get the geometry of the second monitor         
            self.framelessWindow.move(rect.left(), rect.top())
            # self.framelessWindow.showFullScreen()
            self.framelessWindow.show()
        else:
            self.framelessWindow.show()

    def closeFramelessWindow(self):
        if hasattr(self, 'framelessWindow'):
            self.framelessWindow.close()
            self.framelessWindow = None

    # Close the frameless window if the main window is closed
    def closeEvent(self, event):
        self.closeFramelessWindow()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainApp = MainApp()
    mainApp.show()
    sys.exit(app.exec_())