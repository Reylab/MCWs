
from qtpy.QtWidgets import * 
from qtpy.QtGui import QStandardItemModel
from qtpy.QtCore import Qt

class CheckableComboBox(QComboBox):
    
    # constructor
    def __init__(self, parent=None, width=140):
        super(CheckableComboBox, self).__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.count = 0
        self.setMinimumWidth(width) # pixels

    # action called when item get checked
    def do_action(self):

        print("Checked number : " +str(self.count))

    # when any item get pressed
    def handleItemPressed(self, index):

        # getting the item
        item = self.model().itemFromIndex(index)

        # checking if item is checked
        if item.checkState() == Qt.Checked:

            # making it unchecked
            item.setCheckState(Qt.Unchecked)

        # if not checked
        else:
            # making the item checked
            item.setCheckState(Qt.Checked)

            self.count += 1

            # call the action
            self.do_action()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    combo = CheckableComboBox(window)
    combo.addItem("Item 1")
    combo.addItem("Item 2")
    combo.addItem("Item 3")
    combo.addItem("Item 4")
    layout.addWidget(combo)
    window.show()
    sys.exit(app.exec_())