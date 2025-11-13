# Author: Sunil Mathew
# Date: 20 June 2024
# Description: A custom QComboBox with a QTreeView that contains checkable items in a tree structure.
import sys
from qtpy.QtWidgets import (QApplication, QWidget, QVBoxLayout, QComboBox, 
                            QTreeView, QStyledItemDelegate, QStyleOptionViewItem)
from qtpy.QtGui import QStandardItemModel, QStandardItem, QFontMetrics
from qtpy.QtCore import Qt, QRect

class CheckableTreeComboBox(QComboBox):
    def __init__(self, c, parent=None):
        self.c = c
        self.channel_dict = {}
        super(CheckableTreeComboBox, self).__init__(parent)
        self.view = QTreeView()
        self.view.setHeaderHidden(True)
        self.view.setItemsExpandable(True)
        self.view.setRootIsDecorated(True)
        self.setView(self.view)

        self.model = QStandardItemModel()
        self.setModel(self.model)

        self.delegate = CheckBoxDelegate()
        self.view.setItemDelegate(self.delegate)

        self.view.expanded.connect(self.showPopup)
        self.view.collapsed.connect(self.showPopup)

        self.model.itemChanged.connect(self.handleItemChanged)

    # def clear(self) -> None:
    #     self.model.clear()
    #     # return super().clear()

    def addItems(self, channel_dict, parent=None):
        self.channel_dict = channel_dict
        if not parent:
            parent = self.model.invisibleRootItem()
        for key, value in channel_dict.items():
            check_state = Qt.Checked if value[0] else Qt.Unchecked
            item = QStandardItem(key)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setData(check_state, Qt.CheckStateRole)
            parent.appendRow(item)
            self.handleItemChanged(item, update_selection=False)
            if isinstance(value[1], dict):
                self.addItems(value[1], item)

    def update_channel_dict(self):
        self.channel_dict = {}
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            self.update_channel_dict_recursive(item, self.channel_dict)

    def update_channel_dict_recursive(self, item, parent_dict):
        check_state = item.checkState()
        parent_dict[item.text()] = (check_state, {})
        if item.hasChildren():
            for row in range(item.rowCount()):
                child_item = item.child(row)
                self.update_channel_dict_recursive(child_item, parent_dict[item.text()][1])

    def get_children(self, item):
        return [item.child(i) for i in range(item.rowCount())]

    def handleItemChanged(self, item, update_selection=True):
        try:
            self.model.itemChanged.disconnect(self.handleItemChanged)
        except:
            pass
        
        if item.hasChildren():
            self.checkChildren(item, item.checkState())
        
        self.checkParent(item.parent())

        if hasattr(self, 'channel_dict') and update_selection:
            self.update_channel_dict()
            self.c.sig_view_acq_channels_selection.emit(self.channel_dict)
        
        self.model.itemChanged.connect(self.handleItemChanged)
        
        self.updateText()

        

    def checkChildren(self, parent, check_state):
        for row in range(parent.rowCount()):
            child = parent.child(row)
            child.setCheckState(check_state)
            if child.hasChildren():
                self.checkChildren(child, check_state)

    def checkParent(self, parent):
        if not parent:
            return
        checked_count = 0
        total_count = parent.rowCount()
        for row in range(total_count):
            if parent.child(row).checkState() != Qt.Unchecked:
                checked_count += 1
        if checked_count == 0:
            parent.setCheckState(Qt.Unchecked)
        elif checked_count == total_count:
            parent.setCheckState(Qt.Checked)
        else:
            parent.setCheckState(Qt.PartiallyChecked)
        self.checkParent(parent.parent())

    def adjustPopup(self):
        font_metrics = QFontMetrics(self.view.font())
        checkbox_width = self.style().pixelMetric(self.style().PM_IndicatorWidth)
        max_width = self.calculateMaxWidth(self.model.invisibleRootItem(), font_metrics, self.view.indentation(), checkbox_width)
        
        # Add some margin to the width
        width = max_width + 20  # Extra margin
        
        # Calculate the total height of all visible items
        total_height = 0
        for row in range(self.model.rowCount()):
            index = self.model.index(row, 0)
            total_height += self.calculateItemHeight(index)
        
        # Add some margin to the height
        total_height += 20
        
        # Set a maximum height (e.g., 70% of screen height)
        max_height = int(QApplication.primaryScreen().size().height() * 0.7)
        height = min(total_height, max_height)
        
        self.view.setFixedSize(width, height)

    def setPopupRect(self):
        self.adjustPopup()

    def calculateMaxWidth(self, parent_item, font_metrics, indentation, checkbox_width, level=0):
        max_width = 0
        for row in range(parent_item.rowCount()):
            item = parent_item.child(row)
            text_width = font_metrics.horizontalAdvance(item.text())
            item_width = checkbox_width + text_width + (level * indentation) + 50  # 20 for some padding
            max_width = max(max_width, item_width)
            if item.hasChildren():
                child_max_width = self.calculateMaxWidth(item, font_metrics, indentation, checkbox_width, level + 1)
                max_width = max(max_width, child_max_width)
        return max_width

    def calculateItemHeight(self, index, level=0):
        height = self.view.sizeHintForIndex(index).height()
        if self.view.isExpanded(index):
            for row in range(self.model.rowCount(index)):
                child_index = self.model.index(row, 0, index)
                height += self.calculateItemHeight(child_index, level + 1)
        return height

    def showPopup(self):
        self.setPopupRect()
        super().showPopup()

    def hidePopup(self):
        self.updateText()
        super().hidePopup()

    def updateText(self):
        checked_items = self.getCheckedItems()
        if checked_items:
            self.setEditText(", ".join(checked_items))
        else:
            self.setEditText("")

    def getCheckedItems(self, parent=None):
        checked = []
        if not parent:
            parent = self.model.invisibleRootItem()
        for row in range(parent.rowCount()):
            item = parent.child(row)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
            elif item.checkState() == Qt.PartiallyChecked:
                checked.extend(self.getCheckedItems(item))
        return checked

class CheckBoxDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return None

    def sizeHint(self, option, index):
        size = super(CheckBoxDelegate, self).sizeHint(option, index)
        size.setHeight(size.height() + 4)  # Add a little vertical padding
        return size

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tree Checkboxes ComboBox')
        self.setGeometry(300, 300, 300, 200)

        layout = QVBoxLayout()

        self.combo = CheckableTreeComboBox(self)
        layout.addWidget(self.combo)

        # Define the tree structure
        channel_dict = {
            "PortA": (False, {
                "Bundle1": (False, {
                    "Bundle1_Ch1": (False, {}),
                    "Bundle1_Ch2": (False, {}),
                    "Bundle1_Ch3": (False, {})
                }),
                "Bundle2": (False, {
                    "Bundle2_Ch1": (False, {}),
                    "Bundle2_Ch2": (False, {}),
                    "Bundle2_Ch3": (False, {})
                })
            }),
            "PortB": (False, {
                "Bundle3": (False, {
                    "Bundle3_Ch1": (False, {}),
                    "Bundle3_Ch2": (False, {}),
                    "Bundle3_Ch3": (False, {})
                }),
                "Bundle4": (False, {
                    "Bundle4_Ch1": (False, {}),
                    "Bundle4_Ch2": (False, {}),
                    "Bundle4_Ch3": (False, {})
                })
            })
        }

        self.combo.addItems(channel_dict)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())