from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from utils import load_pixmap_to_label
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QFileDialog
from PyQt5.QtCore import QTimer


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('ui.ui', self)

        # input images
        self.input_img1.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img1)
        self.input_img2.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img2)

        # output images
            # output_img1_GV ==> graphics view
            # output_img2_GV ==> graphics view

        # slider
        self.threshold_slider.valueChanged.connect(
            lambda: self.threshold_label.setText(f"{self.threshold_slider.value()}"))

        # button
            # self.button

        #mode
            # self.mode_combo        

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
