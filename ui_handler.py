from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step
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

        # mode
        # self.mode_combo

        # Harris Tab_________________________________________________________________________
        # input images Harris
        self.Harris_input_img.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.Harris_input_img)
        # output images Harris

        # Harris_output_img1_GV ==> graphics view
        # Harris_output_img2_GV ==> graphics view

        # Harris slider
        self.k_slider.valueChanged.connect(
            lambda: self.K_label.setText(f"{self.k_slider.value()/1000}"))
        # range for slider from 40 to 60 so real value is /1000

        self.Harris_threshold_slider.valueChanged.connect(
            lambda: self.Harris_threshold_label.setText(f"{self.Harris_threshold_slider.value()}"))
        self.Window_size_slider.valueChanged.connect(
            lambda: self.Window_size_label.setText(f"{self.Window_size_slider.value()}"))
        self.Window_size_slider.valueChanged.connect(
            lambda: enforce_slider_step(self.Window_size_slider, 2, 3))

        # time labels

        # Harris_input_label
        # Harris_outpu_label

        # Harris button
        # button_2

    def enforceWindowSizeSliderStep(self):
        value = self.Window_size_slider.value()
        if value % 2 == 0:
            corrected_value = round((value - 3) / 2) * 2 + 3
            self.Window_size_slider.setValue(corrected_value)

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
