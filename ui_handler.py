from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import sys
import time
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QFileDialog
from PyQt5.QtCore import QTimer

from Harris import Harris


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
            lambda: self.Harris_threshold_label.setText(f"{self.Harris_threshold_slider.value()/1000}"))
        # range for slider from 5 to 50 so real value is /1000
        

        self.Window_size_slider.valueChanged.connect(
            lambda: self.Window_size_label.setText(f"{self.Window_size_slider.value()}"))
        self.Window_size_slider.valueChanged.connect(
            lambda: enforce_slider_step(self.Window_size_slider, 2, 3))

        # time labels

        # Harris_input_label
        # Harris_outpu_label

        # Harris button
        # button_2
        self.button_2.clicked.connect(self.processHarrisImage)

    def enforceWindowSizeSliderStep(self):
        value = self.Window_size_slider.value()
        if value % 2 == 0:
            corrected_value = round((value - 3) / 2) * 2 + 3
            self.Window_size_slider.setValue(corrected_value)

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)

    def processHarrisImage(self):
        # Check if an image has been loaded
        if self.img_path is None:
            QMessageBox.warning(self, "No Image", "Please load an image first by double-clicking on the input widget.")
            return

        # Read the image using OpenCV:
        img = cv2.imread(self.img_path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Image not found at {self.img_path}")
            return

        gray_image = Harris.manual_gray_conversion(img)

        # Get parameters from your sliders:
        k = self.k_slider.value() / 1000.0
        window_size = self.Window_size_slider.value() 
        threshold_ratio = self.Harris_threshold_slider.value() / 1000.0

        # Run Harris Corner Detection:
        start_harris = time.perf_counter()
        R = Harris.compute_harris_response(gray_image, k=k, window_size=window_size)
        corners_harris = Harris.get_corner_points(R, threshold_ratio=threshold_ratio, window_size=window_size)
        end_harris = time.perf_counter()
        print(f"Harris operator computation time: {end_harris - start_harris:.4f} seconds")
        print(f"Number of Harris corners: {len(corners_harris)}")

        # Run Lambda-min corner detection:
        start_lambda = time.perf_counter()
        lambda_response = Harris.compute_lambda_response(gray_image, window_size=window_size)
        corners_lambda = Harris.get_corner_points(lambda_response, threshold_ratio=threshold_ratio, window_size=window_size)
        end_lambda = time.perf_counter()
        print(f"Lambda-min operator computation time: {end_lambda - start_lambda:.4f} seconds")
        print(f"Number of Lambda-min corners: {len(corners_lambda)}")

        # Mark corners on images:
        output_harris = Harris.mark_corners_on_image(img.copy(), corners_harris, color=(0, 0, 255))
        output_lambda = Harris.mark_corners_on_image(img.copy(), corners_lambda, color=(0, 255, 0))

        display_image_Graphics_scene(self.Harris_output_img1_GV, output_harris)
        display_image_Graphics_scene(self.Harris_output_img2_GV, output_lambda)
        self.Harris_input_label.setText(f"Harris operator computation time: {end_harris - start_harris:.4f} seconds")
        self.Harris_outpu_label.setText(f"Lambda-min operator computation time: {end_lambda - start_lambda:.4f} seconds")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
