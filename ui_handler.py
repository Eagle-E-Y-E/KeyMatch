import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import sys
import time
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step
from Harris import Harris
from ImageMatcher import ImageMatcher
import sift


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.scored_image = None
        uic.loadUi('ui.ui', self)
        self.matcher = ImageMatcher()
        self.matcher_image = None
        self.template = None

        self.input_img1.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img1)
        self.input_img2.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img2)
        print(self.input_img1)
        # output images
        # output_img1_GV ==> graphics view
        # output_img2_GV ==> graphics view

        # slider
        self.threshold_slider.valueChanged.connect(
            lambda: self.threshold_label.setText(f"{self.threshold_slider.value() / 100}"))
        self.threshold_slider.valueChanged.connect(self.update_output)
        self.calculate.clicked.connect(self.run_matching)

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
            lambda: self.K_label.setText(f"{self.k_slider.value() / 1000}"))
        # range for slider from 40 to 60 so real value is /1000

        self.Harris_threshold_slider.valueChanged.connect(
            lambda: self.Harris_threshold_label.setText(f"{self.Harris_threshold_slider.value() / 1000}"))
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

        # SIFT Tab_________________________________________________________________________
        # input images SIFT
        self.SIFT_input_img1.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.SIFT_input_img1)
        self.SIFT_input_img2.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.SIFT_input_img2)
        
        # output images SIFT
             # SIFT_output_img1_GV ==> graphics view 

        self.handle_sift_status_label()
        # Match_btn
        self.match_btn.clicked.connect(self.sift_match)


    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)
        if widget == self.input_img1:
            self.matcher_image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.colored_image = cv2.imread(self.img_path)
        elif widget == self.input_img2:
            self.template = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        elif widget == self.SIFT_input_img1:
            self.sift_img1 = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        elif widget == self.SIFT_input_img2:
            self.sift_img2 = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

    def handle_sift_status_label(self, status=None):
        if status == 'success':
            self.status_label.setText("Succes, Match Found")
            self.status_label.setStyleSheet("color: green;")
        elif status == 'fail':
            self.status_label.setText("Fail, No Match Found")
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setText("status")
            self.status_label.setStyleSheet("color: white;")

    def processHarrisImage(self):
        # Check if an image has been loaded
        if self.img_path is None:
            QMessageBox.warning(self, "No Image", "Double-click on the input widget to load an image")
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
        corners_lambda = Harris.get_corner_points(lambda_response, threshold_ratio=threshold_ratio,
                                                  window_size=window_size)
        end_lambda = time.perf_counter()
        print(f"Lambda-min operator computation time: {end_lambda - start_lambda:.4f} seconds")
        print(f"Number of Lambda-min corners: {len(corners_lambda)}")

        # Mark corners on images:
        output_harris = Harris.mark_corners_on_image(img.copy(), corners_harris, color=(0, 0, 255))
        output_lambda = Harris.mark_corners_on_image(img.copy(), corners_lambda, color=(0, 255, 0))

        display_image_Graphics_scene(self.Harris_output_img1_GV, output_harris)
        display_image_Graphics_scene(self.Harris_output_img2_GV, output_lambda)
        self.Harris_input_label.setText(f"Harris operator computation time: {end_harris - start_harris:.4f} seconds")
        self.Harris_outpu_label.setText(
            f"Lambda-min operator computation time: {end_lambda - start_lambda:.4f} seconds")

    def SSD(self, image, template):
        self.scored_image = self.matcher.ssd_matcher(image, template)
        display_image_Graphics_scene(self.output_img1_GV, self.scored_image)

    def NCC(self, image, template):
        self.scored_image = self.matcher.ncc_matcher(image, template)
        display_image_Graphics_scene(self.output_img1_GV, self.scored_image)

    def run_matching(self):
        if self.matcher_image is not None and self.template is not None:
            method = self.mode_combo.currentText()
            if method == 'SSD':
                self.threshold_slider.setValue(0)
                self.SSD(self.matcher_image, self.template)
            elif method == 'NCC':
                self.threshold_slider.setValue(100)
                self.NCC(self.matcher_image, self.template)
            self.update_output()

    def update_output(self):
        if self.scored_image is None:
            return
        threshold = self.threshold_slider.value() / 100
        is_ssd = self.mode_combo.currentText() == 'SSD'
        thresholded_scored_image = np.where(
            self.scored_image < threshold if is_ssd else self.scored_image > threshold, 255, 0).astype(np.uint8)
        display_image_Graphics_scene(self.output_img1_GV, thresholded_scored_image)
        marked_image = self.marker(self.matcher_image, self.scored_image, self.template.shape, threshold)
        display_image_Graphics_scene(self.output_img2_GV, marked_image)

    def marker(self, image, output_image, template_shape, threshold):
        marked_image = self.colored_image.copy()
        tmpl_h, tmpl_w = template_shape
        match_locations = np.argwhere(
            output_image < threshold) if self.mode_combo.currentText() == 'SSD' else np.argwhere(
            output_image > threshold)
        for y, x in match_locations:
            top_left = (x, y)
            bottom_right = (x + tmpl_w, y + tmpl_h)
            cv2.rectangle(marked_image, top_left, bottom_right, (255, 0, 0), 2)
        return marked_image

    def sift_match(self):
        if self.sift_img1 is None or self.sift_img2 is None:
            QMessageBox.warning(self, "No Images", "Double-click on the input widgets to load images for matching")
            return
         # 1) Extract keypoints+descriptors
        kp1, des1 = sift.computeKeypointsAndDescriptors(self.sift_img1)
        kp2, des2 = sift.computeKeypointsAndDescriptors(self.sift_img2)

        # 2) Match and test
        is_match, good_matches, match_vis = sift.match_descriptors(
            self.sift_img1, kp1, des1, self.sift_img2, kp2, des2, matcher='FLANN', ratio_thresh=0.7, min_matches=5, draw_matches=True
        )

        if match_vis is not None:
            display_image_Graphics_scene(self.SIFT_output_img1_GV, match_vis)
            # update labels
            self.num_features_img1.setText(f'Number of Keypoints: {len(kp1)}')
            self.num_features_img2.setText(f'Number of Keypoints: {len(kp2)}')
            self.num_matches_label.setText(f'Number of Matches: {len(good_matches)}')
            self.handle_sift_status_label('success' if is_match else 'fail') #  handle after processing

            print(f"Match found? {is_match}, #good matches = {len(good_matches)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
