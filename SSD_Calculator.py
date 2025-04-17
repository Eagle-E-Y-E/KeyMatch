import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class SSD_Calculator(QWidget):
    def __init__(self):
        super().__init__()

        self.image = None
        self.template = None
        self.output = None

        self.init_ui()

    def init_ui(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Label to show the original image
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Label to show the SSD output
        self.output_label = QLabel(self)
        layout.addWidget(self.output_label)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Button to upload image
        upload_image_button = QPushButton('Upload Image', self)
        upload_image_button.clicked.connect(self.upload_image)
        button_layout.addWidget(upload_image_button)

        # Button to upload template
        upload_template_button = QPushButton('Upload Template', self)
        upload_template_button.clicked.connect(self.upload_template)
        button_layout.addWidget(upload_template_button)

        layout.addLayout(button_layout)

        # Create label for slider value
        self.slider_value_label = QLabel('Threshold: 0', self)
        layout.addWidget(self.slider_value_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 1000)
        self.slider.setSingleStep(1)  # Set the step size to 100
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_output)
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.setWindowTitle('Image Matcher App')

    def ssd_matcher(self, image, template):
        img_h, img_w = image.shape
        tmpl_h, tmpl_w = template.shape
        image = cv2.GaussianBlur(image, (11, 11), 0)

        out_h = img_h - tmpl_h + 1
        out_w = img_w - tmpl_w + 1

        output_image = np.zeros((out_h, out_w), dtype=np.float32)
        template_sq_sum = np.sum(template.astype(np.float32) ** 2)

        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i + tmpl_h, j:j + tmpl_w].astype(np.float32)
                region_sq_sum = np.sum(region ** 2)
                ssd = np.sum((region - template) ** 2)

                denom = np.sqrt(region_sq_sum * template_sq_sum)
                output_image[i, j] = ssd / denom if denom != 0 else 0

        # Normalize to range [0, 1]
        min_val = np.min(output_image)
        max_val = np.max(output_image)
        if max_val != min_val:
            output_norm = (output_image - min_val) / (max_val - min_val)
        else:
            output_norm = np.zeros_like(output_image)

        return output_norm

    def upload_image(self):
        # Open file dialog to select image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image, self.image_label)
            if self.template is not None:  # Run matching if template is already loaded
                self.run_matching()

    def upload_template(self):
        # Open file dialog to select template
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Template", "", "Images (*.png *.xpm *.jpg)")
        if file_path:
            self.template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is not None:  # Run matching if image is already loaded
                self.run_matching()

    def run_matching(self):
        # Run the SSD matcher
        if self.image is not None and self.template is not None:
            # Check if template is larger than image
            if self.template.shape[0] > self.image.shape[0] or self.template.shape[1] > self.image.shape[1]:
                print("Error: Template is larger than the image!")
                return
            self.output = self.ssd_matcher(self.image, self.template)
            self.update_output()  # Auto-update display

    def draw_matches_on_image(self, image, output_image, template_shape, threshold):
        marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tmpl_h, tmpl_w = template_shape

        match_locations = np.argwhere(output_image <= threshold)
        for y, x in match_locations:
            top_left = (x, y)
            bottom_right = (x + tmpl_w, y + tmpl_h)
            cv2.rectangle(marked_image, top_left, bottom_right, (255, 0, 0), 2)

        return marked_image

    def update_output(self):
        if self.output is not None:
            threshold = self.slider.value() / 100
            self.slider_value_label.setText(f'Threshold: {threshold}')

            # Binary thresholded map
            thresholded_output = np.where(self.output <= threshold, 255, 0).astype(np.uint8)
            self.display_image(thresholded_output, self.output_label)

            # Draw rectangles on original image
            marked_image = self.draw_matches_on_image(self.image, self.output, self.template.shape, threshold)
            self.display_image(marked_image, self.image_label)

    def display_image(self, img, label):
        if len(img.shape) == 2:  # Grayscale
            height, width = img.shape
            bytes_per_line = width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color (assumed BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SSD_Calculator()
    window.show()
    sys.exit(app.exec_())
