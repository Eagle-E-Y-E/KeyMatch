import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QHBoxLayout, \
    QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class SSD_Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.template = None
        self.output = None
        self.matching_method = 'SSD'  # Default method

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.output_label = QLabel(self)
        layout.addWidget(self.output_label)

        button_layout = QHBoxLayout()

        upload_image_button = QPushButton('Upload Image', self)
        upload_image_button.clicked.connect(self.upload_image)
        button_layout.addWidget(upload_image_button)

        upload_template_button = QPushButton('Upload Template', self)
        upload_template_button.clicked.connect(self.upload_template)
        button_layout.addWidget(upload_template_button)

        layout.addLayout(button_layout)

        # Combo box to select matching method
        self.method_selector = QComboBox(self)
        self.method_selector.addItems(['SSD', 'NCC'])
        self.method_selector.currentTextChanged.connect(self.run_matching)
        layout.addWidget(self.method_selector)

        self.slider_value_label = QLabel('Threshold: 0', self)
        layout.addWidget(self.slider_value_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 1000)
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

        return self.normalize_output(output_image)

    def ncc_matcher(self, image, template):
        img_h, img_w = image.shape
        tmpl_h, tmpl_w = template.shape
        image = image.astype(np.float32)
        template = template.astype(np.float32)

        template_mean = np.mean(template)
        template_std = np.std(template)*np.size(template)

        out_h = img_h - tmpl_h + 1
        out_w = img_w - tmpl_w + 1
        output_image = np.zeros((out_h, out_w), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i + tmpl_h, j:j + tmpl_w]
                region_mean = np.mean(region)
                region_std = np.std(region)*np.size(region)
                if region_std == 0 or template_std == 0:
                    output_image[i, j] = 0
                else:
                    norm_region = (region - region_mean) / region_std
                    norm_template = (template - template_mean) / template_std
                    output_image[i, j] = np.mean(norm_region * norm_template)

        return self.normalize_output(output_image)

    def normalize_output(self, output):
        min_val = np.min(output)
        max_val = np.max(output)
        if max_val != min_val:
            return (output - min_val) / (max_val - min_val)
        return np.zeros_like(output)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image, self.image_label)
            if self.template is not None:
                self.run_matching()

    def upload_template(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Template", "", "Images (*.png *.xpm *.jpg)")
        if file_path:
            self.template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is not None:
                self.run_matching()

    def run_matching(self):
        if self.image is not None and self.template is not None:
            method = self.method_selector.currentText()
            if method == 'SSD':
                self.output = self.ssd_matcher(self.image, self.template)
            elif method == 'NCC':
                self.output = self.ncc_matcher(self.image, self.template)
            self.update_output()

    def draw_matches_on_image(self, image, output_image, template_shape, threshold):
        marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tmpl_h, tmpl_w = template_shape

        match_locations = np.argwhere(output_image <= threshold) if self.method_selector.currentText() == 'SSD' else np.argwhere(output_image >= threshold)
        for y, x in match_locations:
            top_left = (x, y)
            bottom_right = (x + tmpl_w, y + tmpl_h)
            cv2.rectangle(marked_image, top_left, bottom_right, (255, 0, 0), 2)

        return marked_image

    def update_output(self):
        if self.output is not None:
            threshold = self.slider.value() / 100
            self.slider_value_label.setText(f'Threshold: {threshold}')

            is_ssd = self.method_selector.currentText() == 'SSD'
            binary_output = np.where(self.output <= threshold if is_ssd else self.output >= threshold, 255, 0).astype(np.uint8)
            self.display_image(binary_output, self.output_label)

            marked_image = self.draw_matches_on_image(self.image, self.output, self.template.shape, threshold)
            self.display_image(marked_image, self.image_label)

    def display_image(self, img, label):
        if len(img.shape) == 2:
            height, width = img.shape
            bytes_per_line = width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
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