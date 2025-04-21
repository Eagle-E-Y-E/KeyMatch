import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QHBoxLayout, \
    QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class ImageMatcher(QWidget):
    def __init__(self):
        super().__init__()

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
        template_std = np.std(template) * np.size(template)

        out_h = img_h - tmpl_h + 1
        out_w = img_w - tmpl_w + 1
        output_image = np.zeros((out_h, out_w), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i + tmpl_h, j:j + tmpl_w]
                region_mean = np.mean(region)
                region_std = np.std(region) * np.size(region)
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
