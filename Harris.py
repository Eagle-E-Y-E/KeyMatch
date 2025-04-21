import numpy as np
import cv2
from scipy import signal as sig

class Harris:
    @staticmethod
    def manual_gray_conversion(img):
        """
        Convert a colored image to grayscale manually.
        If the image already is single-channel, just convert to float.
        Uses the weighted sum: Gray = 0.299*R + 0.587*G + 0.114*B.
        """
        # Check if image has more than one channel (assumes BGR order)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Do the conversion manually without cv2.cvtColor
            # Note: OpenCV loads color images in BGR order.
            gray = img[:,:,2] * 0.299 + img[:,:,1] * 0.587 + img[:,:,0] * 0.114
            return gray.astype(np.float32)
        else:
            return img.astype(np.float32)
        
    # @staticmethod
    # def compute_gradients(gray):
    #     """
    #     Compute image gradients using simple finite differences.
    #     Uses a central difference approximation (ignores borders).
    #     """
    #     Ix = np.zeros_like(gray)
    #     Iy = np.zeros_like(gray)
    #     # Central difference: compute gradient in x direction
    #     Ix[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    #     # Central difference: compute gradient in y direction
    #     Iy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    #     return Ix, Iy
    
    def sobel_x(imggray):
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        return sig.convolve2d(imggray, kernel_x, mode='same')
    
    def sobel_y(imggray):
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return sig.convolve2d(imggray, kernel_y, mode='same')


    @staticmethod
    def compute_harris_response(gray, k=0.04, window_size=3):
        """
        Compute only the Harris response (R = det - k*(trace)²).
        """
        # Ix, Iy = Harris.compute_gradients(gray)
        Ix = Harris.sobel_x(gray)
        Iy = Harris.sobel_y(gray)

        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        offset = window_size // 2
        rows, cols = gray.shape
        R = np.zeros_like(gray)
        
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                # Sum products over the window.
                Sxx = np.sum(Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1])
                Syy = np.sum(Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1])
                Sxy = np.sum(Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1])
                
                det = Sxx * Syy - Sxy * Sxy
                trace = Sxx + Syy
                R[i, j] = det - k * (trace ** 2)
                
        return R
    
    @staticmethod
    def compute_lambda_response(gray, window_size=3):
        """
        Compute only the minimum eigenvalue response (λ₋).
        """
        # Ix, Iy = Harris.compute_gradients(gray)
        Ix = Harris.sobel_x(gray)
        Iy = Harris.sobel_y(gray)
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        offset = window_size // 2
        rows, cols = gray.shape
        lambda_min = np.zeros_like(gray)
        
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                Sxx = np.sum(Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1])
                Syy = np.sum(Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1])
                Sxy = np.sum(Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1])
                
                det = Sxx * Syy - Sxy * Sxy
                trace = Sxx + Syy
                
                # Compute eigenvalues using the analytic formula.
                temp = (trace / 2) ** 2 - det
                if temp < 0:
                    eigen1 = eigen2 = trace / 2
                else:
                    sqrt_val = np.sqrt(temp)
                    eigen1 = trace / 2 + sqrt_val
                    eigen2 = trace / 2 - sqrt_val
                lambda_min[i, j] = min(eigen1, eigen2)
                
        return lambda_min

    @staticmethod
    def get_corner_points(response, threshold_ratio=0.01):
        """
        Given a response image (Harris or minimum eigenvalue),
        threshold it by a ratio of the maximum value then perform
        basic non-maximum suppression in a 3x3 window.
        
        Returns a list of (x, y) tuples corresponding to corner points.
        """
        threshold = threshold_ratio * np.max(response)
        points = []
        rows, cols = response.shape
        offset = 1  # for a 3x3 neighborhood
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                if response[i, j] > threshold:
                    # Check if this value is the maximum in its 3x3 neighborhood
                    local_patch = response[i-offset:i+offset+1, j-offset:j+offset+1]
                    if response[i, j] == np.max(local_patch):
                        points.append((j, i))  # (x, y) format for cv2.circle
        return points

    @staticmethod
    def mark_corners_on_image(img, points, color=(0, 0, 255)):
        """
        Draw small circles on the input image at the provided point locations.
        """
        for pt in points:
            cv2.circle(img, pt, 2, color, -1)
        return img