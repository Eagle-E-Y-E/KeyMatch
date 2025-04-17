import cv2
import numpy as np

# Load the main image and template in grayscale
image = cv2.imread('Data/SSD_testing/meta.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('Data/SSD_testing/metaicon.png', cv2.IMREAD_GRAYSCALE)

# Check if images loaded properly
if image is None or template is None:
    raise ValueError("Could not load one or both images")

# Perform template matching using SSD
result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)

# Normalize the result for visualization
result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Show the result
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Since we're using TM_SQDIFF, the best match has the **minimum** value
top_left = min_loc
h, w = template.shape
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw a rectangle around the best match
matched_image = image.copy()
cv2.rectangle(matched_image, top_left, bottom_right, 0, 2)

cv2.imshow('Best Match Location', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
