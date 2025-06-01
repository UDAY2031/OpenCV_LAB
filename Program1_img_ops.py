import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('c.png', 0)  # Already in grayscale

# No need to convert again to grayscale (img is already grayscale)
gray = img.copy()

# Histogram Equalization to improve contrast
eq = cv2.equalizeHist(gray)

# Apply binary thresholding
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Edge detection using the Canny method
ed = cv2.Canny(gray, 100, 200)

# Resize the image to width=100 and height=130
re = cv2.resize(gray, (100, 130))

# Flip the image horizontally (1 means horizontal flip)
flip = cv2.flip(gray, 1)

# Apply morphological closing (dilation followed by erosion)
kernel = np.ones((5, 5), np.uint8)
mor = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# List of processed images and their titles
imgs = [img, gray, eq, th, ed, re, flip, mor]
names = ['original', 'gray', 'equalized', 'threshold', 'edge', 'resize', 'flip', 'morphology']

# Plot all images using matplotlib
plt.figure(figsize=(16, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)  # Better layout: 2 rows x 4 columns
    cmap = 'gray' if len(imgs[i].shape) == 2 else None  # Use grayscale colormap for 2D images
    plt.imshow(imgs[i], cmap=cmap)
    plt.title(names[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
