import cv2
import matplotlib.pyplot as plt

# Load the image using OpenCV (in BGR format by default)
img = cv2.imread('s2.png')

# Convert the image from BGR to RGB for correct color display in matplotlib
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert RGB image to different color spaces
hsv   = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)       # RGB to HSV
lab   = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)       # RGB to LAB
ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)     # RGB to YCrCb

# Split each color space into individual channels
imgs = [rgb] + list(cv2.split(hsv)) + list(cv2.split(lab)) + list(cv2.split(ycrcb))

# Titles for each subplot
titles = [
    'RGB',
    'HSV - H', 'HSV - S', 'HSV - V',
    'LAB - L', 'LAB - A', 'LAB - B',
    'YCrCb - Y', 'YCrCb - Cr', 'YCrCb - Cb'
]

# Plot all images
plt.figure(figsize=(12, 6))
for i in range(len(imgs)):
    plt.subplot(2, 5, i + 1)                         # Create subplot grid (2 rows x 5 columns)
    cmap = None if i == 0 else 'gray'               # Use color for RGB, grayscale for channels
    plt.imshow(imgs[i], cmap=cmap)
    plt.title(titles[i])

plt.tight_layout()
plt.show()
