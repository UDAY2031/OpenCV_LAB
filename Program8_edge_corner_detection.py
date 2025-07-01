import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image once
img = cv2.imread('s2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save the original for display
original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# 2. Harris Corner Detection (on a copy of the original)

corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
img[corners > 0.01 * corners.max()] = [255, 0, 0]  # Red corners
harris_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. Hough Line Detection (on a copy of the original)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
hough_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Prepare images and titles
images = [original_rgb, edges, harris_rgb, hough_rgb]
titles = ['Original Image', 'Canny Edge Detection', 'Harris Corner Detection', 'Hough Line Detection']

# Plot all 4 stages
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    if i == 1:  # Edge image is grayscale
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
