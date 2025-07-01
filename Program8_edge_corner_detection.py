import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('s2.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# 2. Hough Line Detection
line_img = np.zeros_like(img)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 3. Harris Corner Detection (modifies original img directly)
corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
# This will modify `img` directly, as requested
img[corners > 0.01 * corners.max()] = [255, 0, 0]  # Mark corners in red

# Convert images to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)

# List of images and titles
images = [img_rgb, edges, line_img_rgb, img_rgb]  # Note: img_rgb now has corners marked too
titles = ['Original Image with Corners', 'Canny Edge Detection', 'Hough Line Detection', 'Harris Corner Detection']

# Plot using loop
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    if i == 1:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
