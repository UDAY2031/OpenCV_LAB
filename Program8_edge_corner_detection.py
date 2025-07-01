import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('s2.png')
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply Canny edge detection
edge = cv2.Canny(gray, 100, 200)
# Convert grayscale to float32 for Harris corner detection
grays = np.float32(gray)
# Apply Harris corner detection
harris = cv2.cornerHarris(grays, 2, 3, 0.04)
# Dilate the result to enhance corner points
harris = cv2.dilate(harris, None)
# Mark corners in red on a copy of the original image
corner_img = img.copy()
corner_img[harris > 0.01 * harris.max()] = [0, 0, 255]
# Detect lines using Hough Transform
hough_img = corner_img.copy()
lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Convert images to RGB for displaying with matplotlib
corner_img_rgb = cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB)
hough_rgb = cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB)

# Prepare images and titles for display
imgs = [corner_img_rgb, edge, hough_rgb, harris]
titles = ['Original with Corners', 'Canny Edges', 'Hough Lines', 'Harris Response']

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    if i == 1 or i == 3:
        plt.imshow(imgs[i], cmap='gray')  # Display grayscale images
    else:
        plt.imshow(imgs[i])  # Display color images
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
