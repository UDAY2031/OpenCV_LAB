import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
img = cv2.imread('image.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding (inverse binary) to create a binary image
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define a 3x3 kernel for morphological operations
k = np.ones((3, 3), np.uint8)

# Perform morphological opening to remove small noise
op = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)

# Dilate to get sure background area
bg = cv2.dilate(op, k, iterations=3)

# Use distance transform to get sure foreground area
dist = cv2.distanceTransform(op, cv2.DIST_L2, 5)
_, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
fg = np.uint8(fg)

# Subtract foreground from background to get unknown region
unk = cv2.subtract(bg, fg)

# Label the connected components in the foreground
_, mk = cv2.connectedComponents(fg)

# Increment all labels so background is not 0, but 1
mk = mk + 1

# Mark the unknown region as 0
mk[unk == 255] = 0

# Apply the watershed algorithm
mk = cv2.watershed(img, mk)

# Mark the watershed boundaries with green color in the original image
img[mk == -1] = [0, 255, 0]

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Boundaries')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mk, cmap='jet')
plt.title('Markers')
plt.axis('off')

plt.tight_layout()
plt.show()
