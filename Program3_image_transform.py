import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and convert the image to RGB
img = cv2.imread('c.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image dimensions
rows, cols = img.shape[:2]

# Apply translation
M_translate = np.float32([[1, 0, 50], [0, 1, 50]])
translated = cv2.warpAffine(img, M_translate, (cols, rows))

# Apply rotation
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

# Apply scaling
scaled = cv2.resize(img, None, fx=0.5, fy=0.5)

# Store images and titles in lists
images = [img, translated, rotated, scaled]
titles = ["Original", "Translated", "Rotated", "Scaled"]

# Plot using a loop
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
