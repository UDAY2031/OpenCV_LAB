import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and convert image to RGB
img = cv2.imread('c.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Image dimensions
h, w = img.shape[:2]

# Define source (original corners) and destination (warped) points
pts1 = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])  # Corners of original image
pts2 = np.float32([                                          # New distorted corners
    [w*0.1, h*0.2],
    [w*0.9, h*0.1],
    [w*0.9, h*0.9],
    [w*0.2, h*0.8]
])

# Compute perspective transform matrix
mat = cv2.getPerspectiveTransform(pts1, pts2)

# Apply perspective warp
warp = cv2.warpPerspective(img, mat, (w, h))
warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)  # Optional: Ensures it's RGB if warped changes channels

# Display original and warped images using a loop
titles = ['Original Image', 'Warped Image']
images = [img, warp]

plt.figure(figsize=(10, 5))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
