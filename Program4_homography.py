import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('c.png')
img2 = cv2.imread('s.png')

# Convert to RGB for display
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Corresponding points (manual)
pts1 = np.float32([[100, 100], [200, 100], [200, 200], [100, 200]])
pts2 = np.float32([[120, 130], [220, 120], [230, 230], [110, 240]])

# Compute homography and warp
H, _ = cv2.findHomography(pts1, pts2)
warp = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)

# Display using loop
titles = ['Image 1 (Source)', 'Image 2 (Destination)', 'Warped Image 1']
images = [img1, img2, warp]
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
