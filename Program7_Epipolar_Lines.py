import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo image and split into left and right images
img = cv2.imread('image.jpg')
h, w = img.shape[:2]
imgL = img[:, :w//2]  # Left image
imgR = img[:, w//2:]  # Right image

# Generate random matching points
x = np.random.randint(50, w//2 - 50, (10, 2)).astype(float)
y = x + np.random.normal(0, 2, x.shape)

# Find fundamental matrix using RANSAC
F, m = cv2.findFundamentalMat(x, y, cv2.FM_RANSAC)
x, y = x[m.ravel() == 1], y[m.ravel() == 1]  # Keep inliers only

# Compute epipolar lines for left image based on points in right image
lines = cv2.computeCorrespondEpilines(y.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

# Convert images to RGB
imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

# Plot using loop
imgs = [imgL_rgb, imgR_rgb]
titles = ["Epipolar lines in Left Image", "Matching Points in Right Image"]
points = [x, y]

plt.figure(figsize=(10, 5))
for i in range(2):
    ax = plt.subplot(1, 2, i+1)
    ax.imshow(imgs[i])
    ax.set_title(titles[i])
    ax.axis('off')
    ax.scatter(points[i][:, 0], points[i][:, 1], color='red')
    
    if i == 0:  # Draw lines only on left image
        for r in lines:
            x0, y0 = 0, int(-r[2]/r[1])
            x1, y1 = imgL.shape[1], int(-(r[2] + r[0]*x1)/r[1])
            ax.plot([x0, x1], [y0, y1], color='blue')

plt.tight_layout()
plt.show()
