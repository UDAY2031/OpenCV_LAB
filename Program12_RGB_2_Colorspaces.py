import cv2
import matplotlib.pyplot as plt

# Load and convert image
img = cv2.imread('sample.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to other color spaces
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Combine all channels
images = [img_rgb] + list(cv2.split(hsv)) + list(cv2.split(lab)) + list(cv2.split(ycbcr))
titles = [
    'Original RGB',
    'HSV - H', 'HSV - S', 'HSV - V',
    'LAB - L', 'LAB - A', 'LAB - B',
    'YCbCr - Y', 'YCbCr - Cb', 'YCbCr - Cr'
]

# Display images
plt.figure(figsize=(15, 6))
for i in range(len(images)):
    plt.subplot(2, 5, i + 1)
    cmap = None if i == 0 else 'gray'
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
