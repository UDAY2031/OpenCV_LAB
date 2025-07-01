import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('sample.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image (invert: white objects on black background)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Count connected regions
num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

# Draw labels and boxes
for i in range(1, num):  # Skip background (label 0)
    x, y, w, h, _ = stats[i]
    cx, cy = map(int, centroids[i])

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

print(f"Total regions (excluding background): {num - 1}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Show result
plt.imshow(img)
plt.axis('off')
plt.show()
