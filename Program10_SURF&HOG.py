import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Load image and convert to grayscale
img = cv2.imread('sample.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SURF or fallback to ORB
try:
    detector = cv2.xfeatures2d.SURF_create(400)
except:
    detector = cv2.ORB_create()

# Detect keypoints
kp, _ = detector.detectAndCompute(gray, None)
surf_img = cv2.drawKeypoints(img, kp, None, (0, 255, 0))
surf_img = cv2.cvtColor(surf_img, cv2.COLOR_BGR2RGB)

# HOG feature image
_, hog_image = hog(gray, visualize=True)
hog_image = exposure.rescale_intensity(hog_image)

# Show both
titles = ['SURF/ORB', 'HOG']
images = [surf_img, hog_image]

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], cmap=None if i == 0 else 'gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
