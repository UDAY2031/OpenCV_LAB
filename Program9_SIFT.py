#9
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("birds.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img_with_keypoints = cv2.drawKeypoints(
        gray, kp, img)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT KEYPOINTS')
plt.axis('off')
plt.show()
