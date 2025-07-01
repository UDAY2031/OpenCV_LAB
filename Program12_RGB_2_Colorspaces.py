import cv2
import numpy as np
import matplotlib.pyplot as plt

image_rgb= cv2.imread('sample.png')
image_rgb=cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB)
# Convert RGB image to different color spaces
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

colors=[image_rgb,image_hsv,image_lab,image_ycrcb]
tlts=["RGB","HSV","LAB","YCbCr"]
plt.figure(figsize=(12,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(colors[i])
    plt.title(tlts[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
