# 6. Camera Calibration using Chessboard
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert to grayscale
x = cv2.imread('chess.png')                # x: original image
y = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)    # y: grayscale image

# Try to find chessboard corners (7x7 pattern)
found, corners = cv2.findChessboardCorners(y, (7, 7), None)

# Draw corners
x = cv2.drawChessboardCorners(x, (7, 7), corners, found)

# Show image
plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
plt.title("Chessboard Corners")
plt.show()
