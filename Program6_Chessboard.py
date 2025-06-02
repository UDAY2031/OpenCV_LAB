# 6. Camera Calibration using Chessboard
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert to grayscale
x = cv2.imread('chess.png')                # x: original image
y = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)    # y: grayscale image

# Try to find chessboard corners (7x7 pattern)
found, corners = cv2.findChessboardCorners(y, (7, 7), None)

if found:
    # Create 3D points (z = 0)
    obj = np.zeros((49, 3), np.float32)
    obj[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Calibrate the camera
    _, m, d, r, t = cv2.calibrateCamera([obj], [corners], y.shape[::-1], None, None)
    # m: camera matrix, d: distortion, r: rotation, t: translation

    # Project 3D points back to image
    proj, _ = cv2.projectPoints(obj, r[0], t[0], m, d)

    # Compute reprojection error
    e = cv2.norm(corners, proj) / len(proj)

    # Show results
    print("Camera Matrix:\n", m)
    print("Distortion Coefficients:\n", d)
    print("Reprojection Error:", e)

    # Draw corners
    x = cv2.drawChessboardCorners(x, (7, 7), corners, found)

    # Show image
    plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    plt.title("Chessboard Corners")
    plt.axis('off')
    plt.show()
else:
    print("Chessboard not found.")
