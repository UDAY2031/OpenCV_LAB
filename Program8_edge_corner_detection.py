import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('c.png')  # Read the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
edge = cv2.Canny(img, 100, 200)  # Apply Canny edge detection

grays = np.float32(gray)  # Convert grayscale image to float32 for Harris corner

harris = cv2.cornerHarris(grays, 2, 3, 0.04)  # Detect corners using Harris
harris = cv2.dilate(harris, None)  # Dilate the corner regions

img[harris > 0.01 * harris.max()] = [0, 0, 255]  # Mark corners in red on the original image

hough = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib
h = cv2.cvtColor(harris, cv2.COLOR_BGR2RGB)  # Convert Harris output (optional, may not show well)

imgs = [img, edge, hough, h]  # List of images to display
names = ['Original', 'Edge', 'Hough lines', 'Harris Corner']  # Titles for subplots

plt.figure(figsize=(10, 5))  # Create a plotting figure

for i in range(4):  # Loop to plot each image
    plt.subplot(2, 2, i + 1)  # Define subplot grid position
    plt.imshow(imgs[i])  # Show the image
    plt.title(names[i])  # Set the title
    
plt.tight_layout()  # Adjust layout for clean spacing
plt.show()  # Display the plots
