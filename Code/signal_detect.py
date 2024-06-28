import cv2
import numpy as np

# Load the image
img = cv2.imread('P3Data/Seq_Frames/interpolated_frames/scene4/frame260.jpg')

# Define color ranges for red and green signals in HSV color space
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])
green_lower = np.array([45, 60, 60])
green_upper = np.array([75, 255, 255])

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create masks for red and green signals
red_mask = cv2.inRange(hsv, red_lower, red_upper)
green_mask = cv2.inRange(hsv, green_lower, green_upper)

# Find the contours in the masks
red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Check which signal color has the largest contour area
if len(red_contours) > 0:
    red_area = max([cv2.contourArea(c) for c in red_contours])
else:
    red_area = 0
    
if len(green_contours) > 0:
    green_area = max([cv2.contourArea(c) for c in green_contours])
else:
    green_area = 0
    
# Determine the signal color based on the largest contour area
if red_area > green_area:
    print('Red Signal')
elif green_area > red_area:
    print('Green Signal')
else:
    print('No Signal Found')
