# Trash
import cv2
import numpy as np

img = cv2.imread("cube.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_range = np.array([30,30, 30])
upper_range = np.array([130, 255, 255])

mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('Image', img)
cv2.imshow('Mask', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()