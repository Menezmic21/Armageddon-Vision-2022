import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('powerCubes.jfif',0)

# # Initiate ORB detector
# orb = cv.ORB_create()

# # find the keypoints with ORB
# kp = orb.detect(img,None)

# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)

# # draw only keypoints location,not size and orientation
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()

#Create default parametrization LSD
lsd = cv.createLineSegmentDetector(0)

#Detect lines in the image
lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines

#Draw detected lines in the image
drawn_img = lsd.drawSegments(img,lines)

#Show image
plt.imshow(drawn_img), plt.show()