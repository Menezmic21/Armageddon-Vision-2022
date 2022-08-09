import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([30,30, 30])
    upper_range = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(0)
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()