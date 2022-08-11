import cv2
import numpy as np 
import math

red = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -10) 
callibration = cv2.imread("/Users/omarafzal/TexasTorque/Armageddon/Armageddon-Vision-2022/code/BlobDetection/cube.png")

def find_marker(callibration):
    hsv = cv2.cvtColor(callibration, cv2.COLOR_BGR2HSV)

    lowerYellow = np.array([26,100, 12])
    upperYellow = np.array([130, 100, 100])

    mask = cv2.inRange(hsv, lowerYellow, upperYellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)

    return cv2.minAreaRect(c)

def linearConstraint(max, min, val):

    if val > max:
        return max
    elif val < min:
        return min
    return val


def getYawResidual(x, X, theta):

    x = linearConstraint(X, -X, x)
    theta = math.radians(theta)

    return x * theta / (2 * X)


while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerYellow = np.array([20, 20, 20])
    upperYellow = np.array([90, 80, 90])

 
    mask = cv2.inRange(hsv, lowerYellow, upperYellow)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    maxContour = []

    for contour in contours:
        if len(contour) > len(maxContour):
            maxContour = contour

    x, y, w, h = 0, 0, 0, 0
    if len(maxContour) > 0:
        x, y, w, h = cv2.boundingRect(maxContour)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

        p1 = [x, y]
        p2 = [x, y + h]
        p3 = [x + w, y]
        p4 = [x + w, y + h]

        centerX = (p3[0] - p1[0]) / 2
        centerY = (p2[1] - p1[1]) / 2
        centerPoint = [p1[0] + centerX, p1[1] + centerY]

        #imageCircle = cv2.circle(frame, center = (int(centerPoint[0]), int(centerPoint[1])), radius = 5, color = (56, 247, 35), thickness = 2)
        imageSquare = cv2.rectangle(frame, p1, p4, (56, 247, 35), thickness = -1)

        frameWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cpirto = [centerPoint[0] - (frameWidth / 2), -(centerPoint[1] - (frameHeight / 2))]

        c = max(maxContour, key = cv2.contourArea)
        marker = cv2.minAreaRect(c)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
     

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break




cv2.imshow('image', frame)
cv2.waitKey()
#if the loop breaks; the instance breaks 
cv2.destroyAllWindows()
cap.release() 
