import cv2 as cv
import numpy as np
import torch
from time import time
import math

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default

# Images
frame = cv.imread('inputVideos/fixedBox.png')
frame = cv.resize(frame, (960, 480))
frame2 = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model(frame2, size=640)  # includes NMS
labels = results.xyxyn[0][:, -1].numpy()
cord = results.xyxyn[0][:, :-1].numpy()

def linearConstraint(max, min, val):

    if val > max:

        return max
    
    elif val < min:

        return min

    return val

def getLineLength(line):

    x0, y0, x1, y1 = line.flatten()

    return pow((x1 - x0)**2 + (y1 - y0)**2, .5)

def getHorizontalDistance(h, phi, theta, y, Y):

    phi = math.radians(phi - theta / 2)
    theta = math.radians(theta)
    y = linearConstraint(Y, 0, y)

    return h * math.tan(linearConstraint(math.radians(89.99999999999), 0, phi + theta * y / Y))


def plot_boxes(model, results, frame):

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    bluredFrame = cv.bilateralFilter(frame, 7, 50, 50) #blur spaces while keeping edges
    bluredFrame = cv.cvtColor(bluredFrame, cv.COLOR_BGR2HSV)
    tol = 5
    threshold = cv.inRange(bluredFrame, (23-tol, 41-tol, 133-tol), (40+tol, 255, 255))
    bluredFrame = cv.bitwise_and(bluredFrame, bluredFrame, mask=threshold)

    margin = 20

    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue

        x1 = linearConstraint(x_shape, 0, int(row[0]*x_shape) - margin)
        y1 = linearConstraint(y_shape, 0, int(row[1]*y_shape) - margin)
        x2 = linearConstraint(x_shape, 0, int(row[2]*x_shape) + margin)
        y2 = linearConstraint(y_shape, 0, int(row[3]*y_shape) + margin)

        bgr = (0, 255, 0) # color of the box
        classes = model.names # Get the name of label index
        label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.

        cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
        cv.putText(frame, "Cube", (x1, y1), label_font, 0.9, bgr, 2) #Put a label over box.; classes[labels[i]] vs "cube"
        
        miniFrame = frame[y1:y2, x1:x2] #getting subimage
        miniFrame = cv.cvtColor(miniFrame, cv.COLOR_BGR2GRAY)

        miniFrame = cv.Canny(bluredFrame[y1:y2, x1:x2], 100, 200) #Edge detection
        kernel = np.ones((5,5), np.uint8)
        edges = cv.dilate(miniFrame, kernel, iterations=1)

        # lsd = cv.createLineSegmentDetector(0) #Line detection
        # lines = lsd.detect(edges)[0]
        # lines = np.array([line for line in lines if getLineLength(line) > (y2-y1) * 0.15])

        # linearImage = np.zeros((y2-y1, x2-x1,3), np.uint8)
        # lsd.drawSegments(linearImage, lines)
        # linearImage = cv.cvtColor(linearImage, cv.COLOR_BGR2GRAY)

        #x, y, w, h = cv.boundingRect(edges)
        contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #Convex hull
        contours = [cv.convexHull(cnt) for cnt in contours]
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

        rect = cv.minAreaRect(contours[0]) #min area rect
        box = cv.boxPoints(rect)
        box = np.int0(box)

        miniFrame = cv.cvtColor(miniFrame, cv.COLOR_GRAY2BGR)

        cv.drawContours(miniFrame, [contours[0]], 0, (255, 0, 255), 1)

        vertexLst = sorted(box, key=lambda x: x[1], reverse=True)

        first = vertexLst[0]
        second = vertexLst[1]
        third = vertexLst[2]
        fourth = vertexLst[3]

        cv.circle(miniFrame, first, 8, (255, 50, 0), -1)
        cv.circle(miniFrame, second, 8, (255, 255, 0), -1)
        cv.circle(miniFrame, third, 8, (255, 255, 255), -1)
        cv.circle(miniFrame, fourth, 8, (255, 255, 255), -1)

        dist1 = getHorizontalDistance(25, 80, 60, y_shape - (y1 + first[1]), y_shape) #height, angle fr vert, fov
        dist2 = getHorizontalDistance(25, 80, 60, y_shape - (y1 + second[1]), y_shape)

        print(f"Distance: ({dist1}, {dist2})")

        frame[y1:y2, x1:x2] = miniFrame

    return frame

start_time = time()
frame = plot_boxes(model, (labels, cord), frame)
end_time = time()
fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
print(f"Frames Per Second : {fps}")

# Results
results.print()  
cv.imshow('power cubes', frame)

cv.waitKey(0)
cv.destroyAllWindows()