import cv2 as cv
import numpy as np
import torch
from time import time
import math

# Model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
model = torch.hub.load('code/yolov5-6.1', 'custom', path='code/best.pt', source="local")

#Start total timer
init_time = time()

# Images
frame = cv.imread('code/inputVideos/fixedBox.png')
frame = cv.resize(frame, (960, 480))
frame2 = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model(frame2, size=640)  # includes NMS
labels = results.xyxyn[0][:, -1].cpu().numpy()
cord = results.xyxyn[0][:, :-1].cpu().numpy()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# results = model(frame2, size=640)  # includes NMS
# labels = results.xyxyn[0][:, -1].cpu().numpy()
# cord = results.xyxyn[0][:, :-1].cpu().numpy()

def linearConstraint(max, min, val):

    if val > max:

        return max
    
    elif val < min:

        return min

    return val

def getHorizontalDistance(h, phi, theta, y, Y):

    phi = math.radians(phi - theta / 2)
    theta = math.radians(theta)
    y = linearConstraint(Y, 0, y)

    return h * math.tan(linearConstraint(math.radians(89.99999999999), 0, phi + theta * y / Y))

def getYawResidual(x, X, theta):

    x = linearConstraint(X, -X, x)
    theta = math.radians(theta)

    return x * theta / (2 * X)

def getLateralDistance(h, phi, theta1, y, Y, x, X, theta2):

    return math.tan(getYawResidual(x, X, theta2)) * getHorizontalDistance(h, phi, theta1, y, Y)

def getDirectDistance(h, phi, theta1, y, Y, x, X, theta2):

    return math.pow(getHorizontalDistance(h, phi, theta1, y, Y) ** 2 + getLateralDistance(h, phi, theta1, y, Y, x, X, theta2) ** 2, .5)

def isSignificant(row):

    if row[4] < 0.2: 
        
        return False
    
    return True

def extractDirectDistance(row, frame):

    x_shape, y_shape = frame.shape[1], frame.shape[0]

    x1 = int(row[0]*x_shape)
    y1 = int(row[1]*y_shape)
    x2 = int(row[2]*x_shape)
    y2 = int(row[3]*y_shape)

    h = 25
    phi = 80
    theta1 = 34.3
    y = y_shape - (y1 + y2) / 2
    Y = y_shape
    x = (x1 + x2) / 2 - x_shape /2
    X = x_shape / 2
    theta2 = 60

    return getDirectDistance(h, phi, theta1, y, Y, x, X, theta2)

def plot_boxes(model, results, frame):
    
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    cubesLst = sorted(filter(isSignificant, cord), key=lambda x: extractDirectDistance(x, frame), reverse=False)

    for i in range(n):
        row = cubesLst[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        #print(row[4])
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        bgr = (0, 0, 255) if i == 0 else (0, 255, 0)# color of the box
        classes = model.names # Get the name of label index
        label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.

        idStr = "Cube ID: "+ (str) (i)

        h = 25
        phi = 80
        theta1 = 34.3
        y = y_shape - max(y1, y2)
        Y = y_shape
        x = (x1 + x2) / 2 - x_shape /2
        X = x_shape / 2
        theta2 = 60

        print(idStr)
        print(f"Horizontal Distance: {getHorizontalDistance(h, phi, theta1, y, Y)}")
        print(f"Yaw Residual: {getYawResidual(x, X, theta2)}")
        print(f"Lateral Distance: {getLateralDistance(h, phi, theta1, y, Y, x, X, theta2)}")
        print(f"Direct Distance: {getDirectDistance(h, phi, theta1, y, Y, x, X, theta2)}\n")

        cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
        cv.putText(frame, idStr, (x1, y1), label_font, 0.9, bgr, 2) #Put a label over box.; classes[labels[i]] vs "cube"
        
    return frame

start_time = time()
frame = plot_boxes(model, (labels, cord), frame)
end_time = time()
fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
print(f"Frames Per Second: {fps}")
print(f"Total Time: {np.round(end_time - init_time, 3)}")

# Results
results.print()  
cv.imshow('power cubes', frame)

cv.waitKey(0)
cv.destroyAllWindows()