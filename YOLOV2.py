import cv2 as cv
import numpy as np
import torch

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default

# Images
frame = cv.imread('inputVideos/powerCubes.jfif')
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

    return (x1 - x0)**2 + (y1 - y0)**2


def plot_boxes(model, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    bluredFrame = cv.bilateralFilter(frame, 7, 50, 50)
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
        miniFrame = frame[y1:y2, x1:x2]
        miniFrame = cv.cvtColor(miniFrame, cv.COLOR_BGR2GRAY)
        # miniFrame = cv.normalize(miniFrame,  miniFrame, 0, 255, cv.NORM_MINMAX)
        # ret, threshold = cv.threshold(miniFrame, 75, 255, cv.THRESH_BINARY_INV)
        # miniFrame = cv.inpaint(miniFrame, threshold, 3, cv.INPAINT_TELEA)
        miniFrame = cv.Canny(bluredFrame[y1:y2, x1:x2], 100, 200)
        kernel = np.ones((5,5), np.uint8)
        edges = cv.dilate(miniFrame, kernel, iterations=1)
        lsd = cv.createLineSegmentDetector(0)
        lines = lsd.detect(edges)[0]
        print(type(lines))
        lines = np.array([line for line in lines if getLineLength(line) > 25])
        print(type(lines))
        
        #x, y, w, h = cv.boundingRect(edges)
        contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cv.convexHull(cnt) for cnt in contours]
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        rect = cv.minAreaRect(contours[0])
        box = cv.boxPoints(rect)
        box = np.int0(box)

        miniFrame = cv.cvtColor(miniFrame, cv.COLOR_GRAY2BGR)

        for cnt in contours:
            cv.drawContours(miniFrame, [cnt], 0, (255, 0, 255), 1)
        
        lsd.drawSegments(miniFrame, lines)

        left = box[1]
        right = box[3]
        top = box[2] 
        bottom = box[0]

        cv.circle(miniFrame, left, 8, (0, 50, 255), -1)
        cv.circle(miniFrame, right, 8, (0, 255, 255), -1)
        cv.circle(miniFrame, top, 8, (255, 50, 0), -1)
        cv.circle(miniFrame, bottom, 8, (255, 255, 0), -1)

        frame[y1:y2, x1:x2] = miniFrame

    return frame

frame = plot_boxes(model, (labels, cord), frame)

# Results
results.print()  
cv.imshow('power cubes', frame)

cv.waitKey(0)
cv.destroyAllWindows()