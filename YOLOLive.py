import cv2 as cv
import torch
import numpy as np
from time import time

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default

def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    #frame = [torch.tensor(frame)]
    results = model([frame])
    #print(results)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def plot_boxes(model, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        bgr = (0, 255, 0) # color of the box
        classes = model.names # Get the name of label index
        label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.

        cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
        cv.putText(frame, "Cube", (x1, y1), label_font, 0.9, bgr, 2) #Put a label over box.; classes[labels[i]] vs "cube"
        
    return frame

cap = cv.VideoCapture('inputVideos/cubeWalkthrough.mp4')
# Obtain frame size information using get() method
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
FPS = 60
# Initialize video writer object
output = cv.VideoWriter('oputputVideos/output2.avi', cv.VideoWriter_fourcc('M','J','P','G'), FPS, frame_size)
while(cap.isOpened()):
    ret, frame = cap.read()

    frame2 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    start_time = time() # We would like to measure the FPS.
    results = score_frame(frame2, model) # Score the Frame
    frame = plot_boxes(model, results, frame) # Plot the boxes.
    end_time = time()
    fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
    print(f"Frames Per Second : {fps}")
    #cv.imshow('power cubes', frame)
    output.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()

cv.destroyAllWindows()