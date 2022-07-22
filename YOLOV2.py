import cv2 as cv
import torch

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default

# Images
img1 = cv.imread('powerCubes.jfif')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1]  # batch of images

# Inference
results = model(imgs, size=640)  # includes NMS

# Results
results.print()  
results.show()  # or .show()