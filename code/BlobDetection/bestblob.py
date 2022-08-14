import cv2 as cv
import numpy as np
import threading
from networktables import NetworkTables, NetworkTablesInstance
from math import tan, radians, sqrt

diagonalFOV = 114.184
#armageddon camera = 68.5
#webcam on my computer = 114.184?

resolution = (1280,720)
scale = 0.7

boxWidth = 13/12

def CV():
  width = resolution[0]*scale
  height = resolution[1]*scale
  
  cap = cv.VideoCapture(0)
  cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
  
  diagonalPixelLength = sqrt(width**2+height**2)
  pixelDegree = diagonalFOV / diagonalPixelLength
  
  if not cap.isOpened():
    print("Oooga booga ur camera dead bruh")
    exit()
  while True:
    ret, frame = cap.read()

    #frame = cv.flip(frame, 0)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lowerbound = np.array([20, 80, 130])
    upperbound = np.array([60, 255, 255])

    mask = cv.inRange(hsv, lowerbound, upperbound)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    if len(sorted_contours) == 0:
      continue
    contour = sorted_contours[0]

    rect = cv.minAreaRect(contour)  
    box = cv.boxPoints(rect)
    box = np.int0(box)

    left = width
    right = 0
    meanX = 0
    meanY = 0
    
    for x in box:
      if (x[0] > right):
        right = x[0]
      if (x[0]<left):
        left = x[0]
      meanX += x[0]
      meanY += x[1]
    meanX /= 4
    meanY /= 4

    yawResidual = abs(meanX-width/2)*pixelDegree
    pitchResidual = abs(meanY-height/2)*pixelDegree
    
    pixelWidth = right-left
    distance = 0.0
    if (pixelWidth > 0):
      distance = boxWidth/2/tan(radians(pixelWidth*pixelDegree/2))

    if cv.waitKey(1) & 0xFF == ord('w'):
      print("Screen Position:",meanX,meanY)
      print("Distance:", distance,"ft")
      print("Pitch Residual:", pitchResidual)
      print("Yaw Residual:", yawResidual)
    
    
    cv.drawContours(frame, [box], 0, (0, 0, 255), 3)
    cv.imshow("frame", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv.destroyAllWindows()

def main():
    team = 1477
    ip = "10.14.77.2"
    notified = False
    condition = threading.Condition()
    
    # notify as soon as connection is made
    def connection_listener(connected, info):
        with condition:
            notified = True
            condition.notify()
    
    NetworkTables.initialize(server=ip)
    NetworkTables.addConnectionListener(connection_listener, immediateNotify=True)
    with condition:
        if not notified:
            condition.wait()

    print("Connected to Network Tables")
    ntinst = NetworkTablesInstance.getDefault()
    tb = ntinst.getTable("cube_detection")
    tb.getEntry("yawResidual").forceSetValue(1477)
    
    CV()
  

if __name__ == "__main__":
    main()
