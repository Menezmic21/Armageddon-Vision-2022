# Rewritten by Sri and Dawson, detects cubes pretty good
import cv2 as cv
import numpy as np
import threading
from networktables import NetworkTables, NetworkTablesInstance

def CV():
  cap = cv.VideoCapture(0)
  if not cap.isOpened():
    print("Oooga booga ur camera dead bruh")
    exit()
  while True:
    ret, frame = cap.read()

    frame = cv.flip(frame, 0)

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

    cv.drawContours(frame, [box], 0, (0, 0, 255), 3)

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv.destroyAllWindows()


def main():
    print("About to Connect to Network Tables")
    team = 1477
    ip = "10.14.77.2"
    notified = False
    condition = threading.Condition()
    
    # notify as soon as connection is made
    def connection_listener(connected, info):
        with condition:
            notified = False
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