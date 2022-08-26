# This file gets yaw and pitch residual and depth distance
import cv2 as cv
import numpy as np
import threading
from networktables import NetworkTables, NetworkTablesInstance
from math import tan, radians, sqrt

diagonalFOV = 114.184

resolution = (1280,720)
scale = 0.7
boxWidth = 13/12



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

    ntinst = NetworkTablesInstance.getDefault()
    tb = ntinst.getTable("cube_detection")
    tb.getEntry("depthDistance").forceSetValue(1477)
    tb.getEntry("pitchResidual").forceSetValue(1477)
    tb.getEntry("yawResidual").forceSetValue(1477)
    print("Connected to Network Tables")
    
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

        frame = cv.flip(frame, 0)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lowerbound = np.array([0, 200, 200])
        upperbound = np.array([80, 255, 255])

        mask = cv.inRange(hsv, lowerbound, upperbound)

        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        if len(sorted_contours) != 0:

            contour = sorted_contours[0]
            area  = cv.contourArea(contour)

            if area > frame.shape[0] * frame.shape[1] * 0:

                rect = cv.minAreaRect(contour)  
                box = cv.boxPoints(rect)
                box = np.int0(box)

                cv.drawContours(frame, [box], 0, (0, 0, 255), 3)

                left = width
                right = 0
                meanX = 0
                meanY = 0
            
                for x in box:
                    if (x[0] > right):
                        right = x[0]
                    if (x[0] < left):
                        left = x[0]   
                    meanX += x[0]
                    meanY += x[1]
                meanX /= 4
                meanY /= 4

                yawResidual = (meanX-width/2)*pixelDegree
                pitchResidual = (meanY-height/2)*pixelDegree
            
                pixelWidth = right-left
                distance = 0.0
                
                if (pixelWidth > 0):
                    distance = boxWidth/2/tan(radians(pixelWidth*pixelDegree/2))

                print("Screen Position:",meanX,meanY)
                print("Distance:", distance,"ft")
                
                print("Pitch Residual:", pitchResidual)
                '''tb.getEntry("pitchResidual").forceSetValue(pitchResidual)
                tb.getEntry("depthDistance").forceSetValue(distance)
                tb.getEntry("yawResidual").forceSetValue(yawResidual)'''
                print("Yaw Residual:", yawResidual)
                
            else: # If it can't see any cubes, return 0 so the PID won't do anything
                tb.getEntry("yawResidual").forceSetValue(0)
                tb.getEntry("depthDistance").forceSetValue(0)
                tb.getEntry("pitchResidual").forceSetValue(0)
                

        cv.imshow("frame", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()
