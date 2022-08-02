import numpy as np
import cv2 as cv
from sympy import true

def getAspectRatio(cnt):

	x, y, w, h = cv.boundingRect(cnt)

	return w/h

def getSquareness(cnt):

	return pow(getAspectRatio(cnt) - 1, 2)

def getArea(cnt):

	x, y, w, h = cv.boundingRect(cnt)

	return w * h

def getRectangleness(cnt):

	cntArea = cv.contourArea(cnt)
	boxArea = getArea(cnt)

	return abs((cntArea - boxArea)/ ((cntArea + boxArea) / 2))

cap = cv.VideoCapture('cubeWalkthrough.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()

	# Reading image
	_, baseIMG = cap.read()
	_, copyIMG = cap.read()

	# Reading same image in another variable and
	# converting to gray scale.
	_, grayIMG = cap.read()
	grayIMG = cv.cvtColor(grayIMG, cv.COLOR_BGR2GRAY)

	# Converting image to a binary image
	# (black and white only image).
	frame_HSV = cv.cvtColor(baseIMG, cv.COLOR_BGR2HSV)
	h, s, v = cv.split(frame_HSV)
	tol = 0.
	kernel = np.ones((5,5), np.uint8)
	threshold = cv.inRange(frame_HSV, (23, 41, 133), (40, 255, 255))#(26.5 - tol * 180, 50 - tol * 255, 175.95 - tol * 255), (38.5 + tol * 180, 200 + tol * 255, 255 + tol * 255))
	# threshold = cv.dilate(threshold, kernel, iterations=3)
	threshold = cv.erode(threshold, kernel, iterations=3)
	threshold = cv.dilate(threshold, kernel, iterations=3+3)
	threshold = cv.erode(threshold, kernel, iterations=3)

	# Masking
	maskedIMG = cv.bitwise_and(baseIMG, baseIMG, mask=threshold)
	#maskedSAT = cv.bitwise_and(s, s, mask=threshold)

	# Edge Detection
	grayMaskedIMG = cv.cvtColor(maskedIMG, cv.COLOR_BGR2GRAY)
	grayMaskedIMG = cv.dilate(grayMaskedIMG, kernel, iterations=5)
	grayMaskedIMG = cv.erode(grayMaskedIMG, kernel, iterations=5)
	normalizedIMG = np.zeros((800, 800))
	normalizedIMG = cv.normalize(grayMaskedIMG,  normalizedIMG, 0, 255, cv.NORM_MINMAX, mask=threshold)
	#blurredNorm = cv.GaussianBlur(normalizedIMG, (25, 25), 0)
	blurredNorm = cv.bilateralFilter(normalizedIMG, 11, 71, 21)
	maskedBLUR = cv.bitwise_and(blurredNorm, blurredNorm, mask=threshold)
	edgedIMG = cv.Canny(maskedBLUR, 0, 200)
	edgedIMG = cv.dilate(edgedIMG, kernel, iterations=1)

	# Detecting shapes in image by selecting region
	# with same colors or intensity.
	contours,hierarchy=cv.findContours(edgedIMG, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	# Searching through every region selected to
	# find the required polygon.

	#print(hierarchy)

	simpleContours = []

	index = 0
	for cnt in contours:
		area = cv.contourArea(cnt)

		# Shortlisting the regions based on there area.
		if area > 3000:
			approx = cv.convexHull(cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True))

			# Checking if the no. of sides of the selected region is greater than 4.
			if(len(approx) >= 4 and hierarchy[0][index][-1] != -1 and cv.isContourConvex(approx)):

				# Simplified Contour
				epsilon = 0.05*cv.arcLength(approx,True)
				approx2 = cv.approxPolyDP(approx,epsilon,True)

				if len(approx2) == 4: 
					simpleContours.append((hierarchy[0][index][-1], approx2))
					cv.drawContours(copyIMG, [approx2], 0, (0, 0, 255), 2) # Squarest in blob, BGR

				#print(len(approx2))

				#OG Contour
				#cv.drawContours(copyIMG, [approx], 0, (0, 0, 255), 2)

		index += 1

	simpleContours = sorted(simpleContours, key=lambda x: x[0], reverse=False)

	if len(simpleContours) > 0:

		index = 1
		simpleContourHierarchy = [[simpleContours[0][1]]]
		parents = 0

		while index < len(simpleContours):

			prevParent = simpleContours[index-1][0]
			currentParent = simpleContours[index][0]

			if currentParent == prevParent:

				simpleContourHierarchy[parents].append(simpleContours[index][1])

			else:

				simpleContourHierarchy.append([])
				parents += 1
				simpleContourHierarchy[parents].append(simpleContours[index][1])

			index += 1

		numCubes = len(simpleContourHierarchy)
		print((str) (numCubes) + " cubes detected!")

		# for cnt in simpleContourHierarchy[1]:
		# 	cv.drawContours(copyIMG, [cnt], 0, (0, 255, 0), 2)

		squareHierarchy = []

		for parent in simpleContourHierarchy:

			squarenessLst = sorted(parent, key=lambda x: getSquareness(x), reverse=False)
			squareHierarchy.append(squarenessLst[0])
			cv.drawContours(copyIMG, [squarenessLst[0]], 0, (0, 255, 0), 2) # Squarest in blob, BGR
			
			squarestContour = squarenessLst[0]
			x, y, w, h = cv.boundingRect(squarestContour)

			font = cv.FONT_HERSHEY_SIMPLEX
			cv.putText(copyIMG, (str) (((int)(getSquareness(squarestContour) * 100))/100), (x,y), font, 1, (0, 255, 0), 2, cv.LINE_AA)

		closenessLst = sorted(squareHierarchy, key=lambda x: getArea(x), reverse=True)
		#print(getRectangleness(closenessLst[0]))
		if getRectangleness(closenessLst[0]) < .25: cv.drawContours(copyIMG, [closenessLst[0]], 0, (255, 0, 0), 2) # Closest blob, BGR

	# Showing the image along with outlined arrow.
	cv.imshow('polygons', copyIMG)
  
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()