import numpy as np
import cv2 as cv
from sympy import true

def getAspectRatio(cnt):

	x, y, w, h = cv.boundingRect(cnt)

	return w/h

# Reading image
baseIMG = cv.imread('powerCubes.jfif', cv.IMREAD_COLOR)
copyIMG = cv.imread('powerCubes.jfif', cv.IMREAD_COLOR)

# Reading same image in another variable and
# converting to gray scale.
grayIMG = cv.imread('powerCubes.jfif', cv.IMREAD_GRAYSCALE)

# Converting image to a binary image
# (black and white only image).
frame_HSV = cv.cvtColor(baseIMG, cv.COLOR_BGR2HSV)
h, s, v = cv.split(frame_HSV)
tol = 0.2
kernel = np.ones((5,5), np.uint8)
threshold = cv.inRange(frame_HSV, (27.5 - tol * 180, 138.975 - tol * 255, 175.95 - tol * 255), (30 + tol * 180, 165.24 + tol * 255, 255 + tol * 255))
threshold = cv.dilate(threshold, kernel, iterations=3)
threshold = cv.erode(threshold, kernel, iterations=3)

# Masking
maskedIMG = cv.bitwise_and(baseIMG, baseIMG, mask=threshold)
#maskedSAT = cv.bitwise_and(s, s, mask=threshold)

# Edge Detection
grayMaskedIMG = cv.cvtColor(maskedIMG, cv.COLOR_BGR2GRAY)
grayMaskedIMG = cv.dilate(grayMaskedIMG, kernel, iterations=3)
grayMaskedIMG = cv.erode(grayMaskedIMG, kernel, iterations=3)
normalizedIMG = np.zeros((800, 800))
normalizedIMG = cv.normalize(grayMaskedIMG,  normalizedIMG, 0, 255, cv.NORM_MINMAX, mask=threshold)
#blurredNorm = cv.GaussianBlur(normalizedIMG, (25, 25), 0)
blurredNorm = cv.bilateralFilter(normalizedIMG, 11, 41, 21)
maskedBLUR = cv.bitwise_and(blurredNorm, blurredNorm, mask=threshold)
edgedIMG = cv.Canny(maskedBLUR, 10, 200)
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
	if area > 400:
		approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)

		# Checking if the no. of sides of the selected region is greater than 4.
		if(len(approx) >= 4 and hierarchy[0][index][-1] != -1):

			# Simplified Contour
			epsilon = 0.05*cv.arcLength(approx,True)
			approx2 = cv.approxPolyDP(approx,epsilon,True)

			simpleContours.append((hierarchy[0][index][-1], approx2))

			#print(len(approx2))

			#OG Contour
			#cv.drawContours(copyIMG, [approx], 0, (0, 0, 255), 2)

	index += 1

simpleContours = sorted(simpleContours, key=lambda x: x[0], reverse=False)

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

for parent in simpleContourHierarchy:

	squarenessLst = sorted(parent, key=lambda x: pow(getAspectRatio(x) - 1, 2), reverse=False)
	cv.drawContours(copyIMG, [squarenessLst[0]], 0, (0, 0, 255), 2) # Squarest in blob

# Showing the image along with outlined arrow.
cv.imshow('polygons', copyIMG)

# Exiting the window if 'q' is pressed on the keyboard.
if cv.waitKey(0) & 0xFF == ord('q'):
	cv.destroyAllWindows()