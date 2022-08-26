
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;

double diagonalFOV = 114.184;

int resolution[2] = {1280, 720};
double scale = 0.7;
double boxWidth = 13 / 12;
double PI = 3.1415926535897932384626433832795028841971;
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}

int main(const int argc, const char **argv)
{
	int team = 1477;
	string ip = "10.14.77.2";
	bool notified = false;

	float width = resolution[0] * scale;
	float height = resolution[1] * scale;

	Mat frame;
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_WIDTH, width);
	cap.set(CAP_PROP_FRAME_HEIGHT, height);
	namedWindow("Display window");

	double diagonalPixelLength = sqrt(pow(width, 2) + pow(height, 2));
	double pixelDegree = diagonalFOV / diagonalPixelLength;

	if (!cap.isOpened())
	{
		std::cout << ("Oooga booga ur camera dead bruh");
		exit(-1);
	}
	while (true)
	{
		cap >> frame;
		if (frame.empty())
			break;
		// frame = cv.flip(frame, 0)
		Mat hsv;
		Mat mask;
		cvtColor(frame, hsv, COLOR_BGR2HSV);
		inRange(hsv, Scalar(0, 200, 200), Scalar(80, 255, 255), mask);

		vector<vector<Point>> contours1;
		vector<Vec4i> hierarchy1;
		findContours(mask, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_SIMPLE);

		std::sort(contours1.begin(), contours1.end(), compareContourAreas);
		std::vector<cv::Point> biggestContour = contours1[contours1.size() - 1];
		drawContours(frame, biggestContour, 0, Scalar(0, 255, 0), 2);

		float left = width;
		float right = 0;
		float meanX = 0;
		float meanY = 0;

		for (int x; x = 0; x++)
		{
			Point point = biggestContour[x];
			if (point.x > right)
			{
				right = point.x;
			}
			if (point.x < left)
			{
				left = point.x;
			}
			meanX += point.x;
			meanY += point.y;
		}
		meanX /= 4;
		meanY /= 4;

		float yawResidual = (meanX - width / 2) * pixelDegree;
		float pitchResidual = (meanY - height / 2) * pixelDegree;

		float pixelWidth = right - left;
		float distance = 0.0;

		if (pixelWidth > 0)
		{
			distance = boxWidth / 2 / tan((pixelWidth * pixelDegree * PI) / 360);
		}

		cout << "Screen PositionK ", meanX, meanY;
		cout << "Distance: ", distance, "ft";
		cout << "Pitch Residual: ", pitchResidual;
		cout << "Yaw Residual: ", yawResidual;

		imshow("Display window", frame);

		waitKey(1);
	}
	cap.release();
	destroyAllWindows();
	return 0;
}