#include <iostream>
#include <stdio.h>

int main(const int argc, const char** argv) {
    int resolution[2] = {1280, 720};
    double diagonalFOV = 114.184;
    double scale = 0.7;
    double boxWidth = 13/12;
    std::printf("%f\n", boxWidth);

        
    double width = resolution[0]*scale;
    double height = resolution[1]*scale;

    // cap = cv.VideoCapture(0)
    // cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    // cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    return 0;
} 