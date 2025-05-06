#include <iostream>
#include <opencv2/opencv.hpp>
#include "proj.h"

using namespace std;
using namespace cv;

int main() {
    Mat source = imread("C:/Users/diana/Desktop/anul3/sem2/PI/Project/photo1.bmp");
    if (source.empty()) {
        cout << "Can't open image!" << endl;
        return -1;
    }

    Mat grayscale = rgb_2_grayscale(source);
    imshow("Grayscale", grayscale);

    Mat blurred = gaussian_blur(grayscale);
    imshow("Gaussian Blur", blurred);

    Mat float_blurred;
    blurred.convertTo(float_blurred, CV_32F);
    GradientData gradient_data = high_pass_filter(float_blurred);

    Mat grad_magnitude_display;
    normalize(gradient_data.magnitude, grad_magnitude_display, 0, 255, NORM_MINMAX);
    grad_magnitude_display.convertTo(grad_magnitude_display, CV_8UC1);
    imshow("Gradient Magnitude", grad_magnitude_display);

    Mat suppressed = non_max_suppression(gradient_data);
    Mat suppressed_display;
    normalize(suppressed, suppressed_display, 0, 255, NORM_MINMAX);
    suppressed_display.convertTo(suppressed_display, CV_8UC1);
    imshow("Non-Max Suppression", suppressed_display);

    Mat hysteresis = hysteresis_thresholding(suppressed);
    imshow("Canny final", hysteresis);

    vector<vector<int>> accumulator;
    Mat output = grayscale.clone();
    cvtColor(output, output, COLOR_GRAY2BGR);
    vector<pair<float, float>> lines = hough_transform(hysteresis,accumulator);
    display_hough_space(accumulator);
    draw_hough_lines(output, lines);
    imshow("Hough Lines final", output);
    
    waitKey(0);
    return 0;
}
