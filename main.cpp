#include <iostream>
#include <opencv2/opencv.hpp>
#include "proj.h"

using namespace std;
using namespace cv;

int main() {
    Mat source = imread("C:/Users/diana/Desktop/anul3/sem2/PI/Project/photo5.bmp");
    if (source.empty()) {
        cout << "Can't open image!" << endl;
        return -1;
    }

    Mat grayscale = rgb_2_grayscale(source);
    //imshow("Grayscale", grayscale);

    Mat blurred = gaussian_blur(grayscale);
    //imshow("Gaussian Blur", blurred);

    Mat float_blurred;
    blurred.convertTo(float_blurred, CV_32F);
    GradientData gradient_data = high_pass_filter(float_blurred);

    Mat grad_magnitude_display;
    normalize(gradient_data.magnitude, grad_magnitude_display, 0, 255, NORM_MINMAX);
    grad_magnitude_display.convertTo(grad_magnitude_display, CV_8UC1);
    //imshow("Gradient Magnitude", grad_magnitude_display);

    Mat suppressed = non_max_suppression(gradient_data);
    Mat suppressed_display;
    normalize(suppressed, suppressed_display, 0, 255, NORM_MINMAX);
    suppressed_display.convertTo(suppressed_display, CV_8UC1);
    //imshow("Non-Max Suppression", suppressed_display);

    Mat hysteresis = hysteresis_thresholding(suppressed);
    imshow("Custom Canny", hysteresis);

    vector<vector<int>> accumulator;
    Mat output = grayscale.clone();
    cvtColor(output, output, COLOR_GRAY2BGR);
    vector<pair<float, float>> lines = hough_transform(hysteresis,accumulator);
    display_hough_space(accumulator);
    draw_hough_lines(output, lines);
    imshow("Custom Hough Lines", output);


    // ----- OpenCV Implementation for comparison -----
    Mat gray_cv;
    cvtColor(source, gray_cv, COLOR_BGR2GRAY);
    Mat blurred_cv;
    GaussianBlur(gray_cv, blurred_cv, Size(5, 5), 1.4);

    // === Gradient & magnitudine ===
    Mat grad_x, grad_y;
    Sobel(blurred_cv, grad_x, CV_32F, 1, 0, 3);
    Sobel(blurred_cv, grad_y, CV_32F, 0, 1, 3);

    Mat magnitude;
    magnitude.create(blurred_cv.size(), CV_32F);

    for (int i = 0; i < blurred_cv.rows; ++i) {
        for (int j = 0; j < blurred_cv.cols; ++j) {
            float gx = grad_x.at<float>(i, j);
            float gy = grad_y.at<float>(i, j);
            magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);
        }
    }
    Mat magnitude_norm;
    normalize(magnitude, magnitude_norm, 0, 255, NORM_MINMAX);
    magnitude_norm.convertTo(magnitude_norm, CV_8UC1);

    // === Histogram ===
    vector<int> hist(256, 0);
    int nonZero = 0;
    for (int i = 1; i < magnitude_norm.rows - 1; ++i) {
        for (int j = 1; j < magnitude_norm.cols - 1; ++j) {
            uchar val = magnitude_norm.at<uchar>(i, j);
            if (val > 0) {
                hist[val]++;
                nonZero++;
            }
        }
    }
    float p = 0.95f; // top 5%
    int cumulative = 0, highT = 0;
    for (int i = 0; i < 256; ++i) {
        cumulative += hist[i];
        if (cumulative >= nonZero * p) {
            highT = i;
            break;
        }
    }
    int lowT = static_cast<int>(0.4 * highT);
    Mat edges;
    Canny(blurred_cv, edges, lowT, highT, 3, true);
    imshow("OpenCV Canny", edges);

    // Hough transform OpenCV
    int edgePixels = countNonZero(edges);
    int threshold = max(60, edgePixels / 100);

    vector<Vec2f> lines_cv;
    HoughLines(edges, lines_cv, 1, CV_PI / 180, threshold);
    Mat hough_grayscale;
    cvtColor(gray_cv, hough_grayscale, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines_cv.size(); ++i) {
        float rho = lines_cv[i][0];
        float theta = lines_cv[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        line(hough_grayscale, pt1, pt2, Scalar(0, 255, 0), 1);
    }
    imshow("OpenCV Hough Lines", hough_grayscale);


    // ----- Comparison & Metrics -----
    compare_hough_lines(lines, lines_cv, grayscale);

    waitKey(0);
    return 0;
}
