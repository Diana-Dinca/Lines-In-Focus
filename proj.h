#ifndef PROJ_H
#define PROJ_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
struct GradientData {
    Mat magnitude;
    Mat direction;
};

Mat rgb_2_grayscale(Mat source);

//Canny
Mat applyKernel(const Mat& source, const vector<vector<float>>& kernel);
Mat gaussian_blur(Mat source);
GradientData high_pass_filter(Mat source);
Mat non_max_suppression(GradientData source);
//pair<int, int> compute_thresholds(const Mat& norm);
Mat hysteresis_thresholding(Mat input);

//Hough
vector<pair<float, float>> hough_transform(Mat input, vector<vector<int>>& accumulator_out);
void draw_hough_lines(Mat& image,vector<pair<float, float>> lines);
void display_hough_space(const std::vector<std::vector<int>>& accumulator);

#endif
