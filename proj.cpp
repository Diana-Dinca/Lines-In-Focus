#include "proj.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <stack>

using namespace cv;
using namespace std;

Mat rgb_2_grayscale(Mat source) {
    /* Converts an RGB image to a grayscale image, returning the result image. */

    int rows = source.rows;
    int cols = source.cols;
    Mat gray(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            gray.at<uchar>(i, j) =(pixel[2] + pixel[1] + pixel[0]) / 3;
        }
    }
    return gray;
}

Mat applyKernel(const Mat& source, const vector<vector<float>>& kernel) {
    /* Applies the kernel by multiplying its values with the corresponding
     * neighborhood values in the image. The result of the convolution is stored
     * new output image.
    */

    Mat result = Mat::zeros(source.size(), CV_32F);
    int ksize = kernel.size();
    int kcenter = ksize / 2;
    int rows = source.rows;
    int cols = source.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0;
            for (int ki = 0; ki < ksize; ki++) {
                for (int kj = 0; kj < ksize; kj++) {
                    int ni = i + ki - kcenter;
                    int nj = j + kj - kcenter;
                    // Replicate border pixels
                    ni = min(max(ni, 0), rows - 1);
                    nj = min(max(nj, 0), cols - 1);
                    sum += source.at<float>(ni, nj) * kernel[ki][kj];
                }
            }
            result.at<float>(i, j) = sum;
        }
    }
    return result;
}

Mat gaussian_blur(Mat source) {
    /* Applies Gaussian blur to reduce noise, while preserving edges. */
    source.convertTo(source, CV_32F);

    vector<vector<float>> kernel = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    float sum = 273.0f;

    // Normalize the kernel values to preserve the local average of the pixels
    for (int i = 0; i < kernel.size(); i++) {
        for (int j = 0; j < kernel[i].size(); j++) {
            kernel[i][j] /= sum;
        }
    }

    Mat blurred = applyKernel(source, kernel);
    blurred.convertTo(blurred, CV_8UC1);
    return blurred;
}

GradientData high_pass_filter(Mat source) {
    /* Computes the gradient magnitude and gradient direction of an image, using Sobel filters,
     * a type of high-pass filter that highlights areas with rapid intensity change.
    */
    vector<vector<float>> sobelX = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    vector<vector<float>> sobelY = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    Mat gradX = applyKernel(source, sobelX);
    Mat gradY = applyKernel(source, sobelY);

    GradientData result;
    result.magnitude = Mat::zeros(source.size(), CV_32FC1);
    result.direction = Mat::zeros(source.size(), CV_32FC1);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            float gx = gradX.at<float>(i, j);
            float gy = gradY.at<float>(i, j);
            result.magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);
            result.direction.at<float>(i, j) = atan2(gy, gx);
        }
    }

    return result;
}

Mat non_max_suppression(GradientData grad) {
    /* Compares the magnitude of each pixel to the magnitude of its neighbors along the
     * gradient direction. If the pixel is not the local maximum, it is suppressed.
    */
    Mat suppressed = Mat::zeros(grad.magnitude.size(), CV_32FC1);

    for (int i = 1; i < grad.magnitude.rows - 1; ++i) {
        for (int j = 1; j < grad.magnitude.cols - 1; ++j) {
            float angle = grad.direction.at<float>(i, j);
            float mag = grad.magnitude.at<float>(i, j);

            // Convert to degrees in the range [0, 180)
            angle = angle * 180.0 / CV_PI;
            if (angle < 0) angle += 180;

            float m1 = 0, m2 = 0;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                m1 = grad.magnitude.at<float>(i, j - 1);
                m2 = grad.magnitude.at<float>(i, j + 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                m1 = grad.magnitude.at<float>(i - 1, j + 1);
                m2 = grad.magnitude.at<float>(i + 1, j - 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                m1 = grad.magnitude.at<float>(i - 1, j);
                m2 = grad.magnitude.at<float>(i + 1, j);
            } else if (angle >= 112.5 && angle < 157.5) {
                m1 = grad.magnitude.at<float>(i - 1, j - 1);
                m2 = grad.magnitude.at<float>(i + 1, j + 1);
            }

            if (mag >= m1 && mag >= m2) {
                suppressed.at<float>(i, j) = mag;
            }
        }
    }

    return suppressed;
}

Mat hysteresis_thresholding(Mat input) {
    /* It performs hysteresis thresholding to decide which gradient magnitude
     * pixels become edges in the final output.
    */
    int rows = input.rows;
    int cols = input.cols;

    Mat norm;
    input.convertTo(norm, CV_8UC1, 255.0 / 1024.0);

    vector<int> hist(256, 0);
    int nonZero = 0;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            uchar val = norm.at<uchar>(i, j);
            if (val > 0) {
                hist[val]++;
                nonZero++;
            }
        }
    }

    // The high threshold will capture only the strongest 5% of edges
    float p = 0.95f;
    int cumulative = 0, highThreshold = 0;
    for (int i = 0; i < 256; ++i) {
        cumulative += hist[i];
        if (cumulative >= nonZero * p) {
            highThreshold = i;
            break;
        }
    }
    int lowThreshold = static_cast<int>(0.4f * highThreshold);

    Mat edges = Mat::zeros(rows, cols, CV_8UC1);
    queue<Point> q;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            uchar val = norm.at<uchar>(i, j);
            if (val >= highThreshold) {
                edges.at<uchar>(i, j) = 255; //strong edge
                q.push(Point(j, i));
            } else if (val >= lowThreshold) {
                edges.at<uchar>(i, j) = 128; //weak edge
            }
        }
    }

    const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    // Strong edges "activate" weak edges connected to them (8-connected neighbors)
    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        for (int k = 0; k < 8; ++k) {
            int ni = p.y + dy[k];
            int nj = p.x + dx[k];
            if (edges.at<uchar>(ni, nj) == 128) {
                // These connected weak edges are upgraded to strong and also pushed further
                edges.at<uchar>(ni, nj) = 255;
                q.push(Point(nj, ni));
            }
        }
    }

    // All non-strong edges are suppressed to 0.
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (edges.at<uchar>(i, j) != 255)
                edges.at<uchar>(i, j) = 0;

    return edges;
}

vector<pair<float, float>> hough_transform(Mat input, vector<vector<int>>& accumulator_out) {
    int width = input.cols;
    int height = input.rows;

    int diag = sqrt(width * width + height * height);
    int rho_max = diag;
    int rho_steps = 2 * rho_max;
    int theta_steps = 180;

    vector<vector<int>> accumulator(rho_steps, vector<int>(theta_steps, 0));
    vector<pair<float, float>> lines;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (input.at<uchar>(y, x) == 255) { //edge pixel
                for (int t = 0; t < theta_steps; t++) { //for each angle we compute ρ
                    float theta = t * CV_PI / 180;
                    int rho = cvRound(x * cos(theta) + y * sin(theta)) + rho_max; //add rho_max to shift negatives to valid indices
                    if (rho >= 0 && rho < rho_steps)
                        accumulator[rho][t]++; //this pixel votes for this line
                }
            }
        }
    }
    int edgePixels = countNonZero(input);
    int threshold = max(60, edgePixels / 100); // dynamic threshold = at least 50, or 1% of edge pixels

    for (int r = 0; r < rho_steps; r++) {
        for (int t = 0; t < theta_steps; t++) {
            if (accumulator[r][t] > threshold) { //wherever votes exceed threshold, it’s a line
                float rho = r - rho_max;
                float theta = t * CV_PI / 180;
                lines.emplace_back(rho, theta);
            }
        }
    }
    accumulator_out = accumulator;
    return lines;
}

void display_hough_space(const vector<vector<int>>& accumulator) {
    /* Visualize the Hough Transform accumulator as an image */
    int rows = accumulator.size();
    int cols = accumulator[0].size();

    Mat hough_image(rows, cols, CV_32F);
    float max_val = 0;

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            hough_image.at<float>(i, j) = static_cast<float>(accumulator[i][j]);
            max_val = max(max_val, hough_image.at<float>(i, j)); // keep track of the maximum value to normalize
        }

    Mat hough_display;
    hough_image.convertTo(hough_display, CV_8U, 255.0 / max_val);

    resize(hough_display, hough_display, Size(400, 400));
    imshow("Hough Space", hough_display);
}

void draw_hough_lines(Mat& image, vector<pair<float, float>> lines) {
    for (const auto& line : lines) {
        float rho = line.first;
        float theta = line.second;

        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point pt1(cvRound(x0 + 1000*(-b)), cvRound(y0 + 1000*(a)));
        Point pt2(cvRound(x0 - 1000*(-b)), cvRound(y0 - 1000*(a)));
        cv::line(image, pt1, pt2, Scalar(0, 0, 255), 1);
    }
}


