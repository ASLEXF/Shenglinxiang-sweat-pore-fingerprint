#pragma once
#include <windows.h>
#include <vector>
#include <time.h>
#include <cmath>
#include <chrono>

#ifndef OPENCV_HEADS
#define OPENCV_HEADS
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/utils/logger.hpp>
#endif // !OPENCV_HEADS

#ifndef CUDA_RUNTIME
#define CUDA_RUNTIME
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif // !CUDA_RUNTIME

#include "cuda_kernel.h"

#define PI              CV_PI

#define DEFAULT         0
#define ARCH            1
#define TENTED_ARCH     2
#define WHORL           3
#define LEFT_LOOP       4
#define RIGHT_LOOP      5
#define TWIN_LOOP       6

#ifndef GABOR_FILTER_PARAMETERS
#define GABOR_FILTER_PARAMETERS
#define GABOR_SIZE      10
#define GABOR_ANGLE_NUM 36
#define GABOR_FREQ_NUM  20
#endif // !GABOR_FILTER_PARAMETERS

using namespace cv;
using namespace cv::cuda;
using namespace std;
using namespace chrono;

class fp {
public:
    fp(void);
    //	~fp(void);
    void set_number(const unsigned int n);
    void set_fp_size(const int width, const int length); // cm
    void set_pattern(const int pattern);
    void set_resolution(const unsigned int input);
    void set_image_size(const int width, const int height);
    void set_shape(int l, int r, int t, int m, int b);
    void set_points(int nc, Point2d* c, int nd, Point2d* d, int nl = 0, Point2d* l = NULL);
    void set_orientation_correction(int piece_num);

    void generate_filter();
    void generate_shape();
    void generate_orientation();
    void generate_density();
    void generate_ridge();
    void generate_master_fp();

    void show_fp_size();
    void show_image_size();

    void show_shape();
    void show_density();
    void show_orientation();
    void show_ridge();
    void show_master_fp();
    void show_gabor_filters();
    void show_all();

    double orientation_correcting(double alpha);

    void save_img();
    void save_as(string filename);

private:
    unsigned int number;
    int fp_width, fp_length; // 指纹宽度、长度
    int pattern; // 纹形: 1:arch, 2:tented arch, 3:whorl, 4:left loop, 5:right loop, 6:twin loop
    unsigned int L, R, T, M, B;	// 五参数模型：left, right, top, middle, bottom
    unsigned int reso; // 传感器分辨率(dpi)
    double minF, maxF; // 脊线密度
    double correction[9]; // 对方向场的方向矫正

    Size size;
    int nCore, nDelta, nLoop;
    Point2d * cores, * deltas, * loops;

    Mat shape;
    Mat density;
    Mat orientation;
    Mat ridge;
    Mat master_fp;
    Mat gabor_filters;
};

void triangle(Mat& img, Point center, double distance);
