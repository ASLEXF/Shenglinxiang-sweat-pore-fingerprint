#pragma once

#include <Windows.h>

#ifndef CUDA_RUNTIME
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "curand_kernel.h"
#endif // !CUDA_RUNTIME

#ifndef OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#endif // !OPENCV

#ifdef CUDA_KERNEL_EXPORTS
#define CUDA_KERNEL_API __declspec(dllexport)
#else
#define CUDA_KERNEL_API __declspec(dllimport)
#endif // CUDA_KERNEL_EXPORTS

#define CHECK_ERROR(call){\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}

#define uint32 unsigned int 
#define warp_size 32

#ifndef GABOR_FILTER_PARAMETERS
#define GABOR_FILTER_PARAMETERS
#define GABOR_SIZE      10
#define GABOR_ANGLE_NUM 36
#define GABOR_FREQ_NUM  20
#endif // !GABOR_FILTER_PARAMETERS

using namespace cv;
using namespace cv::cuda;

extern "C" CUDA_KERNEL_API void add_image(Mat a, Mat b, Mat dst, Size size);
extern "C" CUDA_KERNEL_API void mul_image(Mat a, Mat b, Mat dst, Size size);
extern "C" CUDA_KERNEL_API void div_double(Mat a, double b, Mat dst, Size size);
extern "C" CUDA_KERNEL_API void div_image(Mat a, Mat b, Mat dst, Size size);
extern "C" CUDA_KERNEL_API void if_image(Mat a, double lessThan, double result1, 
    double result2, Mat dst, Size size);
extern "C" CUDA_KERNEL_API void cpy_image(GpuMat dst, GpuMat src, int x, int y, Size size);
extern "C" CUDA_KERNEL_API void generate_gabor_filter(GpuMat dst, double maxF, 
    double minF, int i, int j, Size size);
extern "C" CUDA_KERNEL_API void generate_gabor_filters(Mat dst, double maxF, 
    double minF, Size fil_size);
extern "C" CUDA_KERNEL_API void image_core_density(Mat dst, Point2d center, 
    double maxF, double minF, double min_distance, Size size);
extern "C" CUDA_KERNEL_API void generate_ridge_layer(Mat dst, Mat ridge, Mat orientation, 
    Mat density, Mat filter, double maxF, double minF, Size size);