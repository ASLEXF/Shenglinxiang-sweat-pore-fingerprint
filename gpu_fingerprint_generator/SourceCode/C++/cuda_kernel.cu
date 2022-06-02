#define CUDA_KERNEL_EXPORTS
#include "cuda_kernel.h"

#define WRONG_ACCESS() { \
    printf("Error:%s,%d,",__FILE__,__LINE__);\
    printf("illegal memory access\b"); \
    exit(1); \
}
#define WRONG_TYPE() { \
    printf("Error:%s,%d,",__FILE__,__LINE__);\
    printf("unsupported image type\b"); \
    exit(1); \
}
#define max(x, y) \
    (x > y ? x : y)
#define min(x, y) \
    (x < y ? x : y)
#define isEqual3(a, b, c) \
    a == b ? (b == c ? true : false) : false
#define isEqual4(a, b, c, d) \
    a == b ? (b == c ? (c == d ? true : false) : false) : false

__global__ void generate_gabor_filter_kernel(PtrStepSz<double> dst, double maxF, double minF, int i, int j, Size size, double gamma = 1, double psi = 0) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        double theta = CV_PI * 2 / GABOR_ANGLE_NUM * i + CV_PI / 2;
        double F = (maxF - minF) / GABOR_FREQ_NUM * j + minF;
        double lambd = 1.0 / F;
        double sigma = sqrt(3.0 / (8.0 * F * F));

        double sigma_x = sigma;
        double sigma_y = sigma / gamma;
        double ex = -0.5 / (sigma_x * sigma_x);
        double ey = -0.5 / (sigma_y * sigma_y);
        double cscale = CV_PI * 2 / lambd;

        int d_w = size.width / 2, d_h = size.height / 2;
        int p_x = -x + d_w, p_y = -y + d_h;
        double _x = cos(theta) * p_x + sin(theta) * p_y;
        double _y = -sin(theta) * p_x + cos(theta) * p_y;
        dst(y, x) = exp((ex * _x * _x + ey * _y * _y) / 2.0 * sigma * sigma) * cos(cscale * _x + psi);
    }
}
__global__ void add_image_kernel(PtrStepSz<double> a, PtrStepSz<double> b, PtrStepSz<double> dst, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        dst(y, x) = a(y, x) + b(y, x);
    }
}
__global__ void mul_image_kernel(PtrStepSz<double> a, PtrStepSz<double> b, PtrStepSz<double> dst, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        dst(y, x) = a(y, x) * b(y, x);
    }
}
__global__ void div_double_kernel(PtrStepSz<double> a, double b, PtrStepSz<double> dst, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        dst(y, x) = a(y, x) / b;
    }
}
__global__ void div_image_kernel(PtrStepSz<double> a, PtrStepSz<double> b, PtrStepSz<double> dst, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        dst(y, x) = a(y, x) / b(y, x);
    }
}
__global__ void if_image_kernel(PtrStepSz<double> a, double lessThan, double result1, double result2, PtrStepSz<double> dst, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        if (dst(y, x) < lessThan) {
            dst(y, x) = result1;
        }
        else {
            dst(y, x) = result2;
        }
    }
}
__global__ void cpy_image_kernel(PtrStepSz<double> dst, PtrStepSz<double> src, int dst_x, int dst_y, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        dst(y + dst_y, x + dst_x) = src(y, x);
    }
}
__global__ void image_core_density_kernel(PtrStepSz<double> dst, Point2d center, double maxF, double minF, double min_distance, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        if (pow(center.x - x, 2) + pow(center.y - y, 2) < min_distance) {
            /*
            curandState localState;
            double rand_double = curand_normal_double(&localState);
            */
            dst(y, x) += (maxF - minF) / 2;
        }
    }
}
/*
__global__ void generate_orientation_kernel(PtrStepSz<double> dst, int nCore, Point2d cores, int nDelta, Point2d deltas, int pattern, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        double theta = 0;
        for (int k = 0; k < nCore; k++) {
            Vec2d v = cores[k] - Point2d(x, y);
            theta += orientation_correcting(atan2(v[1], v[0])) / nCore;
        }
        for (int k = 0; k < nDelta; k++) {
            Vec2d v = deltas[k] - Point2d(x, y);
            theta -= orientation_correcting(atan2(v[1], v[0])) / nDelta;

        }
        theta *= nDelta / 2.0;
        if (pattern == 1 || pattern == 2)
            theta -= CV_PI;
        dst(y, x) = theta;
    }
}
*/
__global__ void generate_ridge_layer_kernel(PtrStepSz<double> dst, PtrStepSz<double> ridge, PtrStepSz<double> orientation, PtrStepSz<double> density, PtrStepSz<double> filters, double maxF, double minF, Size size) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < size.width && y < size.height) {
        double theta = orientation(y, x) < 0 ? orientation(y, x) + CV_2PI : orientation(y, x);
        double F = density(y, x);
        int a = theta / CV_PI / 2 * GABOR_ANGLE_NUM;
        int b = (F - minF) / (maxF - minF) * 5;// *GABOR_FREQ_NUM;
        int w = size.width, h = size.height;
        int _x = int(x), _y = int(y);
        int width = min(min(2 * GABOR_SIZE + 1, w + GABOR_SIZE - _x), GABOR_SIZE + 1 + _x);
        int height = min(min(2 * GABOR_SIZE + 1, h + GABOR_SIZE - _y), GABOR_SIZE + 1 + _y);
        int fil_x_begin = max(GABOR_SIZE - _x, 0) + a * (2 * GABOR_SIZE + 1), fil_y_begin = max(GABOR_SIZE - _y, 0) + b * (2 * GABOR_SIZE + 1);
        int fng_x_begin = max(_x - GABOR_SIZE, 0), fng_y_begin = max(_y - GABOR_SIZE, 0);
        double sum = 0;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                sum += ridge(fng_y_begin + j, fng_x_begin + i) * filters(fil_y_begin + j, fil_x_begin + i);
            }
        }
        dst(y, x) = sum > 0;
    }
}
/*
__global__ void setup_kernel(unsigned long long* sobolDirectionVectors,
    unsigned long long* sobolScrambleConstants,
    curandStateScrambledSobol64* state)
{
    int id = threadIdx.x + blockIdx.x * warp_size;
    int dim = 3 * id;
//    Each thread uses 3 different dimensions 
    curand_init(sobolDirectionVectors + VECTOR_SIZE * dim,
        sobolScrambleConstants[dim],
        1234,
        &state[dim]);

    curand_init(sobolDirectionVectors + VECTOR_SIZE * (dim + 1),
        sobolScrambleConstants[dim + 1],
        1234,
        &state[dim + 1]);

    curand_init(sobolDirectionVectors + VECTOR_SIZE * (dim + 2),
        sobolScrambleConstants[dim + 2],
        1234,
        &state[dim + 2]);
}
*/
CUDA_KERNEL_API void generate_gabor_filter(GpuMat dst, double maxF, double minF, int i, int j, Size size) {
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        generate_gabor_filter_kernel << < blocksPerGrid, threadsPerBlock >> > (dst, maxF, minF, i, j, size);
    }
    else WRONG_TYPE();
    CHECK_ERROR(cudaDeviceSynchronize());
}
CUDA_KERNEL_API void generate_gabor_filters(Mat dst, double maxF, double minF, Size fil_size) {
    if (dst.type() == CV_64FC1) {
        GpuMat g_dst(dst);
        for (int i = 0; i < GABOR_ANGLE_NUM; i++) {
            for (int j = 0; j < GABOR_FREQ_NUM; j++) {
                GpuMat filter(fil_size, CV_64FC1);
                generate_gabor_filter(filter, maxF, minF, i, j, fil_size);
                cpy_image(g_dst, filter, fil_size.width * i, fil_size.height * j, fil_size);
            }
        }
        g_dst.download(dst);
    }
    else WRONG_TYPE();
}
CUDA_KERNEL_API void add_image(Mat a, Mat b, Mat dst, Size size) {
    assert(isEqual4(a.rows, b.rows, dst.rows, size.height) && isEqual4(a.cols, b.cols, dst.cols, size.width));
    assert(isEqual3(a.type(), b.type(), dst.type()));
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_a(a), g_b(b), g_dst(dst);
        add_image_kernel << < blocksPerGrid, threadsPerBlock >> > (g_a, g_b, g_dst, size);
        g_dst.download(dst);
    }
    else WRONG_TYPE();
}
CUDA_KERNEL_API void mul_image(Mat a, Mat b, Mat dst, Size size) {
    assert(isEqual4(a.rows, b.rows, dst.rows, size.height) && isEqual4(a.cols, b.cols, dst.cols, size.width));
    assert(isEqual3(a.type(), b.type(), dst.type()));
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_a(a), g_b(b), g_dst(dst);
        mul_image_kernel << < blocksPerGrid, threadsPerBlock >> > (g_a, g_b, g_dst, size);
        g_dst.download(dst);
    }
    else WRONG_TYPE();
    
}
CUDA_KERNEL_API void div_double(Mat a, double b, Mat dst, Size size) {
    assert(isEqual3(a.rows, dst.rows, size.height) && isEqual3(a.cols, dst.cols, size.width));
    assert(a.type() == dst.type());
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_a(a), g_dst(dst);
        div_double_kernel << < blocksPerGrid, threadsPerBlock >> > (g_a, b, g_dst, size);
        g_dst.download(dst);
    }
    else WRONG_TYPE();
    
}
CUDA_KERNEL_API void div_image(Mat a, Mat b, Mat dst, Size size) {
    assert(isEqual4(a.rows, b.rows, dst.rows, size.height) && isEqual4(a.cols, b.cols, dst.cols, size.width));
    assert(isEqual3(a.type(), b.type(), dst.type()));
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_a(a), g_b(b), g_dst(dst);
        div_image_kernel << < blocksPerGrid, threadsPerBlock >> > (g_a, g_b, g_dst, size);
        g_a.download(a);
        g_b.download(b);
        g_dst.download(dst);
    }
    else WRONG_TYPE();
    
}
CUDA_KERNEL_API void if_image(Mat a, double lessThan, double result1, double result2, Mat dst, Size size) {
    assert(isEqual3(a.rows, dst.rows, size.height) && isEqual3(a.cols, dst.cols, size.width));
    assert(a.type() == dst.type());
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_a(a), g_dst(dst);
        if_image_kernel << < blocksPerGrid, threadsPerBlock >> > (g_a, lessThan, result1, result2, g_dst, size);
        g_dst.download(dst);
    }
    else WRONG_TYPE();

}
CUDA_KERNEL_API void cpy_image(GpuMat dst, GpuMat src, int dst_x, int dst_y, Size src_size) {
    assert(src.type() == dst.type());
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((src_size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (src_size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        cpy_image_kernel << < blocksPerGrid, threadsPerBlock >> > (dst, src, dst_x, dst_y, src_size);
    }
    else WRONG_TYPE();
}
CUDA_KERNEL_API void image_core_density(Mat dst, Point2d center, double maxF, double minF, double min_distance, Size size) {
    assert(pow(dst.rows, 2) + pow(dst.cols, 2) > min_distance);
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_dst(dst);
        image_core_density_kernel << < blocksPerGrid, threadsPerBlock >> > (g_dst, center, maxF, minF, min_distance, size);
        g_dst.download(dst);
    }
    else WRONG_TYPE();
    
}
/*
CUDA_KERNEL_API void generate_orientation(GpuMat dst, int nCore, Point2d cores, int nDelta, Point2d deltas, int pattern, Size size) {
    assert(dst.cols == size.width && dst.rows == size.height);
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {

        generate_orientation_kernel << < blocksPerGrid, threadsPerBlock >> > (dst, nCore, cores, nDelta, deltas, pattern, size);
    }
    else WRONG_TYPE();
}
*/
CUDA_KERNEL_API void generate_ridge_layer(Mat dst, Mat ridge, Mat orientation, Mat density, Mat filters, double maxF, double minF, Size size) {
    assert(isEqual4(dst.cols, orientation.cols, density.cols, size.width) && isEqual4(dst.rows, orientation.rows, density.rows, size.height));
    assert(isEqual4(ridge.type(), orientation.type(), density.type(), dst.type()));
    dim3 threadsPerBlock(warp_size, warp_size);
    dim3 blocksPerGrid((size.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (size.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (dst.type() == CV_64FC1) {
        GpuMat g_dst(dst), g_ridge(ridge), g_orientation(orientation), g_density(density), g_filters(filters);
        generate_ridge_layer_kernel << < blocksPerGrid, threadsPerBlock >> > (g_dst, g_ridge, g_orientation, g_density, g_filters, maxF, minF, size);
        g_dst.download(dst);
    }
    else WRONG_TYPE();
    
}