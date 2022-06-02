#include "fp.h"
#include <iostream>

fp::fp(void) {
    number = 0;
    fp_width = 2;
    fp_length = 3;
    pattern = 0;
    reso = 1200;
    minF = 1.0 / 15 * 2 / sqrt(reso / 300);
    maxF = 1.0 / 5 * 2 / sqrt(reso / 300);
    L = R = T = M = B = 0;
    size = Size(2 * 0.3937 * reso, 3 * 0.3937 * reso);
    nCore = nDelta = nLoop = 0;
    cores = deltas = NULL;
    shape = Mat::zeros(size, CV_64FC1);
    density = Mat::zeros(size, CV_64FC1);
    orientation = Mat::zeros(size, CV_64FC1);
    ridge = Mat::zeros(size, CV_64FC1);
    master_fp = Mat::zeros(size, CV_64FC1);
    gabor_filters = Mat::zeros(GABOR_FREQ_NUM * (2 * GABOR_SIZE + 1), GABOR_ANGLE_NUM * (2 * GABOR_SIZE + 1), CV_64FC1);
}
//fp::~fp(void) {}

void fp::set_number(const unsigned int n) {
    number = n;
}
void fp::set_fp_size(const int width, const int length) {
    fp_width = width;
    fp_length = length;
    show_fp_size();
}
void fp::set_pattern(const int input) {
    pattern = input;
    if (pattern < 1 || pattern > 6) {
        std::cout << "[ERROR] wrong pattern" << std::endl;
        system("pause");
        exit(1);
    }
}
void fp::set_resolution(const unsigned int input) {
    reso = input;
    minF = 1.0 / 15 * 2 / sqrt(reso / 300);
    maxF = 1.0 / 5 * 2 / sqrt(reso / 300);
    int img_w = fp_width * 0.3937 * reso;
    int img_h = fp_length * 0.3937 * reso;
    /*
    size = Size(img_w, img_h);
    Mat tmp = Mat::zeros(size, CV_64FC1);
    tmp = Mat::zeros(size, CV_64FC1);
    */
    std::cout << "F: " << minF << "~" << maxF << std::endl;
    std::cout << "[INFO] recommended image size: " << img_w << ", " << img_h << std::endl;
}
void fp::set_image_size(const int width, const int height) {
    if (L + R > width || T + M + B > height) {
        std::cout << "[ERROR] confliction between size and shape!" << std::endl;
        system("pause");
        exit(1);
    }
    size = Size(width, height);
    std::cout << "image size reset to " << size.width << ", " << size.height << std::endl;
    shape = Mat::zeros(size, CV_64FC1);
    density = Mat::zeros(size, CV_64FC1);
    orientation = Mat::zeros(size, CV_64FC1);
    ridge = Mat::zeros(size, CV_64FC1);
    master_fp = Mat::zeros(size, CV_64FC1);
}
void fp::set_shape(int l, int r, int t, int m, int b) {
    size = Size(l + r, t + m + b);
    show_image_size();
    L = l;
    R = r;
    T = t;
    M = m;
    B = b;
    std::cout << "image size reset to " << size.width << ", " << size.height << std::endl;
    shape = Mat::zeros(size, CV_64FC1);
    density = Mat::zeros(size, CV_64FC1);
    orientation = Mat::zeros(size, CV_64FC1);
    ridge = Mat::zeros(size, CV_64FC1);
    master_fp = Mat::zeros(size, CV_64FC1);
}
void fp::set_points(int nc, Point2d* c, int nd, Point2d* d, int nl, Point2d* l) {
    nCore = nc;
    nDelta = nd;
    nLoop = nl;
    cores = c;
    deltas = d;
    loops = l;
}
void fp::set_orientation_correction(int piece_num) {
    correction[0] = -PI;
    for (int i = 1; i <= piece_num; i++) {
        correction[i] = correction[i - 1] + 2 * PI / piece_num;
    }

    srand(time(0));
    correction[piece_num / 2] += ((double)rand() / 32767 * 2 - 1) * 2 * PI / piece_num;
    if (piece_num >= 5) {
        correction[piece_num / 2 - 1] = (correction[piece_num / 2 - 2]
            + correction[piece_num / 2]) / 2;
        correction[piece_num / 2 + 1] = (correction[piece_num / 2 + 2]
            + correction[piece_num / 2]) / 2;
    }

}

void fp::generate_filter() {
    std::cout << "generating gabor filters... 0%";
//    generate_gabor_filters(gabor_filters, maxF, minF, Size(2 * GABOR_SIZE + 1, 2 * GABOR_SIZE + 1));
    for (int i = 0; i < GABOR_ANGLE_NUM; i++) {
        for (int j = 0; j < GABOR_FREQ_NUM; j++) {
            double theta = PI * 2 / GABOR_ANGLE_NUM * i + PI / 2;
            double F = (maxF - minF) / GABOR_FREQ_NUM * j + minF;
            double sigma = sqrt(-9.0 / (8.0 * F * F * log(0.001)));
            Mat filter = getGaborKernel(Size(2 * GABOR_SIZE, 2 * GABOR_SIZE), sigma, theta, 1.0 / F, 1, 0);

            filter.copyTo(gabor_filters(Rect(i * (GABOR_SIZE * 2 + 1), j * (GABOR_SIZE * 2 + 1), (GABOR_SIZE * 2 + 1), (GABOR_SIZE * 2 + 1))));
        }
    }
//    imshow("gabor_filters", gabor_filters);
    std::cout << "\b\b\b" << "100%" << std::endl;
}
void fp::generate_shape() {
    std::cout << "generating shape... 0%";
    ellipse(shape, Point(L, T), Size(L, T), 0, 180, 270, 1, -1);
    ellipse(shape, Point(L, T), Size(R, T), 0, 270, 360, 1, -1);
    ellipse(shape, Point(L, T + M), Size(L, B), 0, 90, 180, 1, -1);
    ellipse(shape, Point(L, T + M), Size(R, B), 0, 0, 90, 1, -1);
    rectangle(shape, Rect(0, T, L + R, M), 1, -1);
    std::cout << "\b\b\b" << "100%" << std::endl;
}
void fp::generate_orientation() {
    std::cout << "generating orientation... 0%";
    if (pattern == 1) {
        double k_arch = 2;
        double arch_fact1, arch_fact2;
        arch_fact1 = 1.0;  // ÇúÏß×óÓÒÆ«ÒÆ
        arch_fact2 = 0.8;  // ÇúÏß±ÈÀý
        for (int i = 0; i < size.width; i++) {
            for (int j = 0; j < size.height; j++) {
                double temp = k_arch - k_arch * j / size.height / arch_fact2;
                double alpha = temp > 0.0 ? temp : 0.0;
                orientation.at<double>(j, i) = atan(alpha) * cos(i * CV_PI / size.width / arch_fact1) * -1;
            }
        }
    }
    else {
        for (int i = 0; i < size.width; i++) {
            for (int j = 0; j < size.height; j++) {
                double theta = 0;
                for (int k = 0; k < nCore; k++) {
                    Vec2d v = cores[k] - Point2d(i, j);
                    theta += orientation_correcting(atan2(v[1], v[0])) / nCore; // 2 * 
                }
                for (int k = 0; k < nDelta; k++) {
                    Vec2d v = deltas[k] - Point2d(i, j);
                    theta -= orientation_correcting(atan2(v[1], v[0])) / nDelta;
                }
                for (int k = 0; k < nLoop; k++) {
                    Vec2d v = loops[k] - Point2d(i, j);
                    theta += orientation_correcting(atan2(v[1], v[0])) / nLoop;
                }
                theta *= nDelta / 2.0;
                if (pattern == 2)
                    theta -= PI;
                orientation.at<double>(j, i) = theta;
            }
        }
    }

    std::cout << "\b\b\b" << "100%" << std::endl;
}
void fp::generate_density() {
    std::cout << "generating density... 0%";

    Mat rand_density = Mat::zeros(size, CV_64FC1);
    std::vector<int> rand_n;
    rand_n.push_back(3);
    rand_n.push_back(4);
    RNG rng(time(NULL));
    for (int each : rand_n) {
        Mat rand_image(each, each, CV_64FC1);
        rng.fill(rand_image, RNG::UNIFORM, minF, maxF);
        resize(rand_image, rand_image, size, 0, 0, INTER_CUBIC);
//        add_image(rand_density, rand_image, rand_density, size);
        rand_density += rand_image;
    }
    rand_density /= rand_n.size();
    /*
    Mat core_density = Mat::zeros(size, CV_64FC1);
    double x = 0.0, y = 0.0;
    for (int i = 0; i < nCore; i++) {
        x += cores[i].x;
        y += cores[i].y;
    }
    Point2d center(x / nCore, y / nCore);
    double min_distance = pow(size.width, 2) + pow(size.height, 2); // minimiun distance between cores and deltas
    for (int i = 0; i < nDelta; i++) {
        double temp = pow(center.x - deltas[i].x, 2) + pow(center.y - deltas[i].y, 2);
        if (temp < min_distance)
            min_distance = temp;
    }
    image_core_density(core_density, center, maxF / 2, minF / 2, min_distance, size);
    */
    //    core_density = (core_density + 1) / 2 * maxF + (minF + maxF) / 2;

//    add_image(rand_density, core_density, density, size);
    density = rand_density;
    std::cout << "\b\b\b" << "100%" << std::endl;
}
void fp::generate_ridge() {
    std::cout << "generating ridge... 0%";
    randu(ridge, 0, 1);
    if_image(ridge, 0.001, 1, 0, ridge, size);

    int count = 0;
    while (count < 10) {
        Mat layer(size, CV_64FC1);
        generate_ridge_layer(layer, ridge, orientation, density, gabor_filters, maxF, minF, size);
        cv::GaussianBlur(layer, ridge, cv::Size(21, 21), 0, 0);
        if (double(abs(countNonZero(ridge) - countNonZero(layer))) / size.area() > 0.001) {
            ridge = layer;
        }
        else {
            break;
        }
        count++;
        std::cout << "\b\b\b" << count * 10 << "%";
    }
    std::cout << std::endl;
}
void fp::generate_master_fp() {
    std::cout << "generating master fingerprint... 0%";
    mul_image(ridge, shape, master_fp, size);
    std::cout << "\b\b\b" << "100%" << std::endl;
    std::cout << "generate done" << std::endl;
}

void fp::show_fp_size() {
    std::cout << "finger size = " << fp_width << "cm, " << fp_length << "cm" << std::endl;
}
void fp::show_image_size() {
    std::cout << "image size = " << size.width << ", " << size.height << std::endl;
}

void fp::show_shape() {
    imshow("shape", shape);
    waitKey(0);
}
void fp::show_orientation() {
    Mat orientation_display(size, CV_64FC1);
    double line_length = reso / 70;
    for (int i = line_length; i < size.width; i += line_length * 2) {
        for (int j = line_length; j < size.height; j += line_length * 2) {
            Point center(i, j);
            double theta = orientation.at<double>(j, i);
            Point2d p1(i - line_length * cos(theta), j - line_length * sin(theta));
            Point2d p2(i + line_length * cos(theta), j + line_length * sin(theta));
            line(orientation_display, p1, p2, 1);
        }
    }
    mul_image(orientation_display, shape, orientation_display, size);
    Mat m_[] = { orientation_display,orientation_display,orientation_display };
    merge(m_, 3, orientation_display);
    for (int k = 0; k < nCore; k++) {
        circle(orientation_display, cores[k], 5, Scalar(0, 0, 1), 1);
    }
    for (int k = 0; k < nDelta; k++) {
        triangle(orientation_display, deltas[k], 5);
    }
    imshow("orientation", orientation_display);
    waitKey(0);
}
void fp::show_density() {
    imshow("density", density);
    waitKey(0);
}
void fp::show_ridge() {
    imshow("ridge", ridge);
    waitKey(0);
}
void fp::show_master_fp() {
    imshow("master fingerprint", master_fp);
    waitKey(0);
}
void fp::show_gabor_filters() {
    imshow("gabor filters", gabor_filters);
    waitKey(0);
}
void fp::show_all() {
//    imshow("Ê¾Àý1_gpu gaborÂË²¨Æ÷ 36*20", gabor_filters);
    Mat orientation_display(size, CV_64FC1);
    double line_length = reso / 70;
    for (int i = line_length; i < size.width; i += line_length * 2) {
        for (int j = line_length; j < size.height; j += line_length * 2) {
            Point center(i, j);
            double theta = orientation.at<double>(j, i);
            Point2d p1(i - line_length * cos(theta), j - line_length * sin(theta));
            Point2d p2(i + line_length * cos(theta), j + line_length * sin(theta));
            line(orientation_display, p1, p2, 1);
        }
    }
    Mat plt(size.height + 10, size.width * 5 + 30, CV_64FC1);
    plt.cols;
    plt.rows;
    shape.copyTo(plt(Rect(5, 5, size.width, size.height)));
    orientation_display.copyTo(plt(Rect(10 + size.width, 5, size.width, size.height)));
    density.copyTo(plt(Rect(15 + size.width * 2, 5, size.width, size.height)));
    ridge.copyTo(plt(Rect(20 + size.width * 3, 5, size.width, size.height)));
    master_fp.copyTo(plt(Rect(25 + size.width * 4, 5, size.width, size.height)));
    imshow("Ê¾Àý_gpu", plt);
    waitKey(0);
}

double fp::orientation_correcting(double alpha) {
    // piecewise linear function
    int piece_num = 8;
    double ans = 0.0;
    int i = 0;
    for (i = 0; i < piece_num; i++) {
        double l = -PI + 2 * PI / piece_num * i;
        double r = l + 2 * PI / piece_num;
        if (alpha <= r) {
            ans = correction[i]
                + (alpha - l) / (2 * PI / piece_num)
                * (correction[i + 1] - correction[i]);
            break;
        }
    }
    if (i == piece_num) {
        std::cout << "error while orientation_correcting" << std::endl;
        system("pause");
        exit(2);
    }
    return ans;
}

void fp::save_img() {
    Mat saved_master_fp;
    master_fp.convertTo(saved_master_fp, CV_8UC1, 255);
    imwrite("master fingerprint.jpg", saved_master_fp);
}
void fp::save_as(string filename) {
    imwrite(filename, master_fp);
}

void triangle(Mat& img, Point center, double distance) {
    Point2d p1(center.x, center.y - distance);
    Point2d p2(center.x - distance / 2 * sqrt(3), center.y + distance / 2);
    Point2d p3(center.x + distance / 2 * sqrt(3), center.y + distance / 2);
    line(img, p1, p2, Scalar(0, 1, 0), 1);
    line(img, p2, p3, Scalar(0, 1, 0), 1);
    line(img, p3, p1, Scalar(0, 1, 0), 1);
}