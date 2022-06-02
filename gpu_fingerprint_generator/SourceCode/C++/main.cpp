#include "proc.h"
#include "fp.h"
#include <iostream>

void show_help() {
    cout << "usage: gpu_fp_generator -o <path> [options]" << endl;
    cout << "      -num <number>    :    number of fingerprints(1)" << endl;
    cout << "      -dpi <number>    :    dpi of input images(350)" << endl;
    exit(0);
}

int main(int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_WARNING);//不再输出日志

    string output_dir = "";
    int num = 1, dpi = 350;
    string file_type = ".jpg";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            show_help();
        }
        else if (!strcmp(argv[i], "-o")) {
            output_dir = argv[++i];
        }
        else if (!strcmp(argv[i], "-num")) {
            num = atoi(argv[i]);
        }
        else if (!strcmp(argv[i], "-dpi")) {
            dpi = atoi(argv[i]);
        }
        else {
            show_help();
        }
    }

    for (int i = 1; i <= num; i++) {
        auto start = system_clock::now();

        fp a;

        //参数设置部分
        a.set_fp_size(2, 3); // 指纹大小（厘米）
        a.set_pattern(4); // 指纹纹形
        //特征点的设置需要符合纹形
        //350dpi指纹设置
        Point2d cores[] = { Point2d(100,125.5) }; // core位置
        Point2d deltas[] = { Point2d(30.5,217.5) }; // delta位置
        //Point2d loops = NULL;
        a.set_resolution(dpi); // 指纹分辨率
        a.set_image_size(512, 512); // 1.5cm 2.1cm // 图片大小
        a.set_shape(100, 110, 110, 100, 80); // 指纹形状

        //1200dpi指纹设置
        //Point2d cores[] = { Point2d(572, 450) }; // Point2d(350, 620), Point2d(360, 695)
        //Point2d deltas[] = { Point2d(572, 800) }; // Point2d(130.5, 1100), Point2d(730.5, 1100)
        //a.set_resolution(1200); // dpi
        //a.set_image_size(944, 1417); // 2cm -> 2 * 0.3937 * 1200 = 944, 3cm -> 3 * 0.3937 * 1200 = 1417
        //a.set_shape(440, 504, 700, 250, 360);

        //参数传递
        a.set_orientation_correction(8);
        a.set_points(sizeof(cores) / sizeof(cores[0]), cores, sizeof(deltas) / sizeof(deltas[0]), deltas);

        //生成部分
        a.generate_filter();
        a.generate_shape();
        a.generate_orientation();
        a.generate_density();
        a.generate_ridge();
        a.generate_master_fp();

        a.show_all();

        //string outfile = output_dir + std::to_string(i) + file_type;
        //a.save_as(outfile);

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "指纹"
            << i
            << "用时："
            << double(duration.count()) * microseconds::period::num / microseconds::period::den
            << "秒" << endl;
    }

    return 0;
}