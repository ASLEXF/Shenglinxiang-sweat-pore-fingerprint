#ifndef THIN
#define THIN

#include <opencv2/opencv.hpp>
#include <string>
#include <windows.h>

//patterns defined in "Image Processing, Analysis, and Machine Vision Fourth Edition, Milan Sonka et. al."

void createFolder(std::string dirName) {
	std::string commandMd = "mkdir " + dirName;
	std::string commandRd = "rmdir /s /q " + dirName;
	if (std::system(commandMd.c_str()))
	{
		std::cout << "remove dir:dst, and re-makedir" << std::endl;
		std::system(commandRd.c_str());
	}
	std::system(commandMd.c_str());
}

bool compare(cv::Mat src, const int pattern[][3]) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (pattern[i][j] == 2)
				continue;
			if (!src.at<uchar>(i, j) && pattern[i][j] == 1 || src.at<uchar>(i, j) && pattern[i][j] == 0)
				return false;
		}
	}
	return true;
}
void thinning_it(cv::Mat &src, cv::Mat& ridge) {
	const int S1[3][3] = {
		{ 0,0,0 },
		{ 2,1,2 },
		{ 1,1,1 } };
	const int S2[3][3] = {
		{ 1,2,0 },
		{ 1,1,0 },
		{ 1,2,0 } };
	const int S3[3][3] = {
		{ 1,1,1 },
		{ 2,1,2 },
		{ 0,0,0 } };
	const int S4[3][3] = {
		{ 0,2,1 },
		{ 0,1,1 },
		{ 0,2,1 } };
	const int S5[3][3] = {
		{ 2,0,0 },
		{ 1,1,0 },
		{ 1,1,2 } };
	const int S6[3][3] = {
		{ 1,1,2 },
		{ 1,1,0 },
		{ 2,0,0 } };
	const int S7[3][3] = {
		{ 2,1,1 },
		{ 0,1,1 },
		{ 0,0,2 } };
	const int S8[3][3] = {
		{ 0,0,2 },
		{ 0,1,1 },
		{ 2,1,1 } };
	auto T_Set = { S1, S5, S2, S6, S3, S7, S4, S8 };

	ridge = src.clone();
	int count = 0;
	while (count < T_Set.size()) {
		for (auto pattern : T_Set) {
			bool flag = true;
			for (int i = 1; i < src.rows - 2; i++) {
				for (int j = 1; j < src.cols - 2; j++) {
					cv::Mat roi = src(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2));
					if (compare(roi, pattern)) {
						ridge.at<uchar>(i, j) = 0;
						flag = false;
					}
				}
			}
			if (flag)
				count++;
			else
				count = 0;
		}
	}
}

void thinning(cv::Mat &src, cv::Mat& ridge) {
//	createFolder("dst");
	ridge = src.clone();
	thinning_it(ridge, ridge);
//	std::string name = "dst/result" + std::to_string(i++) + ".jpg";
//	cv::imwrite(name, ridge);
}

void cutting(cv::Mat &src, cv::Mat &dst, int num) {
	const int T1[3][3] = {
		{ 0,0,0 },
		{ 1,1,0 },
		{ 0,0,0 } };
	const int T2[3][3] = {
		{ 0,1,0 },
		{ 0,1,0 },
		{ 0,0,0 } };
	const int T3[3][3] = {
		{ 0,0,0 },
		{ 0,1,1 },
		{ 0,0,0 } };
	const int T4[3][3] = {
		{ 0,0,0 },
		{ 0,1,0 },
		{ 0,1,0 } };
	const int T5[3][3] = {
		{ 1,0,0 },
		{ 0,1,0 },
		{ 0,0,0 } };
	const int T6[3][3] = {
		{ 0,0,1 },
		{ 0,1,0 },
		{ 0,0,0 } };
	const int T7[3][3] = {
		{ 0,0,0 },
		{ 0,1,0 },
		{ 0,0,1 } };
	const int T8[3][3] = {
		{ 0,0,0 },
		{ 0,1,0 },
		{ 1,0,0 } };
	auto C_Set = { T1, T2, T3, T4, T5, T6, T7, T8 };

	dst = src.clone();
	for (int i = 0; i < num; i++) {
		for (auto pattern : C_Set) {
			for (int i = 1; i < src.rows - 2; i++) {
				for (int j = 1; j < src.cols - 2; j++) {
					cv::Mat roi = src(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2));
					if (compare(roi, pattern)) {
						dst.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
}

void process1(cv::Mat &src, cv::Mat& dst) {
	const int P1[3][3] = {
		{ 1,1,1 },
		{ 1,0,0 },
		{ 1,0,0 }
	};
	const int P2[3][3] = {
		{ 1,1,1 },
		{ 0,0,1 },
		{ 0,0,1 }
	};
	const int P3[3][3] = {
		{ 0,0,1 },
		{ 0,0,1 },
		{ 1,1,1 }
	};
	const int P4[3][3] = {
		{ 1,0,0 },
		{ 1,0,0 },
		{ 1,1,1 }
	};
	auto P_Set = { P1, P2, P3, P4 };

	dst = src.clone();
	for (int i = 1; i < src.rows - 2; i++) {
		for (int j = 1; j < src.cols - 2; j++) {
			cv::Mat roi = src(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2));
			if (compare(roi, P1)) {
				dst.at<uchar>(i - 1, j) = 0;
				dst.at<uchar>(i, j - 1) = 0;
				dst.at<uchar>(i - 1, j - 1) = 0;
				dst.at<uchar>(i, j) = 255;
			}
			if (compare(roi, P2)) {
				dst.at<uchar>(i - 1, j) = 0;
				dst.at<uchar>(i, j + 1) = 0;
				dst.at<uchar>(i - 1, j + 1) = 0;
				dst.at<uchar>(i, j) = 255;
			}
			if (compare(roi, P3)) {
				dst.at<uchar>(i + 1, j) = 0;
				dst.at<uchar>(i, j + 1) = 0;
				dst.at<uchar>(i + 1, j + 1) = 0;
				dst.at<uchar>(i, j) = 255;
			}
			if (compare(roi, P4)) {
				dst.at<uchar>(i + 1, j) = 0;
				dst.at<uchar>(i, j - 1) = 0;
				dst.at<uchar>(i + 1, j - 1) = 0;
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

void process2(cv::Mat &src, cv::Mat& dst) {
	const int D[3][3] = {
		{ 0,0,0 },
		{ 0,1,0 },
		{ 0,0,0 }
	};

	dst = src.clone();
	for (int i = 1; i < src.rows - 2; i++) {
		for (int j = 1; j < src.cols - 2; j++) {
			cv::Mat roi = src(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2));
			if (compare(roi, D)) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
}

inline int rand_int(int high = RAND_MAX, int low = 0)
{
//	srand((unsigned)time(NULL));
	return low + (int)(double(high - low) * (rand() / (RAND_MAX + 1.0)));
}
void set_pore_position(cv::Mat& src, cv::Mat& dst, const int reso) {
	dst = cv::Mat::zeros(src.size(), src.type());
	int size = reso / 70 + 1;
	for (int i = size / 2; i < src.rows - 1 - size / 2; i += size) {
		for (int j = size / 2; j < src.cols - 1 - size / 2; j += size) {
			if (src.at<uchar>(i, j) != 0) {
				dst.at<uchar>(i + rand_int(size / 2, -size / 2),
							  j + rand_int(size / 2, -size / 2)) = 255;
			}
		}
	}
}

void _fill3x3(cv::Mat &dst, int y, int x, const int se[3][3]) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (se[i][j] == 1)
				dst.at<uchar>(y + i - 1, x + j - 1) = 255;
		}
	}
}

void set_pore(cv::Mat &src, cv::Mat& dst, int max_size = 8) {
	const int D1[3][3]= {
		{ 1,1,1 },
		{ 1,1,1 },
		{ 1,1,1 }
	};
	const int D2[3][3] = {
		{ 0,1,0 },
		{ 1,1,1 },
		{ 0,1,1 }
	};
	const int D3[3][3] = {
		{ 0,1,0 },
		{ 1,1,1 },
		{ 1,1,0 }
	};
	const int D4[3][3] = {
		{ 1,1,0 },
		{ 1,1,1 },
		{ 0,1,0 }
	};
	const int D5[3][3] = {
		{ 0,1,1 },
		{ 1,1,1 },
		{ 0,1,0 }
	};


	dst = src.clone();
	
	for (int i = 1; i < src.rows - 2; i++) {
		for (int j = 1; j < src.cols - 2; j++) {
			int times = max_size;
			if (src.at<uchar>(i, j) != 0) {
				while(times > 0) {
					//if (times > 7 && rand() > RAND_MAX / 4 * 3) {
					//	_fill3x3(dst, i, j, D1);
					//	times -= 8;
					//}
					if (times > 4 && rand() > RAND_MAX / 4 * 3) {
						int rand_a = rand();
						if (rand_a > RAND_MAX / 4 * 3) {
							_fill3x3(dst, i, j, D2);
						}
						else if (rand_a > RAND_MAX / 2) {
							_fill3x3(dst, i, j, D3);
						}
						else if (rand_a > RAND_MAX / 4) {
							_fill3x3(dst, i, j, D4);
						}
						else {
							_fill3x3(dst, i, j, D5);
						}
						times -= 5;
					}
					else {
						int randi = rand_int(1, -1);
						int randj = rand_int(1, -1);
						dst.at<uchar>(i + randi, j + randj) = 255;
						times -= 1;
					}
				}
			}
		}
	}
}

void add_pore(cv::Mat& src, cv::Mat& dst, const int pore_density = 350, const int width_padding = 0, const int height_padding = 0) {
	cv::Mat grayImage, binaryImage, ridge;
	cv::Mat pore_seed, pore;

	src = src(cv::Rect(width_padding, height_padding, src.cols - width_padding * 2, src.rows - height_padding * 2));
	if (src.type() == 0) {
		grayImage = src;
	}
	else {
		cv::cvtColor(src, grayImage, cv::COLOR_RGB2GRAY);
	}
	cv::threshold(grayImage, binaryImage, 150, 255, cv::THRESH_BINARY);
	cv::bitwise_not(binaryImage, binaryImage);

	thinning(binaryImage, ridge);

	//imshow("thinned", ridge);

	process1(ridge, ridge);
	process2(ridge, ridge);
	cutting(ridge, ridge, 20);

	//imshow("cutted", ridge);

	set_pore_position(ridge, pore_seed, pore_density);

	//imshow("pore_seed", pore_seed);

	set_pore(pore_seed, pore, 8);

	//imshow("pore", pore);

	dst = grayImage + pore;
}

void non_contacted(cv::Mat &src, cv::Mat& dst) {
	dst = cv::Mat::zeros(src.size(), src.type());
	cv::Mat tmp1 = src.clone(), tmp2;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	while (true) {
		erode(tmp1, tmp1, element);
		morphologyEx(tmp1, tmp2, cv::MORPH_OPEN, element);
		dst += (tmp1 - tmp2);
		tmp1 = tmp2;
		if (cv::sum(tmp1)[0] == 0)
			break;
	}
	cv::bitwise_not(dst, dst);
}

inline double rand_double(double high = 1.0, double low = 0)
{
	return low + ((high - low) * (rand() / (RAND_MAX + 1.0)));
}

bool _fill_filter(cv::Mat& dst, int y, int x, cv::Mat &fill, bool flag_height, bool flag_width) {
	bool ret = false;
	if (flag_height && flag_width) {
		x -= 21;
	}
	else if (flag_height && !flag_width) {
		y -= 21;
		x -= 21;
	}
	else if (!flag_height && !flag_width) {
		y -= 21;
	}
	for (int i = 0; i < 21; i++) {
		for (int j = 0; j < 21; j++) {
			if (fill.at<double>(i, j) < 0.4) continue;
			int _y = y + i;
			int _x = x + j;
			if (_y < 0 || _y > dst.rows - 1 || _x < 0 || _x > dst.cols - 1) {
				ret = true;
				continue;
			}
			dst.at<double>(_y, _x) = 1;
		}
	}
	return ret;
}

void add_scratch(cv::Mat &src, cv::Mat &dst, int time = 1) {

	cv::Mat scratches = cv::Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < time; i++) {
		cv::Mat scratch = cv::Mat::zeros(src.size(), CV_64F);

		double theta = rand_double(CV_PI / 12, -CV_PI / 12);
		double F = rand_double(1.0 / 5, 1.0 / 10);
		double sigma = sqrt(-9.0 / (8.0 * F * F * log(0.001)));

		cv::Mat filter;
		double rand_x = rand_double(); // height
		double rand_y = rand_double(); // width
		double max_size = 0;
		int size = 0;
		int turning = 0;
		if (rand_y < 0.4 || rand_y > 0.6) {
			max_size = 0.5;
			turning = 1;
		}
		else {
			max_size = 0.2;
		}
		max_size = rand_double(max_size);
		int rand_orientation = rand();
		int count = 21;
		if (rand_orientation & 0x1) { // ºáÏò
			size = max_size * src.cols;
		}
		else { // ×ÝÏò
			size = max_size * src.rows;
			theta += CV_PI / 2;
		}
		int next_x = rand_x * src.rows, next_y = rand_y * src.cols;
		while (1) {
			theta += rand_double(CV_PI / 36, -CV_PI / 36);

			filter = cv::getGaborKernel(cv::Size(20, 20), sigma, theta - CV_PI / 2, 1.0 / F, 1, 0);
			bool ret = _fill_filter(scratch, next_x, next_y, filter, rand_y > 0.5, rand_x < 0.5);
			if (ret) break;
			if (rand_y < 0.4) {
				next_x += 7 * sin(theta);
				next_y += 7 * cos(theta);
			}
			else {
				next_x -= 7 * sin(theta);
				next_y -= 7 * cos(theta);
			}
			count += 7;
			if (count >= size) break;
			else {
				//cv::line(scratch, cv::Point(x, y), cv::Point(next_x, next_y), 1.0 / F);
			}
		}
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat out;
		morphologyEx(scratch, out, cv::MORPH_CLOSE, element);

		out.convertTo(out, CV_8UC1, 255);
		scratches += out;
	}
//	cv::imshow("scratches", scratches);
	if (src.type() == 0) {
		dst = src + scratches;
	}
	else {
		cv::Mat src_gray;
		cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);
		dst = src_gray + scratches;
	}
}

#endif // !THIN