#include "thin.h"
#include <iostream>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
using namespace std;
using namespace chrono;

int main() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
	srand((unsigned)time(NULL));

	string file_dir = "C:\\Users\\27976\\Downloads\\�ο�\\Anguli-MinGw\\Fingerprints\\Fingerprints\\fp_1\\";
	string file_type = ".jpg";
	int num = 1;
	for (; num <= 30; num++) {
		auto start = system_clock::now();

		string input = file_dir + std::to_string(num) + file_type;
		cv::Mat src = cv::imread(input);
//		src = src(cv::Rect(15, 15, src.cols - 30, src.rows - 30));
		//cv::imshow("input", src);

		//��Ӻ���
		cv::Mat pored;
		add_pore(src, pored, 350); // ���룬�����dpiֵ����ȷ�����ߴ�ϸ
		//cv::imshow("pored", pored);

		//��ӻ���
		cv::Mat scratched;
		add_scratch(pored, scratched, 10); // ���룬�������������
		//cv::imshow("scratched", scratched);

		string out_file = std::to_string(num) + file_type;
		cv::imwrite(out_file, scratched);

		//����ģ��
		//cv::Mat damaged;
		//non_contacted(src, damaged);
		//cv::imshow("damaged", damaged);

		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		double time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
		cout << "ָ��" << num << "��ʱ:" << time << "��" << endl;

		//cv::waitKey();
	}
	//cv::waitKey();
	cv::destroyAllWindows();
	return(0);
}
