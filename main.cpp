
#pragma warning(disable:4996)

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>


#include "MLV.h"
#include "feature_utility.h"


int main(int argc, char* argv[])
{

	std::string file1 = "church1.jpg";

	std::string file2 = "church2.jpg";


	cv::Mat img1 = cv::imread(file1, cv::IMREAD_COLOR);

	cv::Mat img2 = cv::imread(file2, cv::IMREAD_COLOR);

	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	std::vector<cv::DMatch> finalMatches;

	runMLV(img1, img2, keypoints1, keypoints2, finalMatches);

	cv::Mat finalMatchingResult = cvg::draw_horizontal_matches(img1, keypoints1, img2, keypoints2, finalMatches,
		cvg::LineColor::LINE_COLOR_YELLOW, cvg::LineStyle::NO_POINT_LINE, cvg::LineThickness::LINE_THICKNESS_TWO);

	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::imshow("result", finalMatchingResult);

	cv::waitKey(-1);

	cv::destroyAllWindows();

	return EXIT_SUCCESS;


}
