
#pragma warning(disable:4996)

#include <vector>

#include <opencv2/opencv.hpp>

#include "MLV.h"

void detecteFeatures(
	cv::Mat img1, cv::Mat img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& knnMatches12)
{
	cv::Mat descriptors1, descriptors2;
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
	sift->detect(img1, keypoints1);
	sift->detect(img2, keypoints2);
	sift->compute(img1, keypoints1, descriptors1);
	sift->compute(img2, keypoints2, descriptors2);

	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches;
	matcher.knnMatch(descriptors1, descriptors2, knnMatches12, 2);
}


void ratio_test_verification(
	std::vector<std::vector<cv::DMatch>>& knnMatches,
	std::vector<cv::DMatch>& goodMatche, float r)
{
	assert(knnMatches.size() > 0);
	std::size_t cnt = knnMatches.size();
	goodMatche.clear();
	for (std::size_t i = 0; i < cnt; i++)
	{
		cv::DMatch match1 = knnMatches[i][0];
		cv::DMatch match2 = knnMatches[i][1];
		if (match1.distance < r * match2.distance)
		{
			goodMatche.push_back(match1);
		}
	}

}

void symmetric_verification(
	std::vector<cv::DMatch> mKnnMatches12,
	std::vector<cv::DMatch> mKnnMatches21,
	std::vector<cv::DMatch>& mSymmetricMatches)
{
	for (int i = 0; i < mKnnMatches12.size(); i++)
	{

		for (int j = 0; j < mKnnMatches21.size(); j++)
		{
			if (mKnnMatches12[i].queryIdx == mKnnMatches21[j].trainIdx &&
				mKnnMatches12[i].trainIdx == mKnnMatches21[j].queryIdx)
			{
				mSymmetricMatches.push_back(cv::DMatch(mKnnMatches12[i].queryIdx,
					mKnnMatches12[i].trainIdx, mKnnMatches12[i].distance));
				break;
			}
		}
	}

}

void geometry_verification(
	std::vector<cv::DMatch>& mSymmetricMatches,
	std::vector<cv::KeyPoint> keypoints1,
	std::vector<cv::KeyPoint> keypoints2,
	std::vector<cv::DMatch>& mGeometryMatches)
{

	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;

	for (int i = 0; i < mSymmetricMatches.size(); i++)
	{
		int queryInx = mSymmetricMatches[i].queryIdx;
		int trainInx = mSymmetricMatches[i].trainIdx;

		points1.push_back(cv::Point2f(keypoints1[queryInx].pt.x, keypoints1[queryInx].pt.y));
		points2.push_back(cv::Point2f(keypoints2[trainInx].pt.x, keypoints2[trainInx].pt.y));
	}

	std::vector<unsigned char> inliers(points1.size(), 0);

	cv::Mat fundemental = cv::findFundamentalMat(
		points1, points2,         // matching points
		cv::FM_RANSAC,       // RANSAC method
		1.0,               // distance to epipolar line
		0.99,             // confidence probability
		inliers);              // match status (inlier or outlier)


	int inxMatcher = 0;

	//#pragma omp parallel for num_threads(mThreads)
	for (std::vector<unsigned char>::const_iterator inlierIterator = inliers.begin();
		inlierIterator != inliers.end(); inlierIterator++)
	{
		if (*inlierIterator)
		{
			int first = mSymmetricMatches[inxMatcher].queryIdx;
			int second = mSymmetricMatches[inxMatcher].trainIdx;
			mGeometryMatches.push_back(cv::DMatch(first, second, 0));
			//mFinalRefPoints.push_back(points1[inxMatcher]);
			//mFinalQueryPoints.push_back(points2[inxMatcher]);
		}

		inxMatcher++;
	}


}

void runMLV(
	cv::Mat& img1, cv::Mat& img2,
	std::vector<cv::KeyPoint>& kp1,
	std::vector<cv::KeyPoint>& kp2,
	std::vector<cv::DMatch>& finalMatches)
{
	std::vector<cv::KeyPoint> keypoints2_1;
	std::vector<cv::KeyPoint> keypoints2_2;

	std::vector<std::vector<cv::DMatch>> knnMatches12;
	std::vector<std::vector<cv::DMatch>> knnMatches21;

	std::vector<cv::DMatch> ratioMatches12;
	detecteFeatures(img1, img2, kp1, kp2, knnMatches12);
	ratio_test_verification(knnMatches12, ratioMatches12, 0.85);

	std::vector<cv::DMatch> ratioMatches21;
	detecteFeatures(img2, img1, keypoints2_1, keypoints2_2, knnMatches21);
	ratio_test_verification(knnMatches21, ratioMatches21, 0.85);

	std::vector<cv::DMatch> crossDMatch;
	symmetric_verification(ratioMatches12, ratioMatches21, crossDMatch);

	geometry_verification(crossDMatch, kp1, kp2, finalMatches);

}