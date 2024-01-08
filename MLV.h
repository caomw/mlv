#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>


/**
 * \brief 特征点检测和计算描述子
 * \param img1              输入， 查询图像
 * \param img2              输入， 参考图像
 * \param keypoints1        输出， 查询图像特征点
 * \param keypoints2        输出， 参考图像特征点
 * \param knnMatches12      输出， KNN候选匹配
 */
void detecteFeatures(
	cv::Mat img1, cv::Mat img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& knnMatches12);


/**
 * \brief 比例测试
 * \param knnMatches        输入， KNN候选匹配
 * \param goodMatche        输出， 比例测试后的特征匹配结果
 * \param r                 输入， 比例测试阈值
 */
void ratio_test_verification(
	std::vector<std::vector<cv::DMatch>>& knnMatches,
	std::vector<cv::DMatch>& goodMatche, float r);


/**
 * \brief 交叉验证
 * \param mKnnMatches12     输入， KNN候选匹配1
 * \param mKnnMatches21     输入， KNN候选匹配2
 * \param mSymmetricMatches 输出， 交叉验证结果
 */
void symmetric_verification(
	std::vector<cv::DMatch> mKnnMatches12,
	std::vector<cv::DMatch> mKnnMatches21,
	std::vector<cv::DMatch>& mSymmetricMatches);


/**
 * \brief 几何验证
 * \param mSymmetricMatches 输入， 交叉验证结果
 * \param keypoints1        输入， 查询图像特征点
 * \param keypoints2        输入， 参考图像特征点
 * \param mGeometryMatches  输出， 几何约束结果
 */
void geometry_verification(
	std::vector<cv::DMatch>& mSymmetricMatches,
	std::vector<cv::KeyPoint> keypoints1,
	std::vector<cv::KeyPoint> keypoints2,
	std::vector<cv::DMatch>& mGeometryMatches);


/**
 * \brief 运行MLV算法
 * \param img1              输入， 查询图像
 * \param img2              输入， 参考图像
 * \param kp1               输入， 查询图像特征点
 * \param kp2               输入， 参考图像特征点
 * \param geometryMatches   输出， 最终匹配结果
 */
void runMLV(
	cv::Mat& img1, cv::Mat& img2,
	std::vector<cv::KeyPoint>& kp1,
	std::vector<cv::KeyPoint>& kp2,
	std::vector<cv::DMatch>& finalMatches);