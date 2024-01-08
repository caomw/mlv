#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>


/**
 * \brief ��������ͼ���������
 * \param img1              ���룬 ��ѯͼ��
 * \param img2              ���룬 �ο�ͼ��
 * \param keypoints1        ����� ��ѯͼ��������
 * \param keypoints2        ����� �ο�ͼ��������
 * \param knnMatches12      ����� KNN��ѡƥ��
 */
void detecteFeatures(
	cv::Mat img1, cv::Mat img2,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& knnMatches12);


/**
 * \brief ��������
 * \param knnMatches        ���룬 KNN��ѡƥ��
 * \param goodMatche        ����� �������Ժ������ƥ����
 * \param r                 ���룬 ����������ֵ
 */
void ratio_test_verification(
	std::vector<std::vector<cv::DMatch>>& knnMatches,
	std::vector<cv::DMatch>& goodMatche, float r);


/**
 * \brief ������֤
 * \param mKnnMatches12     ���룬 KNN��ѡƥ��1
 * \param mKnnMatches21     ���룬 KNN��ѡƥ��2
 * \param mSymmetricMatches ����� ������֤���
 */
void symmetric_verification(
	std::vector<cv::DMatch> mKnnMatches12,
	std::vector<cv::DMatch> mKnnMatches21,
	std::vector<cv::DMatch>& mSymmetricMatches);


/**
 * \brief ������֤
 * \param mSymmetricMatches ���룬 ������֤���
 * \param keypoints1        ���룬 ��ѯͼ��������
 * \param keypoints2        ���룬 �ο�ͼ��������
 * \param mGeometryMatches  ����� ����Լ�����
 */
void geometry_verification(
	std::vector<cv::DMatch>& mSymmetricMatches,
	std::vector<cv::KeyPoint> keypoints1,
	std::vector<cv::KeyPoint> keypoints2,
	std::vector<cv::DMatch>& mGeometryMatches);


/**
 * \brief ����MLV�㷨
 * \param img1              ���룬 ��ѯͼ��
 * \param img2              ���룬 �ο�ͼ��
 * \param kp1               ���룬 ��ѯͼ��������
 * \param kp2               ���룬 �ο�ͼ��������
 * \param geometryMatches   ����� ����ƥ����
 */
void runMLV(
	cv::Mat& img1, cv::Mat& img2,
	std::vector<cv::KeyPoint>& kp1,
	std::vector<cv::KeyPoint>& kp2,
	std::vector<cv::DMatch>& finalMatches);