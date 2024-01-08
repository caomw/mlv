#pragma once

#pragma warning(disable:4996)

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>


namespace cvg
{

	enum class LineStyle
	{
		POINT_LINE,//ԭ�����
		NO_POINT_LINE,//������
		CROSS_POINT_LINE//ʮ�ֽ�������
	};

	enum class LineThickness
	{
		LINE_THICKNESS_ONE = 1,
		LINE_THICKNESS_TWO,
		LINE_THICKNESS_THREE,
		LINE_THICKNESS_FOUR,
		LINE_THICKNESS_FIVE,
		LINE_THICKNESS_SIX,
		LINE_THICKNESS_SEVEN,
		LINE_THICKNESS_EIGHT,
		LINE_THICKNESS_NINE,
		LINE_THICKNESS_TEN
	};


	enum class LineColor
	{
		LINE_COLOR_BLUE_GREEN,//����
		LINE_COLOR_DEFAULT,//ī��ɫ
		LINE_COLOR_RED,//��ɫ
		LINE_COLOR_BLUE,//��ɫ
		LINE_COLOR_GREEN,//��ɫ
		LINE_COLOR_PINK,//�ۺ�ɫ
		LINE_COLOR_YELLOW,//��ɫ
		LINE_COLOR_BLUE_RED,//��ɫ
		LINE_COLOR_LIGHT_GREEN,//ǳ��ɫ
		LINE_COLOR_BLACK,//��ɫ
		LINE_COLOR_WHILTE,//��ɫ

	};


	/**
	 * \brief ����ˮƽƥ������
	 * \param img1      ���룬��ѯͼ��
	 * \param ktp1      ���룬��ѯ������
	 * \param img2      ���룬�ο�ͼ��
	 * \param ktp2      ���룬�ο�������
	 * \param matches   ���룬����ƥ����
	 * \param color     ���룬������ɫ
	 * \param lineStyle ���룬��������
	 * \param lineThickness ���룬������ϸ
	 */
	cv::Mat draw_horizontal_matches(
		cv::Mat& img1,
		std::vector<cv::KeyPoint>& kpt1,
		cv::Mat& img2,
		std::vector<cv::KeyPoint>& kpt2,
		std::vector<cv::DMatch>& matches,
		const LineColor lineColor = LineColor::LINE_COLOR_DEFAULT,
		const LineStyle lineStyle = LineStyle::NO_POINT_LINE,
		const LineThickness lineThickness = LineThickness::LINE_THICKNESS_ONE);


	/**
	 * \brief ���ƴ�ֱƥ������
	 * \param img1      ���룬��ѯͼ��
	 * \param ktp1      ���룬��ѯ������
	 * \param img2      ���룬�ο�ͼ��
	 * \param ktp2      ���룬�ο�������
	 * \param matches   ���룬����ƥ����
	 * \param color     ���룬������ɫ
	 * \param lineStyle ���룬��������
	 * \param lineThickness ���룬������ϸ
	 */
	cv::Mat draw_vertical_matches(
		cv::Mat& img1,
		std::vector<cv::KeyPoint>& kpt1,
		cv::Mat& img2,
		std::vector<cv::KeyPoint>& kpt2,
		std::vector<cv::DMatch>& matches,
		LineColor lineColor = LineColor::LINE_COLOR_DEFAULT,
		const LineStyle lineStyle = LineStyle::NO_POINT_LINE,
		const LineThickness lineThickness = LineThickness::LINE_THICKNESS_ONE);


	/**
	 * \brief �ı�ͼ��ķֱ���
	 * \param src      ���룬ԭʼͼ��
	 * \param height   ���룬Ŀ��ͼ��ĸ߶�
	 * \param dst      �����Ŀ��ͼ��
	 */
	void imresize(const cv::Mat& src, const int height, cv::Mat& dst);



	void draw_cross_flag(
		cv::Mat& img, cv::Point point,
		cv::Scalar color, int size, int thickness);



	/**
	 * \brief ����������
	 * \param fileName ���룬�ļ���
	 * \param kpts     ���룬 ������
	*/
	void save_feature_points(std::string fileName, const std::vector<cv::KeyPoint>& kpts);


	/**
	 * \brief ��������������
	 * \param fileName ���룬�ļ���
	 * \param desc     ���룬 ����������
	*/
	void save_feature_descriptors(std::string fileName, const cv::Mat& desc);


	/**
	 * \brief ��ȡ������
	 * \param fileName ���룬�ļ���
	 * \param kpts     �����������
	*/
	void load_feature_points(std::string fileName, std::vector<cv::KeyPoint>& kpts);


	/**
	 * \brief ��ȡ����������
	 * \param fileName ���룬�ļ���
	 * \param desc     ���������������
	*/
	void load_feature_descriptors(std::string fileName, cv::Mat& desc);


	/**
	 * \brief ��������ƥ����
	 * \param fileName ���룬�ļ���
	 * \param matches  ���룬����ƥ����
	*/
	void save_feature_matches(std::string fileName, const std::vector<cv::DMatch>& matches);



	/**
	 * \brief ��ȡ����ƥ����
	 * \param fileName ���룬�ļ���
	 * \param matches  ���������ƥ����
	*/
	void load_feature_matches(std::string fileName, std::vector<cv::DMatch>& matches);

	/**
	 * \brief ����������
	 * \param img    ���룬ԭʼͼ��
	 * \param ktps   ���룬������
	 * \param outImg �����������������ͼ��
	*/
	void draw_keypoints(cv::Mat img, const std::vector<cv::Point2f>& kpts, cv::Mat& outImg);


}//namespace cvg

