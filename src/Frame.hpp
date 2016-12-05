#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <stdlib.h>

class LearnerParameters;

enum class Labels {
	Background,
	LeftShoulder
};

class Feature {
public:
	Feature(int row, int col, int label, int im_id);
	void evaluate(LearnerParameters & params);

	bool operator< (const Feature& f) const {
		return _value < f._value;
	}

	const float & getValue() { return _value; }
	const int & getLabel() { return _label; }
private:
	int _row;
	int _col;
	Label _label;
	float _value;
	int _image_id;
};

class Frame {
public:

	Frame(std::string,std::string);

	void load(std::string depth_path,
		  std::string gt_path);

	Label getLabel(int row, int col) const {
		return _labels.at<uchar>(row, col);
	}		
	
	const float& operator()(int row, int col);

	cv::Size getImageSize() const {
		return _depth.size();
	}

private:

	cv::Mat _depth;
	cv::Mat _labels;
};

class FramePool {
public:

	static std::vector<Feature> computeFeatures();	
	static void create();
	static std::vector<Frame> image_vector;
	
private:

};
