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
	Feature() {}
	Feature(int row, int col, Label label, int im_id);
	void evaluate(const LearnerParameters & params);

	bool operator< (const Feature& f) const {
		return _value < f._value;
	}

	const float & getValue() { return _value; }
	const Label & getLabel() { return _label; }

private:
	int _row;
	int _col;
	Label _label;
	float _value;
	int _image_id;
};

typedef std::vector<Feature> Data;
typedef std::shared_ptr<Data> DataPtr;
typedef Data::iterator FeatureIterator;


class Frame {
public:

	Frame(std::string,std::string);

	void load(std::string depth_path,
		  std::string gt_path);

	Label getLabel(int row, int col) const {
		return (Label)_labels.at<uchar>(row, col);
	}

	const cv::Mat & getLabelImage() const {
		return _labels;
	}

	float operator()(int row, int col);

	cv::Size getImageSize() const {
		return _depth.size();
	}

private:

	cv::Mat _depth;
	cv::Mat _labels;
};

class FramePool {
public:

	static void computeFeatures(DataPtr);
	static void create();
	static std::vector<Frame> image_vector;

private:

};
