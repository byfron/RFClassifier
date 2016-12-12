#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <stdlib.h>

class LearnerParameters;
class Frame;

enum class Labels {
	Background,
	LeftShoulder
};

class Feature {
public:
	Feature() {}
	Feature(int row, int col, Label label, const Frame* im);
	void evaluate(const LearnerParameters & params);

	bool operator< (const Feature& f) const {
		return _value < f._value;
	}

	const Frame *getFrame() const;

	const float & getValue() { return _value; }
	const Label & getLabel() { return _label; }

	const int row() { return _row; }
	const int col() { return _col; }

private:
	int _row;
	int _col;
	Label _label;
	float _value;
	const Frame* _image;
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

	void show();
	void computeForegroundFeatures(Data & features);

	float operator()(int row, int col) const ;

	void setLabel(int row, int col, Label value) {
		_labels.at<uchar>(row, col) = (uchar)value;
	}
	cv::Size getImageSize() const {
		return _depth.size();
	}

	size_t getFrameSizeInBytes() const {
		size_t depthInBytes = _depth.step[0] * _depth.rows;
		size_t labelsInBytes = _labels.step[0] * _labels.rows;
		return depthInBytes + labelsInBytes;
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
