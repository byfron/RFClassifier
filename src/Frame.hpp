#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <stdlib.h>

class LearnerParameters;
class Frame;

namespace FrameUtils {
	void setBackgroundToMaxDepth(cv::Mat & depth, const cv::Mat & mask);
}

enum class Labels {
	Background,
	Foreground
};

class Feature {
public:
	Feature() {}
	Feature(int row, int col, Label label, const Frame* im);
	void evaluate(const LearnerParameters & params);

	inline bool operator< (const Feature& f) const {
		return _value < f._value;
	}

	inline const Frame *getFrame() const;
	inline const float & getValue() { return _value; }
	inline const Label & getLabel() { return _label; }
	inline const int row() { return _row; }
	inline const int col() { return _col; }

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

	Frame() {}
	Frame(std::string,std::string);

	void load(std::string depth_path,
		  std::string gt_path);

	inline Label getLabel(int row, int col) const {
		return (Label)_labels.at<uchar>(row, col);
	}

	inline const cv::Mat & getLabelImage() const {
		return _labels;
	}

	inline const cv::Mat & getDepthImage() const {
		return _depth;
	}

	inline void setDepthImage(cv::Mat depth) {
		_depth = depth;
	}

	inline void setLabelImage(cv::Mat fw_mask) {
		_labels = fw_mask;
	}

	void show();
	void computeForegroundFeatures(Data & features);

	inline float operator()(int row, int col) const ;

	inline void setLabel(int row, int col, Label value) {
		_labels.at<uchar>(row, col) = (uchar)value;
	}

	inline cv::Size getImageSize() const {
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
	static void create(float);
	static std::vector<Frame> image_vector;

private:

};
