#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <stdlib.h>

class LearnerParameters;
class Feature;

enum Labels {
	Background = 255,
	Foreground = 0
};

typedef std::vector<Feature> Data;
typedef std::shared_ptr<Data> DataPtr;
typedef Data::iterator FeatureIterator;

class Frame : public std::enable_shared_from_this<Frame>{
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

	inline bool inForeground(int row, int col) const {
	        return (int)_labels.at<uchar>(row,col) != (int)Labels::Background;
	}

	inline float operator()(int row, int col) const {
		return _depth.at<float>(row, col);
	}

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

typedef std::shared_ptr<Frame> FramePtr;
typedef std::shared_ptr<const Frame> ConstFramePtr;


namespace FrameUtils {
	void setBackgroundToMaxDepth(cv::Mat & depth, const cv::Mat & mask);
	void sampleFromForeground(const FramePtr frame, int & row, int & col);
}


class Feature {
public:
	Feature() {}
	Feature(int row, int col, Label label, ConstFramePtr im);
	void evaluate(const LearnerParameters & params);

	inline bool operator< (const Feature& f) const {
		return _value < f._value;
	}

	inline ConstFramePtr getFrame() const;
	inline const float & getValue() const { return _value; }
	inline const Label & getLabel() const { return _label; }
	inline const int row() const { return _row; }
	inline const int col() const { return _col; }

private:
	int _row;
	int _col;
	Label _label;
	float _value;
	ConstFramePtr _image;
};

class FramePool {
public:

	static void computeFeatures(DataPtr);
	static bool create(float);
	static std::vector<FramePtr> image_vector;

private:

};
