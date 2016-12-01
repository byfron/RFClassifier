#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>

class Frame {
public:

	void load(std::string depth_path,
		  std::string gt_path);

	const float& operator()(int row, int col);

	static std::vector<Vec3> color_map;

private:

	cv::Mat _depth;
	cv::Mat _labels;
};

class FramePool {
public:

	static void create() {
	}

	static Frame & getFrame(int id) {
		return _image_vector[id];
	}

	static std::vector<Frame> _image_vector;
};
