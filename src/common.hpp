#pragma once
#include <opencv2/opencv.hpp>
#include <random>

#define MAIN_DB_PATH "DB_PATH"

typedef cv::Vec<uchar,3> color;
typedef int Label;

class Settings {
public:
	static int num_trees;
	static int num_images_per_tree;
	static int num_thresholds_per_feature;
	static int num_offsets_per_pixel;
	static int num_pixels_per_image;
	static int max_tree_depth;
	static float maximum_depth_difference;
	static int offset_box_size;
	static int num_labels;
};
