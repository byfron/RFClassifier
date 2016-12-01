#pragma once
#include <random>
#include <opencv2/opencv.hpp>

typedef cv::Vec<float,3> Vec3;

class RandomNumberGenerator {

public:
	RandomNumberGenerator() : gen(rd()),
				  dis(0.0, 1.0) {
	}

	float generateUniform() {
		return dis(gen);
	}

private:

	std::uniform_real_distribution<> dis;
        std::random_device rd;
	std::mt19937 gen;
};

class RandomGenerator {
public:
	static float generateUniform() {
		return random.generateUniform();
	}
	static RandomNumberGenerator random;
};


class Settings {
public:
	static std::string data_path;
	static int num_trees;
	static int num_images_per_tree;
	static int num_thresholds_per_feature;
	static int num_offsets_per_pixel;
	static int max_tree_depth;
	static float max_offset;
};
