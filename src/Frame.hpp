#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <stdlib.h>

class Frame {
public:

	Frame(std::string,std::string);

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

		std::string main_db_path = getenv(MAIN_DB_PATH);
		int num_images_per_seq = 100;
		int num_max_sequences = 180;
		int num_camera = 1;
		int size = 500;

		std::unique_ptr<char[]> buf( new char[ size ] );

		for (int num_seq = 1; num_seq <= num_max_sequences; num_seq++) {
			for (int num_im = 1; num_im < num_images_per_seq; num_im++) {
				std::snprintf( buf.get(), size,
					       "%s/train/%d/images/depthRender/Cam%d/mayaProject.%06d.png",
					       main_db_path.c_str(),
					       num_seq, num_camera, num_im);

				std::string path_depth(buf.get());

				std::snprintf( buf.get(), size,
					       "%s/train/%d/images/groundtruth/Cam%d/mayaProject.%06d.png",
					       main_db_path.c_str(),
					       num_seq, num_camera, num_im);

				std::string path_gt(buf.get());

				FramePool::image_vector.push_back(Frame(path_depth, path_gt));
			}
		}

	}

	static std::vector<Frame> image_vector;

private:

};
