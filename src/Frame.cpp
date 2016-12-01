#include "Frame.hpp"

std::vector<Frame> FramePool::image_vector = std::vector<Frame>();

std::vector<color> Frame::color_map =  {
	 {255,106,  0},
	 {0, 0, 0},
	 {255,  0,  0},
	 {255,178,127},
	 {255,127,127},
	 {182,255,  0},
	 {218,255, 127},
	 {255,216, 0},
	 {255,233, 127},
	 {0  ,148, 255},
	 {72 , 0, 255},
	 {48 ,48, 48},
	 {76 ,255, 0},
	 {0  ,255, 33},
	 {0  ,255, 255},
	 {0  ,255, 144},
	 {178, 0, 255},
	 {127, 116, 63},
	 {127, 63, 63},
	 {127, 201, 255},
	 {127, 255, 255},
	 {165, 255, 127},
	 {127, 255, 197},
	 {214, 127, 255},
	 {161, 127, 255},
	 {107, 63, 127},
	 {63 ,73, 127},
	 {63 ,127, 127},
	 {109, 127, 63},
	 {255, 127, 237},
	 {127, 63, 118},
	 {0 ,74, 127},
	 {255, 0, 110},
	 {0 ,127, 70},
	 {127, 0, 0},
	 {33 ,0, 127},
	 {127, 0, 55},
	 {38, 127, 0},
	 {127, 51, 0},
	 {64, 64, 64},
	 {73, 73, 73},
	 {191, 168, 247},
	 {192, 192, 192},
	 {127, 63, 63},
	 {127, 116, 63}};

Frame::Frame(std::string depth_path,
		  std::string gt_path) {
	load(depth_path, gt_path);
}

void Frame::load(std::string depth_path,
		 std::string gt_path) {

	_depth = cv::imread(depth_path, CV_LOAD_IMAGE_GRAYSCALE);
	
	cv::Mat gt_image = cv::imread(gt_path, -1);
	cv::Mat label_image = cv::Mat(gt_image.size(), CV_8UC1);
	label_image.setTo(0);
	cv::Mat rgba[4];
	cv::split(gt_image, rgba);

	int num_labels = Frame::color_map.size();

	// Generate label image
	for (int label = 0; label < num_labels; label++) {

		cv::Mat Labels =
			(abs(rgba[0] - Frame::color_map[label][2]) < 3) &
			(abs(rgba[1] - Frame::color_map[label][1]) < 3) &
			(abs(rgba[2] - Frame::color_map[label][0]) < 3) &
			rgba[3];

		std::vector<cv::Point2i> locations;
		cv::findNonZero(Labels, locations);

		for (auto p : locations) {
			label_image.at<uchar>(p) = (char)label;
		}
	}

	_labels = label_image;
}

const float& Frame::operator()(int row, int col) {
	return _depth.at<float>(row, col);
}
