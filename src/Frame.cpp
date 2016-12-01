#include "Frame.hpp"

std::vector<Frame> FramePool::image_vector = std::vector<Frame>();

namespace FrameUtils{
	
std::vector<color> color_map =  {
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

	cv::Mat cropForeground(cv::Mat im, cv::Mat mask) {
		
		cv::Mat fw_locations;
		const int WIDTH = 500;
		const int HEIGHT = 500;       
		int max_row = 0;
		int max_col = 0;
		int min_row = im.rows;
		int min_col = im.cols;
	
		cv::findNonZero(mask, fw_locations);
		for (int i = 0; i < fw_locations.total(); i++ ) {
			cv::Point p = fw_locations.at<cv::Point>(i); 
			if (min_col > p.x) min_col = p.x;
			if (max_col < p.x) max_col = p.x; 
			if (min_row > p.y) min_row = p.y;
			if (max_row < p.y) max_row = p.y; 
		}

		cv::Rect ROI = cv::Rect(min_col, min_row,
					max_col - min_col,
					max_row - min_row);

		return im(ROI);
	}
}

Frame::Frame(std::string depth_path,
		  std::string gt_path) {
	load(depth_path, gt_path);
}

void Frame::load(std::string depth_path,
		 std::string gt_path) {	
	
	cv::Mat gt_image = cv::imread(gt_path, -1);
	cv::Mat label_image = cv::Mat(gt_image.size(), CV_8UC1);
	label_image.setTo(0);
	cv::Mat rgba[4];
	cv::split(gt_image, rgba);

	int num_labels = FrameUtils::color_map.size();

	// Generate label image
	for (int label = 0; label < num_labels; label++) {

		cv::Mat Labels =
			(abs(rgba[0] - FrameUtils::color_map[label][2]) < 3) &
			(abs(rgba[1] - FrameUtils::color_map[label][1]) < 3) &
			(abs(rgba[2] - FrameUtils::color_map[label][0]) < 3) &
			rgba[3];

		std::vector<cv::Point2i> locations;
		cv::findNonZero(Labels, locations);

		for (auto p : locations) {
			label_image.at<uchar>(p) = (char)label;
		}
	}

	// Crop foreground data
	_depth = FrameUtils::cropForeground(
		cv::imread(depth_path, CV_LOAD_IMAGE_GRAYSCALE), rgba[3]);
	_labels = FrameUtils::cropForeground(label_image, rgba[3]);
}

const float& Frame::operator()(int row, int col) {
	return _depth.at<float>(row, col);
}
