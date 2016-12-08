#include "Frame.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"

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

	void sampleFromForeground(const Frame & frame, int & row, int & col) {

		int sampled = 0;
		cv::Size size = frame.getImageSize();

		do {
			//generate row
			row = random_int(0, size.height);
			col = random_int(0, size.width);


		} while (frame.getLabel(row, col) == (int)Labels::Background);
		//TODO: Is Background=0 the correct bg label?
	}
}

Feature::Feature(int row, int col, int label, int image_id) :
	_row(row), _col(col), _label(label), _image_id(image_id) {
}

void Feature::evaluate(LearnerParameters & params) {

	Frame & im = FramePool::image_vector[_image_id];
	float z = im(_row, _col);

	//If offset is outside the image project it back
	int row = std::min(std::max(0, _row + int(params.offset_1[0]/z)),
			   im.getImageSize().height);
	int col = std::min(std::max(0, _col + int(params.offset_1[1]/z)),
			   im.getImageSize().width);

	_value = im(row, col);

	if (!params.is_unary) {
		row = std::max(std::min(0, _row + int(params.offset_2[0]/z)),
			       im.getImageSize().height);
		col = std::max(std::min(0, _col + int(params.offset_2[1]/z)),
					im.getImageSize().width);
		z = im(row, col);
	}

	_value -= z;
}

void FramePool::create() {

	std::string main_db_path = getenv(MAIN_DB_PATH);
	int num_images_per_seq = 10;
	int num_max_sequences = 1;
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

std::vector<Feature> FramePool::computeFeatures() {

	int row, col;
	std::vector<Feature> features;

	// For each image sample uniformly pixels in the foreground
	for (int im_id = 0; im_id < FramePool::image_vector.size(); im_id++) {

		for (int i = 0; i < Settings::num_pixels_per_image; i++) {

			Frame & image = FramePool::image_vector[im_id];

			FrameUtils::sampleFromForeground(image, row, col);
			features.push_back(Feature(row, col, image.getLabel(row, col), im_id));
		}
	}

	return features;
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
	for (uchar label = 0; label < num_labels; label++) {

		cv::Mat Labels =
			(abs(rgba[0] - FrameUtils::color_map[label][2]) < 3) &
			(abs(rgba[1] - FrameUtils::color_map[label][1]) < 3) &
			(abs(rgba[2] - FrameUtils::color_map[label][0]) < 3) &
			rgba[3];

		std::vector<cv::Point2i> locations;
		int count = countNonZero(Labels);
		if (count > 0)
			cv::findNonZero(Labels, locations);

		for (auto p : locations) {
			label_image.at<uchar>(p) = label;
		}
	}

	// Crop foreground data
	cv::Mat depth = FrameUtils::cropForeground(
		cv::imread(depth_path, CV_LOAD_IMAGE_GRAYSCALE), rgba[3]);



	// Transform to centimeters
	depth.convertTo(_depth, CV_32FC1);
	_depth = (_depth/255. * (800-50) + 50)*1.03;

	// TODO:Set background to maximum value

	_labels = FrameUtils::cropForeground(label_image, rgba[3]);
}

const float& Frame::operator()(int row, int col) {
	return _depth.at<float>(row, col);
}
