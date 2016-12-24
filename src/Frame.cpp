#include "Frame.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"
#include <fstream>

double bytesToGigabytes(long bytes)
{
	return bytes * 9.31322574615479e-10;
}

std::vector<FramePtr> FramePool::image_vector = std::vector<FramePtr>();

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

	void sampleFromForeground(const FramePtr frame, int & row, int & col) {

		int sampled = 0;
		cv::Size size = frame->getImageSize();

		do {
			//generate row
			row = random_int(0, size.height-1);
			col = random_int(0, size.width-1);


		} while (frame->getLabel(row, col) == Background);
		assert(frame->operator()(row,col) != 0);
	}

	inline bool outsideFrame(int row, int col, ConstFramePtr im) {
		return row < 0 || row > im->getImageSize().height-1 ||
			col < 0 || col > im->getImageSize().width-1;
	}

	float getOutsideOfFrameDepth() {
		switch(Settings::bmode) {

		case BackgroundMode::DEFAULT:
			return MAX_DEPTH;
			break;

		case BackgroundMode::RANDOM_MIDRANGE:
			//random uniform in range
			return random_real_in_range(Settings::bg_mid_range);
			break;

		case BackgroundMode::RANDOM_LONGRANGE:
			return random_real_in_range(Settings::bg_long_range);
			break;

		default:
			return MAX_DEPTH;
		}
	}

	void setBackgroundToMaxDepth(cv::Mat & depth, const cv::Mat & mask) {
		std::vector<cv::Point2i> locations;
		cv::findNonZero(1 - mask, locations);
		for (auto p : locations) {
			depth.at<float>(p) = getOutsideOfFrameDepth();
		}
	}
}

Feature::Feature(int row, int col, Label label, ConstFramePtr image) :
	_row(row)
	, _col(col)
	, _label(label)
	, _value(0.0)
	,_image(image) {
}

ConstFramePtr Feature::getFrame() const {
	return _image;
}

void Feature::evaluate(const LearnerParameters & params) {

	ConstFramePtr im = getFrame();
	float z = im->operator()(_row, _col);

	assert(z != 0);

	int row = _row + int(params.offset_1[0]/z);
	int col = _col + int(params.offset_1[1]/z);

	if (FrameUtils::outsideFrame(row, col, im)) {
		_value = FrameUtils::getOutsideOfFrameDepth();
	}
	else {
		_value = im->operator()(row, col);
	}

	if (!params.is_unary) {

		row = _row + int(params.offset_2[0]/z);
		col = _col + int(params.offset_2[1]/z);

		if (FrameUtils::outsideFrame(row, col, im)) {
			z = FrameUtils::getOutsideOfFrameDepth();
		}
		else {
			z = im->operator()(row, col);
		}
	}

	_value -= z;
}

bool file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

bool FramePool::create(float max_size) {

	std::string main_db_path = getenv(MAIN_DB_PATH);
	int max_images_per_cam = 1000;
	int max_sequences = 175;
	int max_cams = 3;
	int charbuffsize = 500;

	std::unique_ptr<char[]> buf( new char[ charbuffsize ] );

	static int num_seq = 1;
	static int num_im = 1;
	static int num_cam = 1;

	int total_frames = 0;
	float size = 0;

	FramePool::image_vector.clear();

	while(size <= max_size) {

		std::snprintf( buf.get(), charbuffsize,
			       "%s/train/%d/images/depthRender/Cam%d/mayaProject.%06d.png",
			       main_db_path.c_str(),
			       num_seq, num_cam, num_im);

		std::string path_depth(buf.get());

		std::snprintf( buf.get(), charbuffsize,
			       "%s/train/%d/images/groundtruth/Cam%d/mayaProject.%06d.png",
			       main_db_path.c_str(),
			       num_seq, num_cam, num_im);

		std::string path_gt(buf.get());

		std::cout << "Loading image " << buf.get() << std::endl;
		if (!file_exist(buf.get())) {
			std::cout << "Error: " << buf.get() << " does not exist" << std::endl;
			return false;
		}

		FramePtr frame = std::make_shared<Frame>(path_depth, path_gt);

		FramePool::image_vector.push_back(frame);

		if (num_im <= max_images_per_cam) {
			num_im++;
		}
		else {
			num_im = 1;
			if (num_cam < max_cams) {
				num_cam++;
			}
			else {
				num_cam = 1;
				if (num_seq < max_sequences) {
					num_seq++;
				}
				else break;
			}
		}


		size += bytesToGigabytes(frame->getFrameSizeInBytes());
		total_frames++;
		std::cout << "Total images: " << total_frames << " - Memory used:" << size << "G" << std::endl;
	}

	return true;
}

void FramePool::computeFeatures(DataPtr features) {

	int row = 0;
	int col = 0;
	int idx = 0;
	features->resize(FramePool::image_vector.size()*Settings::num_pixels_per_image);

	// For each image sample uniformly pixels in the foreground
	for (int im_id = 0; im_id < FramePool::image_vector.size(); im_id++) {

		for (int i = 0; i < Settings::num_pixels_per_image; i++) {

			FramePtr image = FramePool::image_vector[im_id];
			FrameUtils::sampleFromForeground(image, row, col);
			features->operator[](idx) = Feature(row, col, image->getLabel(row, col), image);
			idx++;
		}
	}
}

Frame::Frame(std::string depth_path,
	     std::string gt_path) {
	load(depth_path, gt_path);
}

void Frame::show() {

	cv::Mat depth = (_depth/1.03 - 50)*255./(800.-50.);
	depth.convertTo(depth, CV_8UC1);
	cv::imshow("depth2", depth);

	cv::Mat labels(_labels.size(), CV_8UC3);

	for (int row = 0; row < _labels.rows; row++) {
		for (int col = 0; col < _labels.cols; col++) {
			uchar label = _labels.at<uchar>(row, col);
			labels.at<color>(row,col) = FrameUtils::color_map[label];
		}
	}

	cv::imshow("labels", labels);
	cv::waitKey(1);
}

void Frame::computeForegroundFeatures(Data & features) {

	features.clear();
	cv::Size s = getImageSize();
	for (int row = 0; row < s.height; row++) {
		for (int col = 0; col < s.width; col++) {
			Label l = getLabel(row, col);
			if (l != Background) {
				features.push_back(Feature(row, col, l, shared_from_this()));
			}
		}
	}
}

void Frame::load(std::string depth_path,
		 std::string gt_path) {

	cv::Mat gt_image = cv::imread(gt_path, -1);
	cv::Mat label_image = cv::Mat(gt_image.size(), CV_8UC1);
	label_image.setTo(0);
	cv::Mat rgba[4];
	cv::split(gt_image, rgba);

	cv::Mat fw_mask = cv::Mat(gt_image.size(), CV_8UC1);
	fw_mask.setTo(0);
	fw_mask.setTo(1,rgba[3] > 0);

	int num_labels = FrameUtils::color_map.size();

	// Generate label image
	for (uchar label = 0; label < num_labels; label++) {

		cv::Mat Labels =
			(abs(rgba[0] - FrameUtils::color_map[label][2]) < 3) &
			(abs(rgba[1] - FrameUtils::color_map[label][1]) < 3) &
			(abs(rgba[2] - FrameUtils::color_map[label][0]) < 3) &
			fw_mask;

		cv::Size s = label_image.size();
		for (int row = 0; row < s.height; row++) {
			for (int col = 0; col < s.width; col++) {
				if ((int)fw_mask.at<uchar>(row,col) == 0)
					label_image.at<uchar>(row,col) = Background;
				else
					label_image.at<uchar>(row,col) = label;
			}
		}
	}

	// Crop foreground data
	cv::Mat depth = FrameUtils::cropForeground(
		cv::imread(depth_path, CV_LOAD_IMAGE_GRAYSCALE), rgba[3]);

	// Crop mask
	cv::Mat cropped_mask = FrameUtils::cropForeground(fw_mask, fw_mask);

	// Transform to centimetersa
	depth.convertTo(_depth, CV_32FC1);
	_depth = (_depth/255. * (800-50) + 50)*1.03;


	// zero depth?
	cv::Size s = _depth.size();
	for (int row = 0; row < s.height; row++)
		for (int col = 0; col < s.width; col++)
			assert(_depth.at<float>(row,col) > 0);


	FrameUtils::setBackgroundToMaxDepth(_depth, cropped_mask);
	_labels = FrameUtils::cropForeground(label_image, rgba[3]);
}
