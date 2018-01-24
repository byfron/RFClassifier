#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <vector>
#include <memory>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <RandomForest.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

struct DepthFrame {
	uint16_t width, height;
	std::vector<uint8_t> data;
};


namespace {

	Frame convertToFrame(DepthFrame & in) {

		Frame f;
		cv::Mat depth(cv::Size(in.width, in.height), CV_32FC1, in.data.data());

		// Convert to cm:
		depth /= 10.;

		// Mask foreground < 2m
//		FrameUtils::setBackgroundToMaxDepth(depth, depth > 200.);

		cv::Mat bw_mask = depth > 200;
		FrameUtils::setBackgroundToMaxDepth(depth, 1 - bw_mask);

		cv::Mat masked_depth = cv::Mat(depth.size(), CV_32FC1);
		masked_depth.setTo(0);

		depth.copyTo(masked_depth, 1 - bw_mask);

		cv::imshow("mask", masked_depth);
		cv::waitKey(1);

		f.setDepthImage(depth);

		//TODO: In practice we would like to predict also for background features
		std::vector<cv::Point2i> locations;
		cv::findNonZero(depth > 200, locations);
		cv::Mat labels(depth.size(), CV_8UC1);
		labels.setTo((uchar)Labels::Foreground);
		for (auto p : locations) {
			labels.at<uchar>(p) = (uchar)Labels::Background;
		}


		// static std::vector<color> mask_colmap =  { {0,0,0}, {255,255,255} };

		// for (int row = 0; row < labels.rows; row++) {
		// 	for (int col = 0; col < labels.cols; col++) {
		// 		if (depth.at<float>(row,col) < 100.) {
		// 			labels.at<uchar>(row,col) = 1;
		// 		}
		// 	}
		// }

		f.setLabelImage(labels);

		return f;
	}
}

class KinectCamera {

public:

	KinectCamera() {
		_serial = _freenect2.getDefaultDeviceSerialNumber();
		_dev = _freenect2.openDevice(_serial);
		if(_dev == 0) throw std::runtime_error ("Error opening device");
		int types = 0;
		types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
		_listener = std::make_shared<libfreenect2::SyncMultiFrameListener>(types);
		_dev->setIrAndDepthFrameListener(_listener.get());
	}

	void start() {
		_dev->start();
	}

	void capture(DepthFrame & frame) {

		libfreenect2::FrameMap frames;
		if (!_listener->waitForNewFrame(frames, 10*1000)) // 10 sconds
		{
			throw std::runtime_error ("Camera time-out");
		}

		libfreenect2::Frame *freenect_depth = frames[libfreenect2::Frame::Depth];

		int numbytes =
			freenect_depth->width *
			freenect_depth->height *
			freenect_depth->bytes_per_pixel;

		frame.width = freenect_depth->width;
		frame.height = freenect_depth->height;
		frame.data.resize(numbytes);
		std::memcpy(frame.data.data(), freenect_depth->data, numbytes);
		_listener->release(frames);
	}

private:
	std::string _serial;
	std::shared_ptr<libfreenect2::SyncMultiFrameListener> _listener;
	libfreenect2::Freenect2 _freenect2;
	libfreenect2::Freenect2Device *_dev = 0;
};


int main(int argc, char **argv) {

	if (!cmdOptionExists(argv, argv + argc, "-i")) {
		std::cout << "Usage: ./predict -i <forest_input_file>" << std::endl;
		return 0;
	}

	KinectCamera camera;
	char * input_file = getCmdOption(argv, argv + argc, "-i");


	std::ifstream file(input_file);
	cereal::BinaryInputArchive ar(file);
	RandomForest forest;
	forest.serialize<cereal::BinaryInputArchive>(ar);


	camera.start();

	for (;;) {

		DepthFrame frame;
		camera.capture(frame);

		Frame f = convertToFrame(frame);
//		f.show();
		std::cout << "running prediction" << std::endl;
		Frame output = forest.predict(f);
		std::cout << "DONE" << std::endl;

		output.show();


	}
}
