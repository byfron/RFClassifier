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

	FramePtr convertToFrame(DepthFrame & in) {

		float threshold = 1.5f;
		
		FramePtr f = std::make_shared<Frame>();
		cv::Mat depth(cv::Size(in.width, in.height), CV_32FC1, in.data.data());

		// Convert to m:
		depth /= 1000.0f;


		// Mask foreground < 2m
		//FrameUtils::setBackgroundToMaxDepth(depth, depth > 1500);

		cv::Mat fw_mask = cv::Mat(depth.size(), CV_8UC1);
		fw_mask.setTo(0);
		fw_mask.setTo(1, depth != 0 & depth < threshold);
		
		// cv::Mat bw_mask = depth > threshold;
		// // FrameUtils::setBackgroundToMaxDepth(depth, 1 - bw_mask);

		// cv::Mat masked_depth = cv::Mat(depth.size(), CV_32FC1);
		// masked_depth.setTo(0);
		// depth.copyTo(masked_depth, 1 - bw_mask);
	

		
		

		//depth = FrameUtils::cropForeground(depth, fw_mask);
		//FrameUtils::setBackgroundToMaxDepth(depth, fw_mask);

		

		//TODO: In practice we would like to predict also for background features
		// std::vector<cv::Point2i> locations;
		// cv::findNonZero(depth > threshold, locations);
		cv::Mat labels(depth.size(), CV_8UC1);
		labels.setTo((uchar)Labels::Background);
		labels.setTo((uchar)Labels::Foreground, fw_mask);
		// for (auto p : locations) {
		// 	labels.at<uchar>(p) = (uchar)Labels::Background;
		// }


		cv::Mat cropped_mask = FrameUtils::cropForeground(fw_mask, fw_mask);
		labels = FrameUtils::cropForeground(labels, fw_mask);
		depth = FrameUtils::cropForeground(depth, fw_mask);
		FrameUtils::setBackgroundToMaxDepth(depth, cropped_mask);		
		
		cv::Mat depth_3c;
		cv::Mat scaled_depth = (depth-0.5)*200;
		cv::Mat ind[] = {scaled_depth, scaled_depth, scaled_depth};
		cv::merge(ind, 3, depth_3c);
		depth_3c.convertTo(depth_3c, CV_8UC3);
		cv::imshow("depth", depth_3c);
		
		f->setLabelImage(labels);
		f->setDepthImage(depth);

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

	FramePool::initializeColorMap();		

	KinectCamera camera;
	char * input_file = getCmdOption(argv, argv + argc, "-i");


	std::ifstream file(input_file);
	cereal::BinaryInputArchive ar(file);
	RandomForest forest;
	forest.serialize<cereal::BinaryInputArchive>(ar);
	forest.createGPUBuffers();
	
	//RandomForest forest;
	//forest.serialize<cereal::BinaryInputArchive>(ar);


	camera.start();
	int idx = 0;
	
	for (;;) {

		DepthFrame frame;
		camera.capture(frame);

		FramePtr f = convertToFrame(frame);
		//f->show();
		std::cout << "running prediction" << std::endl;
		Frame output = forest.predictGPU(f);
		std::cout << "DONE" << std::endl;

		//output.show();

		cv::Size size = output.getImageSize();
		cv::Mat result = cv::Mat(cv::Size(size.width*2, size.height), CV_8UC3);

		cv::Mat roi_gt = result(cv::Rect(size.width, 0, size.width, size.height));					
		output.getColoredLabels().copyTo(roi_gt);

		cv::Mat depth_3c;
		cv::Mat scaled_depth = (output.getDepthImage()-0.5)*200;
		cv::Mat in[] = {scaled_depth, scaled_depth, scaled_depth};
		cv::merge(in, 3, depth_3c);
		depth_3c.convertTo(depth_3c, CV_8UC3);
		cv::Mat roi_depth = result(cv::Rect(0, 0, size.width, size.height));
		depth_3c.copyTo(roi_depth);

		cv::imshow("res", result);
		cv::waitKey(1);
		
		char buff[50];
		sprintf(buff, "kinect_res%03d.png", idx);
		cv::imwrite(buff, result);
		idx++;

	}
}
