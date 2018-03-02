#include <RandomForest.hpp>
#include <Frame.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

int main(int argc, char **argv) {

	char * input_file = getCmdOption(argv, argv + argc, "-i");

	if (!input_file) {
		std::cout << "Usage: ./predict -i <forest_input_file>" << std::endl;
	}

	FramePool::initializeColorMap();
	
	std::ifstream file(input_file);
	cereal::BinaryInputArchive ar(file);
	RandomForest forest;
	forest.serialize<cereal::BinaryInputArchive>(ar);
//	forest.createGPUBuffers();
	
	
	
 	//RandomTree tree;
	//tree.serialize<cereal::BinaryInputArchive>(ar);

	int num_seq = 200;
	int num_im = 2;
	int max_images = 20;

	for (int i = 0; i < max_images; i++) {

		num_im += i;
		
		int charbuffsize = 500;
		std::string main_db_path = getenv(MAIN_DB_PATH);
		std::unique_ptr<char[]> buf( new char[charbuffsize] );
		std::snprintf( buf.get(), charbuffsize,
					   "%s/%d/capturedframe%d.png",				   
					   main_db_path.c_str(),
					   num_seq, num_im);

		std::string path_depth(buf.get());

		std::snprintf( buf.get(), charbuffsize,
					   "%s/%d/labelsframe%d.png",
					   main_db_path.c_str(),
					   num_seq, num_im);

		std::string path_gt(buf.get());

		std::cout << "Loading :" << path_depth << std::endl;
		
		Settings::bmode = BackgroundMode::DEFAULT;

		FramePtr test_frame = std::make_shared<Frame>(path_depth, path_gt);

		using nano = std::chrono::nanoseconds;
		auto start = std::chrono::high_resolution_clock::now();	   
		Frame output = forest.predict(test_frame);
		auto finish = std::chrono::high_resolution_clock::now();
		std::cout << "GPU took "
				  << std::chrono::duration_cast<nano>(finish - start).count()
				  << " nanoseconds\n";
		
		start = std::chrono::high_resolution_clock::now();	   
		Frame output2 = forest.predict(test_frame);
		finish = std::chrono::high_resolution_clock::now();
		std::cout << "CPU took "
				  << std::chrono::duration_cast<nano>(finish - start).count()
				  << " nanoseconds\n";
		
		
		
		cv::Size size = output.getImageSize();
		cv::Mat result = cv::Mat(cv::Size(size.width*3, size.height), CV_8UC3);

		cv::Mat roi_rgb = result(cv::Rect(0, 0, size.width, size.height));					
		test_frame->getColoredLabels().copyTo(roi_rgb);

		cv::Mat roi_gt = result(cv::Rect(size.width, 0, size.width, size.height));					
		output.getColoredLabels().copyTo(roi_gt);

		cv::Mat depth_3c;
		cv::Mat scaled_depth = (test_frame->getDepthImage()-0.5)*200;
		cv::Mat in[] = {scaled_depth, scaled_depth, scaled_depth};
		cv::merge(in, 3, depth_3c);
		depth_3c.convertTo(depth_3c, CV_8UC3);
		cv::Mat roi_depth = result(cv::Rect(size.width*2, 0, size.width, size.height));
		depth_3c.copyTo(roi_depth);
		
		//cv::imshow("depth", test_frame->getDepthImage()-0.5);
		//cv::imshow("GT", test_frame->getColoredLabels());
		//cv::imshow("result", output.getColoredLabels());
		cv::imshow("output", result);
		cv::waitKey(0);

		char buff[50];
		sprintf(buff, "res%03d.png", i);
		cv::imwrite(buff, result);

	}
	

}
