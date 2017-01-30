#include <RandomForest.hpp>
#include <Frame.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

int main(int argc, char **argv) {

	char * input_file = getCmdOption(argv, argv + argc, "-i");

	if (!input_file) {
		std::cout << "Usage: ./predict -i <forest_input_file>" << std::endl;
	}

	std::ifstream file(input_file);
	cereal::BinaryInputArchive ar(file);
//	RandomForest forest;
//	forest.serialize<cereal::BinaryInputArchive>(ar);

	RandomTree tree;
	tree.serialize<cereal::BinaryInputArchive>(ar);


	int num_seq = 4;
	int num_im = 7;
	int num_camera = 1;
	int charbuffsize = 500;
	std::string main_db_path = getenv(MAIN_DB_PATH);
	std::unique_ptr<char[]> buf( new char[charbuffsize] );
	std::snprintf( buf.get(), charbuffsize,
		       "%s/test/%d/images/depthRender/Cam%d/mayaProject.%06d.png",
		       main_db_path.c_str(),
		       num_seq, num_camera, num_im);

	std::string path_depth(buf.get());

	std::snprintf( buf.get(), charbuffsize,
		       "%s/test/%d/images/groundtruth/Cam%d/mayaProject.%06d.png",
		       main_db_path.c_str(),
		       num_seq, num_camera, num_im);

	std::string path_gt(buf.get());

	Settings::bmode = BackgroundMode::DEFAULT;

	FramePtr test_frame = std::make_shared<Frame>(path_depth, path_gt);

	Frame output = tree.predict(test_frame);

	output.show();

}
