#include <RandomForest.hpp>
#include <Frame.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

int main(int argc, char **argv) {

	std::ifstream file("tree.dat");
	cereal::BinaryInputArchive ar(file);
	RandomTree tree;
	tree.serialize<cereal::BinaryInputArchive>(ar);

	std::cout << "Loaded tree with " << tree.getNumNodes() << " nodes." << std::endl;

	int num_seq = 1;
	int num_im = 7;
	int num_camera = 1;
	int charbuffsize = 500;
	std::string main_db_path = getenv(MAIN_DB_PATH);
	std::unique_ptr<char[]> buf( new char[charbuffsize] );
	std::snprintf( buf.get(), charbuffsize,
		       "%s/train/%d/images/depthRender/Cam%d/mayaProject.%06d.png",
		       main_db_path.c_str(),
		       num_seq, num_camera, num_im);

	std::string path_depth(buf.get());

	std::snprintf( buf.get(), charbuffsize,
		       "%s/train/%d/images/groundtruth/Cam%d/mayaProject.%06d.png",
		       main_db_path.c_str(),
		       num_seq, num_camera, num_im);

	std::string path_gt(buf.get());

	Frame test_frame(path_depth, path_gt);

	Frame output = tree.predict(test_frame);

	output.show();

}
