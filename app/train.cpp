#include <RandomForest.hpp>
#include <Frame.hpp>
#include <fstream>

int main(int argc, char **argv) {

	char * mem_limit = getCmdOption(argv, argv + argc, "-l");
	char * ntrees = getCmdOption(argv, argv + argc, "-n");
	char * output_file = getCmdOption(argv, argv + argc, "-o");
	int charbuffsize = 500;

	if (!output_file) {
		std::cout << "Usage: ./train -o <forest_output_file> [-l <max. num of gigabytes per tree> -n <num. trees>] " << std::endl;
		return -1;
	}

	int num_trees = 1;
	float limit = 0.01;
	if (ntrees) {
		num_trees = atoi(ntrees);
	}
	if (mem_limit) {
		limit = atof(mem_limit);
	}

	RandomForest forest;

	for (int i = 0; i < num_trees; i++) {

		std::unique_ptr<char[]> buf( new char[ charbuffsize ] );
		std::snprintf( buf.get(), charbuffsize, "tree_%02d_%s", i, output_file);

		// Load training
		if (FramePool::create(limit)) {

			DataPtr data = std::make_shared<Data>();
			FramePool::computeFeatures(data);

			RandomTree tree;
			tree.train(data);

			forest.push_tree(tree);

			// save tree
			tree.save(buf.get());
		}
		else {
			std::cout << "Can't create Frame Pool. Exiting..." << std::endl;
			return -1;
		}
	}

	std::unique_ptr<char[]> buf( new char[ charbuffsize ] );
	std::snprintf( buf.get(), charbuffsize, "forest_%s", output_file);
	forest.save(buf.get());

	return 0;
}
