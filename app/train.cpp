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

	for (int i = 0; i < num_trees; i++) {

		std::unique_ptr<char[]> buf( new char[ charbuffsize ] );
		std::snprintf( buf.get(), charbuffsize, "tree_%02d_%s", i, output_file);

		// Load training
		FramePool::create(limit);

		DataPtr data = std::make_shared<Data>();
		FramePool::computeFeatures(data);

		RandomTree tree;
		tree.train(data);


		// save tree
		std::ofstream file(buf.get(), std::ios::binary);
		cereal::BinaryOutputArchive ar(file);
		ar(tree);
	}

	// // Train forest
	// RandomForest random_forest;
	// random_forest.train(data);

	// std::ofstream file(output_file, std::ios::binary);
	// cereal::BinaryOutputArchive ar(file);
	// ar(random_forest);

	//Check result on traning
	// std::vector<Label> labels = random_forest.predict(data);
	// assert(labels.size() == data->size());
	// int idx = 0;
	// for (auto l : labels) {
	// 	std::cout << (int)l << "-" << (int)data->operator[](idx).getLabel() << std::endl;
	// 	idx++;
	// }

	return 0;
}
