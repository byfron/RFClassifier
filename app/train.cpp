#include <RandomForest.hpp>
#include <Frame.hpp>
#include <fstream>

int main(int argc, char **argv) {

	char * mem_limit = getCmdOption(argv, argv + argc, "-l");
	char * output_file = getCmdOption(argv, argv + argc, "-o");

	if (!output_file) {
		std::cout << "Usage: ./train -o <forest_output_file> [-l <max. num of gigabytes>] " << std::endl;
		return -1;
	}

	float limit = 0.01;
	if (mem_limit) {
		limit = atof(mem_limit);
	}

	// Load training
	FramePool::create(limit);

	DataPtr data = std::make_shared<Data>();
	FramePool::computeFeatures(data);

	// Train forest
	RandomForest random_forest;
	random_forest.train(data);

	std::ofstream file(output_file, std::ios::binary);
	cereal::BinaryOutputArchive ar(file);
	ar(random_forest);

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
