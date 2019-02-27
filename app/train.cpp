#include <../byfron-utils/src/Profiler.hpp>
#include <RandomForest.hpp>
#include <Frame.hpp>
#include <fstream>
#include <thread>

bool train_tree(std::shared_ptr<RandomTree> tree, std::shared_ptr<FramePool> pool) {
	DataPtr data = std::make_shared<Data>();
	pool->computeFeatures(data);
	tree->train(data);
	return true;
}

int main(int argc, char **argv) {

	char * mem_limit = getCmdOption(argv, argv + argc, "-l");
	char * ntrees = getCmdOption(argv, argv + argc, "-n");
	char * output_file = getCmdOption(argv, argv + argc, "-o");

	bool paralel_trees = true;

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

	std::shared_ptr<RandomForest> forest = std::make_shared<RandomForest>();

	std::vector<BackgroundMode> bg_modes = {
		BackgroundMode::DEFAULT,
		BackgroundMode::RANDOM_MIDRANGE,
		BackgroundMode::DEFAULT,
		BackgroundMode::RANDOM_LONGRANGE,
		BackgroundMode::DEFAULT};

	bool exit = false;
	std::vector<std::shared_ptr<RandomTree> > tree_vector(num_trees);
	std::vector<std::shared_ptr<FramePool> > data_pools(num_trees);
	for (int i = 0; i < num_trees; i++) {
		data_pools[i] = std::make_shared<FramePool>();
		tree_vector[i] = std::make_shared<RandomTree>();
		if (!data_pools[i]->create(limit)) {
			std::cout << "Can't create Frame Pool. Exiting..." << std::endl;
			exit = true;
		}
		std::cout << "Pool/Tree " << i << " created" << std::endl;
	}

	
	if (paralel_trees) {		
		std::vector< std::thread> learning_threads(num_trees);
		for (int i = 0; i < num_trees; i++) {
			learning_threads[i] = std::thread(train_tree, tree_vector[i], data_pools[i]);
		}

		for (int i = 0; i < num_trees; i++) {
			learning_threads[i].join();
		}
		
	}
	else {
		for (int i = 0; i < num_trees; i++) {
			train_tree(tree_vector[i], data_pools[i]);
		}
	}

	int charbuffsize = 500;
	for (int i = 0; i < num_trees; i++) {
		std::unique_ptr<char[]> buf( new char[ charbuffsize ] );
		std::snprintf( buf.get(), charbuffsize, "%s_tree_%02d.dat", output_file, i);
		forest->push_tree(*tree_vector[i]);

		// save tree
		tree_vector[i]->save(buf.get());	
	}

	if (exit) return -1;

	Profiler::print();
	Profiler::clear();

	//save whole forest
	std::unique_ptr<char[]> buf( new char[ charbuffsize ] );
	std::snprintf( buf.get(), charbuffsize, "%s_forest.dat", output_file);
	forest->save(buf.get());

	return 0;
}
