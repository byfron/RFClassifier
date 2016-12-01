#include <RandomForest.hpp>

int main(int argc, char **argv) {

	// Load training
	ImagePool::create();

	// Train forest
	RandomForest random_forest;
	random_forest.train();

	// Check result on test


	return 0;
}
