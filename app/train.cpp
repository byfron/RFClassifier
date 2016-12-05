#include <RandomForest.hpp>
#include <Frame.hpp>

int main(int argc, char **argv) {

	// Load training
	FramePool::create();

	std::vector<Feature> data = FramePool::computeFeatures();

	// Train forest
	RandomForest random_forest;
	random_forest.train(data);

	// Check result on test

	return 0;
}
