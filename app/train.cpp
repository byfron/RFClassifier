#include <RandomForest.hpp>
#include <Frame.hpp>

int main(int argc, char **argv) {

	// Load training
	FramePool::create();

	DataPtr data = std::make_shared<Data>();
	FramePool::computeFeatures(data);

	// Train forest
	RandomForest random_forest;
	random_forest.train(data);

	// Check result on traning
	std::vector<Label> labels = random_forest.predict(data);


	assert(labels.size() == data->size());

	int idx = 0;
	for (auto l : labels) {
		std::cout << l << "-" << data->operator[](idx).getLabel() << std::endl;
		idx++;
	}

	return 0;
}
