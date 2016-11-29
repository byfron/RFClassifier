#include "RandomForest.hpp"

bool Feature::evaluateFeature(LearnerParameters & params) {

}

boid Node::sampleParameters(LearnerParameterds & params) {

}

void Node::train(const std::vector<Feature> & data) {


	// Check if we are finished


	// Sample a new set of parameters

	
	// Evaluate all features with each set of parameters


	// Propose data partition
	
	
	// evaluateCost function in each partition

	
	// Keep best partition and propagate to children
	
}


float Node::evaluateCostfunction(const std::vector<Feature> &data,
				 const std::vector<int> & left_partition,
				 const std::vector<int> & right_partition) {


	//shannon entropy 
	
	
}


RandomTree::train(const std::vector<Feature> & data) {


	std::vector<int> left_nodes, right_nodes;
	std::queue<Node> untrained_nodes;
	Node root_node;
	untrained_nodes.push_back(root_node);
	root_node.train(data);
	
	
	while(untrained_nodes.size() > 0) {

		Node node = untrained_nodes.front();


	}
}
