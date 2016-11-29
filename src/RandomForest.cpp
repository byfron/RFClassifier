#include "RandomForest.hpp"

void Feature::evaluate(LearnerParameters & params) {

	Image & im = ImagePool::getImage(_image_id);
	float z = im(_row, _col);
	value = im(_row + params.offset1[0]/z, _column + params.offset1[1]/z);
	if (!params.is_unary) {
		value -= im(_row + params.offset2[0]/z, _column + params.offset2[1]/z);
	}
}

boid Node::sampleParameters(std::vector<LearnerParameters> & params) {

	//Sample offsets from a uniform distribution

}

void Node::train(DataSplit & ds) {

	// Check if we are finished (reached max depth)
	if (_depth == settings.max_tree_depth) {
		_is_leaf = true;

		//what here?
		return;
	}

	// Sample a new set of parameters
	std::vector<LearnerParameters> sampled_learners;
	sampleParameters(sampled_learers);

	float best_cost = INF;
	float best_threshold;
	LearnerParameters best_learner;

	// Evaluate all features with each set of parameters
	for (auto learner : sampled_learners) {
		for (FeatureIterator it = ds.start; it != ds.end; it++) {
			it->evaluateFeature(learner);
		}

		// Order features acoording to (Eq) function
		std::sort(ds.data, ds.start, ds.end);

		//sample thresholds from a uniform distribution
		std::vector<float> learner_thresholds;
		sampleThresholds(learner_thresholds);

		for (foat threshold : learner_thresholds) {

			if (evaluateCostFunction(ds, threshold) < best_cost) {
				best_threshold = threshold;
				best_learner = learner;
			}
		}
	}

	// save learned node parameters
	_node_params = best_learner;
	_threshold = threshold;

	//sort data acoording to the best learner
	for (FeatureIterator it = ds.start; it != ds.end; it++) {
		it->evaluateFeature(best_learner);
	}
	std::sort(data, start, end);
}


FeatureIterator Node::getSplitIterator(const DataSplit &ds) const {

	FeatureIterator it;
	for (it = ds.start; it != ds.end; it++) {
		if (it->value <= _best_threshold)
			break;
	}

	return it;
}

float Node::evaluateCostFunction(const DataSplit & ds, float theshold) {

	//shannon entropy
}

RandomTree::train(std::vector<Feature> & data) {

	int depth = 0;
	std::vector<int> left_nodes, right_nodes;
	std::vector<Node> trained_nodes;
	std::queue<NodeContructor> queue;

	// Train root node
	Node root_node(depth);
	DataSplit root_ds(data, data.begin(), data.end());
	root_node.train(ds);
	trained_nodes.push_back(root_node);
	depth++;

	if (!root_node.isLeaf()) {
		queue.push_back(
			NodeConstructor(0,
					data.begin(),
					data.end());
	}

	while(queue.size() > 0) {

		int node_id = queue.front().node_id;
		FeatureIterator start = queue.front().start;
		FeatureIterator end = queue.front().end;
		queue.pop_front();

		FeatureIterator split_it = trained_nodes[node_id].
			getSplitIterator(DataSplit(data, start, end));

		// Train left child
		Node left_node(depth);
		left_node.train(DataSplit(data, start, split_it));
		int left_id = trained_nodes.size();
		trained_nodes.push_back(left_node);

		// Train right child
		Node right_node(depth);
		right_node.train(DataSplit(data, split_it, end));
		int right_id = trained_nodes.size();
		trained_nodes.push_back(right_node);

		// Assign child indices to parent
		trained_nodes[node_id].left_child = left_id;
		trained_nodes[node_id].right_child = right_id;

		// Generate constructor for future nodes
		if (!left_node.isLeaf()) {
			queue.push_back(
				NodeConstructor(left_id,
						start,
						split_it));
		}

		if (!right_node.isLeaf()) {
			queue.push_back(
				NodeConstructor(left_id,
						split_it,
						end));
		}

		depth++;
	}
}
