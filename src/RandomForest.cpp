#include "RandomForest.hpp"
#include <vector>
#include <queue>
#include <algorithm>

namespace {

	std::vector<LearnerParameters> sampleParameters() {

	//Sample offsets from a uniform distribution
		RandomGenerator::generateUniform();
	}

	std::vector<float> sampleThresholds() {

		RandomGenerator::generateUniform();
	}
}

void Feature::evaluate(LearnerParameters & params) {

	Frame & im = FramePool::getFrame(_image_id);
	float z = im(_row, _col);
	_value = im(_row + params.offset_1[0]/z, _col + params.offset_1[1]/z);
	if (!params.is_unary) {
		_value -= im(_row + params.offset_2[0]/z, _col + params.offset_2[1]/z);
	}
}



void Node::train(DataSplit ds) {

	// Check if we are finished (reached max depth)
	if (_depth == Settings::max_tree_depth) {
		_is_leaf = true;

		//what here?
		return;
	}

	// Sample a new set of parameters
	std::vector<LearnerParameters> sampled_learners =
		sampleParameters();

	float best_cost = INF;
	float best_threshold;
	LearnerParameters best_learner;

	// Evaluate all features with each set of parameters
	for (auto learner : sampled_learners) {
		for (FeatureIterator it = ds.start; it != ds.end; it++) {
			it->evaluate(learner);
		}

		// Order features acoording to (Eq) function
		std::sort(ds.start, ds.end);

		//sample thresholds from a uniform distribution
		std::vector<float> learner_thresholds =
			sampleThresholds();

		for (float threshold : learner_thresholds) {

			if (evaluateCostFunction(ds, threshold) < best_cost) {
				best_threshold = threshold;
				best_learner = learner;
			}
		}
	}

	// save learned node parameters
	_node_params = best_learner;
	_threshold = best_threshold;

	// sort data acoording to the best learner,
	// so that we know where to split in the next children nodes
	for (FeatureIterator it = ds.start; it != ds.end; it++) {
		it->evaluate(best_learner);
	}
	std::sort(ds.start, ds.end);
}


FeatureIterator Node::getSplitIterator(DataSplit ds) const {

	FeatureIterator it;
	for (it = ds.start; it != ds.end; it++) {
		if (it->getValue() <= _threshold)
			break;
	}

	return it;
}

float Node::evaluateCostFunction(const DataSplit ds,
				 float theshold) {

	//shannon entropy
}

void RandomTree::train(std::vector<Feature> & data) {

	_nodes.clear();
	int depth = 0;
	std::queue<NodeConstructor> queue;

	// Train root node
	Node root_node(depth);
	DataSplit root_ds(data, data.begin(), data.end());
	root_node.train(root_ds);
	_nodes.push_back(root_node);
	depth++;

	if (!root_node.isLeaf()) {
		queue.push(
			NodeConstructor(0,
					data.begin(),
					data.end()));
	}

	while(queue.size() > 0) {

		int node_id = queue.front().node_id;
		FeatureIterator start = queue.front().start;
		FeatureIterator end = queue.front().end;
		queue.pop();

		FeatureIterator split_it = _nodes[node_id].
			getSplitIterator(DataSplit(data, start, end));

		// Train left child
		Node left_node(depth);
		left_node.train(DataSplit(data, start, split_it));
		int left_id = _nodes.size();
		_nodes.push_back(left_node);

		// Train right child
		Node right_node(depth);
		right_node.train(DataSplit(data, split_it, end));
		int right_id = _nodes.size();
		_nodes.push_back(right_node);

		// Assign child indices to parent
		_nodes[node_id].left_child = left_id;
		_nodes[node_id].right_child = right_id;

		// Generate constructor for future nodes
		if (!left_node.isLeaf()) {
			queue.push(
				NodeConstructor(left_id,
						start,
						split_it));
		}

		if (!right_node.isLeaf()) {
			queue.push(
				NodeConstructor(left_id,
						split_it,
						end));
		}

		depth++;
	}
}

void RandomForest::train() {



}
