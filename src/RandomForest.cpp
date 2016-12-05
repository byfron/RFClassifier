#include "common.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <queue>
#include <algorithm>

int Settings::max_tree_depth = 20;
int Settings::num_pixels_per_image = 1000;
int Settings::num_thresholds_per_feature = 10;
int Settings::num_offsets_per_pixel = 2000;
float Settings::maximum_depth_difference = 2.0; //defines range thresholds
int Settings::offset_box_size = 150;
int Settings::num_labels = 32;
	
namespace {

	FeatureIterator computeSplitIterator(DataSplit ds, float threshold) {

		FeatureIterator it;
		for (it = ds.start; it != ds.end; it++) {
			if (it->getValue() <= threshold)
				break;
		}

		return it;
	}
	
	void sampleOffset(float * offset) {

		offset[0] = Settings::offset_box_size*2*random_real(0.0,1.0) -
				Settings::offset_box_size/2;
		offset[1] = Settings::offset_box_size*2*random_real(0.0,1.0) -
				Settings::offset_box_size/2;
	}
	
	std::vector<LearnerParameters> sampleParameters() {
		
		std::vector<LearnerParameters> param_vec;
		
		for (int i = 0; i < Settings::num_offsets_per_pixel; i++) {

			LearnerParameters param;
			
			//Sample offsets from a uniform distribution
			sampleOffset(param.offset_1);
			param.is_unary = random_real(0.0,1.0) > 0.5;
			if (param.is_unary)
				sampleOffset(param.offset_2);
						
			param_vec.push_back(param);
		}

		return param_vec;
	}

	std::vector<float> sampleThresholds() {

		std::vector<float> thresh_vec;
		
		for (int i = 0; i < Settings::num_thresholds_per_feature; i++) {
			thresh_vec.push_back(Settings::maximum_depth_difference*2*random_real(0.0,1.0)
					     - Settings::maximum_depth_difference);
		}

		return thresh_vec;
	}
}

void Node::train(DataSplit ds) {

	// Check if we are finished (reached max depth)
	if (_depth == Settings::max_tree_depth) {
		_is_leaf = true;

		//what here?
		return;
	}

	std::cout << "Sampling learners..." << std::endl;
	// Sample a new set of parameters
	std::vector<LearnerParameters> sampled_learners =
		sampleParameters();

	float best_cost = INF;
	float best_threshold;
	LearnerParameters best_learner;

	std::cout << "Evaluating features..." << std::endl;
	
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


	std::cout << "Finished training node. Entropy:" << std::endl;
	
	// sort data acoording to the best learner,
	// so that we know where to split in the next children nodes
	for (FeatureIterator it = ds.start; it != ds.end; it++) {
		it->evaluate(best_learner);
	}
	std::sort(ds.start, ds.end);
}

FeatureIterator Node::getSplitIterator(DataSplit ds) const {
	return computeSplitIterator(ds, _threshold);
}

float Node::evaluateCostFunction(const DataSplit ds,
				 float threshold) const {

	//divide datasplit with threshold
	FeatureIterator split_it = computeSplitIterator(ds, threshold);
	DataSplit l_split = DataSplit(ds.data, ds.start, split_it);
	DataSplit r_split = DataSplit(ds.data, split_it, ds.end);
	LabelHistogram hist_left(l_split);
	LabelHistogram hist_right(r_split);
	float entr_left = hist_left.computeEntropy();
	float entr_right = hist_left.computeEntropy();

	return (l_split.getSize()/ds.getSize()) * entr_left +
		(r_split.getSize()/ds.getSize()) * entr_right;
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
		std::cout << _nodes.size() << "nodes with depth :" << depth << std::endl;
	}
}

void RandomForest::train(std::vector<Feature> & data) {


	RandomTree tree;
	tree.train(data);

}
