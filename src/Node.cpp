#include "Node.hpp"
#include <time.h>

namespace {

	FeatureIterator computeSplitIterator(DataSplit ds, float threshold) {

		FeatureIterator it = ds.start;
		for (;it != ds.end; it++) {
			if (it->getValue() > threshold)
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
			if (!param.is_unary)
				sampleOffset(param.offset_2);

			param_vec.push_back(param);
		}

		return param_vec;
	}

	std::vector<float> sampleThresholds(float min, float max) {

		std::vector<float> thresh_vec;
		for (int i = 0; i < Settings::num_thresholds_per_feature; i++) {
			thresh_vec.push_back(random_real(min, max));
		}

		return thresh_vec;
	}
}

void Node::train(DataSplit ds) {

	assert(ds.getSize() > 0);

	std::cout << ">> Training node with " << ds.end - ds.start << " features." << std::endl;

	struct timespec start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);

	// Check if we are finished (reached max depth)
	if (_depth == Settings::max_tree_depth) {
		_is_leaf = true;
		_label = LabelHistogram::getMostLikelyLabel(ds);
		return;
	}

	// Sample a new set of parameters
	std::vector<LearnerParameters> sampled_learners =
		sampleParameters();

	float best_cost = INF;
	float best_threshold = 0.0f;
	LearnerParameters best_learner;

	// Evaluate all features with each set of parameters
	for (auto learner : sampled_learners) {

		for (FeatureIterator it = ds.start; it != ds.end; it++) {
			it->evaluate(learner);
		}
		std::sort(ds.start, ds.end);

		FeatureIterator last = ds.end - 1;

		//sample thresholds from a uniform distribution between
		//min and max values of the split
		std::vector<float> learner_thresholds =
			sampleThresholds(ds.start->getValue(), last->getValue());

//		std::cout << "range:" << ds.start->getValue() << "," << last->getValue() << std::endl;

		for (float threshold : learner_thresholds) {

//			std::cout << threshold << std::endl;
			float cost = evaluateCostFunction(ds, threshold);
			if (cost < best_cost) {
				best_threshold = threshold;
				best_learner = learner;
				best_cost = cost;
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

	clock_gettime(CLOCK_MONOTONIC, &finish);
	double elapsed;
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	std::cout << ">> Finished training node in " << elapsed << " seconds. Entropy:" <<
		evaluateCostFunction(ds, best_threshold) << std::endl;
	FeatureIterator split_it = computeSplitIterator(ds, best_threshold);
	std::cout << "Best split:" << (split_it - ds.start) << "/" <<
		(ds.end - split_it) << std::endl;

	//if the best split leaves all nodes here, mark as leaf
	if (split_it == ds.start ||
	    split_it == ds.end) {
		_label = LabelHistogram::getMostLikelyLabel(ds);
		_is_leaf = true;
	}

}

bool Node::fallsToLeftChild(Feature & feat) const {
	feat.evaluate(_node_params);
	if (feat.getValue() < _threshold)
		return true;

	return false;
}

FeatureIterator Node::getSplitIterator(DataSplit ds) const {

	FeatureIterator it = computeSplitIterator(ds, _threshold);
	std::cout << "thresh:" << _threshold << std::endl;
	std::cout << "left split:" << ds.end - it << std::endl;
	return it;
}

float Node::evaluateCostFunction(const DataSplit ds,
				 float threshold) const {

	//divide datasplit with threshold
	FeatureIterator split_it = computeSplitIterator(ds, threshold);
	DataSplit l_split = DataSplit(ds.data, ds.start, split_it);
	DataSplit r_split = DataSplit(ds.data, split_it, ds.end);

	float entr_left = LabelHistogram::computeEntropy(l_split);
	float entr_right = LabelHistogram::computeEntropy(r_split);

	float cost = (float(l_split.getSize())/ds.getSize()) * entr_left +
		(float(r_split.getSize())/ds.getSize()) * entr_right;

	return cost;
}
