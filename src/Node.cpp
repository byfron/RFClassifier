#include "Node.hpp"
#include <ska_sort.hpp>
#include <time.h>
#include <Profiler.hpp>

FeatureIterator computeSplitIterator(DataSplit ds, float threshold) {

	FeatureIterator it = ds.start;

//#pragma omp parallel for TODO: instead of breaking compute the histograms/entropies directly here?
	for (;it != ds.end; it++) {
		if (it->getValue() > threshold)
			break;
	}

	return it;
}

void sampleOffset(float * offset) {

	offset[0] = Settings::offset_box_size*random_real(0.0,1.0) -
		Settings::offset_box_size/2;
	offset[1] = Settings::offset_box_size*random_real(0.0,1.0) -
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

int Node::leaf_counter = -1;

void Node::train(DataSplit ds) {

	{

//	Profiler p("Node train");

	assert(ds.getSize() > 0);

	std::cout << ">> Training node with " << ds.end - ds.start << " features." << std::endl;

	struct timespec start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);

	// Check if we are finished (reached max depth)
	if (_depth == Settings::max_tree_depth) {
		LabelHistogram::getMostLikelyLabel(ds, _label, _probability);
		_leaf_id = ++Node::leaf_counter;
		std::cout << "LEAF: MAX tree depth reached!" << std::endl;
		return;
	}	

	// Check if we have just one feature per node
	if (ds.getSize() == 1) {
		LabelHistogram::getMostLikelyLabel(ds, _label, _probability);
		_leaf_id = ++Node::leaf_counter;
		std::cout << "LEAF: one feature!" << std::endl;
		return;
	}

	// Check if the entropy is already zero
	// if (evaluateCostFunction(ds, INF) == 0) {
	// 	LabelHistogram::getMostLikelyLabel(ds, _label, _probability);
	// 	_leaf_id = ++Node::leaf_counter;
	// 	std::cout << "LEAF: entropy of partition zero!" << std::endl;
	// 	return;
	// }
					 
	// Sample a new set of parameters
	std::vector<LearnerParameters> sampled_learners =
		sampleParameters();

	float best_cost = INF;
	float best_threshold = 0.0f;
	LearnerParameters best_learner;

	// Evaluate all features with each set of parameters
	for (auto learner : sampled_learners) {
		{
//		Profiler p("Evaluate feat.");
		#pragma omp parallel for
		for (FeatureIterator it = ds.start; it < ds.end; it++) {
			it->evaluate(learner);
		}
		}

		{
//		Profiler p("Sorting feat.");
//		std::sort(ds.start, ds.end);
		ska_sort(ds.start, ds.end, [](const Feature & feat) { return feat.getValue(); } );
		}

		FeatureIterator last = ds.end - 1;

//		Profiler p("Sample thresholds");
		//sample thresholds from a uniform distribution between
		//min and max values of the split

		std::vector<float> learner_thresholds =
			sampleThresholds(ds.start->getValue(), last->getValue());
//		p.stop();


		{
//		Profiler p("Evaluate cost func.");
		for (float threshold : learner_thresholds) {

			float cost = evaluateCostFunction(ds, threshold);
			if (cost < best_cost) {
				best_threshold = threshold;
				best_learner = learner;
				best_cost = cost;
			}
		}
		}
	}

	// save learned node parameters
	_node_params = best_learner;
	_threshold = best_threshold;

	// sort data acoording to the best learner,
	// so that we know where to split in the next children nodes
	#pragma omp parallel for
	for (FeatureIterator it = ds.start; it < ds.end; it++) {
		it->evaluate(best_learner);
	}
//	std::sort(ds.start, ds.end);
	ska_sort(ds.start, ds.end, [](const Feature & feat) { return feat.getValue(); } );

	clock_gettime(CLOCK_MONOTONIC, &finish);
	double elapsed;
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	{
//	Profiler p("Cost function");
	float cost = evaluateCostFunction(ds, best_threshold);
	std::cout << ">> Finished training node in " << elapsed << " seconds. Entropy:" << cost << std::endl;
	FeatureIterator split_it = computeSplitIterator(ds, best_threshold);
	std::cout << "Best split:" << (split_it - ds.start) << "/" <<
		(ds.end - split_it) << std::endl;

	//if entropy is zero, mark as leaf
	if (cost < 1.0e-4) {
		LabelHistogram::getMostLikelyLabel(ds, _label, _probability);
		_leaf_id = ++Node::leaf_counter;

		std::cout << "LEAF: entropy zero!" << std::endl;

		// FeatureIterator split_it = computeSplitIterator(ds, best_threshold);
		// DataSplit l_split = DataSplit(ds.data, ds.start, split_it);
		// DataSplit r_split = DataSplit(ds.data, split_it, ds.end);

		// std::cout << "left:";
		// for (FeatureIterator it = l_split.start; it < l_split.end; it++) {
		// 	std::cout << (int)it->getLabel() << " ";
		// }
		// std::cout << std::endl << "right:";
		// for (FeatureIterator it = r_split.start; it < r_split.end; it++) {
		// 	std::cout << (int)it->getLabel() << " ";
		// }
		// std::cout << std::endl;

		return;
	}

	//if the best split leaves all nodes here, mark as leaf
	if (split_it == ds.start ||
	    split_it == ds.end) {
		LabelHistogram::getMostLikelyLabel(ds, _label, _probability);
		_leaf_id = ++Node::leaf_counter;
		std::cout << "LEAF: split leaves all nodes in one side!" << std::endl;
		return;
	}
	}

	}
}

bool Node::fallsToLeftChild(Feature & feat) const {
	feat.evaluate(_node_params);
	if (feat.getValue() < _threshold)
		return true;

	return false;
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

	float entr_left = LabelHistogram::computeEntropy(l_split);
	float entr_right = LabelHistogram::computeEntropy(r_split);

	float cost = (float(l_split.getSize())/ds.getSize()) * entr_left +
		(float(r_split.getSize())/ds.getSize()) * entr_right;

	//TODO: make sure that when the cost is zero is because the labels are the same in each side

	return cost;
}
