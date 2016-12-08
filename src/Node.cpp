#include "Node.hpp"

namespace {

	FeatureIterator computeSplitIterator(DataSplit ds, float threshold) {

		FeatureIterator it;
		for (it = ds.start; it != ds.end; it++) {
			if (it->getValue() > threshold)
				break;
		}

		return it;
		return it+1;
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

LabelHistogram::LabelHistogram(DataSplit & ds) {

	_hist = std::vector<float>(Settings::num_labels, 0.0);
	for (FeatureIterator it = ds.start; it != ds.end; it++) {
		_hist[it->getLabel()]+=1;
	}

	std::cout << "<" << ds.end - ds.start << ">" << std::endl;
	print();
	getchar();
	normalize();
}

void Node::train(DataSplit ds) {

	struct timespec start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);

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

		//sample thresholds from a uniform distribution between
		//min and max values of the split
		std::vector<float> learner_thresholds =
			sampleThresholds(ds.start->getValue(), ds.end->getValue());

		for (float threshold : learner_thresholds) {

			float cost = evaluateCostFunction(ds, threshold);
			std::cout << "cost:" << cost << std::endl;
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

	std::cout << "Finished training node in " << elapsed << " seconds. Entropy:" <<
		evaluateCostFunction(ds, best_threshold) << std::endl;
	FeatureIterator split_it = computeSplitIterator(ds, best_threshold);
	std::cout << "Best split:" << (split_it - ds.start) << "/" <<
		(ds.end - split_it) << std::endl;

	//if the best split leaves all nodes here, mark as leaf
	if (split_it == ds.start ||
	    split_it == ds.end)
		_is_leaf = true;

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

	std::cout << "first eval" << std::endl;
	
	LabelHistogram hist_left(l_split);
	LabelHistogram hist_right(r_split);
	float entr_left = hist_left.computeEntropy();
	float entr_right = hist_right.computeEntropy();

	std::cout << "lent:" << entr_left << std::endl;
	std::cout << "rent:" << entr_right << std::endl;
	
	assert(ds.getSize() > 0);
	
	return (l_split.getSize()/ds.getSize()) * entr_left +
		(r_split.getSize()/ds.getSize()) * entr_right;
}
