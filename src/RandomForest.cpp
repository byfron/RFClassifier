#include "common.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <queue>
#include <algorithm>

RandomForest::RandomForest() {
}

Label RandomTree::predict(Feature & feature) {
	
	size_t node_idx = 0;

	for(;;) {
		Node & node = _nodes[node_idx];

		if (node.isLeaf()) {
			//	if (node._probability > 0.85)
			{
				return node.getLabel();
			}
			//else return 0;
		}
		else {
			if (node.fallsToLeftChild(feature)) {
				node_idx = node.left_child;
			}
			else {
				node_idx = node.right_child;
			}
		}
	}
}

Frame RandomTree::predict(FramePtr frame) {

	Frame output = *frame;
	Data features;
	//frame->computeForegroundFeatures(features);
	frame->computeImageFeatures(features);
	
	for (auto feature : features) {
		Label l = predict(feature);
		output.setLabel(feature.row(), feature.col(), l);
	}
	return output;
}


std::vector<Label> RandomTree::predict(DataPtr data) {
	
	//TODO: book mem a priory
	std::vector<Label> labels;
	for (auto feature : *data)
		labels.push_back(predict(feature));

	return labels;
}

void RandomTree::train(DataPtr data) {

	_bg_mode = Settings::bmode;

	_nodes.clear();
	std::queue<NodeConstructor> queue;

	size_t depth;

	// Train root node
	Node root_node(0);
	DataSplit root_ds(data, data->begin(), data->end());
	root_node.train(root_ds);

	_nodes.push_back(root_node);

	if (!root_node.isLeaf()) {
		queue.push(
			NodeConstructor(0, //id
					0, //depth
					data->begin(),
					data->end()));
	}

	while(queue.size() > 0) {

		size_t node_id = queue.front().node_id;
		FeatureIterator start = queue.front().start;
		FeatureIterator end = queue.front().end;

		depth = queue.front().depth + 1;
		queue.pop();

		FeatureIterator split_it = _nodes[node_id].
			getSplitIterator(DataSplit(data, start, end));

		// Train left child
		Node left_node(depth);
		left_node.train(DataSplit(data, start, split_it));
		size_t left_id = _nodes.size();
		_nodes.push_back(left_node);

		// Train right child
		Node right_node(depth);
		right_node.train(DataSplit(data, split_it, end));
		size_t right_id = _nodes.size();
		_nodes.push_back(right_node);

		std::cout << "Training tree - current size: " << _nodes.size() << std::endl;

		// Assign child indices to parent
		_nodes[node_id].left_child = left_id;
		_nodes[node_id].right_child = right_id;

		// Generate constructor for future nodes
		if (!left_node.isLeaf()) {
			queue.push(
				NodeConstructor(left_id,
						depth,
						start,
						split_it));
		}

		if (!right_node.isLeaf()) {
			queue.push(
				NodeConstructor(right_id,
						depth,
						split_it,
						end));
		}
	}

	std::cout << "Finished training. Tree has " <<_nodes.size() << " nodes with depth :" << depth << std::endl;
}

std::vector<Label> RandomForest::predict(DataPtr data) {
	for (int i = 0; i < _tree_ensemble.size(); i++) {
		return _tree_ensemble[i].predict(data);
	}
}

Frame RandomForest::majorityVoting(const std::vector<Frame> &frames) {

	// Frame majFrame = frames[0];
	cv::Size size = frames[0].getImageSize();
	cv::Mat best_labels = cv::Mat(size, CV_8UC1);
	best_labels.setTo(0);

	int votes[NUM_LABELS];

	// std::vector<int> votes;

	for (int r = 0; r < size.height; r++) {
		for (int c = 0; c < size.width; c++) {

			Label maj_label = best_labels.at<uchar>(r,c);

			for (int i = 0; i < NUM_LABELS; i++) {
				votes[i] = 0;
			}

			for (int f = 0; f < frames.size(); f++) {
				Label l = frames[f].getLabel(r, c);
				int idx = (int)l;
				votes[idx]++;
			}

			int max_votes = 0;
			Label best_label = 0;
			//find the majority vote
			for (int idx = 0; idx < NUM_LABELS; idx++) {
				if (votes[idx] > max_votes) {
					max_votes = votes[idx];
					best_label = (Label)idx;
				}
			}

			best_labels.at<uchar>(r,c) = best_label;
		}
	}

	Frame frame;
	frame.setDepthImage(frames[0].getDepthImage());
	frame.setLabelImage(best_labels);
	return frame;

}

Frame RandomForest::predict(FramePtr frame) {

	return _tree_ensemble[0].predict(frame);
	
	// std::vector<Frame> frame_predictions;
	// for (int i = 0; i < _tree_ensemble.size(); i++) {
	// 	frame_predictions.push_back(_tree_ensemble[i].predict(frame));
	// }

	// //return frame_predictions[0];

	// return RandomForest::majorityVoting(frame_predictions);
}

void RandomForest::train(DataPtr data) {
	//TODO: refactor timing

	struct timespec start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);

	std::shared_ptr<RandomTree> tree = std::make_shared<RandomTree>();
	tree->train(data);
	_tree_ensemble.push_back(*tree);

	clock_gettime(CLOCK_MONOTONIC, &finish);
	double elapsed;
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	std::cout << ">> Finished training tree of " << tree->getNumNodes() <<
		" in " << elapsed << " seconds." << std::endl;
}
