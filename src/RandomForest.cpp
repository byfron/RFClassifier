#include "common.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <queue>
#include <algorithm>

Label RandomTree::predict(Feature & feature) {

	size_t node_idx = 0;

	for(;;) {
		Node & node = _nodes[node_idx];

		if (node.isLeaf()) {
			return node.getLabel();
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


Frame RandomTree::predict(Frame & frame) {

	Frame output = frame;
	Data features;
	frame.computeForegroundFeatures(features);

	for (auto feature : features) {
		Label l = predict(feature);
		std::cout << (int)l << std::endl;
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
	return _tree_ensemble[0].predict(data);
}

Frame RandomForest::predict(Frame & frame) {
	return _tree_ensemble[0].predict(frame);
}

void RandomForest::train(DataPtr data) {
	RandomTree tree;
	tree.train(data);
	_tree_ensemble.push_back(tree);
}
