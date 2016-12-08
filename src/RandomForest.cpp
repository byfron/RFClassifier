#include "common.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <queue>
#include <algorithm>

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
