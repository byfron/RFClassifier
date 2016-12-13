#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include "Node.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

class RandomTree
{

public:

	std::vector<Label> predict(DataPtr);
	Label predict(Feature &);
	Frame predict(Frame & frame);
	void train(DataPtr);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(_nodes);
	}

	size_t getNumNodes() {
		return _nodes.size();
	}

private:

	std::vector<Node> _nodes;
};

class RandomForest
{

public:
	std::vector<Label> predict(DataPtr);
	Frame predict(Frame & frame);
	void train(DataPtr);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(_tree_ensemble);
	}

	void print() {

		std::cout << "Forest has " << _tree_ensemble.size() << " trees with ";
		for (auto tree : _tree_ensemble) {
			std::cout << tree.getNumNodes() << " ";
		}
		std::cout << "nodes" << std::endl;

	}

private:

	std::vector<RandomTree> _tree_ensemble;

};
