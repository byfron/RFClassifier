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
	void train(DataPtr);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(_nodes);
	}

private:

	std::vector<Node> _nodes;
};

class RandomForest
{

public:
	std::vector<Label> predict(DataPtr);
	void train(DataPtr);

private:

	std::vector<RandomTree> _tree_ensemble;

};
