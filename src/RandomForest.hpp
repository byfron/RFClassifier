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

	std::vector<Label> predict(std::vector<Feature>&);
	Label predict(Feature &);
	void train(std::vector<Feature>&);

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
	std::vector<Label> predict(std::vector<Feature>&);
	void train(std::vector<Feature>&);

private:

	std::vector<RandomTree> _tree_ensemble;

};
