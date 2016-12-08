#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include "Node.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cereal/archives/binary.hpp>

class RandomTree
{

public:

	void train(std::vector<Feature>&);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(a);
	}

private:

	int a;
	std::vector<Node> _nodes;
};

class RandomForest
{

public:

	void train(std::vector<Feature>&);

private:

	std::vector<RandomTree> _tree_ensemble;

};
