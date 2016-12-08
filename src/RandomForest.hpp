#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include "Node.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

class RandomTree
{

public:

	void train(std::vector<Feature>&);

private:

	std::vector<Node> _nodes;
};

class RandomForest
{

public:

	void train(std::vector<Feature>&);

private:

	std::vector<RandomTree> _tree_ensemble;

};
