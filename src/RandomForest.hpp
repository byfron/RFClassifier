#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#define INF 1000000//std::numeric_limits<float>::max()

class Feature;
typedef int Label;
typedef std::vector<Feature>::iterator FeatureIterator;

struct LearnerParameters {
	bool is_unary;
	float offset_1[2];
	float offset_2[2];
};

class Feature {
public:
	void evaluate(LearnerParameters & params);

	bool operator< (const Feature& f) const {
		return _value < f._value;
	}

	const float & getValue() { return _value; }

private:
	int _row;
	int _col;
	Label _label;
	float _value;
	int _image_id;
};


struct DataSplit {

	DataSplit(std::vector<Feature> & d,
		  FeatureIterator s,
		  FeatureIterator e) : data(d), start(s), end(e) {}

	std::vector<Feature> & data;
	FeatureIterator start;
	FeatureIterator end;
};

struct NodeConstructor {

	NodeConstructor(int id,
			FeatureIterator s,
			FeatureIterator e) : node_id(id), start(s), end(e) {}
	int node_id;
	FeatureIterator start, end;
};

class Node
{
public:

	Node(int depth) : _depth(depth) {};

	void train(DataSplit);

	float evaluateCostFunction(const DataSplit, float);

	FeatureIterator getSplitIterator(DataSplit) const;

	bool isLeaf() { return _is_leaf; }

	int left_child;
	int right_child;

private:

	LearnerParameters _node_params;
	float _threshold;
	int _depth;
	bool _is_leaf;
};

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

	void train();

private:

	std::vector<RandomTree> _tree_ensemble;

};
