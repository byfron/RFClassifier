#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#define INF 1000000//std::numeric_limits<float>::max()

typedef std::vector<Feature>::iterator FeatureIterator;

struct LearnerParameters {
	bool is_unary;
	float offset_1[2];
	float offset_2[2];
};

struct DataSplit {

	DataSplit(std::vector<Feature> & d,
		  FeatureIterator s,
		  FeatureIterator e) : data(d), start(s), end(e) {}

	int getSize() const {
		return std::distance(start, end);
	}
	
	std::vector<Feature> & data;
	FeatureIterator start;
	FeatureIterator end;
};

class LabelHistogram {
public:
	LabelHistogram(DataSplit & ds) {
		_hist = std::vector<int>(Settings::num_labels, 0);
		for (FeatureIterator it = ds.start; it != ds.end; it++) {
			_hist[it->getLabel()]++;			
		}	       
	}

	float computeEntropy() const {
		float sum = 0;
		for (int i = 0; i < _hist.size(); i++)
			sum += _hist[i]*log(_hist[i]);
		return -sum;
	}
	
private:
	std::vector<int> _hist;
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

	float evaluateCostFunction(const DataSplit, float) const;

	FeatureIterator getSplitIterator(DataSplit, float threshold) const;
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

	void train(std::vector<Feature>&);

private:

	std::vector<RandomTree> _tree_ensemble;

};
