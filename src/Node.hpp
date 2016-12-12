#pragma once
#include "common.hpp"
#include "Frame.hpp"
#include "RandomGenerator.hpp"
#include <cereal/archives/binary.hpp>

typedef std::vector<Feature> Data;
typedef std::shared_ptr<Data> DataPtr;
typedef Data::iterator FeatureIterator;

struct LearnerParameters {
	bool is_unary;
	float offset_1[2];
	float offset_2[2];

	template <class Archive>
	void serialize( Archive & ar )
	{
		ar(is_unary,
		   cereal::binary_data(offset_1, sizeof(float)*2),
		   cereal::binary_data(offset_2, sizeof(float)*2));
	}
};

class DataSplit {

public:

	DataSplit(DataPtr d,
		  FeatureIterator s,
		  FeatureIterator e) : data(d), start(s), end(e) {

		//	std::cout << "Creating data split(" << end - start << ")" << std::endl;
	}

	int getSize() const {
		return std::distance(start, end);
	}

	DataPtr data;
	FeatureIterator start;
	FeatureIterator end;
};


class LabelHistogram {
public:
	LabelHistogram(DataSplit & ds);
	void print() const {
		std::cout << std::endl << "# ";
		for (auto b : _hist)
			std::cout << b << " ";
		std::cout << "#" << std::endl;
	}

	void normalize() {
		float sum = 0;
		for (auto b : _hist)
			sum += b;

		if (sum  > 0)
			for (int i = 0; i < _hist.size(); i++)
				_hist[i]/=sum;
	}

	Label getMostLikelyLabel() {
		Label best_label;
		float max_prob = 0.0;
		for (int i = 0; i < _hist.size(); i++) {
			if (_hist[i] > max_prob) {
				max_prob = _hist[i];
				best_label = (Label)i;
			}
		}

		return best_label;
	}

	float computeEntropy() const {

		float sum = 0;
		for (int i = 0; i < _hist.size(); i++)
			if (_hist[i] > 0)
				sum += _hist[i]*log(_hist[i]);
		return -sum;
	}

private:
	std::vector<float> _hist;
};


struct NodeConstructor {

	NodeConstructor(int id,
			float d,
			FeatureIterator s,
			FeatureIterator e) : node_id(id), depth(d), start(s), end(e) {}
	int node_id;
	size_t depth;
	FeatureIterator start, end;
};

class Node
{
public:

	Node() {}
	Node(size_t depth) :
		_threshold(0.0),
		_depth(depth),
		_is_leaf(false),
		_label(0),
		left_child(0),
		right_child(0) {};

	void train(DataSplit);

	float evaluateCostFunction(const DataSplit, float) const;

	FeatureIterator getSplitIterator(DataSplit, float threshold) const;
	FeatureIterator getSplitIterator(DataSplit) const;

	bool fallsToLeftChild(Feature & feat) const;
	bool isLeaf() const { return _is_leaf; }
	Label getLabel() const { return _label; }

	template <class Archive>
	void serialize( Archive & archive )
	{
		archive(left_child,
			right_child,
			_node_params,
			_threshold,
			_depth,
			_is_leaf,
			_label);
	}

	size_t left_child;
	size_t right_child;

private:

	LearnerParameters _node_params;
	float _threshold;
	size_t _depth;
	bool _is_leaf;
	Label _label;
};
