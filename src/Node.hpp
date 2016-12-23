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
	static inline void getMostLikelyLabel(DataSplit & ds, Label & best_label, float & prob) {

		std::vector<float> & hist = createHistogram(ds);

		float max_prob = 0.0;
		for (int i = 0; i < hist.size(); i++) {
			if (hist[i] > max_prob) {
				max_prob = hist[i];
				best_label = (Label)i;
			}
		}

		prob = max_prob/hist.size();
	}

	static inline std::vector<float> & createHistogram(DataSplit & ds) {

		static std::vector<float> hist = std::vector<float>(Settings::num_labels);

		for (int i = 0; i < Settings::num_labels; i++)
			hist[i] = 0.0;

		const float normalised_bin = 1./(ds.end - ds.start);

		for (FeatureIterator it = ds.start; it < ds.end; it++) {
			hist[it->getLabel()]+=normalised_bin;
		}

		return hist;
	}


	static inline float computeEntropy(DataSplit & ds) {

		float sum = 0;
		std::vector<float> & hist = createHistogram(ds);

		for (int i = 0; i < Settings::num_labels; i++) {
			if (hist[i] > 0) {
				sum += hist[i]*log(hist[i]);
			}
		}

		return -sum;
	}
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

	static int leaf_counter;

	Node() {}
	Node(size_t depth) :
		_threshold(0.0),
		_depth(depth),
		_label(0),
		_probability(0.0),
		_leaf_id(-1),
		left_child(0),
		right_child(0) {};

	void train(DataSplit);

	float evaluateCostFunction(const DataSplit, float) const;

	FeatureIterator getSplitIterator(DataSplit, float threshold) const;
	FeatureIterator getSplitIterator(DataSplit) const;

	bool fallsToLeftChild(Feature & feat) const;
	bool isLeaf() const { return _leaf_id > -1; }
	Label getLabel() const { return _label; }

	template <class Archive>
	void serialize( Archive & archive )
	{
		archive(left_child,
			right_child,
			_node_params,
			_threshold,
			_depth,
			_leaf_id,
			_label,
			_probability);
	}

	size_t left_child;
	size_t right_child;

private:

	LearnerParameters _node_params;
	float _threshold;
	size_t _depth;
	Label _label;
	float _probability;
	int _leaf_id;
};
