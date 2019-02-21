#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include "RandomGenerator.hpp"
#include <cereal/archives/binary.hpp>
#include <Profiler.hpp>
#include <Eigen/Dense>

using namespace ByfronUtils;

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
		size = std::distance(start, end);
	}

	inline int getSize() const {
		return size;
	}

	DataPtr data;
	int size;
	FeatureIterator start;
	FeatureIterator end;
};


class LabelHistogram {
public:
	static inline void getMostLikelyLabel(DataSplit & ds, Label & best_label, float & prob) {

		Eigen::Matrix<float, 1, NUM_LABELS> hist = createHistogram(ds);

		float max_prob = 0.0;
		for (int i = 0; i < NUM_LABELS; i++) {
			if (hist(i) > max_prob) {
				max_prob = hist(i);
				best_label = (Label)i;
			}
		}

		prob = max_prob;
	}

	static inline Eigen::Matrix<float, 1, NUM_LABELS> createHistogram(DataSplit & ds) {
		Profiler p("create hist");
		static Eigen::Matrix<int, 1, NUM_LABELS> hist;
		hist.setZero();

		for (FeatureIterator it = ds.start; it < ds.end; it++) {
			hist(it->getLabel())++;
		}

		return hist.cast<float>()/ds.size;
	}


	static inline float computeEntropy(DataSplit & ds) {
		Profiler p("compute_entropy");
		float sum = 0.0;

		Eigen::Matrix<float, 1, NUM_LABELS> hist = createHistogram(ds);

		for (int i = 0; i < NUM_LABELS; i++) {
			if (hist(i) > 0 && hist(i) < 1) {
				sum += hist(i)*logf(hist(i));
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
	inline bool isLeaf() const { return _leaf_id > -1; }
	inline Label getLabel() const { return _label; }

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

	friend class RandomTree;
	
	LearnerParameters _node_params;
	float _threshold;
	size_t _depth;
	Label _label;
	float _probability;
	int _leaf_id;
};


/// Interface exposed for testing
FeatureIterator computeSplitIterator(DataSplit ds, float threshold);
void sampleOffset(float * offset);
std::vector<LearnerParameters> sampleParameters();
std::vector<float> sampleThresholds(float min, float max);
