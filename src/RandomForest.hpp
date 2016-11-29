typedef int Label;

struct Settings {

	int num_trees;
	int num_thresholds_per_feature;
	int num_offsets_per_pixel;
	int max_tree_depth;
	float max_offset;

}

struct LearnerParameters {
	bool is_unary;
	float offset_1[2];
	float offset_2[2];
};

class Feature {

	void evaluate(LearnerParameters & params);

	bool operator< (const Feature& f) const {
		return _value < f._value;
	}

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
	FeatureIteator start;
	FeatureIterator end;
}

struct NodeConstrctor {

	NodeConstructor(int id, FeatureIterator s,
			FeatureIterator e) : node_id(id), start(s), end(e) {}
	int node_id;
	FeatureIterator start, end;
};

class Node
{
public:

	Node(int depth) : _depth(depth) {};

	void train(DataSplit &);

	void sampleParameters(std::vector<LearnerParameters> & params);
	float evaluateCostFunction(const DataSplit &, float);

	FeatureIterator getSplitIterator(const DataSplit &);

	int left_child;
	int right_child;

private:

	LearnerParameters _node_params;
	float _threshold;
	int _depth;
	bool _is_leaf;
}

class RandomTree
{

public:

	void train(std::vector<Feature>&, std::vector<Label>&);

private:

	std::vector<Node> _nodes;
};

class RandomForest
{

public:
	predict()
	train()

private:

	std::vector<RandomTree> _tree_ensemble;

};
