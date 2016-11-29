typedef int Label;

struct Settings {

	int num_trees;
	int num_thresholds_per_feature;
	int num_offsets_per_pixel;
	int max_three_depth;
	float max_offset;
	
}

struct LearnerParameters {
	float _threshold;
	float _offset_1;
	float _offset_2;
};
	
class Feature {

	float evaluateFeature(LearnerParameters & params);


	operator  <
	
private:
	int row;
	int column;
	int label;
	float value;
	Image *depth_image;	
};

class Node
{
public:

	void train(const std::vector<Feature> & data);
	void sampleParameters(std::vector<LearnerParameters> & params);	
	float evaluateCostFunction(const std::vector<Feature> &data,
				   const std::vector<int> & left_partition,
				   const std::vector<int> & right_partition);
	void proposeSplit(const LearnerParameters & params,
			  const std::vector<Feature> &data,
			  std::vector<int> & left_partition,
			  std::vector<int> & right_partition);
	
private:
	
	int _index_1;
	int _index_2;
	LearnerParameters _node_params;
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
