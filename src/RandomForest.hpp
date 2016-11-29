#include <opencv2/opencv.hpp>
#include <random>
#include <iostream>

#define INF 1000000//std::numeric_limits<float>::max()

class Feature;
typedef int Label;
typedef std::vector<Feature>::iterator FeatureIterator;

class Image {
public:
	const float& operator()(int row, int col) {
		return _image.at<float>(row, col);
	}

	cv::Mat _image;
};

class ImagePool {
public:

	static Image & getImage(int id) {
		return _image_vector[id];
	}

	static std::vector<Image> _image_vector;
};


class RandomNumberGenerator {

public:
	RandomNumberGenerator() : gen(rd()),
				  dis(0.0, 1.0) {
	}
	
	float generateUniform() {
		return dis(gen);
	}

private:

	std::uniform_real_distribution<> dis;
        std::random_device rd;
	std::mt19937 gen;	
};

class RandomGenerator {
public:
	static float generateUniform() {
		return random.generateUniform();
	}
	static RandomNumberGenerator random;
};


class Settings {
public:
	static int num_trees;
	static int num_images_per_tree;
	static int num_thresholds_per_feature;
	static int num_offsets_per_pixel;
	static int max_tree_depth;
	static float max_offset;
};

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
