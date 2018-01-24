#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include "Node.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

class RandomTree
{

public:

	std::vector<Label> predict(DataPtr);
	Label predict(Feature &);
	Frame predict(FramePtr frame);
	void train(DataPtr);

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(_nodes,
			_bg_mode);

	}

	size_t getNumNodes() {
		return _nodes.size();
	}

	void save(const std::string & file) {
		std::ofstream buf(file, std::ios::binary);
		cereal::BinaryOutputArchive ar(buf);
		ar(*this);
	}

private:

	std::vector<Node> _nodes;
	BackgroundMode _bg_mode;
};

class RandomForest
{

public:

	static Frame majorityVoting(const std::vector<Frame>& frames);

	std::vector<Label> predict(DataPtr);
	Frame predict(FramePtr frame);
	void train(DataPtr);

	void push_tree(const RandomTree & tree) {
		_tree_ensemble.push_back(tree);
	}

	void save(const std::string & file) {
		std::ofstream buf(file, std::ios::binary);
		cereal::BinaryOutputArchive ar(buf);
		ar(*this);
	}

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(_tree_ensemble);
	}

private:

	std::vector<RandomTree> _tree_ensemble;

};
