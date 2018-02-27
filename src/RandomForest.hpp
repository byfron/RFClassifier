#pragma once

#include "common.hpp"
#include "Frame.hpp"
#include "Node.hpp"
#include "OpenCLManager.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

struct GPUTree {
	uint32_t* left_child_arr;
	uint32_t* right_child_arr;
	float* offset1_x_arr;
	float* offset1_y_arr;
	float* offset2_x_arr;
	float* offset2_y_arr;
	float* threshold_arr;
	float* probability_arr;
	uint8_t* is_unary_arr;
	uint8_t* is_leaf_arr;
	uint8_t* label_arr;
};

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


	void computeGPUTree(GPUTree* gpu_tree);	
	
private:

	std::vector<Node> _nodes;
	BackgroundMode _bg_mode;
};

class RandomForest
{

public:

	static Frame majorityVoting(const std::vector<Frame>& frames);

	RandomForest();	
	
	std::vector<Label> predict(DataPtr);
	Frame predict(FramePtr frame);
	Frame predictGPU(FramePtr frame);
	void train(DataPtr);

	void initOpenCL();
	void createGPUBuffers();
	
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
	OpenCLManager _cl_manager;
	CLSharedData _shared_data;

};
