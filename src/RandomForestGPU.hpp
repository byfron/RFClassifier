#pragma once

#include "RandomForest.hpp"

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

class RandomTreeGPU : public RandomTree
{

public:
	void computeGPUTree(GPUTree* gpu_tree);	
};

class RandomForestGPU : public RandomForest 
{

public:
	RandomForestGPU();
	void initOpenCL();
	void createGPUBuffers();

private:
	OpenCLManager _cl_manager;
	CLSharedData _shared_data;
};
