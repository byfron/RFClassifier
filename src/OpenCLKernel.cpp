
#define __CL_ENABLE_EXCEPTIONS

#include "OpenCLKernel.hpp"
#include "OpenCLManager.hpp"
#include "cl_utils.hpp"
#include <fstream>
#include <assert.h>
#include <iostream>
#include <utility>
#include <chrono>
#include <cmath>

#define TILE_SIZE 32

unsigned long roundUp(unsigned long numToRound, int multiple) 
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

unsigned long nextPow2( unsigned long x ) {

	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x++;

	return x;
}

void OpenCLKernel::load() {
		
	assert(_manager);

	std::ifstream cl_file(getFilename());
	std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));

	try {

		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
							      cl_string.length() + 1));
		cl::Program program(_manager->getContext(), source);
		
		// compile opencl source
		program.build(_manager->getDevices());
		
		// load named kernel from opencl source
		cl::Kernel kernel(program, getName().c_str());
		_kernel = kernel;
	}
	catch(cl::Error& er) {
		printf("ERROR loadingKernel:%s -> %s(%s)\n", getName().c_str(), er.what(), oclErrorString(er.err()));
	}
}

void OpenCLKernel::execute(CLSharedData *data) {

	try {
		setParams(data);
		_manager->getCommandQueue().enqueueNDRangeKernel(_kernel,
								 cl::NullRange,
								 _global,
								 _local,
								 NULL,
								 NULL);
	}
	catch(cl::Error& er) {
		printf("ERROR %s: %s(%s)\n", _name.c_str(), er.what(), oclErrorString(er.err()));
	}		
}

void TestKernel::setParams(CLSharedData *data) {

	_kernel.setArg(0, data->width);
	_kernel.setArg(1, data->height);
	_kernel.setArg(2, data->input_image_buffer);
	_kernel.setArg(3, data->output_image_buffer);
	
	int width = nextPow2(data->width);
	int height = nextPow2(data->height);
	
	_global = cl::NDRange(width, height);
	_local = cl::NDRange(32, 32);	
}

void PredictKernel2::setParams(CLSharedData *data) {

	_kernel.setArg(0, data->width);
	_kernel.setArg(1, data->height);
	_kernel.setArg(2, data->left_child_arr);
	_kernel.setArg(3, data->right_child_arr);
	_kernel.setArg(4, data->offset1_x_arr);
	_kernel.setArg(5, data->offset1_y_arr);
	_kernel.setArg(6, data->offset2_x_arr);
	_kernel.setArg(7, data->offset2_y_arr);
	_kernel.setArg(8, data->threshold_arr);
	_kernel.setArg(9, data->probability_arr);
	_kernel.setArg(10, data->is_unary_arr);
	_kernel.setArg(11, data->is_leaf_arr);
	_kernel.setArg(12, data->label_arr);
	_kernel.setArg(13, data->input_image_buffer);
	_kernel.setArg(14, data->test_output_image_buffer);
	
	int width = nextPow2(data->width);
	int height = nextPow2(data->height);
	
	_global = cl::NDRange(width, height);
	_local = cl::NDRange(32, 32);	
}

void HelloKernel::setParams(CLSharedData *data) {
	_kernel.setArg(0, data->hello_buffer);

	_global = cl::NDRange(128);
	_local = cl::NDRange(128);
}

void PredictKernel::setParams(CLSharedData *data) {

	_kernel.setArg(0, data->width);
	_kernel.setArg(1, data->height);
	_kernel.setArg(2, data->left_child_arr);
	_kernel.setArg(3, data->right_child_arr);
	_kernel.setArg(4, data->offset1_x_arr);
	_kernel.setArg(5, data->offset1_y_arr);
	_kernel.setArg(6, data->offset2_x_arr);
	_kernel.setArg(7, data->offset2_y_arr);
	_kernel.setArg(8, data->threshold_arr);
	_kernel.setArg(9, data->is_unary_arr);
	_kernel.setArg(10, data->is_leaf_arr);	
	_kernel.setArg(11, data->label_arr);
	_kernel.setArg(12, data->input_image_buffer);
	_kernel.setArg(13, data->output_image_buffer);

	int width = nextPow2(512);
	int height = nextPow2(424);
	
	_global = cl::NDRange(width, height);
	_local = cl::NDRange(32, 32);	
}
