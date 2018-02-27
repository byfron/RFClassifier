/*
 * Copyright © MindMaze Holding SA 2017 - All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * CONFIDENTIAL: This project is proprietary and confidential. It cannot be
 * copied and/or distributed without the express permission of MindMaze
 * Holding SA.
 */
#pragma once

#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <memory>
#include <map>

class OpenCLManager;

struct CLSharedData {

	//test
	cl::Buffer hello_buffer;
	
	// tree
	cl::Buffer left_child_arr;
	cl::Buffer right_child_arr;
	cl::Buffer offset1_x_arr;
	cl::Buffer offset1_y_arr;
	cl::Buffer offset2_x_arr;
	cl::Buffer offset2_y_arr;
	cl::Buffer threshold_arr;
	cl::Buffer probability_arr;
	cl::Buffer is_unary_arr;
	cl::Buffer is_leaf_arr;
	cl::Buffer label_arr;

	// I/O data
	int width;
	int height;
	cl::Buffer input_image_buffer;
//	cl::Image2D input_image_buffer;
	cl::Buffer output_image_buffer;
	cl::Buffer test_output_image_buffer;	
//	cl::Image2D test_output_image_buffer;	
};

unsigned long nextPow2( unsigned long x );

class OpenCLKernel {

public:
	OpenCLKernel(const std::string& filename, const std::string& name) :
		_manager(nullptr),
		_filename(filename),
		_name(name) {
	}

	void setDimension(cl::NDRange local, cl::NDRange global) {
		_local = local;
		_global = global;
	}

	void setKernel(cl::Kernel & k) { _kernel = k; }
	cl::Kernel & getKernel() { return _kernel; }
	void setManager(OpenCLManager *m) { _manager = m; }

	std::string getName() { return _name; }
	std::string getFilename() { return _filename; }

	virtual void load();
	virtual void execute(CLSharedData *data);
	
protected:

	virtual void setParams(CLSharedData *data) = 0;
	
	cl::NDRange _global;
	cl::NDRange _local;
	
	OpenCLManager *_manager;
	std::string _filename;
	std::string _name;	
	cl::Kernel _kernel;	
};

class TestKernel :  public OpenCLKernel {
public:
	TestKernel(const std::string& filename, const std::string& name) : OpenCLKernel(filename, name) {};
	void setParams(CLSharedData *data);	
};


class PredictKernel2 :  public OpenCLKernel {
public:
	PredictKernel2(const std::string& filename, const std::string& name) : OpenCLKernel(filename, name) {};
	void setParams(CLSharedData *data);	
};

class HelloKernel :  public OpenCLKernel {
public:
	HelloKernel(const std::string& filename, const std::string& name) : OpenCLKernel(filename, name) {};
	void setParams(CLSharedData *data);	
};

class PredictKernel : public OpenCLKernel {
public:
	PredictKernel(const std::string& filename, const std::string& name) : OpenCLKernel(filename, name) {};
	void setParams(CLSharedData *data);
	
};
