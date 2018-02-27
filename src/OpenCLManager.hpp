/*
 * Copyright © MindMaze Holding SA 2017 - All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * CONFIDENTIAL: This project is proprietary and confidential. It cannot be
 * copied and/or distributed without the express permission of MindMaze
 * Holding SA.
 */

#pragma once

#include <CL/opencl.h>
#include <CL/cl.hpp>
#include "OpenCLKernel.hpp"
#include <memory>
#include <map>

class OpenCLManager {

public:
	OpenCLManager();

	void loadKernel(std::shared_ptr<OpenCLKernel> cl_kernel, std::string kernel_map_name);
	void loadKernel(std::shared_ptr<OpenCLKernel> cl_kernel);

	template<typename T>
	void getDeviceInfo(size_t param_name, T * param);

	cl::Kernel& getKernel(std::string name) {
		return _kernels[name]->getKernel();
	}

	void createRWBuffer(cl::Buffer & buffer, size_t bytes, cl_mem_flags flags) {
		buffer = cl::Buffer(_context,
				    flags,
				    bytes,
				    NULL,
				    NULL);
	}

	void createImageBuffer(cl::Image2D & texture, size_t width, size_t height,
						   cl_mem_flags flags, cl::ImageFormat format, unsigned char *data) {
		texture = cl::Image2D(_context,
							  flags,
							  format,
							  width,
							  height,
							  0,
							  data);
	}

	void detectDevice(cl_device_type & dev_type);

	cl::Context & getContext() {
		return _context;
	}

	cl::CommandQueue & getCommandQueue() {
		return _queue;
	}

	cl::CommandQueue* getCommandQueuePtr() {
		return &_queue;
	}

	void executeKernel(std::string name, CLSharedData *data) {
		_kernels[name]->execute(data);
	}

	int fillBufferWithValue(cl::Buffer mem_object, float value, size_t size);

	int copyDataDeviceToDevice(cl::Buffer src,
							   cl::Buffer dst,
							   int numbytes) {

		return _queue.enqueueCopyBuffer(src,
										dst,
										0,
										0,
										numbytes);
	}

	template <typename T>
	int copyImageToHost(cl::Image2D mem_object,
						size_t w, size_t h,
						T *data) {

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		
		cl::size_t<3> region;
		region[0] = w;
		region[1] = h;
		region[2] = 1;
		
		return _queue.enqueueReadImage(mem_object,
									   CL_TRUE, // blocking op. (change if slow?)
									   origin,
									   region,
									   0,
									   0,
									   data);
	}

	template <typename T>
	int copyImageToDevice(const T* data,
						  size_t w, size_t h,
						  cl::Image2D mem_object) {

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		
		cl::size_t<3> region;
		region[0] = w;
		region[1] = h;
		region[2] = 1;
			
		return _queue.enqueueWriteImage(mem_object,
										CL_TRUE, // blocking op. (change if slow?)
										origin,
										region,
										0,
										0,
										data);
	}
	
	template <typename T>
	int copyDataToDevice(const T *data,
						 int numbytes,
						 cl::Buffer mem_object) {

		return _queue.enqueueWriteBuffer(mem_object,
										 CL_TRUE,0,
										 numbytes,
										 data);
	}

	template <typename T>
	int copyDataToDevice(const T *data,
						 int numbytes,
						 int offset,
						 cl::Buffer mem_object) {

		return _queue.enqueueWriteBuffer(mem_object,
										 CL_TRUE, offset,
										 numbytes,
										 data);
	}

	template <typename T>
	int copyDataToHost(cl::Buffer mem_object,
					   int numbytes,
					   T *data) {

		return _queue.enqueueReadBuffer(mem_object,
										CL_TRUE,
										0,
										numbytes,
										data);
	}


	template <typename T>
	int copyDataToHost(cl::Buffer mem_object,
					   int numbytes,
					   int offset,
					   T *data) {
		return _queue.enqueueReadBuffer(mem_object,
										CL_TRUE, offset,
										numbytes,
										data);
	}

	std::vector<cl::Device> & getDevices() { return _devices; }
	cl::Device & getDevice() { return _devices[_device_id]; }


private:

	std::map<std::string, std::shared_ptr<OpenCLKernel> > _kernels;
	std::vector<cl::Platform> _platforms;
	std::vector<cl::Device> _devices;

	int _platform_id;
	int _device_id;

	cl::CommandQueue _queue;
	cl::Context _context;

	std::vector<int> _max_workgroup_sizes;

};
