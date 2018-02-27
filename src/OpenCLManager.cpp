/*
 * Copyright © MindMaze Holding SA 2017 - All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * CONFIDENTIAL: This project is proprietary and confidential. It cannot be
 * copied and/or distributed without the express permission of MindMaze
 * Holding SA.
 */

#define __CL_ENABLE_EXCEPTIONS

#include "OpenCLManager.hpp"
#include "cl_utils.hpp"
#include <iostream>
#include <CL/cl.hpp>

using namespace std;

OpenCLManager::OpenCLManager() : _platform_id(0), _device_id(0) {

	std::cout << "Initializing OpenCL" << std::endl;
	
	// create platform
	int err = cl::Platform::get(&_platforms);
	cl_device_type device_requested;
	detectDevice(device_requested);

	try {

		for (unsigned int i = 0; i < _platforms.size(); i++) {
			std::string platform = _platforms[i].getInfo<CL_PLATFORM_NAME>();

			std::cout << "platform: " << platform << std::endl;
			
			try {
				cl_context_properties properties[] =
					{ CL_CONTEXT_PLATFORM, (cl_context_properties)(_platforms[i])(), 0};
				_context = cl::Context(device_requested, properties);
				_devices = _context.getInfo<CL_CONTEXT_DEVICES>();
				_platform_id = i;

			}catch(cl::Error& er) {
				printf("Platform has no computing capabilities: %s(%s)\n", er.what(), oclErrorString(er.err()));
			}
		}
		
		// create context
		cl_context_properties properties[] =
			{ CL_CONTEXT_PLATFORM, (cl_context_properties)(_platforms[_platform_id])(), 0};
		_context = cl::Context(device_requested, properties);
		_devices = _context.getInfo<CL_CONTEXT_DEVICES>();

		_devices[_device_id].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES,
					     &_max_workgroup_sizes);

		// create command queue
		_queue = cl::CommandQueue(_context, _devices[_device_id], 0, &err);

	}
	catch(cl::Error& er) {
		printf("OpenCLManager: %s(%s)\n", er.what(), oclErrorString(er.err()));
	}

	if (_devices.size() == 0) {
		printf("Error could not detect any valid device. Exiting");
		exit(-1);
	}

	std::cout << "OpenCL platform detected" << std::endl;
}

void OpenCLManager::detectDevice(cl_device_type & dev_type) {

	switch(OPENCL_DEVICE) {
	case 0:
		dev_type = CL_DEVICE_TYPE_GPU;
		break;
	case 1:
		dev_type = CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU;
		break;
	default:
		printf("invalid device id %d\n", OPENCL_DEVICE);
		exit(-1);
	};
}


template<typename T>
void OpenCLManager::getDeviceInfo(size_t param_name, T * param) {

	_devices[_device_id].getInfo<T>(param_name, param);

}

void OpenCLManager::loadKernel(std::shared_ptr<OpenCLKernel> cl_kernel) {
	loadKernel(cl_kernel, cl_kernel->getName());
}

void OpenCLManager::loadKernel(std::shared_ptr<OpenCLKernel> cl_kernel, std::string kernel_map_name) {
	cl_kernel->setManager(this);
	cl_kernel->load();
	_kernels[kernel_map_name] = cl_kernel;
}

int OpenCLManager::fillBufferWithValue(cl::Buffer mem_object, float value, size_t size) {

	return _queue.enqueueFillBuffer(mem_object,
					value,
					0,
					size);
}
