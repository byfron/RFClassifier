#include "common.hpp"
#include "RandomForestGPU.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <queue>
#include <algorithm>

RandomForestGPU::RandomForestGPU() {

	initOpenCL();
	
}

void RandomForestGPU::initOpenCL() {
	
	std::string kernel_path = OPENCL_KERNEL_PATH;	
	_cl_manager.loadKernel(std::make_shared<TestKernel>(
							   kernel_path + "kernels.cl",
							   "test"));

	_cl_manager.loadKernel(std::make_shared<PredictKernel2>(
							   kernel_path + "kernels.cl",
							   "predict2"));

	// _cl_manager.loadKernel(std::make_shared<HelloKernel>(
	// 						   kernel_path + "test.cl",
	// 						   "hello"));

	std::cout << "loaded" << std::endl;

}

void RandomForestGPU::createGPUBuffers() {
	
	size_t num_nodes = _tree_ensemble[0].getNumNodes();

	_cl_manager.createRWBuffer(_shared_data.hello_buffer,
							   sizeof(char) * 14,
							   CL_MEM_READ_WRITE);
	
	// initialize tree memory buffers
	_cl_manager.createRWBuffer(_shared_data.left_child_arr,
							   sizeof(uint32_t) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.right_child_arr,
							   sizeof(uint32_t) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.offset1_x_arr,
							   sizeof(float) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.offset1_y_arr,
							   sizeof(float) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.offset2_x_arr,
							   sizeof(float) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.offset2_y_arr,
							   sizeof(float) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.threshold_arr,
							   sizeof(float) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.probability_arr,
							   sizeof(float) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.is_unary_arr,
							   sizeof(uint8_t) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.is_leaf_arr,
							   sizeof(uint8_t) * num_nodes,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.label_arr,
							   sizeof(uint8_t) * num_nodes,
							   CL_MEM_READ_ONLY);
	
	int width = 512;
	int height = 424;		


	_cl_manager.createRWBuffer(_shared_data.input_image_buffer,
							   sizeof(float) * width * height,
							   CL_MEM_READ_ONLY);

	_cl_manager.createRWBuffer(_shared_data.test_output_image_buffer,
							   sizeof(uint8_t) * width * height,
							   CL_MEM_READ_WRITE);

	_cl_manager.createRWBuffer(_shared_data.output_image_buffer,
							   sizeof(float) * width * height,
							   CL_MEM_READ_WRITE);   

	// copy tree data to shared buffers
	GPUTree gpu_tree;
	_tree_ensemble[0].computeGPUTree(&gpu_tree);
	
    _cl_manager.copyDataToDevice(gpu_tree.left_child_arr,
								 sizeof(uint32_t) * num_nodes,
								 _shared_data.left_child_arr);

	_cl_manager.copyDataToDevice(gpu_tree.right_child_arr,
								 sizeof(uint32_t) * num_nodes,
								 _shared_data.right_child_arr);

	_cl_manager.copyDataToDevice(gpu_tree.offset1_x_arr,
								 sizeof(float) * num_nodes,
								 _shared_data.offset1_x_arr);

	_cl_manager.copyDataToDevice(gpu_tree.offset1_y_arr,
								 sizeof(float) * num_nodes,
								 _shared_data.offset1_y_arr);

	_cl_manager.copyDataToDevice(gpu_tree.offset2_x_arr,
								 sizeof(float) * num_nodes,
								 _shared_data.offset2_x_arr);

	_cl_manager.copyDataToDevice(gpu_tree.offset2_y_arr,
								 sizeof(float) * num_nodes,
								 _shared_data.offset2_y_arr);

	_cl_manager.copyDataToDevice(gpu_tree.threshold_arr,
								 sizeof(float) * num_nodes,
								 _shared_data.threshold_arr);

	_cl_manager.copyDataToDevice(gpu_tree.probability_arr,
								 sizeof(float) * num_nodes,
								 _shared_data.probability_arr);

	_cl_manager.copyDataToDevice(gpu_tree.is_unary_arr,
								 sizeof(uint8_t) * num_nodes,
								 _shared_data.is_unary_arr);

	_cl_manager.copyDataToDevice(gpu_tree.is_leaf_arr,
								 sizeof(uint8_t) * num_nodes,
								 _shared_data.is_leaf_arr);

	_cl_manager.copyDataToDevice(gpu_tree.label_arr,
								 sizeof(uint8_t) * num_nodes,
								 _shared_data.label_arr);
	   	
}


void RandomTreeGPU::computeGPUTree(GPUTree* gpu_tree) {

	int num_nodes = _nodes.size();
	
	gpu_tree->left_child_arr = new uint32_t[num_nodes];
	gpu_tree->right_child_arr = new uint32_t[num_nodes];
	gpu_tree->offset1_x_arr = new float[num_nodes];
	gpu_tree->offset1_y_arr = new float[num_nodes];
	gpu_tree->offset2_x_arr = new float[num_nodes];
	gpu_tree->offset2_y_arr = new float[num_nodes];
	gpu_tree->threshold_arr = new float[num_nodes];
	gpu_tree->probability_arr = new float[num_nodes];
	gpu_tree->is_unary_arr = new uint8_t[num_nodes];
	gpu_tree->is_leaf_arr = new uint8_t[num_nodes];
	gpu_tree->label_arr = new uint8_t[num_nodes];

	for (int i = 0; i < _nodes.size(); i++) {
	   
		const Node& node = _nodes[i];

		if (node.isLeaf()) {
			gpu_tree->is_leaf_arr[i] = 1;
			gpu_tree->left_child_arr[i] = 0;
			gpu_tree->right_child_arr[i] = 0;
			gpu_tree->probability_arr[i] = node._probability;
			gpu_tree->label_arr[i] = node.getLabel();
		}		
		else {
			gpu_tree->is_leaf_arr[i] = 0;			
			gpu_tree->left_child_arr[i] = (uint32_t)node.left_child;			
			gpu_tree->right_child_arr[i] = (uint32_t)node.right_child;
			gpu_tree->offset1_x_arr[i] = node._node_params.offset_1[0];
			gpu_tree->offset1_y_arr[i] = node._node_params.offset_1[1];
			gpu_tree->offset2_x_arr[i] = node._node_params.offset_2[0];
			gpu_tree->offset2_y_arr[i] = node._node_params.offset_2[1];
			gpu_tree->is_unary_arr[i] = (node._node_params.is_unary)?1:0;
			gpu_tree->threshold_arr[i] = node._threshold;
			gpu_tree->probability_arr[i] = node._probability;
			gpu_tree->label_arr[i] = 0;
		}
	}
}
	
Frame RandomForestGPU::predict(FramePtr frame) {

	// copy frame to device memory
	Frame out_frame = *frame;
	cv::Mat depth = frame->getDepthImage();


	
	// cv::Mat in[] = {depth, depth, depth, depth};
	// cv::Mat out;
	// cv::merge(in, 4, out);

	std::cout << depth.cols << " , " << depth.rows << std::endl;
	_shared_data.width = depth.cols;
	_shared_data.height = depth.rows;


	// _cl_manager.createRWBuffer(_shared_data.label_arr,
	// 						   sizeof(uint8_t) * _shared_data.width * _shared_data.height,
	// 						   CL_MEM_READ_ONLY);
	
	// _cl_manager.copyDataToDevice(out_frame.getLabelImage().data,
	// 							 sizeof(uint8_t) * depth.cols * depth.rows,
	// 							 _shared_data.label_arr);
	
	
	// _cl_manager.createRWBuffer(_shared_data.input_image_buffer,
	// 						   sizeof(float) * _shared_data.width * _shared_data.height,
	// 						   CL_MEM_READ_ONLY);

	// _cl_manager.createRWBuffer(_shared_data.test_output_image_buffer,
	// 						   sizeof(uint8_t) * _shared_data.width * _shared_data.height,
	// 						   CL_MEM_READ_WRITE);
	
	// _cl_manager.createRWBuffer(_shared_data.output_image_buffer,
	// 						   sizeof(float) * _shared_data.width * _shared_data.height,
	// 						   CL_MEM_READ_WRITE);
	
	
	

	_cl_manager.copyDataToDevice(depth.data,
								 sizeof(float) * depth.cols * depth.rows,
								 _shared_data.input_image_buffer);
	
	
	// _cl_manager.copyImageToDevice(reinterpret_cast<float*>(depth.data),
	// 							  depth.cols, //w
	// 							  depth.rows, //h
	// 							  _shared_data.input_image_buffer);


	//cv::Mat result = cv::Mat(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));
	
	_cl_manager.executeKernel("predict2", &_shared_data);
//	_cl_manager.executeKernel("test", &_shared_data);


	//uint8_t *mem = new uint8_t[depth.cols * depth.rows];
	
	// copy result to host memory	
	cv::Mat result(depth.rows, depth.cols, CV_8UC1);

	_cl_manager.copyDataToHost<uint8_t>(_shared_data.test_output_image_buffer,
										sizeof(uint8_t) * depth.cols * depth.rows,
										result.data);

	// cv::Mat wrap(depth.rows, depth.cols, CV_8UC1, NULL, CV_AUTOSTEP);
	// wrap.data = mem;
	// cv::Mat result = wrap.clone();
	
	
	// _cl_manager.copyDataToHost<unsigned char>(_shared_data.output_image_buffer,
	// 										  sizeof(float) * depth.cols * depth.rows,
	// 										  result.data);
	
	
	// _cl_manager.copyImageToHost(_shared_data.test_output_image_buffer,
	// 							depth.cols, //w
	// 							depth.rows, //h
	// 							result.data);

	// cv::imshow("depth", depth);
	// cv::imshow("result", result);
	// cv::waitKey(0);

	out_frame.setLabelImage(result);
	return out_frame;
	

	

}
