#include "common.hpp"
#include "RandomForest.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <queue>
#include <algorithm>

RandomForest::RandomForest() {

	initOpenCL();
	
}

void RandomForest::initOpenCL() {
	
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

void RandomForest::createGPUBuffers() {
	
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

Label RandomTree::predict(Feature & feature) {
	
	size_t node_idx = 0;

	for(;;) {
		Node & node = _nodes[node_idx];

		if (node.isLeaf()) {
			//	if (node._probability > 0.85)
			{
				return node.getLabel();
			}
			//else return 0;
		}
		else {
			if (node.fallsToLeftChild(feature)) {
				node_idx = node.left_child;
			}
			else {
				node_idx = node.right_child;
			}
		}
	}
}

void RandomTree::computeGPUTree(GPUTree* gpu_tree) {

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
	

Frame RandomTree::predict(FramePtr frame) {

	Frame output = *frame;
	Data features;
	frame->computeForegroundFeatures(features);

	for (auto feature : features) {
		Label l = predict(feature);
		output.setLabel(feature.row(), feature.col(), l);
	}
	return output;
}


std::vector<Label> RandomTree::predict(DataPtr data) {
	
	//TODO: book mem a priory
	std::vector<Label> labels;
	for (auto feature : *data)
		labels.push_back(predict(feature));

	return labels;
}

void RandomTree::train(DataPtr data) {

	_bg_mode = Settings::bmode;

	_nodes.clear();
	std::queue<NodeConstructor> queue;

	size_t depth;

	// Train root node
	Node root_node(0);
	DataSplit root_ds(data, data->begin(), data->end());
	root_node.train(root_ds);

	_nodes.push_back(root_node);

	if (!root_node.isLeaf()) {
		queue.push(
			NodeConstructor(0, //id
					0, //depth
					data->begin(),
					data->end()));
	}

	while(queue.size() > 0) {

		size_t node_id = queue.front().node_id;
		FeatureIterator start = queue.front().start;
		FeatureIterator end = queue.front().end;

		depth = queue.front().depth + 1;
		queue.pop();

		FeatureIterator split_it = _nodes[node_id].
			getSplitIterator(DataSplit(data, start, end));

		// Train left child
		Node left_node(depth);
		left_node.train(DataSplit(data, start, split_it));
		size_t left_id = _nodes.size();
		_nodes.push_back(left_node);

		// Train right child
		Node right_node(depth);
		right_node.train(DataSplit(data, split_it, end));
		size_t right_id = _nodes.size();
		_nodes.push_back(right_node);

		// Assign child indices to parent
		_nodes[node_id].left_child = left_id;
		_nodes[node_id].right_child = right_id;

		// Generate constructor for future nodes
		if (!left_node.isLeaf()) {
			queue.push(
				NodeConstructor(left_id,
						depth,
						start,
						split_it));
		}

		if (!right_node.isLeaf()) {
			queue.push(
				NodeConstructor(right_id,
						depth,
						split_it,
						end));
		}
	}

	std::cout << "Finished training. Tree has " <<_nodes.size() << " nodes with depth :" << depth << std::endl;
}

std::vector<Label> RandomForest::predict(DataPtr data) {
	for (int i = 0; i < _tree_ensemble.size(); i++) {
		return _tree_ensemble[i].predict(data);
	}
}

Frame RandomForest::majorityVoting(const std::vector<Frame> &frames) {

	// Frame majFrame = frames[0];
	cv::Size size = frames[0].getImageSize();
	cv::Mat best_labels = cv::Mat(size, CV_8UC1);
	best_labels.setTo(0);

	int votes[NUM_LABELS];

	// std::vector<int> votes;

	for (int r = 0; r < size.height; r++) {
		for (int c = 0; c < size.width; c++) {

			Label maj_label = best_labels.at<uchar>(r,c);

			for (int i = 0; i < NUM_LABELS; i++) {
				votes[i] = 0;
			}

			for (int f = 0; f < frames.size(); f++) {
				Label l = frames[f].getLabel(r, c);
				int idx = (int)l;
				votes[idx]++;
			}

			int max_votes = 0;
			Label best_label = 0;
			//find the majority vote
			for (int idx = 0; idx < NUM_LABELS; idx++) {
				if (votes[idx] > max_votes) {
					max_votes = votes[idx];
					best_label = (Label)idx;
				}
			}

			best_labels.at<uchar>(r,c) = best_label;
		}
	}

	Frame frame;
	frame.setDepthImage(frames[0].getDepthImage());
	frame.setLabelImage(best_labels);
	return frame;

}

Frame RandomForest::predictGPU(FramePtr frame) {

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

Frame RandomForest::predict(FramePtr frame) {

	return _tree_ensemble[0].predict(frame);
	
	// std::vector<Frame> frame_predictions;
	// for (int i = 0; i < _tree_ensemble.size(); i++) {
	// 	frame_predictions.push_back(_tree_ensemble[i].predict(frame));
	// }

	// //return frame_predictions[0];

	// return RandomForest::majorityVoting(frame_predictions);
}

void RandomForest::train(DataPtr data) {
	//TODO: refactor timing

	struct timespec start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);

	std::shared_ptr<RandomTree> tree = std::make_shared<RandomTree>();
	tree->train(data);
	_tree_ensemble.push_back(*tree);

	clock_gettime(CLOCK_MONOTONIC, &finish);
	double elapsed;
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	std::cout << ">> Finished training tree of " << tree->getNumNodes() <<
		" in " << elapsed << " seconds." << std::endl;
}
