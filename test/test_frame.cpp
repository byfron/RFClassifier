#include "gtest.h"
#include <Frame.hpp>

TEST(TestFrame, TestInit) {

	std::string depth_path = std::string(DATA_FOLDER) + "/train/1/images/depthRender/Cam1/mayaProject.000001.png";
	std::string gt_path = std::string(DATA_FOLDER) + "/train/1/images/groundtruth/Cam1/mayaProject.000001.png";

	Frame f(depth_path, gt_path);

	f.show();
		
}



