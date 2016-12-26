#include "gtest.h"
#include <Frame.hpp>
#include <Node.hpp>

extern std::string test_depth_path;
extern std::string test_gt_path;

TEST(TestNode, TestOffsets) {

	float offset[2];
	sampleOffset(offset);

	EXPECT_TRUE(offset[0] >= -Settings::offset_box_size/2);
	EXPECT_TRUE(offset[0] <=  Settings::offset_box_size/2);
	EXPECT_TRUE(offset[1] >= -Settings::offset_box_size/2);
	EXPECT_TRUE(offset[1] <=  Settings::offset_box_size/2);
}

TEST(TestNode, TestTrain) {

	Settings::bmode = BackgroundMode::DEFAULT;

	FramePtr frame = std::make_shared<Frame>(test_depth_path, test_gt_path);
	DataPtr features = std::make_shared<Data>();
	features->resize(Settings::num_pixels_per_image);

	int row, col;
	for (int i = 0; i < Settings::num_pixels_per_image; i++) {
		FrameUtils::sampleFromForeground(frame, row, col);
		features->operator[](i) = Feature(row, col, frame->getLabel(row, col), frame);
	}

	DataSplit ds(features, features->begin(), features->end());

	Node node(0);
	node.train(ds);

	//make sure features are sorted

	//make sure split is good?

	//make sure entropy decreases?


}
