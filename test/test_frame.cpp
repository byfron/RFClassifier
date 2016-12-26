#include "gtest.h"
#include <Frame.hpp>
#include <Node.hpp>

extern std::string test_depth_path;
extern std::string test_gt_path;

TEST(TestFrame, TestBackground) {

	Frame f_default(test_depth_path, test_gt_path);

	//make sure bw pixels correspond with gt bw pixels
	cv::Mat depth_image = f_default.getDepthImage();
	cv::Mat labels = f_default.getLabelImage();
	cv::Size s = f_default.getImageSize();

	Settings::bmode = BackgroundMode::DEFAULT;
	for (int row = 0; row < s.height; row++) {
		for (int col = 0; col < s.width; col++) {
			if (!f_default.inForeground(row,col)) {
				EXPECT_EQ(depth_image.at<float>(row,col), MAX_DEPTH);
			}
		}
	}

	Settings::bmode = BackgroundMode::RANDOM_MIDRANGE;
	Frame f_mr(test_depth_path, test_gt_path);
	depth_image = f_mr.getDepthImage();
	labels = f_mr.getLabelImage();
	Range r = Settings::bg_mid_range;
	for (int row = 0; row < s.height; row++) {
		for (int col = 0; col < s.width; col++) {
			if (!f_mr.inForeground(row,col)) {
				EXPECT_TRUE(depth_image.at<float>(row,col) < r.max &&
					    depth_image.at<float>(row,col) > r.min);
			}
		}
	}

	Settings::bmode = BackgroundMode::RANDOM_LONGRANGE;
	Frame f_lr(test_depth_path, test_gt_path);
	depth_image = f_lr.getDepthImage();
	labels = f_lr.getLabelImage();
	r = Settings::bg_long_range;
	for (int row = 0; row < s.height; row++) {
		for (int col = 0; col < s.width; col++) {
			if (!f_lr.inForeground(row,col)) {
				EXPECT_TRUE(depth_image.at<float>(row,col) < r.max &&
					    depth_image.at<float>(row,col) > r.min);
			}
		}
	}
}

TEST(TestFrame, TestFWSampling) {

	int row, col;
	int num_samples = 100;
	Settings::bmode = BackgroundMode::DEFAULT;
	FramePtr frame = std::make_shared<Frame>(test_depth_path, test_gt_path);

	for (int i = 0; i < num_samples; i++) {
		FrameUtils::sampleFromForeground(frame, row, col);
		EXPECT_TRUE(frame->getLabel(row, col) != MAX_DEPTH);
		EXPECT_TRUE(frame->inForeground(row, col));
		EXPECT_TRUE(frame->operator()(row,col) != 0);
	}
}

TEST(TestFrame, TestFeature) {

	FramePtr frame = std::make_shared<Frame>(test_depth_path, test_gt_path);
	DataPtr features = std::make_shared<Data>();
	features->resize(Settings::num_pixels_per_image);
	Settings::bmode = BackgroundMode::DEFAULT;

	int row, col;
	for (int i = 0; i < Settings::num_pixels_per_image; i++) {
		FrameUtils::sampleFromForeground(frame, row, col);
		EXPECT_TRUE(frame->getLabel(row, col) != MAX_DEPTH);
		features->operator[](i) = Feature(row, col, frame->getLabel(row, col), frame);
		EXPECT_TRUE(frame->inForeground(row, col));
	}

	LearnerParameters params;
	params.offset_1[0] = 0;
	params.offset_1[1] = 0;
	params.offset_2[0] = 0;
	params.offset_2[1] = 0;

	for (int i = 0; i < Settings::num_pixels_per_image; i++) {
		Feature feat = features->operator[](i);
		feat.evaluate(params);
		if (frame->inForeground(feat.row(), feat.col()))
			EXPECT_EQ(feat.getValue(), 0.0);
	}
}
