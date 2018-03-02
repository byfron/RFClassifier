#include "common.hpp"

int Settings::max_tree_depth = 20;
int Settings::num_pixels_per_image = 2000; //2000 in original paper
int Settings::num_thresholds_per_feature = 20; //20
int Settings::num_offsets_per_pixel = 500; //500
int Settings::offset_box_size = 350; //129*2 pixel meters
BackgroundMode Settings::bmode = BackgroundMode::DEFAULT;
Range Settings::bg_mid_range = Range(5,50);
Range Settings::bg_long_range = Range(50, 200);
