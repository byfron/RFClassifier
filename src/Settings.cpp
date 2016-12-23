#include "common.hpp"

int Settings::max_tree_depth = 20;
int Settings::num_pixels_per_image = 1000;
int Settings::num_thresholds_per_feature = 10;
int Settings::num_offsets_per_pixel = 500;//2000;
int Settings::offset_box_size = 10000;
int Settings::num_labels = 45; //TODO: Find out this!!
BackgroundMode Settings::bmode = BackgroundMode::DEFAULT;
Range Settings::bg_mid_range = Range(5,50);
Range Settings::bg_long_range = Range(50, 200);
