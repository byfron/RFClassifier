#pragma once
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>

#define MAIN_DB_PATH "DB_PATH"
#define INF std::numeric_limits<float>::max()
#define MAX_DEPTH 10 //max depth: 10 meters
#define NUM_LABELS 13 //11 body parts + ignore + table
//24 //21[0-20] body parts, +1 ignore [21], +1 table [22], +1 background [23]

/// Data types
typedef cv::Vec<uchar,3> color;
typedef uint8_t Label;

/// We have different ways of generating background depth
/// The original paper assigns a fixed (large) constant depth
/// We try to generate different random depths in different trees
/// to synthetically generate more realistic background features
enum class BackgroundMode {
	DEFAULT, //10 meters in all cases
	RANDOM_MIDRANGE, // random 0.5-2.0 meters
	RANDOM_LONGRANGE // random 0.05 - 0.5 meters
};

struct Range {
	Range(float mi, float ma) : min(mi), max(ma) {}
	float min;
	float max;
};

/// Algorithm settings
class Settings {
public:
	static int num_trees;
	static int num_images_per_tree;
	static int num_thresholds_per_feature;
	static int num_offsets_per_pixel;
	static int num_pixels_per_image;
	static int max_tree_depth;
	static float maximum_depth_difference;
	static int offset_box_size;
	static BackgroundMode bmode;
	static Range bg_mid_range;
	static Range bg_long_range;
};

/// Command argument utils
static inline
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

static inline
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

/// Math utils
static inline
float fastlog2 (float x)
{
  union { float f; uint32_t i; } vx = { x };
  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
           - 1.498030302f * mx.f
           - 1.72587999f / (0.3520887068f + mx.f);
}

static inline
float fastlog (float x)
{
  return 0.69314718f * fastlog2 (x);
}
