#pragma once
#include "common.hpp"
#include <random>

#define STATIC_SEEDING 0

float random_real_in_range(Range range);
float random_real(float min, float max);
int random_int(int min, int max);
