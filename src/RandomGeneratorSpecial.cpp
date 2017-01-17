#include "RandomGenerator.hpp"

float random_real_in_range(Range range) {
#if STATIC_SEEDING
	static std::mt19937 mt(0);
#else
	static std::mt19937 mt(std::random_device{}());
#endif
	static std::uniform_real_distribution<float> pick;
	return pick(mt, decltype(pick)::param_type{range.min, range.max});
}

float random_real(float min, float max) {
#if STATIC_SEEDING
	static std::mt19937 mt(0);
#else
	static std::mt19937 mt(std::random_device{}());
#endif
	static std::uniform_real_distribution<float> pick;
	return pick(mt, decltype(pick)::param_type{min, max});
}


int random_int(int min, int max) {
#if STATIC_SEEDING
	static std::mt19937 mt(0);
#else
	static std::mt19937 mt(std::random_device{}());
#endif
	static std::uniform_int_distribution<int> pick;
	return pick(mt, decltype(pick)::param_type{min, max});
}
