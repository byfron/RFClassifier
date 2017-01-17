#include "RandomGenerator.hpp"

float random_real_in_range(Range range) {

    static std::mt19937 mt(std::random_device{}());
    static std::uniform_real_distribution<float> pick;

    return pick(mt, decltype(pick)::param_type{range.min, range.max});
}

float random_real(float min, float max) {

    static std::mt19937 mt(std::random_device{}());
    static std::uniform_real_distribution<float> pick;

    return pick(mt, decltype(pick)::param_type{min, max});
}


int random_int(int min, int max) {

    static std::mt19937 mt(std::random_device{}());
    static std::uniform_int_distribution<int> pick;

    return pick(mt, decltype(pick)::param_type{min, max});

}
