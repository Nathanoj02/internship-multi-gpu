#include "array_ops.hpp"
#include <random>

std::vector<int> generate_array(int elems, int min_value, int max_value) {
    // Generate array with random int values
    std::vector<int> array(elems);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<> dis(min_value, max_value);

    for (int i = 0; i < elems; ++i) {
        array[i] = dis(rng);
    }

    return array;
}

int reduce_array(const std::vector<int>& v, int elems) {
    int sum = 0;
    for (int i = 0; i < elems; ++i) {
        sum += v[i];
    }
    return sum;
}
