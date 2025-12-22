#include "array_ops.hpp"
#include <random>

void fill_array(int* array, int elems, int min_value, int max_value) {
    // Fill array with random int values
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<> dis(min_value, max_value);

    for (int i = 0; i < elems; ++i) {
        array[i] = dis(rng);
    }
}

void array_sum(int* out, const int* a, const int* b, int elems) {
    for (int i = 0; i < elems; ++i) {
        out[i] = a[i] + b[i];
    }
}
