#ifndef ARRAY_OPS_HPP
#define ARRAY_OPS_HPP

#include <vector>

/**
 * Generates an array with random integer values
 * @param elems Number of elements in the array
 * @param min_value Minimum value for random integers
 * @param max_value Maximum value for random integers
 * @return Generated array
 */
std::vector<int> generate_array(int elems, int min_value, int max_value);

/**
 * Reduces an array by summing its elements
 * @param v Input array
 * @param elems Number of elements in the array
 * @returns Sum of the array elements
 */
int reduce_array(const std::vector<int>& v, int elems);

#endif // ARRAY_OPS_HPP