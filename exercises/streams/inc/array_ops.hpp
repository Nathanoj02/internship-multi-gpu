#ifndef ARRAY_OPS_HPP
#define ARRAY_OPS_HPP

#include <vector>

/**
 * Generates an array with random integer values
 * @param array Pointer to the array to be filled
 * @param elems Number of elements in the array
 * @param min_value Minimum value for random integers
 * @param max_value Maximum value for random integers
 */
void fill_array(int* array, int elems, int min_value, int max_value);

/**
 * Sums two arrays element-wise
 * @param out Output array to store the result
 * @param a First input array
 * @param b Second input array
 * @param elems Number of elements in the arrays
 */
void array_sum(int* out, const int* a, const int* b, int elems);

#endif // ARRAY_OPS_HPP