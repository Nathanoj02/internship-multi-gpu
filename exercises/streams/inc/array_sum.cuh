#ifndef ARRAY_SUM_CUH
#define ARRAY_SUM_CUH

/**
 * Performs element-wise addition of two integer arrays using a naive approach.
 * @param out Pointer to the output array where the result will be stored.
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param elems Number of elements in the input arrays.
 */
void array_sum_naive(int* out, const int* a, const int* b, size_t elems);

/**
 * Performs element-wise addition of two integer arrays using CUDA streams for concurrency.
 * @param out Pointer to the output array where the result will be stored.
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param elems Number of elements in the input arrays.
 */
void array_sum_streams(int* out, const int* a, const int* b, size_t elems);

#endif // ARRAY_SUM_CUH