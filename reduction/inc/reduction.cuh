#ifndef REDUCTION_CUH
#define REDUCTION_CUH

/**
 * Performs reduction (sum) of the elements in the array v
 * @param v Pointer to the input array (host memory)
 * @param elems Number of elements in the array
 * @return The sum of the elements in the array
 */
int reduce(const int* v, int elems);

#endif // REDUCTION_CUH