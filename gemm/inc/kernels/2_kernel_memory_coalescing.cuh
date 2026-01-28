#ifndef KERNEL_MEMORY_COALESCING_CUH
#define KERNEL_MEMORY_COALESCING_CUH

#include "../dtype.hpp"

/**
 * Multiplies two matrices using GPU with memory coalescing optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_memory_coalescing (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

#endif // KERNEL_MEMORY_COALESCING_CUH