#ifndef KERNEL_BLOCK_TILING_CUH
#define KERNEL_BLOCK_TILING_CUH

#include "../dtype.hpp"

/**
 * Multiplies two matrices using GPU with block tiling optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_block_tiling (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

#endif // KERNEL_BLOCK_TILING_CUH