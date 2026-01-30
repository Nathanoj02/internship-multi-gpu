#ifndef KERNEL_TENSOR_WARP_TILING_CUH
#define KERNEL_TENSOR_WARP_TILING_CUH

#include "../dtype.cuh"

/**
 * Multiplies two matrices using GPU with tensor core warp tiling optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_tensor_warp_tiling (
    float* result, const half* A, const half* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

#endif // KERNEL_TENSOR_WARP_TILING_CUH