#ifndef KERNEL_2D_BLOCK_TILING_CUH
#define KERNEL_2D_BLOCK_TILING_CUH

#include "../dtype.hpp"

/**
 * CUDA kernel to perform matrix multiplication using 2D block tiling
 * Each thread computes TM2D x TN2D elements of the result matrix C
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A (N)
 * @param cols_a Number of columns in matrix A (K)
 * @param cols_b Number of columns in matrix B (M)
 */
__global__ void gemm_2D_block_tiling_kernel (
    const dtype* A, const dtype* B, dtype* C,
    int rows_a, int cols_a, int cols_b
);

#endif // KERNEL_2D_BLOCK_TILING_CUH