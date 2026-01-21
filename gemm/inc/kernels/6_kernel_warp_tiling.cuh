#ifndef KERNEL_WARP_TILING_CUH
#define KERNEL_WARP_TILING_CUH

#include "../dtype.hpp"

/**
 * CUDA kernel to perform matrix multiplication using warp tiling
 * Each warp computes a WN x WM tile, with each thread computing multiple elements
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A (N)
 * @param cols_a Number of columns in matrix A (K)
 * @param cols_b Number of columns in matrix B (M)
 */
__global__ void gemm_warp_tiling_kernel (
    const dtype* A, const dtype* B, dtype* C,
    int rows_a, int cols_a, int cols_b
);

#endif // KERNEL_WARP_TILING_CUH