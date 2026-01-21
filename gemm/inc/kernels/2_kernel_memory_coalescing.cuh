#ifndef KERNEL_MEMORY_COALESCING_CUH
#define KERNEL_MEMORY_COALESCING_CUH

#include "../dtype.hpp"

/**
 * CUDA kernel to perform matrix multiplication with memory coalescing
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__ void gemm_memory_coalescing_kernel (
    const dtype* A, const dtype* B, dtype* C,
    int rows_a, int cols_a, int cols_b
); 

#endif // KERNEL_MEMORY_COALESCING_CUH