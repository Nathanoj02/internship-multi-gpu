#ifndef GEMM_CUDA_CUH
#define GEMM_CUDA_CUH

#include "dtype.hpp"

#include "kernels/1_kernel_naive.cuh"
#include "kernels/2_kernel_memory_coalescing.cuh"
#include "kernels/3_kernel_shared_memory.cuh"
#include "kernels/4_kernel_block_tiling.cuh"
#include "kernels/5_kernel_2D_block_tiling.cuh"
#include "kernels/6_kernel_warp_tiling.cuh"
#include "kernels/7_kernel_tensor_naive.cuh"
#include "kernels/8_kernel_tensor_warp_tiling.cuh"

/**
 * Initialize device memory and copy input matrices
 * @param d_A Device pointer for the first matrix
 * @param d_B Device pointer for the second matrix
 * @param d_C Device pointer for the resultant matrix
 * @param A Host pointer for the first matrix
 * @param B Host pointer for the second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void init_gemm (
    dtype** d_A, dtype** d_B, dtype** d_C,
    const dtype* A, const dtype* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

/**
 * Copy result back to host and free device memory
 * @param d_A Device pointer for the first matrix
 * @param d_B Device pointer for the second matrix
 * @param d_C Device pointer for the resultant matrix
 * @param result Host pointer for the resultant matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_b Number of columns in the second matrix
 */
void cleanup_gemm (
    dtype* d_A, dtype* d_B, dtype* d_C,
    dtype* result, size_t rows_a, size_t cols_b
);

/**
 * Initialize device memory and copy input matrices for half precision
 * @param d_A Device pointer for the first matrix
 * @param d_B Device pointer for the second matrix
 * @param d_C Device pointer for the resultant matrix
 * @param A Host pointer for the first matrix
 * @param B Host pointer for the second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 */
void init_gemm_tensor (
    half** d_A, half** d_B, float** d_C,
    const half* A, const half* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

/**
 * Copy result back to host and free device memory for half precision
 * @param d_A Device pointer for the first matrix
 * @param d_B Device pointer for the second matrix
 * @param d_C Device pointer for the resultant matrix
 * @param result Host pointer for the resultant matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_b Number of columns in the second matrix
 */
void cleanup_gemm_tensor (
    half* d_A, half* d_B, float* d_C,
    float* result, size_t rows_a, size_t cols_b
);

#endif // GEMM_CUDA_CUH