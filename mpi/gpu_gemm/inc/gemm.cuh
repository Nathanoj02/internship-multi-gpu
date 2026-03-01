#ifndef GEMM_CUDA_CUH
#define GEMM_CUDA_CUH

#include "6_kernel_warp_tiling.cuh"

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
    float** d_A, float** d_B, float** d_C,
    const float* A, const float* B,
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
    float* d_A, float* d_B, float* d_C,
    float* result, size_t rows_a, size_t cols_b
);

/**
 * Assign GPU device based on MPI rank
 * @param mpi_rank The rank of the MPI process
 */
void assign_gpu_device(int mpi_rank);

#endif // GEMM_CUDA_CUH