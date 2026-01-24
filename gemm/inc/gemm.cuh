#ifndef GEMM_CUDA_CUH
#define GEMM_CUDA_CUH

#include "dtype.hpp"

/**
 * Multiplies two matrices using GPU
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_naive(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

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
void gemm_memory_coalescing(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

/**
 * Multiplies two matrices using GPU with shared memory optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_shared_memory(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

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
void gemm_block_tiling(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

/**
 * Multiplies two matrices using GPU with 2D block tiling optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_2D_block_tiling(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

/**
 * Multiplies two matrices using GPU with warp tiling optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_warp_tiling(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

/**
 * Multiplies two matrices using GPU with tensor core optimization
 * @param result Resultant matrix to store the multiplication result
 * @param A First matrix
 * @param B Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_tensor_naive(
    float* result, const half* A, const half* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

#endif // GEMM_CUDA_CUH