#ifndef KERNEL_WARP_TILING_CUH
#define KERNEL_WARP_TILING_CUH

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
void gemm_warp_tiling (
    float* result, const float* A, const float* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

#endif // KERNEL_WARP_TILING_CUH