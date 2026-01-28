#include <cuda_runtime.h>
#include "kernels/2_kernel_memory_coalescing.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <const size_t BLOCK_SIZE>
/**
 * CUDA kernel to perform matrix multiplication with memory coalescing
 * @tparam BLOCK_SIZE Size of the block (assumed square)
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
) {
    int row = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    int col = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row >= rows_a || col >= cols_b) {
        return;
    }

    dtype value = DTYPE_ZERO;
    for (int k = 0; k < cols_a; ++k) {
        value += A[row * cols_a + k] * B[k * cols_b + col];
    }
    C[row * cols_b + col] = value;
}


void gemm_memory_coalescing (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    constexpr size_t BLOCK_SIZE = 32;

    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(cols_b, BLOCK_SIZE), CEIL_DIV(rows_a, BLOCK_SIZE));

    gemm_memory_coalescing_kernel<BLOCK_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}
