#include "kernels/2_kernel_memory_coalescing.cuh"
#include "definitions.hpp"
#include <cuda_runtime.h>

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
