#include "kernels/3_kernel_shared_memory.cuh"
#include "definitions.hpp"
#include <cuda_runtime.h>

__global__ void gemm_shared_memory_kernel (
    const dtype* A, const dtype* B, dtype* C, 
    int rows_a, int cols_a, int cols_b
) {
    int cRow = blockIdx.y, cCol = blockIdx.x;
    int threadRow = threadIdx.x / BLOCK_SIZE;
    int threadCol = threadIdx.x % BLOCK_SIZE;
    
    __shared__ dtype As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ dtype Bs[BLOCK_SIZE * BLOCK_SIZE];
    
    // Advance pointers to the starting position for this block
    A += cRow * BLOCK_SIZE * cols_a;
    B += cCol * BLOCK_SIZE;
    C += cRow * BLOCK_SIZE * cols_b + cCol * BLOCK_SIZE;

    dtype value = DTYPE_ZERO;

    for (int bkIdx = 0; bkIdx < cols_a; bkIdx += BLOCK_SIZE) {
        // Load tiles into shared memory
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * cols_a + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * cols_b + threadCol];
        
        __syncthreads();
        
        // Advance pointers for next iteration
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * cols_b;
        
        // Compute using shared memory
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            value += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        
        __syncthreads();
    }
    
    C[threadRow * cols_b + threadCol] = value;
}
