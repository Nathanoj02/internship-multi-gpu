#include "kernels/4_kernel_block_tiling.cuh"
#include "definitions.hpp"
#include <cuda_runtime.h>

__global__ void gemm_block_tiling_kernel (
    const dtype* A, const dtype* B, dtype* C,
    int rows_a, int cols_a, int cols_b
) {
    int cRow = blockIdx.y, cCol = blockIdx.x;
    
    // Calculate thread's position within the block
    int threadRow = threadIdx.x / (BK * TM);
    int threadCol = threadIdx.x % BM;
    
    // Inner indices for loading data into shared memory
    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BM;
    int innerColB = threadIdx.x % BM;
    
    __shared__ dtype As[BN * BK];
    __shared__ dtype Bs[BK * BM];
    
    // Advance pointers to the starting position for this block
    A += cRow * BN * cols_a;
    B += cCol * BM;
    C += cRow * BN * cols_b + cCol * BM;
    
    // Each thread accumulates TM results
    dtype threadResults[TM];
    for (int i = 0; i < TM; ++i) threadResults[i] = DTYPE_ZERO;
    
    // Loop over tiles along the K dimension
    for (int bkIdx = 0; bkIdx < cols_a; bkIdx += BK) {
        // Load tile from A into shared memory
        As[innerRowA * BK + innerColA] = A[innerRowA * cols_a + innerColA];
        
        // Load tile from B into shared memory
        Bs[innerRowB * BM + innerColB] = B[innerRowB * cols_b + innerColB];
        
        __syncthreads();
        
        // Advance pointers for next iteration
        A += BK;
        B += BK * cols_b;
        
        // Compute partial results for this tile
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            dtype Btmp = Bs[dotIdx * BM + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * cols_b + threadCol] = threadResults[resIdx];
    }
}
