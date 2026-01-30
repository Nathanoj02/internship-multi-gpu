#include <cuda_runtime.h>
#include "kernels/4_kernel_block_tiling.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <size_t BN, size_t BK, size_t BM, size_t TM>
/**
 * CUDA kernel to perform matrix multiplication using block tiling
 * @tparam BN The threadblock size for N dimension SMEM chaching
 * @tparam BK The threadblock size for K dimension SMEM chacing
 * @tparam BM The threadblock size for M dimension SMEM chacing
 * @tparam TM Number of M results computed per thread
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
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


void gemm_block_tiling (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    #ifdef CC80 // A30
    constexpr size_t BM = 64;
    constexpr size_t BN = 64;
    constexpr size_t BK = 8;
    constexpr size_t TM = 8;
    #else
    constexpr size_t BM = 64;
    constexpr size_t BN = 64;
    constexpr size_t BK = 8;
    constexpr size_t TM = 8;
    #endif

    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Number of threads per block = (BN * BM) / TM
    dim3 blockSize((BN * BM) / TM, 1, 1);
    dim3 gridSize(CEIL_DIV(cols_b, BM), CEIL_DIV(rows_a, BN));

    gemm_block_tiling_kernel<BN, BK, BM, TM><<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}
