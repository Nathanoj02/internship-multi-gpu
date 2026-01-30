#include <cuda_runtime.h>
#include "kernels/5_kernel_2D_block_tiling.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(x, y) (((x) + (y) -1) / (y))

template <
    size_t BN2D, size_t BK2D, size_t BM2D,
    size_t TM2D, size_t TN2D
>
/**
 * CUDA kernel to perform matrix multiplication using 2D block tiling
 * Each thread computes TM2D x TN2D elements of the result matrix C
 * @tparam BN2D The threadblock size for rows of matrix A
 * @tparam BK2D The threadblock size for columns of matrix A / rows of matrix B
 * @tparam BM2D The threadblock size for columns of matrix B
 * @tparam TM2D Number of rows computed by each thread
 * @tparam TN2D Number of columns computed by each thread
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A (N)
 * @param cols_a Number of columns in matrix A (K)
 * @param cols_b Number of columns in matrix B (M)
 */
__global__ void gemm_2D_block_tiling_kernel (
    const dtype* A, const dtype* B, dtype* C,
    int rows_a, int cols_a, int cols_b
) {
    int cRow = blockIdx.y, cCol = blockIdx.x;
    
    // Thread positioning within block
    int threadRow = threadIdx.x / (BM2D / TM2D);
    int threadCol = threadIdx.x % (BM2D / TM2D);
    
    // Inner indices for loading data into shared memory
    int innerRowA = threadIdx.x / BK2D;
    int innerColA = threadIdx.x % BK2D;
    int innerRowB = threadIdx.x / BM2D;
    int innerColB = threadIdx.x % BM2D;
    
    // Stride for loading multiple rows
    int strideA = blockDim.x / BK2D;
    int strideB = blockDim.x / BM2D;
    
    __shared__ dtype As[BN2D * BK2D];
    __shared__ dtype Bs[BK2D * BM2D];
    
    // Advance pointers to starting position for this block
    A += cRow * BN2D * cols_a;
    B += cCol * BM2D;
    C += cRow * BN2D * cols_b + cCol * BM2D;
    
    // Each thread accumulates TM2D x TN2D results
    dtype threadResults[TM2D * TN2D];
    dtype regM[TM2D];
    dtype regN[TN2D];
    for (int i = 0; i < TM2D * TN2D; ++i) threadResults[i] = DTYPE_ZERO;
    for (int i = 0; i < TM2D; ++i) regM[i] = DTYPE_ZERO;
    for (int i = 0; i < TN2D; ++i) regN[i] = DTYPE_ZERO;
    
    // Loop over tiles
    for (int bkIdx = 0; bkIdx < cols_a; bkIdx += BK2D) {
        // Load tile from A into shared memory (strided loads)
        for (int loadOffset = 0; loadOffset < BN2D; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK2D + innerColA] = 
                A[(innerRowA + loadOffset) * cols_a + innerColA];
        }
        
        // Load tile from B into shared memory (strided loads)
        for (int loadOffset = 0; loadOffset < BK2D; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BM2D + innerColB] = 
                B[(innerRowB + loadOffset) * cols_b + innerColB];
        }
        
        __syncthreads();
        
        // Advance pointers for next iteration
        A += BK2D;
        B += BK2D * cols_b;
        
        // Compute using register tiling
        for (int dotIdx = 0; dotIdx < BK2D; ++dotIdx) {
            // Load TN2D elements from A into registers
            for (int i = 0; i < TN2D; ++i) {
                regN[i] = As[(threadRow * TN2D + i) * BK2D + dotIdx];
            }
            
            // Load TM2D elements from B into registers
            for (int i = 0; i < TM2D; ++i) {
                regM[i] = Bs[dotIdx * BM2D + threadCol * TM2D + i];
            }
            
            // Compute TM2D x TN2D results
            for (int resIdxM = 0; resIdxM < TM2D; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN2D; ++resIdxN) {
                    threadResults[resIdxN * TM2D + resIdxM] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    for (int resIdxM = 0; resIdxM < TM2D; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN2D; ++resIdxN) {
            C[(threadRow * TN2D + resIdxN) * cols_b + threadCol * TM2D + resIdxM] = 
                threadResults[resIdxN * TM2D + resIdxM];
        }
    }
}


void gemm_2D_block_tiling (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    #ifdef CC80
    constexpr size_t BN2D = 128;
    constexpr size_t BK2D = 16;
    constexpr size_t BM2D = 128;
    constexpr size_t TM2D = 8;
    constexpr size_t TN2D = 8;
    #else
    constexpr size_t BN2D = 64;
    constexpr size_t BK2D = 8;
    constexpr size_t BM2D = 64;
    constexpr size_t TM2D = 4;
    constexpr size_t TN2D = 4;
    #endif

    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize((BN2D * BM2D) / (TM2D * TN2D));
    dim3 gridSize(CEIL_DIV(cols_b, BM2D), CEIL_DIV(rows_a, BN2D));

    gemm_2D_block_tiling_kernel<BN2D, BK2D, BM2D, TM2D, TN2D><<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}
