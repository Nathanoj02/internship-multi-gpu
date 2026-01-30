#include <cuda_runtime.h>
#include "kernels/6_kernel_warp_tiling.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <
    size_t BNWARP, size_t BKWARP, size_t BMWARP,
    size_t WM, size_t WN,
    size_t WMITER, size_t WNITER,
    size_t TMWARPS, size_t TNWARPS,
    size_t WSUBM, size_t WSUBN,
    size_t WARPSIZE
>
/**
 * CUDA kernel to perform matrix multiplication using warp tiling
 * Each warp computes a WN x WM tile, with each thread computing multiple elements
 * @tparam BNWARP Block tile size for rows of matrix A
 * @tparam BKWARP Block tile size for columns of matrix A / rows of matrix B
 * @tparam BMWARP Block tile size for columns of matrix B
 * @tparam WM Warp tile size for columns (M dimension)
 * @tparam WN Warp tile size for rows (N dimension)
 * @tparam WMITER Number of iterations in M dimension within warp tile
 * @tparam WNITER Number of iterations in N dimension within warp tile
 * @tparam TMWARPS Number of elements per thread in M dimension per iteration
 * @tparam TNWARPS Number of elements per thread in N dimension per iteration
 * @tparam WSUBM Number of threads in M dimension within a warp
 * @tparam WSUBN Number of threads in N dimension within a warp
 * @tparam WARPSIZE Number of threads in a warp
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A (N)
 * @param cols_a Number of columns in matrix A (K)
 * @param cols_b Number of columns in matrix B (M)
 */
__global__ void gemm_warp_tiling_kernel (
    const dtype* A, const dtype* B, dtype* C,
    int rows_a, int cols_a, int cols_b
) {
    int cRow = blockIdx.y, cCol = blockIdx.x;
    
    // Indices for loading into shared memory
    int innerRowA = threadIdx.x / BKWARP;
    int innerColA = threadIdx.x % BKWARP;
    int innerRowB = threadIdx.x / BMWARP;
    int innerColB = threadIdx.x % BMWARP;
    
    // Warp identification
    int warpNum = threadIdx.x / WARPSIZE;
    int warpRow = warpNum / (BMWARP / WM);
    int warpCol = warpNum % (BMWARP / WM);

    // Thread position within warp
    int warpId = threadIdx.x % WARPSIZE;
    int threadRowInWarp = warpId / WSUBM;
    int threadColInWarp = warpId % WSUBM;
    
    __shared__ dtype As[BNWARP * BKWARP];
    __shared__ dtype Bs[BKWARP * BMWARP];
    
    // Advance pointers to starting position for this block
    A += cRow * BNWARP * cols_a;
    B += cCol * BMWARP;
    C += cRow * BNWARP * cols_b + cCol * BMWARP;
    
    // Each thread accumulates (WNITER * TNWARPS) x (WMITER * TMWARPS) results
    dtype threadResults[(WNITER * TNWARPS) * (WMITER * TMWARPS)];
    dtype regM[WMITER * TMWARPS];
    dtype regN[WNITER * TNWARPS];
    for (int i = 0; i < (WNITER * TNWARPS) * (WMITER * TMWARPS); ++i) threadResults[i] = DTYPE_ZERO;
    for (int i = 0; i < WMITER * TMWARPS; ++i) regM[i] = DTYPE_ZERO;
    for (int i = 0; i < WNITER * TNWARPS; ++i) regN[i] = DTYPE_ZERO;
    
    // Loop over tiles
    for (int bkIdx = 0; bkIdx < cols_a; bkIdx += BKWARP) {
        // Load tiles into shared memory
        As[innerRowA * BKWARP + innerColA] = A[innerRowA * cols_a + innerColA];
        Bs[innerRowB * BMWARP + innerColB] = B[innerRowB * cols_b + innerColB];
        
        __syncthreads();
        
        // Advance pointers
        A += BKWARP;
        B += BKWARP * cols_b;
        
        // Compute using warp-level tiling
        for (int dotIdx = 0; dotIdx < BKWARP; ++dotIdx) {
            // Load from A into registers
            for (int wSubRowIdx = 0; wSubRowIdx < WNITER; ++wSubRowIdx) {
                for (int i = 0; i < TNWARPS; ++i) {
                    regN[wSubRowIdx * TNWARPS + i] = 
                        As[(warpRow * WN + wSubRowIdx * (WSUBN * TNWARPS) + 
                            threadRowInWarp * TNWARPS + i) * BKWARP + dotIdx];
                }
            }
            
            // Load from B into registers
            for (int wSubColIdx = 0; wSubColIdx < WMITER; ++wSubColIdx) {
                for (int i = 0; i < TMWARPS; ++i) {
                    regM[wSubColIdx * TMWARPS + i] = 
                        Bs[dotIdx * BMWARP + warpCol * WM + wSubColIdx * (WSUBM * TMWARPS) + 
                            threadColInWarp * TMWARPS + i];
                }
            }
            
            // Outer product computation
            for (int wSubRowIdx = 0; wSubRowIdx < WNITER; ++wSubRowIdx) {
                for (int wSubColIdx = 0; wSubColIdx < WMITER; ++wSubColIdx) {
                    for (int resIdxM = 0; resIdxM < TMWARPS; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < TNWARPS; ++resIdxN) {
                            threadResults[(wSubRowIdx * TNWARPS + resIdxN) * (WMITER * TMWARPS) + 
                                         (wSubColIdx * TMWARPS) + resIdxM] += 
                                regM[wSubColIdx * TNWARPS + resIdxM] * regN[wSubRowIdx * TMWARPS + resIdxN];
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    for (int wSubRowIdx = 0; wSubRowIdx < WNITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WMITER; ++wSubColIdx) {
            for (int resIdxN = 0; resIdxN < TNWARPS; ++resIdxN) {
                for (int resIdxM = 0; resIdxM < TMWARPS; ++resIdxM) {
                    int result_row = wSubRowIdx * TNWARPS + resIdxN;
                    int result_col = wSubColIdx * TMWARPS + resIdxM;
                    int result_index = result_row * (WMITER * TMWARPS) + result_col;
                    dtype tmp = threadResults[result_index];
                    
                    int C_row = warpRow * WN + wSubRowIdx * (WSUBN * TNWARPS) + 
                                threadRowInWarp * TNWARPS + resIdxN;
                    int C_col = warpCol * WM + wSubColIdx * (WSUBM * TMWARPS) + 
                                threadColInWarp * TMWARPS + resIdxM;
                    
                    C[C_row * cols_b + C_col] = tmp;
                }
            }
        }
    }
}


void gemm_warp_tiling (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    #ifdef CC80 // A30
    constexpr size_t BNWARP = 64;
    constexpr size_t BKWARP = 8;
    constexpr size_t BMWARP = 64;
    constexpr size_t WM = 8;
    constexpr size_t WN = 32;
    constexpr size_t WMITER = 2;
    constexpr size_t WNITER = 1;
    constexpr size_t TMWARPS = 2;
    constexpr size_t TNWARPS = 2;
    constexpr size_t WSUBM = 2;
    constexpr size_t WSUBN = 16;
    constexpr size_t WARPSIZE = 32;
    #else
    constexpr size_t BNWARP = 64;
    constexpr size_t BKWARP = 8;
    constexpr size_t BMWARP = 64;
    constexpr size_t WM = 8;
    constexpr size_t WN = 32;
    constexpr size_t WMITER = 2;
    constexpr size_t WNITER = 1;
    constexpr size_t TMWARPS = 2;
    constexpr size_t TNWARPS = 2;
    constexpr size_t WSUBM = 2;
    constexpr size_t WSUBN = 16;
    constexpr size_t WARPSIZE = 32;
    #endif


    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BNWARP * BKWARP);
    dim3 gridSize(CEIL_DIV(cols_b, BMWARP), CEIL_DIV(rows_a, BNWARP));

    gemm_warp_tiling_kernel<
        BNWARP, BKWARP, BMWARP, WM, WN, WMITER, WNITER, TMWARPS, TNWARPS, WSUBM, WSUBN, WARPSIZE
        ><<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}
