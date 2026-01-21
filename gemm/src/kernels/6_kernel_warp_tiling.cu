#include "kernels/6_kernel_warp_tiling.cuh"
#include "definitions.hpp"
#include <cuda_runtime.h>

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
