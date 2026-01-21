#include "gemm.cuh"
#include "dtype.hpp"
#include "../utils/error.cuh"
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_SIZE 32

#define BM 64
#define BN 64
#define BK 8
#define TM 8

#define BN2D 64
#define BK2D 8
#define BM2D 64
#define TM2D 4
#define TN2D 4

#define BNWARP 64
#define BKWARP 8
#define BMWARP 64
#define TMWARPS 2
#define TNWARPS 2

#define WARPNUM 16
#define WARPSN 2
#define WARPSM 8
#define WN 32
#define WM 8

#define WARPSIZE 32
#define WSUBN 16
#define WSUBM 2
#define WNITER 1
#define WMITER 2

// -- Function declarations --
void init_gemm(
    dtype** d_A, dtype** d_B, dtype** d_C,
    const dtype* A, const dtype* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

void cleanup_gemm(
    dtype* d_A, dtype* d_B, dtype* d_C,
    dtype* result, size_t rows_a, size_t cols_b
);

// -- CUDA Kernels --
/**
 * CUDA kernel to perform matrix multiplication
 * Each thread computes one element of the result matrix C
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_naive_kernel(const dtype* A, const dtype* B, dtype* C, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows_a || col >= cols_b) {
        return;
    }

    dtype value = DTYPE_ZERO;
    for (int k = 0; k < cols_a; ++k) {
        value += A[row * cols_a + k] * B[k * cols_b + col];
    }
    C[row * cols_b + col] = value;
}

/**
 * CUDA kernel to perform matrix multiplication with memory coalescing
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_memory_coalescing_kernel(const dtype* A, const dtype* B, dtype* C, int rows_a, int cols_a, int cols_b) {
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

/**
 * CUDA kernel to perform matrix multiplication using shared memory
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_shared_memory_kernel(const dtype* A, const dtype* B, dtype* C, int rows_a, int cols_a, int cols_b) {
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

/**
 * CUDA kernel to perform matrix multiplication using block tiling
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_block_tiling_kernel(const dtype* A, const dtype* B, dtype* C, int rows_a, int cols_a, int cols_b) {
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

/**
 * CUDA kernel to perform matrix multiplication using 2D block tiling
 * Each thread computes TM2D x TN2D elements of the result matrix C
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A (N)
 * @param cols_a Number of columns in matrix A (K)
 * @param cols_b Number of columns in matrix B (M)
 */
__global__
void gemm_2D_block_tiling_kernel(const dtype* A, const dtype* B, dtype* C, int rows_a, int cols_a, int cols_b) {
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

/**
 * CUDA kernel to perform matrix multiplication using warp tiling
 * Each warp computes a WN x WM tile, with each thread computing multiple elements
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A (N)
 * @param cols_a Number of columns in matrix A (K)
 * @param cols_b Number of columns in matrix B (M)
 */
__global__
void gemm_warp_tiling_kernel(const dtype* A, const dtype* B, dtype* C, int rows_a, int cols_a, int cols_b) {
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


// -- Host Functions --
void gemm_naive(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    // Initialize and copy data to device
    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_a, blockSize.x), CEIL_DIV(cols_b, blockSize.y));

    gemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy result back and cleanup
    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_memory_coalescing(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(cols_b, BLOCK_SIZE), CEIL_DIV(rows_a, BLOCK_SIZE));

    gemm_memory_coalescing_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_shared_memory(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(cols_b, BLOCK_SIZE), CEIL_DIV(rows_a, BLOCK_SIZE));

    gemm_shared_memory_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_block_tiling(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Number of threads per block = (BN * BM) / TM
    dim3 blockSize((BN * BM) / TM, 1, 1);
    dim3 gridSize(CEIL_DIV(cols_b, BM), CEIL_DIV(rows_a, BN));

    gemm_block_tiling_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_2D_block_tiling(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize((BN2D * BM2D) / (TM2D * TN2D));
    dim3 gridSize(CEIL_DIV(cols_b, BM2D), CEIL_DIV(rows_a, BN2D));

    gemm_2D_block_tiling_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_warp_tiling(
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BNWARP * BKWARP);
    dim3 gridSize(CEIL_DIV(cols_b, BMWARP), CEIL_DIV(rows_a, BNWARP));

    gemm_warp_tiling_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}

/**
 * Initialize device memory and copy input matrices
 */
void init_gemm(
    dtype** d_A, dtype** d_B, dtype** d_C,
    const dtype* A, const dtype* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    // Dimension check
    if (cols_a != rows_b) {
        fprintf(stderr, "Error: Matrix dimensions don't match for multiplication. "
                "A is %zux%zu, B is %zux%zu\n", rows_a, cols_a, rows_b, cols_b);
        return;
    }

    size_t size_a = rows_a * cols_a * sizeof(dtype);
    size_t size_b = rows_b * cols_b * sizeof(dtype);
    size_t size_res = rows_a * cols_b * sizeof(dtype);

    CUDA_CHECK( cudaMalloc((void**)d_A, size_a) );
    CUDA_CHECK( cudaMalloc((void**)d_B, size_b) );
    CUDA_CHECK( cudaMalloc((void**)d_C, size_res) );

    CUDA_CHECK( cudaMemcpy(*d_A, A, size_a, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(*d_B, B, size_b, cudaMemcpyHostToDevice) );
}

/**
 * Copy result back to host and free device memory
 */
void cleanup_gemm(
    dtype* d_A, dtype* d_B, dtype* d_C,
    dtype* result, size_t rows_a, size_t cols_b
) {
    size_t size_res = rows_a * cols_b * sizeof(dtype);

    CUDA_CHECK( cudaMemcpy(result, d_C, size_res, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
}