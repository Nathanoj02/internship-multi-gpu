#include "gemm.cuh"
#include "../utils/error.cuh"
#include "definitions.hpp"
#include <cuda_runtime.h>

#include "kernels/1_kernel_naive.cuh"
#include "kernels/2_kernel_memory_coalescing.cuh"
#include "kernels/3_kernel_shared_memory.cuh"
#include "kernels/4_kernel_block_tiling.cuh"
#include "kernels/5_kernel_2D_block_tiling.cuh"
#include "kernels/6_kernel_warp_tiling.cuh"


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