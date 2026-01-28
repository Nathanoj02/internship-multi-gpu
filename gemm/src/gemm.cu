#include "gemm.cuh"
#include "../utils/error.cuh"
#include <cuda_runtime.h>

void init_gemm (
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


void cleanup_gemm (
    dtype* d_A, dtype* d_B, dtype* d_C,
    dtype* result, size_t rows_a, size_t cols_b
) {
    size_t size_res = rows_a * cols_b * sizeof(dtype);

    CUDA_CHECK( cudaMemcpy(result, d_C, size_res, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
}


void init_gemm_tensor (
    half** d_A, half** d_B, float** d_C,
    const half* A, const half* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    // Dimension check
    if (cols_a != rows_b) {
        fprintf(stderr, "Error: Matrix dimensions don't match for multiplication. "
                "A is %zux%zu, B is %zux%zu\n", rows_a, cols_a, rows_b, cols_b);
        return;
    }

    size_t size_a = rows_a * cols_a * sizeof(half);
    size_t size_b = rows_b * cols_b * sizeof(half);
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMalloc((void**)d_A, size_a) );
    CUDA_CHECK( cudaMalloc((void**)d_B, size_b) );
    CUDA_CHECK( cudaMalloc((void**)d_C, size_res) );

    CUDA_CHECK( cudaMemcpy(*d_A, A, size_a, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(*d_B, B, size_b, cudaMemcpyHostToDevice) );
}


void cleanup_gemm_tensor (
    half* d_A, half* d_B, float* d_C,
    float* result, size_t rows_a, size_t cols_b
) {
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMemcpy(result, d_C, size_res, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
}
