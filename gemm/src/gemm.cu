#include "gemm.cuh"
#include "../utils/error.cuh"
#include <cuda_runtime.h>

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
void gemm_kernel(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows_a || col >= cols_b) {
        return;
    }

    float value = 0.0f;
    for (int k = 0; k < cols_a; ++k) {
        value += A[row * cols_a + k] * B[k * cols_b + col];
    }
    C[row * cols_b + col] = value;
}


void gemm(
    float* result, const float* A, const float* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    // Dimension check
    if (cols_a != rows_b) {
        fprintf(stderr, "Error: Matrix dimensions don't match for multiplication. "
                "A is %zux%zu, B is %zux%zu\n", rows_a, cols_a, rows_b, cols_b);
        return;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    size_t size_a = rows_a * cols_a * sizeof(float);
    size_t size_b = rows_b * cols_b * sizeof(float);
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMalloc((void**)&d_A, size_a) );
    CUDA_CHECK( cudaMalloc((void**)&d_B, size_b) );
    CUDA_CHECK( cudaMalloc((void**)&d_C, size_res) );

    CUDA_CHECK( cudaMemcpy(d_A, A, size_a, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_B, B, size_b, cudaMemcpyHostToDevice) );

    dim3 blockSize(16, 16);
    dim3 gridSize((cols_b + blockSize.x - 1) / blockSize.x,
                  (rows_a + blockSize.y - 1) / blockSize.y);

    gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );

    CUDA_CHECK( cudaDeviceSynchronize() );

    CUDA_CHECK( cudaMemcpy(result, d_C, size_res, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
}