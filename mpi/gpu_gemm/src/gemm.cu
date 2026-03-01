#include "gemm.cuh"
#include "../../../utils/error.cuh"
#include <cuda_runtime.h>
#include <iostream>

void init_gemm (
    float** d_A, float** d_B, float** d_C,
    const float* A, const float* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    // Dimension check
    if (cols_a != rows_b) {
        fprintf(stderr, "Error: Matrix dimensions don't match for multiplication. "
                "A is %zux%zu, B is %zux%zu\n", rows_a, cols_a, rows_b, cols_b);
        return;
    }

    size_t size_a = rows_a * cols_a * sizeof(float);
    size_t size_b = rows_b * cols_b * sizeof(float);
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMalloc((void**)d_A, size_a) );
    CUDA_CHECK( cudaMalloc((void**)d_B, size_b) );
    CUDA_CHECK( cudaMalloc((void**)d_C, size_res) );

    CUDA_CHECK( cudaMemcpy(*d_A, A, size_a, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(*d_B, B, size_b, cudaMemcpyHostToDevice) );
}


void cleanup_gemm (
    float* d_A, float* d_B, float* d_C,
    float* result, size_t rows_a, size_t cols_b
) {
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMemcpy(result, d_C, size_res, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
}

void assign_gpu_device(int mpi_rank) {
    int num_devices;
    CUDA_CHECK( cudaGetDeviceCount(&num_devices) );
    CUDA_CHECK( cudaSetDevice(mpi_rank % num_devices) );
    std::cout << "MPI Rank " << mpi_rank << " assigned to GPU " << (mpi_rank % num_devices) << std::endl;
}
