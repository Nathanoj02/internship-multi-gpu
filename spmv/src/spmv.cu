#include "spmv.cuh"
#include "error.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 512

// Each thread computes an element of the final vector
// So 1 row of the matrix
__global__
void spmvKernel(float* out, float* arr, size_t* row_offset, size_t* cols, float* values, size_t rows) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;   // = row index
    if (tid >= rows)    return;
    
    size_t elem_offset = row_offset[tid];
    size_t elem_last = row_offset[tid + 1];

    float sum = 0;
    for (size_t i = elem_offset; i < elem_last; i++) {
        sum += values[i] * arr[cols[i]];
    }
    out[tid] = sum;
}

void spmv(
    float* out, float* arr, 
    size_t* row_offset, size_t* cols, float* values, 
    size_t rows, size_t num_values
) {
    float *d_out, *d_arr, *d_values;
    size_t *d_row_offset, *d_cols;

    // Allocate device memory
    CUDA_CHECK( cudaMalloc((void**)&d_out, rows * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&d_arr, rows * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&d_values, num_values * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&d_row_offset, (rows + 1) * sizeof(size_t)) );
    CUDA_CHECK( cudaMalloc((void**)&d_cols, num_values * sizeof(size_t)) );

    // Copy data
    CUDA_CHECK( cudaMemcpy(d_arr, arr, rows * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_values, values, num_values * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_row_offset, row_offset, (rows + 1) * sizeof(size_t), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_cols, cols, num_values * sizeof(size_t), cudaMemcpyHostToDevice) );

    // Compute blocks
    int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    spmvKernel<<<blocks, THREADS_PER_BLOCK>>>(d_out, d_arr, d_row_offset, d_cols, d_values, rows);
    
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy result back to host
    CUDA_CHECK( cudaMemcpy(out, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free memory
    CUDA_CHECK( cudaFree(d_out) );
    CUDA_CHECK( cudaFree(d_arr) );
    CUDA_CHECK( cudaFree(d_values) );
    CUDA_CHECK( cudaFree(d_row_offset) );
    CUDA_CHECK( cudaFree(d_cols) );
}
