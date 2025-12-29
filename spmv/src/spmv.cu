#include "spmv.cuh"
#include "error.cuh"
#include "load_balancing.hpp"

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <mpi.h>

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


void spmv_streams(
    float* out, float* arr, 
    size_t* row_offset, size_t* cols, float* values, 
    size_t rows, size_t num_values, const size_t stream_num, size_t *row_mapping
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
    
    // Create streams
    cudaStream_t streams[stream_num];
    for (size_t i = 0; i < stream_num; i++) {
        CUDA_CHECK( cudaStreamCreate(&streams[i]) );
    }

    // Loop for each stream
    for (size_t i = 0; i < stream_num; i++) {
        size_t row_start = row_mapping[i];
        size_t row_end = row_mapping[i + 1];
        size_t row_count = row_end - row_start;
        size_t value_start = row_offset[row_start];
        size_t value_end = row_offset[row_end];
        size_t value_count = value_end - value_start;

        // Copy segment data to device
        CUDA_CHECK( cudaMemcpyAsync(d_values + value_start, values + value_start, value_count * sizeof(float), cudaMemcpyHostToDevice, streams[i]) );
        CUDA_CHECK( cudaMemcpyAsync(d_row_offset + row_start, row_offset + row_start, (row_count + 1) * sizeof(size_t), cudaMemcpyHostToDevice, streams[i]) );
        CUDA_CHECK( cudaMemcpyAsync(d_cols + value_start, cols + value_start, value_count * sizeof(size_t), cudaMemcpyHostToDevice, streams[i]) );

        // Compute blocks for this segment
        int blocks = (row_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Launch kernel for this segment
        spmvKernel<<<blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
            d_out + row_start, d_arr, 
            d_row_offset + row_start, d_cols, 
            d_values, 
            row_count
        );

        CUDA_CHECK( cudaGetLastError() );

        // Copy result back to host for this segment
        CUDA_CHECK( cudaMemcpyAsync(out + row_start, d_out + row_start, row_count * sizeof(float), cudaMemcpyDeviceToHost, streams[i]) );
    }
    
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Destroy streams
    for (size_t i = 0; i < stream_num; i++) {
        CUDA_CHECK( cudaStreamDestroy(streams[i]) );
    }

    // Free memory
    CUDA_CHECK( cudaFree(d_out) );
    CUDA_CHECK( cudaFree(d_arr) );
    CUDA_CHECK( cudaFree(d_values) );
    CUDA_CHECK( cudaFree(d_row_offset) );
    CUDA_CHECK( cudaFree(d_cols) );
}

// TODO: needs Allgather on array
// TODO: check kernel correctness
// TODO: check MPI -> can we run CUDA directives with MPI? (last time was 8 kB max)
void spmv_multi_horizontal(
    float* out, float* arr, 
    size_t* row_offset, size_t* cols, float* values, 
    size_t rows, size_t num_values, const size_t* row_mapping
) {
    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get device count
    int device_count;
    CUDA_CHECK( cudaGetDeviceCount(&device_count) );
    int gpu = rank % device_count;
    CUDA_CHECK( cudaSetDevice(gpu) );

    float *d_out, *d_arr, *d_values;
    size_t *d_row_offset, *d_cols;

    // Data partitioning
    size_t row_start = row_mapping[rank];
    size_t row_end = row_mapping[rank + 1];
    size_t row_count = row_end - row_start;
    size_t value_start = row_offset[row_start];
    size_t value_end = row_offset[row_end];
    size_t value_count = value_end - value_start;

    // Allocate device memory
    CUDA_CHECK( cudaMalloc((void**)&d_out, row_count * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&d_arr, row_count * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&d_values, value_count * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&d_row_offset, (row_count + 1) * sizeof(size_t)) );
    CUDA_CHECK( cudaMalloc((void**)&d_cols, value_count * sizeof(size_t)) );

    // Copy data
    CUDA_CHECK( cudaMemcpy(d_arr, arr + row_start, row_count * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_values, values + value_start, value_count * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_row_offset, row_offset + row_start, (row_count + 1) * sizeof(size_t), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_cols, cols + value_start, value_count * sizeof(size_t), cudaMemcpyHostToDevice) );

    // Compute blocks
    int blocks = (row_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    spmvKernel<<<blocks, THREADS_PER_BLOCK>>>(d_out, d_arr, d_row_offset, d_cols, d_values, row_count);
    
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy result back to host
    CUDA_CHECK( cudaMemcpy(out + row_start, d_out, row_count * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free memory
    CUDA_CHECK( cudaFree(d_out) );
    CUDA_CHECK( cudaFree(d_arr) );
    CUDA_CHECK( cudaFree(d_values) );
    CUDA_CHECK( cudaFree(d_row_offset) );
    CUDA_CHECK( cudaFree(d_cols) );
}
