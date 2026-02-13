#include "array_sum.cuh"
#include "../../../utils/error.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define STREAM_NUM 4

__global__ void array_sum_kernel(int* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}


void array_sum_naive(int* out, const int* a, const int* b, size_t elems) {
    int *d_a, *d_b, *d_out;

    CUDA_CHECK( cudaMalloc(&d_a, elems * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_b, elems * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_out, elems * sizeof(int)) );

    CUDA_CHECK( cudaMemcpy(d_a, a, elems * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_b, b, elems * sizeof(int), cudaMemcpyHostToDevice) );

    size_t threads_per_block = 256;
    size_t blocks = (elems + threads_per_block - 1) / threads_per_block;

    array_sum_kernel<<<blocks, threads_per_block>>>(d_out, d_a, d_b, elems);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    CUDA_CHECK( cudaMemcpy(out, d_out, elems * sizeof(int), cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_a) );
    CUDA_CHECK( cudaFree(d_b) );
    CUDA_CHECK( cudaFree(d_out) );
}


void array_sum_streams(int* out, const int* a, const int* b, size_t elems) {
    int *d_a, *d_b, *d_out;
    size_t threads_per_block = 256;

    CUDA_CHECK( cudaMalloc(&d_a, elems * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_b, elems * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_out, elems * sizeof(int)) );

    // Create streams
    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++)
        CUDA_CHECK( cudaStreamCreate(&stream[i]) );
    
    const int elems_per_stream = elems / STREAM_NUM;
    const int remaining_elems = elems % STREAM_NUM;   // Last stream gets more elements

    // Loop for each Stream
    for (int i = 0; i < STREAM_NUM; i++) {
        const size_t offset = i * elems_per_stream;
        const int this_elems = elems_per_stream + (i == STREAM_NUM - 1 ? remaining_elems : 0);

        CUDA_CHECK( cudaMemcpyAsync(
            d_a + offset,
            a + offset, 
            this_elems* sizeof(int), 
            cudaMemcpyHostToDevice,
            stream[i])
        );

        CUDA_CHECK( cudaMemcpyAsync(
            d_b + offset, 
            b + offset, 
            this_elems* sizeof(int), 
            cudaMemcpyHostToDevice,
            stream[i])
        );

        const size_t blocks = (this_elems + threads_per_block - 1) / threads_per_block;
        array_sum_kernel<<<blocks, threads_per_block, 0, stream[i]>>>(
            d_out + offset, 
            d_a + offset,
            d_b + offset,
            this_elems
        );
        CUDA_CHECK( cudaGetLastError() );

        CUDA_CHECK( cudaMemcpyAsync(
            out + offset, 
            d_out + offset, 
            this_elems * sizeof(int), 
            cudaMemcpyDeviceToHost,
            stream[i])
        );
    }

    CUDA_CHECK( cudaDeviceSynchronize() );

    // Destroy streams
    for (int i = 0; i < STREAM_NUM; i++)
        CUDA_CHECK( cudaStreamDestroy(stream[i]) );

    CUDA_CHECK( cudaFree(d_a) );
    CUDA_CHECK( cudaFree(d_b) );
    CUDA_CHECK( cudaFree(d_out) );
}