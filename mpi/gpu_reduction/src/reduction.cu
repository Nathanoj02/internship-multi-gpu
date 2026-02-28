#include "reduction.cuh"
#include "../../../utils/error.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <mpi.h>

#define THREADS_PER_BLOCK 256

template <unsigned int blockSize>
__device__
void warpReduce(volatile int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

/**
 * Kernel function to perform reduction on an array
 * Each block computes a partial sum and writes it to the output array
 * @param v Input array
 * @param out Output array for partial sums
 * @param n Number of elements in the input array
 */
template <unsigned int blockSize>
__global__
void reduceKernel(int* v, int *out, size_t n) {
    extern __shared__ int sdata[];

    // Perform first level of reduction
    // Reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    int tmp = 0;
    // Load up to two elements per thread, guarding bounds correctly
    if (i < n) tmp += v[i];
    if (i + blockDim.x < n) tmp += v[i + blockDim.x];
    sdata[tid] = tmp;

    __syncthreads();

    // Completely unrolled reduction
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }

    // Write result for this block to global memory
    if (tid == 0) out[blockIdx.x] = sdata[0];
}


int reduce(const int* v, int elems) {
    int *d_v;
    int *d_out;

    // Allocate device memory for input
    CUDA_CHECK( cudaMalloc((void**)&d_v, elems * sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(d_v, v, elems * sizeof(int), cudaMemcpyHostToDevice) );

    // Compute blocks and allocate output
    int blocks = (elems + (THREADS_PER_BLOCK * 2 - 1)) / (THREADS_PER_BLOCK * 2);
    CUDA_CHECK( cudaMalloc((void**)&d_out, blocks * sizeof(int)) );

    // Launch kernel
    size_t sharedMemSize = THREADS_PER_BLOCK * sizeof(int);
    reduceKernel<THREADS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(d_v, d_out, elems);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy result from device to host
    int *h_partial = (int *) malloc(blocks * sizeof(int));
    CUDA_CHECK( cudaMemcpy(h_partial, d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost) );

    int result = 0;
    for (int i = 0; i < blocks; ++i) {
        result += h_partial[i];
    }

    // Free memory
    free(h_partial);
    CUDA_CHECK( cudaFree(d_v) );
    CUDA_CHECK( cudaFree(d_out) );

    return result;
}

int reduce_multi_cpu_mediated(const int* v, int elems) {
    int deviceCount;
    CUDA_CHECK( cudaGetDeviceCount(&deviceCount) );

    int result = 0;

    // Split work between GPUs
    int elems_per_gpu = elems / deviceCount;
    int remaining_elems = elems % deviceCount;

    int **d_v = new int*[deviceCount];
    int **d_out = new int*[deviceCount];

    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK( cudaSetDevice(gpu) );

        // Calculate elements and offset for this GPU
        int current_elems = elems_per_gpu + (gpu == 0 ? remaining_elems : 0);
        int offset = gpu * elems_per_gpu + (gpu == 0 ? 0 : remaining_elems);
        
        // Allocate device memory for input
        CUDA_CHECK( cudaMalloc((void**)&d_v[gpu], current_elems * sizeof(int)) );
        CUDA_CHECK( cudaMemcpy(d_v[gpu], v + offset, current_elems * sizeof(int), cudaMemcpyHostToDevice) );
        
        // Compute blocks and allocate output
        int blocks = (current_elems + (THREADS_PER_BLOCK * 2 - 1)) / (THREADS_PER_BLOCK * 2);
        CUDA_CHECK( cudaMalloc((void**)&d_out[gpu], blocks * sizeof(int)) );
        
        // Launch kernel
        size_t sharedMemSize = THREADS_PER_BLOCK * sizeof(int);
        reduceKernel<THREADS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(d_v[gpu], d_out[gpu], current_elems);
        CUDA_CHECK( cudaGetLastError() );
    }

    // Synchronize and copy results back to host
    // Also perform final reduction on host
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK( cudaSetDevice(gpu) );
        CUDA_CHECK( cudaDeviceSynchronize() );

        int current_elems = elems_per_gpu + (gpu == 0 ? remaining_elems : 0);
        
        // Copy result from device to host
        int blocks = (current_elems + (THREADS_PER_BLOCK * 2 - 1)) / (THREADS_PER_BLOCK * 2);
        int *h_partial = (int *) malloc(blocks * sizeof(int));
        CUDA_CHECK( cudaMemcpy(h_partial, d_out[gpu], blocks * sizeof(int), cudaMemcpyDeviceToHost) );
        
        for (int i = 0; i < blocks; ++i) {
            result += h_partial[i];
        }
        
        free(h_partial);

        // Free device memory
        CUDA_CHECK( cudaSetDevice(gpu) );
        CUDA_CHECK( cudaFree(d_v[gpu]) );
        CUDA_CHECK( cudaFree(d_out[gpu]) );
    }

    // Cleanup host arrays
    delete[] d_v;
    delete[] d_out;

    return result;
}

