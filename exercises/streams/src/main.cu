#include "array_ops.hpp"
#include "array_sum.cuh"
#include <time.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define WARMUP_RUNS 3
#define TIMED_RUNS 5

int main(int argc, char** argv) {
    int elems = 1e8;
    int min_value = 0;
    int max_value = 9;

    clock_t start, end;

    int *a, *b, *out;

    // Allocate pinned memory for host capies of a, b, and out
    cudaMallocHost((void**)&a, elems * sizeof(int), cudaHostAllocDefault);
    cudaMallocHost((void**)&b, elems * sizeof(int), cudaHostAllocDefault);
    cudaMallocHost((void**)&out, elems * sizeof(int), cudaHostAllocDefault);

    // Fill input arrays with random values
    fill_array(a, elems, min_value, max_value);
    fill_array(b, elems, min_value, max_value);

    // CPU sum
    double cpu_time = 0.0;
    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        start = clock();
        array_sum(out, a, b, elems);
        end = clock();

        if (i >= 0)
            cpu_time += double(end - start) / CLOCKS_PER_SEC;
    }

    cpu_time /= TIMED_RUNS;
    std::cout << "CPU Sum Time: " << cpu_time << " seconds" << std::endl;


    // Naive GPU sum
    int *out_gpu;
    cudaMallocHost((void**)&out_gpu, elems * sizeof(int), cudaHostAllocDefault);
    double gpu_time = 0.0;
    
    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        clock_t start = clock();
        array_sum_naive(out_gpu, a, b, elems);
        clock_t end = clock();

        if (i >= 0)
            gpu_time += double(end - start) / CLOCKS_PER_SEC;
    }

    gpu_time /= TIMED_RUNS;
    std::cout << "Naive GPU Sum Time: " << gpu_time << " seconds" << std::endl;


    // Verify results
    for (int i = 0; i < elems; ++i) {
        if (out[i] != out_gpu[i]) {
            std::cout << "Mismatch at index " << i << ": CPU = " << out[i]
                      << ", GPU = " << out_gpu[i] << std::endl;
            return -1;
        }
    }
    

    // Stream based GPU sum
    int *out_streams;
    cudaMallocHost((void**)&out_streams, elems * sizeof(int), cudaHostAllocDefault);
    double stream_time = 0.0;

    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        clock_t start = clock();
        array_sum_streams(out_streams, a, b, elems);
        clock_t end = clock();

        if (i >= 0)
            stream_time += double(end - start) / CLOCKS_PER_SEC;
    }
    stream_time /= TIMED_RUNS;
    std::cout << "Stream GPU Sum Time: " << stream_time << " seconds" << std::endl;

    // Verify results
    for (int i = 0; i < elems; ++i) {
        if (out[i] != out_streams[i]) {
            std::cout << "Mismatch at index " << i << ": CPU = " << out[i]
                      << ", Stream GPU = " << out_streams[i] << std::endl;
            return -1;
        }
    }
    
    std::cout << "Results match!" << std::endl;

    // Free pinned memory
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(out);
    cudaFreeHost(out_gpu);
    cudaFreeHost(out_streams);

    return 0;
}