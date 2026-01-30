#include "gemm.hpp"
#include "gemm.cuh"
#include "dtype.cuh"
#include <iostream>
#include <vector>
#include <chrono>

#define WARMUP_RUNS 3
#define TIMED_RUNS 10

int main() {
    int rows = 1024;
    int cols = 1024;
    float min_value = 0;
    float max_value = 10;

    // Generate matrices
    std::vector<float> a = generate_matrix(rows, cols, min_value, max_value);
    std::vector<float> b = generate_matrix(rows, cols, min_value, max_value);

    // Convert to dtype for GPU
    std::vector<dtype> a_gpu = float_to_dtype_vec(a);
    std::vector<dtype> b_gpu = float_to_dtype_vec(b);
    std::vector<dtype> result_gpu(rows * cols, DTYPE_ZERO);

    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        gemm_warp_tiling(result_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    }

    // Timed runs
    using Clock = std::chrono::high_resolution_clock;
    std::vector<double> times;
    times.reserve(TIMED_RUNS);

    for (int i = 0; i < TIMED_RUNS; ++i) {
        auto start = Clock::now();
        gemm_warp_tiling(result_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        auto end = Clock::now();
        std::chrono::duration<double> diff = end - start;
        times.push_back(diff.count());
    }

    // Calculate mean
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean = sum / times.size();

    // Calculate GFLOPS: 2 * N^3 operations for NxN matrix multiply
    double flops = 2.0 * rows * cols * cols;
    double gflops = (flops / mean) / 1e9;

    std::cout << "WARP_TILING: " << mean * 1000 << " ms, " << gflops << " GFLOPS" << std::endl;

    return 0;
}
