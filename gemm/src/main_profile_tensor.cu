#include "gemm.hpp"
#include "gemm.cuh"
#include "dtype.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <utility>  // for std::forward
#include <fstream>
#include <tuple>

#define WARMUP_RUNS 3
#define TIMED_RUNS 10

// Functions declarations
template <typename Func, typename... Args>
std::tuple<double, double, double> benchmark(const char* name, Func&& func, size_t size, Args&&... args);

int main() {
    float min_value = 0;
    float max_value = 10;

    std::vector<size_t> sizes = {2048}; //{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    for (size_t size : sizes) {
        std::cout << "\n========== Testing size: " << size << "x" << size << " ==========\n";
        std::flush(std::cout);

        double bench_avg, bench_std, bench_gflops;
        size_t rows = size;
        size_t cols = size;
    
        // Generate matrices
        std::vector<float> a = generate_matrix(rows, cols, min_value, max_value);
        std::vector<float> b = generate_matrix(rows, cols, min_value, max_value);

        // Convert to half for tensor core
        std::vector<half> a_half = float_to_half_vec(a);
        std::vector<half> b_half = float_to_half_vec(b);

        std::vector<float> result_tensor_gpu(rows * cols, 0.0f);
        std::tie(bench_avg, bench_std, bench_gflops) = benchmark("GEMM CUDA Tensor Core", gemm_tensor_double_buffering, size, result_tensor_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
    }

    return 0;
}

/**
 * Benchmark wrapper definition
 */
template <typename Func, typename... Args>
std::tuple<double, double, double> benchmark(const char* name, Func&& func, size_t size, Args&&... args) {
    using Clock = std::chrono::high_resolution_clock;
    
    // Store individual execution times
    std::vector<double> times;
    times.reserve(TIMED_RUNS);

    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; i++) {
        auto start = Clock::now();
        
        func(std::forward<Args>(args)...); 
        
        auto end = Clock::now();

        // Only record timed runs
        if (i >= 0) {
            std::chrono::duration<double> diff = end - start;
            times.push_back(diff.count());
        }
    }

    // Calculate Mean
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double mean = sum / times.size();

    // Calculate Variance
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double variance = sq_sum / times.size();
    
    // Calculate standard deviation
    double std_dev = std::sqrt(variance);

    std::cout << "\nBenchmark [" << name << "]:\n" 
              << "  Avg: " << mean << " seconds\n"
              << "  Std: " << std_dev << " seconds\n";

    double flops = 2 * static_cast<double>(size) * static_cast<double>(size) * static_cast<double>(size) / mean;
    double gflops = flops / 1e9;
    
    return {mean, std_dev, gflops};
}
