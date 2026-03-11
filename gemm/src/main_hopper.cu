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

double check_difference(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& test
);


int main() {
    float min_value = 0;
    float max_value = 10;

    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

    for (size_t size : sizes) {
        std::cout << "\n========== Testing size: " << size << "x" << size << " ==========\n";
        std::flush(std::cout);
        double bench_avg, bench_std, bench_gflops;
    
        // Generate matrices
        std::vector<float> a = generate_matrix(size, size, min_value, max_value);
        std::vector<float> b = generate_matrix(size, size, min_value, max_value);

        // Convert to dtype for GPU (no-op if dtype=float)
        std::vector<dtype> a_gpu = float_to_dtype_vec(a);
        std::vector<dtype> b_gpu = float_to_dtype_vec(b);

        // Convert to half for tensor core
        std::vector<half> a_half = float_to_half_vec(a);
        std::vector<half> b_half = float_to_half_vec(b);

        // === GPU computations ===
        // Benchmark
        std::vector<dtype> result_warp_tiling_gpu(size * size, DTYPE_ZERO);
        std::tie(bench_avg, bench_std, bench_gflops) = benchmark("GEMM CUDA Warp Tiling", gemm_warp_tiling, size, result_warp_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), size, size, size, size);
        std::vector<float> result_warp_tiling = dtype_to_float_vec(result_warp_tiling_gpu);
        
        std::vector<float> result_tensor(size * size, 0.0f);
        std::tie(bench_avg, bench_std, bench_gflops) = benchmark("GEMM CUDA Tensor Core", gemm_tensor_hopper, size, result_tensor.data(), a_half.data(), b_half.data(), size, size, size, size);
        check_difference("GEMM CUDA Tensor Core", result_warp_tiling, result_tensor);
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

/**
 * Check differences of two vectors
 */
double check_difference(
    const char* name,
    const std::vector<float>& reference, 
    const std::vector<float>& test
) {
    if (reference.size() != test.size()) {
        throw std::invalid_argument("Vectors must be of the same size for difference check.");
    }

    double sum_diff = 0.0;
    double percentage_diff = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        sum_diff += std::abs(reference[i] - test[i]);
        if (reference[i] != 0) {
            percentage_diff += std::abs((reference[i] - test[i]) / reference[i]) * 100.0;
        }
    }
    double avg_diff = sum_diff / reference.size();
    double avg_percentage_diff = percentage_diff / reference.size();

    std::cout << "  Average difference in " << name << ": " 
        << avg_diff << " (" << avg_percentage_diff << "%)" << std::endl;
    
    return avg_percentage_diff;
}
