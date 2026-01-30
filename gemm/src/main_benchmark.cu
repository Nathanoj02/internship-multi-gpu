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
std::pair<double, double> benchmark(const char* name, Func&& func, Args&&... args);

double check_difference(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& test
);


int main() {
    float min_value = 0;
    float max_value = 10;

    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

    for (size_t size : sizes) {
        std::cout << "\n========== Testing size: " << size << "x" << size << " ==========\n";
        
        // Open CSV file for this size
        std::string filename = "results/benchmark_results_" + std::to_string(size) + ".csv";
        std::ofstream csv_file(filename);
        csv_file << "Kernel,Avg_Time(s),Std_Dev(s),Avg_Error(%)\n";
        double bench_avg, bench_std, bench_error;
        size_t rows = size;
        size_t cols = size;
    
        // Generate matrices
        std::vector<float> a = generate_matrix(rows, cols, min_value, max_value);
        std::vector<float> b = generate_matrix(rows, cols, min_value, max_value);

        // CPU computation
        std::vector<float> result(rows * cols, 0.0f);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CPU", gemm_cpu, result, a, b, rows, cols, rows, cols);
        csv_file << "GEMM CPU," << bench_avg << "," << bench_std << ",0.0\n";

        // Convert to dtype for GPU (no-op if dtype=float)
        std::vector<dtype> a_gpu = float_to_dtype_vec(a);
        std::vector<dtype> b_gpu = float_to_dtype_vec(b);

        // Convert to half for tensor core
        std::vector<half> a_half = float_to_half_vec(a);
        std::vector<half> b_half = float_to_half_vec(b);

        // === GPU computations ===
        // Benchmark
        std::vector<dtype> result_naive_gpu(rows * cols, DTYPE_ZERO);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA", gemm_naive, result_naive_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        
        // Convert GPU result back to float for comparison
        std::vector<float> result_naive = dtype_to_float_vec(result_naive_gpu);
        
        // Check difference
        bench_error = check_difference("GEMM CUDA", result, result_naive);
        csv_file << "GEMM CUDA," << bench_avg << "," << bench_std << "," << bench_error << "\n";


        std::vector<dtype> result_coalescing_gpu(rows * cols, DTYPE_ZERO);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Coalescing", gemm_memory_coalescing, result_coalescing_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        std::vector<float> result_coalescing = dtype_to_float_vec(result_coalescing_gpu);
        bench_error = check_difference("GEMM CUDA Coalescing", result, result_coalescing);
        csv_file << "GEMM CUDA Coalescing," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<dtype> result_shared_gpu(rows * cols, DTYPE_ZERO);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Shared Memory", gemm_shared_memory, result_shared_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        std::vector<float> result_shared = dtype_to_float_vec(result_shared_gpu);
        bench_error = check_difference("GEMM CUDA Shared Memory", result, result_shared);
        csv_file << "GEMM CUDA Shared Memory," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<dtype> result_tiling_gpu(rows * cols, DTYPE_ZERO);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Block Tiling", gemm_block_tiling, result_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        std::vector<float> result_tiling = dtype_to_float_vec(result_tiling_gpu);
        bench_error = check_difference("GEMM CUDA Block Tiling", result, result_tiling);
        csv_file << "GEMM CUDA Block Tiling," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<dtype> result_2D_tiling_gpu(rows * cols, DTYPE_ZERO);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA 2D Block Tiling", gemm_2D_block_tiling, result_2D_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        std::vector<float> result_2D_tiling = dtype_to_float_vec(result_2D_tiling_gpu);
        bench_error = check_difference("GEMM CUDA 2D Block Tiling", result, result_2D_tiling);
        csv_file << "GEMM CUDA 2D Block Tiling," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<dtype> result_warp_tiling_gpu(rows * cols, DTYPE_ZERO);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Warp Tiling", gemm_warp_tiling, result_warp_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
        std::vector<float> result_warp_tiling = dtype_to_float_vec(result_warp_tiling_gpu);
        bench_error = check_difference("GEMM CUDA Warp Tiling", result, result_warp_tiling);
        csv_file << "GEMM CUDA Warp Tiling," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<float> result_tensor_gpu(rows * cols, 0.0f);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Tensor Core", gemm_tensor_naive, result_tensor_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
        bench_error = check_difference("GEMM CUDA Tensor Core", result, result_tensor_gpu);
        csv_file << "GEMM CUDA Tensor Core," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<float> result_tensor_warp_gpu(rows * cols, 0.0f);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Tensor Core Warp Tiling", gemm_tensor_warp_tiling, result_tensor_warp_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
        bench_error = check_difference("GEMM CUDA Tensor Core Warp Tiling", result, result_tensor_warp_gpu);
        csv_file << "GEMM CUDA Tensor Core Warp Tiling," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        std::vector<float> result_tensor_buffering_gpu(rows * cols, 0.0f);
        std::tie(bench_avg, bench_std) = benchmark("GEMM CUDA Tensor Core Double Buffering", gemm_tensor_double_buffering, result_tensor_buffering_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
        bench_error = check_difference("GEMM CUDA Tensor Core Double Buffering", result, result_tensor_buffering_gpu);
        csv_file << "GEMM CUDA Tensor Core Double Buffering," << bench_avg << "," << bench_std << "," << bench_error << "\n";

        csv_file.close();
    }

    return 0;
}

/**
 * Benchmark wrapper definition
 */
template <typename Func, typename... Args>
std::pair<double, double> benchmark(const char* name, Func&& func, Args&&... args) {
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
    
    return {mean, std_dev};
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
