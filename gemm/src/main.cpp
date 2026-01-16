#include "gemm.hpp"
#include "gemm.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <utility>  // for std::forward

#define WARMUP_RUNS 3
#define TIMED_RUNS 10

#define EPS 1e-1

// Functions declarations
template <typename Func, typename... Args>
void benchmark(const char* name, Func&& func, Args&&... args);

bool check_correctness(
    const char* name,
    const std::vector<float>& reference, 
    const std::vector<float>& test
);


int main() {
    int rows = 2048;
    int cols = 2048;
    float min_value = 0;
    float max_value = 10;

    std::vector<float> a = generate_matrix(rows, cols, min_value, max_value);
    std::vector<float> b = generate_matrix(rows, cols, min_value, max_value);
    
    std::vector<float> result(rows * cols, 0.0f);
    benchmark("GEMM CPU", gemm_cpu, result, a, b, rows, cols, rows, cols);

    std::vector<float> result_naive(rows * cols, 0.0f);
    benchmark("GEMM CUDA", gemm_naive, result_naive.data(), a.data(), b.data(), rows, cols, rows, cols);

    std::vector<float> result_coalescing(rows * cols, 0.0f);
    benchmark("GEMM CUDA Coalescing", gemm_memory_coalescing, result_coalescing.data(), a.data(), b.data(), rows, cols, rows, cols);

    std::vector<float> result_shared(rows * cols, 0.0f);
    benchmark("GEMM CUDA Shared Memory", gemm_shared_memory, result_shared.data(), a.data(), b.data(), rows, cols, rows, cols);

    std::vector<float> result_tiling(rows * cols, 0.0f);
    benchmark("GEMM CUDA Block Tiling", gemm_block_tiling, result_tiling.data(), a.data(), b.data(), rows, cols, rows, cols);

    // Check correctness
    bool correct = true;
    correct &= check_correctness("GEMM CUDA", result, result_naive);
    correct &= check_correctness("GEMM CUDA Coalescing", result, result_coalescing);
    correct &= check_correctness("GEMM CUDA Shared Memory", result, result_shared);
    correct &= check_correctness("GEMM CUDA Block Tiling", result, result_tiling);
    
    if (!correct) {
        return -1;
    }

    std::cout << "\nCUDA result matches CPU result" << std::endl;
    
    return 0;
}

/**
 * Benchmark wrapper definition
 */
template <typename Func, typename... Args>
void benchmark(const char* name, Func&& func, Args&&... args) {
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
              << "  Std: " << std_dev << " seconds" << std::endl;
}

/**
 * Check correctness of two vectors
 */
bool check_correctness(
    const char* name,
    const std::vector<float>& reference, 
    const std::vector<float>& test
) {
    if (reference.size() != test.size()) {
        throw std::invalid_argument("Vectors must be of the same size for correctness check.");
    }

    for (size_t i = 0; i < reference.size(); ++i) {
        if (std::abs(reference[i] - test[i]) > EPS) {
            std::cerr << "Mismatch at index " << i << " in " << name <<
                                     ": reference " << reference[i] << 
                                     ", test " << test[i] << std::endl;
            return false;
        }
    }
    return true;
}