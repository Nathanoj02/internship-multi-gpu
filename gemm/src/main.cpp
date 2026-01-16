#include "gemm.hpp"
#include "gemm.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <utility>  // for std::forward

#define WARMUP_RUNS 3
#define TIMED_RUNS 10

#define EPS 1e-1

// Benchmark wrapper declaration
template <typename Func, typename... Args>
void benchmark(const char* name, Func&& func, Args&&... args);


int main() {
    int rows = 2048;
    int cols = 2048;
    float min_value = 0;
    float max_value = 10;

    std::vector<float> a = generate_matrix(rows, cols, min_value, max_value);
    std::vector<float> b = generate_matrix(rows, cols, min_value, max_value);
    
    std::vector<float> result(rows * cols, 0.0f);
    benchmark("GEMM CPU", gemm_cpu, result, a, b, rows, cols, rows, cols);

    std::vector<float> result_cuda(rows * cols, 0.0f);
    benchmark("GEMM CUDA", gemm, result_cuda.data(), a.data(), b.data(), rows, cols, rows, cols);

    // Check correctness
    for (size_t i = 0; i < result.size(); ++i) {
        if (std::abs(result[i] - result_cuda[i]) > EPS) {
            std::cout << "Mismatch at index " << i << ": CPU result " 
                      << result[i] << ", CUDA result " << result_cuda[i] << std::endl;
            
            return -1;
        }
    }

    std::cout << "\nCUDA result matches CPU result" << std::endl;
    
    return 0;
}

// Benchmark wrapper definition
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
