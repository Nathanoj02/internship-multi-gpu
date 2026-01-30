#include "gemm.hpp"
#include "gemm.cuh"
#include "dtype.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <utility>  // for std::forward
#include <tuple>

#define WARMUP_RUNS 3
#define TIMED_RUNS 10

double check_difference(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& test
);


int main() {
    int rows = 1024;
    int cols = 1024;
    float min_value = 0;
    float max_value = 10;

    // Generate matrices
    std::vector<float> a = generate_matrix(rows, cols, min_value, max_value);
    std::vector<float> b = generate_matrix(rows, cols, min_value, max_value);

    // CPU computation
    std::vector<float> result(rows * cols, 0.0f);

    // Convert to dtype for GPU (no-op if dtype=float)
    std::vector<dtype> a_gpu = float_to_dtype_vec(a);
    std::vector<dtype> b_gpu = float_to_dtype_vec(b);

    // Convert to half for tensor core
    std::vector<half> a_half = float_to_half_vec(a);
    std::vector<half> b_half = float_to_half_vec(b);

    // === GPU computations ===
    // Benchmark
    std::vector<dtype> result_naive_gpu(rows * cols, DTYPE_ZERO);
    gemm_naive(result_naive_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    
    // Convert GPU result back to float for comparison
    std::vector<float> result_naive = dtype_to_float_vec(result_naive_gpu);
    check_difference("GEMM CUDA Naive", result, result_naive);
   
    std::vector<dtype> result_coalescing_gpu(rows * cols, DTYPE_ZERO);
    gemm_memory_coalescing(result_coalescing_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    std::vector<float> result_coalescing = dtype_to_float_vec(result_coalescing_gpu);
    check_difference("GEMM CUDA Coalescing", result, result_coalescing);

    std::vector<dtype> result_shared_gpu(rows * cols, DTYPE_ZERO);
    gemm_shared_memory(result_shared_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    std::vector<float> result_shared = dtype_to_float_vec(result_shared_gpu);
    check_difference("GEMM CUDA Shared Memory", result, result_shared);

    std::vector<dtype> result_tiling_gpu(rows * cols, DTYPE_ZERO);
    gemm_block_tiling(result_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    std::vector<float> result_tiling = dtype_to_float_vec(result_tiling_gpu);
    check_difference("GEMM CUDA Block Tiling", result, result_tiling);

    std::vector<dtype> result_2D_tiling_gpu(rows * cols, DTYPE_ZERO);
    gemm_2D_block_tiling(result_2D_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    std::vector<float> result_2D_tiling = dtype_to_float_vec(result_2D_tiling_gpu);
    check_difference("GEMM CUDA 2D Block Tiling", result, result_2D_tiling);

    std::vector<dtype> result_warp_tiling_gpu(rows * cols, DTYPE_ZERO);
    gemm_warp_tiling(result_warp_tiling_gpu.data(), a_gpu.data(), b_gpu.data(), rows, cols, rows, cols);
    std::vector<float> result_warp_tiling = dtype_to_float_vec(result_warp_tiling_gpu);
    check_difference("GEMM CUDA Warp Tiling", result, result_warp_tiling);

    std::vector<float> result_tensor_gpu(rows * cols, 0.0f);
    gemm_tensor_naive(result_tensor_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
    check_difference("GEMM CUDA Tensor Core", result, result_tensor_gpu);

    std::vector<float> result_tensor_warp_gpu(rows * cols, 0.0f);
    gemm_tensor_warp_tiling(result_tensor_warp_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
    check_difference("GEMM CUDA Tensor Core Warp Tiling", result, result_tensor_warp_gpu);

    std::vector<float> result_tensor_buffering_gpu(rows * cols, 0.0f);
    gemm_tensor_double_buffering(result_tensor_buffering_gpu.data(), a_half.data(), b_half.data(), rows, cols, rows, cols);
    check_difference("GEMM CUDA Tensor Core Double Buffering", result, result_tensor_buffering_gpu);

    return 0;
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

    std::cout << "Average difference in " << name << ": " 
        << avg_diff << " (" << avg_percentage_diff << "%)" << std::endl;
    
    return avg_percentage_diff;
}
