#include "gemm.hpp"
#include <random>
#include <stdexcept>

std::vector<float> generate_matrix (int rows, int cols, float min_value, float max_value) {
    // Generate matrix with random float values
    std::vector<float> matrix(rows * cols);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> dis(min_value, max_value);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dis(rng);
        }
    }

    return matrix;
}

void gemm_cpu (
    std::vector<float>& result,
    const std::vector<float>& a, const std::vector<float>& b, 
    int rows_a, int cols_a, int rows_b, int cols_b
) {
    // Assert sizes matches
    if (cols_a != rows_b) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    }

    // Zero out result before accumulation
    std::fill(result.begin(), result.end(), 0.0f);

    for (int i = 0; i < rows_a; ++i) {
        for (int k = 0; k < cols_a; ++k) {
            float aik = a[i * cols_a + k];
            for (int j = 0; j < cols_b; ++j) {
                result[i * cols_b + j] += aik * b[k * cols_b + j];
            }
        }
    }
}