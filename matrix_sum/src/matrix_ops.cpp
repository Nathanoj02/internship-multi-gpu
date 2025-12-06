#include "matrix_ops.hpp"
#include <random>

std::vector<int> generate_matrix(int rows, int cols, int min_value, int max_value) {
    // Generate matrix with random int values
    std::vector<int> matrix(rows * cols);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<> dis(min_value, max_value);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dis(rng);
        }
    }

    return matrix;
}

std::vector<int> sum_matrices(const std::vector<int>& a, const std::vector<int>& b, int rows) {
    int cols = a.size() / rows;
    std::vector<int> result(rows * cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] = a[i * cols + j] + b[i * cols + j];
        }
    }

    return result;
}