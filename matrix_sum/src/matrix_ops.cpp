#include "matrix_ops.hpp"
#include <random>

std::vector<std::vector<int>> generate_matrix(int rows, int cols, int min_value, int max_value) {
    // Generate matrix with random int values
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<> dis(min_value, max_value);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(rng);
        }
    }

    return matrix;
}

std::vector<std::vector<int>> sum_matrices(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
    int rows = a.size();
    int cols = a[0].size();
    std::vector<std::vector<int>> result(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    return result;
}