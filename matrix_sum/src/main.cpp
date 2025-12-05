#include "matrix_ops.hpp"
#include <iostream>
#include <vector>

int main() {
    int rows = 1000;
    int cols = 1000;
    int min_value = 0;
    int max_value = 9;

    std::vector<std::vector<int>> a = generate_matrix(rows, cols, min_value, max_value);
    std::vector<std::vector<int>> b = generate_matrix(rows, cols, min_value, max_value);

    std::vector<std::vector<int>> result = sum_matrices(a, b);

    return 0;
}