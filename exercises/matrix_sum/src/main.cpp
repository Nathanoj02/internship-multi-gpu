#include "matrix_ops.hpp"
#include "matrix.cuh"
#include <iostream>
#include <vector>

int main() {
    int rows = 1000;
    int cols = 1000;
    int min_value = 0;
    int max_value = 9;

    std::vector<int> a = generate_matrix(rows, cols, min_value, max_value);
    std::vector<int> b = generate_matrix(rows, cols, min_value, max_value);

    std::vector<int> result = sum_matrices(a, b, rows);

    // Use CUDA to perform the same operation
    std::vector<int> result_acc(rows * cols);
    matrix_sum_acc(a.data(), b.data(), result_acc.data(), rows, cols);

    // Check correctness
    for (int i = 0; i < rows * cols; ++i) {
        if (result[i] != result_acc[i]) {
            std::cerr << "Mismatch in 'acc' algorithm at index " << i << ": CPU result " << result[i]
                      << ", GPU result " << result_acc[i] << std::endl;
            return -1;
        }
    }

    std::cout << "Matrix summation successful and verified!" << std::endl;

    // Multi-GPU version
    std::vector<int> result_multi(rows * cols);
    matrix_sum_multi(a.data(), b.data(), result_multi.data(), rows, cols);

    // Check correctness for multi-GPU version
    for (int i = 0; i < rows * cols; ++i) {
        if (result[i] != result_multi[i]) {
            std::cerr << "Mismatch in 'multi' algorithm at index " << i << ": CPU result " << result[i]
                      << ", GPU result " << result_multi[i] << std::endl;
            return -1;
        }
    }

    std::cout << "Multi-GPU matrix summation successful and verified!" << std::endl;

    return 0;
}