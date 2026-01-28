#ifndef GEMM_HPP
#define GEMM_HPP

#include <vector>

/**
 * Generates a matrix with random float values
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @param min_value Minimum value for random floats
 * @param max_value Maximum value for random floats
 */
std::vector<float> generate_matrix (int rows, int cols, float min_value, float max_value);

/**
 * Multiplies two matrices using CPU
 * @param result Resultant matrix to store the multiplication result
 * @param a First matrix
 * @param b Second matrix
 * @param rows_a Number of rows in the first matrix
 * @param cols_a Number of columns in the first matrix
 * @param rows_b Number of rows in the second matrix
 * @param cols_b Number of columns in the second matrix
 */
void gemm_cpu (
    std::vector<float>& result,
    const std::vector<float>& a, const std::vector<float>& b, 
    int rows_a, int cols_a, int rows_b, int cols_b
);

#endif // GEMM_HPP