#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <vector>

/**
 * Generates a matrix with random integer values
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @param min_value Minimum value for random integers
 * @param max_value Maximum value for random integers
 */
std::vector<std::vector<int>> generate_matrix(int rows, int cols, int min_value, int max_value);

/**
 * Sums two matrices element-wise.
 * @param a First matrix
 * @param b Second matrix
 * @return Resulting matrix after summation
 */
std::vector<std::vector<int>> sum_matrices(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);

#endif // MATRIX_OPS_HPP