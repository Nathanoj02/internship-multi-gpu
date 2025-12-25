#ifndef MATRIX_UTILS_CPP
#define MATRIX_UTILS_CPP

#include <vector>
#include <string>

/**
 * Reads a matrix from a Matrix Market (.mtx) file and stores it in CSR format.
 * @param filename The path to the .mtx file.
 * @param row_index Output vector to store the row indices.
 * @param col_index Output vector to store the column indices.
 * @param values Output vector to store the non-zero values.
 * @return 0 on success, negative value on error.
 */
int read_mtx_csr(
    const std::string& path, 
    std::vector<size_t>& row_ptr, 
    std::vector<size_t>& col_ind, 
    std::vector<float>& values
);

/**
 * Generates an array of random floating-point numbers within a specified range.
 * @param elems Number of elements to generate.
 * @param min_value Minimum value (inclusive).
 * @param max_value Maximum value (inclusive).
 * @return A vector containing the generated random floating-point numbers.
 */
std::vector<float> generate_array(int elems, float min_value, float max_value);

#endif // MATRIX_UTILS_CPP