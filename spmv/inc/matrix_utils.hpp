#ifndef MATRIX_UTILS_CPP
#define MATRIX_UTILS_CPP

#include <vector>
#include <string>

/**
 * Reads the dimensions of a matrix from a Matrix Market (.mtx) file.
 * @param filename The path to the .mtx file.
 * @param out_rows Output parameter to store the number of rows.
 * @param out_nnz Output parameter to store the number of non-zero elements.
 * @return 0 on success, negative value on error.
 */
int read_mtx_dimensions(const std::string& path, size_t* out_rows, size_t* out_nnz);

/**
 * Reads a matrix from a Matrix Market (.mtx) file and stores it in CSR format.
 * @param filename The path to the .mtx file.
 * @param row_index Output vector to store the row indices.
 * @param col_index Output vector to store the column indices.
 * @param values Output vector to store the non-zero values.
 * @return 0 on success, negative value on error.
 */
int read_mtx_csr(
    const std::string& path, size_t*& row_ptr, 
    size_t*& col_ind, float*& values
);

/**
 * Generates an array of random floating-point numbers within a specified range.
 * @param elems Number of elements to generate.
 * @param min_value Minimum value (inclusive).
 * @param max_value Maximum value (inclusive).
 * @return A vector containing the generated random floating-point numbers.
 */
float* generate_array(int elems, float min_value, float max_value);

#endif // MATRIX_UTILS_CPP