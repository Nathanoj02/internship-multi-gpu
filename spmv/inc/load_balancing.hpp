#ifndef LOAD_BALANCING_HPP
#define LOAD_BALANCING_HPP

/**
 * Balances the load of non-zero elements across multiple GPUs using binary search.
 * @param nnz Total number of non-zero elements in the matrix.
 * @param device_num Number of GPUs available.
 * @param row_index Array containing the row indices of the matrix (CSR format).
 * @param rows Total number of rows in the matrix.
 * @return An array of size device_num + 1 containing the row offsets for each GPU
 */
size_t* balance_load(size_t nnz, size_t device_num, const size_t* row_index, size_t rows);

/**
 * Test function for load balancing.
 */
void test_load_balancing();

#endif // LOAD_BALANCING_HPP