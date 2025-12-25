#include "load_balancing.hpp"
#include "matrix_utils.hpp"
#include "spmv.cuh"

#include <iostream>


int main() {
    // test_load_balancing();

    // Read matrix in CSR format
    std::vector<size_t> row_offset, cols;
    std::vector<float> values;
    int read_success = read_mtx_csr("data/494_bus.mtx", row_offset, cols, values);
    if (read_success != 0) {
        std::cerr << "Failed to read matrix file." << std::endl;
        return read_success;
    }

    // Get sizes
    size_t nnz = values.size();
    size_t rows = row_offset.size() - 1;
    
    // Generate input array and output array
    std::vector<float> arr = generate_array(rows, 0.0f, 10.0f);
    std::vector<float> result(rows);

    // GPU algorithm
    spmv(result.data(), arr.data(), row_offset.data(), cols.data(), values.data(), rows, nnz);

    return 0;
}