#include "load_balancing.hpp"
#include "matrix_utils.hpp"

#include <iostream>


int main() {
    // test_load_balancing();

    std::vector<size_t> row_ptr, col_ind;
    std::vector<float> values;
    int result = read_mtx_csr("data/494_bus.mtx", row_ptr, col_ind, values);
    if (result != 0) {
        std::cerr << "Failed to read matrix file." << std::endl;
        return result;
    }

    size_t nnz = values.size();
    size_t rows = row_ptr.size() - 1;
    
    // Print some info about the matrix
    std::cout << "Matrix loaded: " << rows << " rows, " << nnz << " non-zeros." << std::endl;

    return 0;
}