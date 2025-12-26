#include "matrix_utils.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <random>

int read_mtx_dimensions(const std::string& path, size_t* out_rows, size_t* out_nnz) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "File opening failed!" << std::endl;
        return -1;
    }

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break; 
    }

    // Read dimensions
    size_t rows, cols, nnz;
    std::istringstream iss(line);
    if (!(iss >> rows >> cols >> nnz)) {
        std::cerr << "Error: Invalid matrix header format" << std::endl;
        return -2;
    }

    *out_rows = rows;
    *out_nnz  = nnz;

    return 0;
}

int read_mtx_csr(
    const std::string& path, size_t*& row_ptr, 
    size_t*& col_ind, float*& values
) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "File opening failed!" << std::endl;
        return -1;
    }

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break; 
    }

    // Read dimensions
    size_t rows, cols, nnz;
    std::istringstream iss(line);
    if (!(iss >> rows >> cols >> nnz)) {
        std::cerr << "Error: Invalid matrix header format" << std::endl;
        return -2;
    }

    // Save file position to read data twice
    std::streampos data_pos = file.tellg();

    // Count non-zeros per row
    size_t r, c;
    float v;
    
    while (file >> r >> c >> v) {
        // MTX is 1-based, we need 0-based row index
        if (r >= 1 && r <= rows) {
            row_ptr[r]++; // Store count in the "next" slot temporarily
        }
    }

    // Prefix sum to get row_ptr
    size_t cumulative = 0;
    for (size_t i = 0; i <= rows; ++i) {
        size_t count = row_ptr[i]; // row_ptr[i] holds the COUNT of row i-1
        row_ptr[i] = cumulative;
        cumulative += count;
    }

    // Fill Data
    file.clear(); // Clear EOF flag
    file.seekg(data_pos); // Go back to data start

    size_t* current_pos = new size_t[rows + 1];
    std::copy(row_ptr, row_ptr + rows + 1, current_pos);

    while (file >> r >> c >> v) {
        size_t row_idx = r - 1;
        size_t col_idx = c - 1;

        size_t dest_idx = current_pos[row_idx];
        
        col_ind[dest_idx] = col_idx;
        values[dest_idx]  = v;

        current_pos[row_idx]++;
    }

    delete[] current_pos;

    return 0;
}


float* generate_array(int elems, float min_value, float max_value) {
    float *array = new float[elems];
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> dis(min_value, max_value);
    
    for (int i = 0; i < elems; ++i) {
        array[i] = dis(rng);
    }

    return array;
}