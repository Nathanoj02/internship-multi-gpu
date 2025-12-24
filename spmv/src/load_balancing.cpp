#include "load_balancing.hpp"
#include <cstdlib>
#include <cstdio>

#include <algorithm> // for std::upper_bound
#include <vector>
#include <cmath>

size_t* balance_load(size_t nnz, size_t device_num, const size_t* row_index, size_t rows) {
    if (device_num == 0) return nullptr;

    size_t* dev_offset = new size_t[device_num + 1];
    
    dev_offset[0] = 0;
    dev_offset[device_num] = rows;

    // Pointers defining the search range within row_index
    const size_t* begin_ptr = row_index;
    const size_t* end_ptr = row_index + rows + 1;

    for (size_t d = 1; d < device_num; ++d) {
        // Calculate global target
        size_t target = (nnz * d) / device_num;

        // std::upper_bound returns iterator to the first element strictly GREATER than target
        const size_t* it = std::upper_bound(begin_ptr, end_ptr, target);

        // Two candidates for the best split
        size_t val_upper = *it; // The first element > target (The "ceiling" or close to it)
        size_t val_lower = *(it - 1);   // The element <= target (The "floor")

        size_t diff_upper = val_upper - target;
        size_t diff_lower = target - val_lower;

        size_t best_idx;

        // Pick the index that provides the smaller error
        if (diff_lower <= diff_upper) {
            best_idx = (it - 1) - row_index;
        } else {
            best_idx = it - row_index;
        }

        dev_offset[d] = best_idx;

        // Update the begin_ptr for the next iteration
        begin_ptr = row_index + best_idx;
    }

    return dev_offset;
}


static void print_offsets(const char* test_name, size_t nnz, size_t device_num, const size_t* row_index, size_t rows, size_t* offsets) {
    printf("--- %s ---\n", test_name);
    printf("Total rows: %zu, Total nnz: %zu, devices: %zu\n", rows, nnz, device_num);
    for (size_t i = 0; i < device_num; ++i) {
        size_t row_start = offsets[i];
        size_t row_end = offsets[i+1];
        size_t rows_assigned = row_end - row_start;
        size_t nnz_assigned = row_index[row_end] - row_index[row_start];
        printf("Device %zu: rows [%zu, %zu) count_rows=%zu nnz=%zu\n",
               i, row_start, row_end, rows_assigned, nnz_assigned);
    }
    printf("Final offset: %zu\n\n", offsets[device_num]);
}


void test_load_balancing() {
    // Test 1
    {
        size_t nnz = 1000;
        size_t device_num = 4;
        size_t rows = 10;
        size_t row_index[] = {0, 100, 250, 400, 550, 700, 800, 850, 900, 950, 1000};

        size_t* offsets = balance_load(nnz, device_num, row_index, rows);
        print_offsets("Test 1 (uniform-ish)", nnz, device_num, row_index, rows, offsets);
        delete[] offsets;
    }

    // Test 2 (skewed, some empty rows)
    {
        size_t device_num = 3;
        size_t rows = 8;
        size_t row_index[] = {0, 5, 15, 16, 46, 46, 100, 150, 200}; // nnz = 200
        size_t nnz = row_index[rows];

        size_t* offsets = balance_load(nnz, device_num, row_index, rows);
        print_offsets("Test 2 (skewed / empty rows)", nnz, device_num, row_index, rows, offsets);
        delete[] offsets;
    }

    // Test 3 (many small + some large rows)
    {
        size_t device_num = 5;
        size_t rows = 12;
        size_t row_index[] = {0, 2, 5, 10, 20, 45, 50, 80, 90, 110, 160, 170, 200}; // nnz = 200
        size_t nnz = row_index[rows];

        size_t* offsets = balance_load(nnz, device_num, row_index, rows);
        print_offsets("Test 3 (mixed row sizes)", nnz, device_num, row_index, rows, offsets);
        delete[] offsets;
    }
}