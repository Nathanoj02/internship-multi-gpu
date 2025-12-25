#ifndef SPMV_CUH
#define SPMV_CUH

/**
 * Sparse Matrix-Vector Multiplication (SpMV) using CSR format.
 * @param out Output vector (result)
 * @param arr Input vector
 * @param row_offset Row offset array of the CSR matrix
 * @param cols Column indices array of the CSR matrix
 * @param values Non-zero values array of the CSR matrix
 * @param rows Number of rows in the matrix
 * @param num_values Number of non-zero values in the matrix
 */
void spmv(
    float* out, float* arr, 
    size_t* row_offset, size_t* cols, float* values, 
    size_t rows, size_t num_values
);

#endif // SPMV_CUH