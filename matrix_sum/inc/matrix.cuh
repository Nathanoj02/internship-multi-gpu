#ifndef MATRIX_CUH
#define MATRIX_CUH

/**
 * @brief Performs element-wise sum of two matrices A and B, storing the result in matrix C
 * @param A Pointer to the first input matrix (host memory)
 * @param B Pointer to the second input matrix (host memory)
 * @param C Pointer to the output matrix (host memory) where the result will be stored
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_sum_acc(const int* A, const int* B, int* C, int rows, int cols);

/**
 * @brief Performs element-wise sum of two matrices A and B using multiple GPUs, storing the result in matrix C
 * @param A Pointer to the first input matrix (host memory)
 * @param B Pointer to the second input matrix (host memory)
 * @param C Pointer to the output matrix (host memory) where the result will be stored
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
void matrix_sum_multi(const int* A, const int* B, int* C, int rows, int cols);

#endif // MATRIX_CUH