#include <cuda_runtime.h>
#include "kernels/3_kernel_shared_memory.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <const size_t BLOCK_SIZE>
/**
 * CUDA kernel to perform matrix multiplication using shared memory
 * @tparam BLOCK_SIZE Size of the block (assumed square)
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__ void gemm_shared_memory_kernel (
    const dtype* A, const dtype* B, dtype* C, 
    int rows_a, int cols_a, int cols_b
) {
    int cRow = blockIdx.y, cCol = blockIdx.x;
    int threadRow = threadIdx.x / BLOCK_SIZE;
    int threadCol = threadIdx.x % BLOCK_SIZE;
    
    __shared__ dtype As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ dtype Bs[BLOCK_SIZE * BLOCK_SIZE];
    
    // Advance pointers to the starting position for this block
    A += cRow * BLOCK_SIZE * cols_a;
    B += cCol * BLOCK_SIZE;
    C += cRow * BLOCK_SIZE * cols_b + cCol * BLOCK_SIZE;

    dtype value = DTYPE_ZERO;

    for (int bkIdx = 0; bkIdx < cols_a; bkIdx += BLOCK_SIZE) {
        // Load tiles into shared memory
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * cols_a + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * cols_b + threadCol];
        
        __syncthreads();
        
        // Advance pointers for next iteration
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * cols_b;
        
        // Compute using shared memory
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            value += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        
        __syncthreads();
    }
    
    C[threadRow * cols_b + threadCol] = value;
}


void gemm_shared_memory (
    dtype* result, const dtype* A, const dtype* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    constexpr size_t BLOCK_SIZE = 32;

    dtype* d_A;
    dtype* d_B;
    dtype* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(cols_b, BLOCK_SIZE), CEIL_DIV(rows_a, BLOCK_SIZE));

    gemm_shared_memory_kernel<BLOCK_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}
