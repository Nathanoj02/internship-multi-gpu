#include <cuda_runtime.h>
#include <mma.h>
#include "kernels/7_kernel_tensor_naive.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

template<size_t WMMA_M, size_t WMMA_N, size_t WMMA_K>
/**
 * CUDA kernel to perform matrix multiplication with tensor cores
 * @tparam WMMA_M The M dimension of the WMMA tile
 * @tparam WMMA_N The N dimension of the WMMA tile
 * @tparam WMMA_K The K dimension of the WMMA tile
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__ void gemm_tensor_naive_kernel (
    const half* A, const half* B, float* C,
    int rows_a, int cols_a, int cols_b
) {
    // Tile using a 2D grid
    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // Loop over cols of A
    for (int i = 0; i < cols_a; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Load the inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &A[aRow * cols_a + aCol], cols_a);
        nvcuda::wmma::load_matrix_sync(b_frag, &B[bRow * cols_b + bCol], cols_b);

        // Perform the matrix multiplication (c_frag = a_frag*b_frag + c_frag)
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&C[cRow * cols_b + cCol], c_frag, cols_b, nvcuda::wmma::mem_row_major);
}


void gemm_tensor_naive (
    float* result, const half* A, const half* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    constexpr size_t WMMA_M = 16;
    constexpr size_t WMMA_N = 16;
    constexpr size_t WMMA_K = 16;
    const size_t BLOCK_SIZE = 32;

    half* d_A;
    half* d_B;
    float* d_C;

    init_gemm_tensor(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Kernel execution
    dim3 dim_block(BLOCK_SIZE, 1);
    dim3 dim_grid(rows_a / WMMA_M, cols_b / WMMA_N);

    gemm_tensor_naive_kernel<WMMA_M, WMMA_N, WMMA_K><<<dim_grid, dim_block>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm_tensor(d_A, d_B, d_C, result, rows_a, cols_b);
}
