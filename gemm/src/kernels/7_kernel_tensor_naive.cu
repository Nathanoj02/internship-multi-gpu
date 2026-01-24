#include "kernels/7_kernel_tensor_naive.cuh"
#include "definitions.hpp"
#include <cuda_runtime.h>
#include <mma.h>


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
