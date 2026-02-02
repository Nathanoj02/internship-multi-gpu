#include <cuda_runtime.h>
#include <mma.h>
#include "kernels/8_kernel_tensor_warp_tiling.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <
    size_t BLOCK_ROW_WARPS,
    size_t BLOCK_COL_WARPS,
    size_t WARP_ROW_TILES,
    size_t WARP_COL_TILES,
    size_t WMMA_M,
    size_t WMMA_N,
    size_t WMMA_K
>
/**
 * CUDA kernel to perform matrix multiplication with tensor cores using warp tiling
 * Each warp computes WARP_ROW_TILES x WARP_COL_TILES WMMA tiles
 * @tparam BLOCK_ROW_WARPS Number of warps per block in row dimension
 * @tparam BLOCK_COL_WARPS Number of warps per block in column dimension
 * @tparam WARP_ROW_TILES Number of WMMA tiles per warp in row dimension
 * @tparam WARP_COL_TILES Number of WMMA tiles per warp in column dimension
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
__global__ void gemm_tensor_warp_tiled_kernel (
    const half* A, const half* B, float* C,
    int rows_a, int cols_a, int cols_b
) {
    const uint warp_id = threadIdx.x / 32;
    const uint warp_row = warp_id / BLOCK_COL_WARPS;
    const uint warp_col = warp_id % BLOCK_COL_WARPS;

    constexpr int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
    constexpr int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;

    constexpr int BM = BLOCK_ROW_TILES * WMMA_M;
    constexpr int BN = BLOCK_COL_TILES * WMMA_N;
    constexpr int BK = WMMA_K;

    // Shared memory: tile_a (BM x BK, row-major), tile_b (BK x BN, column-major)
    __shared__ half tile_a[BM * BK];
    __shared__ half tile_b[BK * BN];

    const half* global_a = A + blockIdx.y * BM * cols_a;
    const half* global_b = B + blockIdx.x * BN;
    float* global_c = C + blockIdx.y * BM * cols_b + blockIdx.x * BN;

    // Declare fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;

    // Accumulator fragments: each warp maintains WARP_ROW_TILES x WARP_COL_TILES tiles
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            nvcuda::wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    constexpr int NUM_THREADS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32;

    // K-loop: iterate by BK
    for (int block_k_idx = 0; block_k_idx < cols_a; block_k_idx += BK) {
        // Load A tile (BM x BK, row-major)
        #pragma unroll
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            tile_a[row * BK + col] = global_a[row * cols_a + col];
        }

        // Load B tile (BK x BN, column-major for WMMA)
        #pragma unroll
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS) {
            int row = idx / BN;
            int col = idx % BN;
            // Store in column-major format: tile_b[col * BK + row]
            tile_b[col * BK + row] = global_b[row * cols_b + col];
        }

        __syncthreads();

        // Advance pointers
        global_a += BK;
        global_b += BK * cols_b;

        // Warp-level tiling: each warp computes WARP_ROW_TILES x WARP_COL_TILES WMMA tiles
        #pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i) {
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j) {
                int a_tile_row = warp_row * WARP_ROW_TILES + i;
                int b_tile_col = warp_col * WARP_COL_TILES + j;

                const half* a_tile_ptr = tile_a + (a_tile_row * WMMA_M) * BK;
                const half* b_tile_ptr = tile_b + (b_tile_col * WMMA_N) * BK;

                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);
                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // Store results to global memory
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            int c_tile_row = warp_row * WARP_ROW_TILES + i;
            int c_tile_col = warp_col * WARP_COL_TILES + j;

            float* c_ptr = global_c + (c_tile_row * WMMA_M) * cols_b + (c_tile_col * WMMA_N);

            nvcuda::wmma::store_matrix_sync(c_ptr, acc_frag[i][j], cols_b, nvcuda::wmma::mem_row_major);
        }
    }
}


void gemm_tensor_warp_tiling (
    float* result, const half* A, const half* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    constexpr size_t WMMA_M = 16;
    constexpr size_t WMMA_N = 16;
    constexpr size_t WMMA_K = 16;
    constexpr size_t BLOCK_ROW_WARPS = 4;
    constexpr size_t BLOCK_COL_WARPS = 2;
    constexpr size_t WARP_ROW_TILES = 4;
    constexpr size_t WARP_COL_TILES = 4;

    constexpr size_t BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
    constexpr size_t BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;
    constexpr size_t BM = BLOCK_ROW_TILES * WMMA_M;
    constexpr size_t BN = BLOCK_COL_TILES * WMMA_N;

    half* d_A;
    half* d_B;
    float* d_C;

    init_gemm_tensor(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Each block has BLOCK_ROW_WARPS * BLOCK_COL_WARPS warps
    dim3 blockSize(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);
    dim3 gridSize(CEIL_DIV(cols_b, BN), CEIL_DIV(rows_a, BM));

    gemm_tensor_warp_tiled_kernel<BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, WMMA_M, WMMA_N, WMMA_K>
        <<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm_tensor(d_A, d_B, d_C, result, rows_a, cols_b);
}