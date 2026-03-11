#include <cuda_runtime.h>
#include <mma.h>
#include "kernels/10_kernel_tensor_hopper.cuh"
#include "gemm.cuh"
#include "../../utils/error.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#ifdef CC90

#include <cuda.h>
#include <cuda/barrier>
#include <cassert>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// =====================================================================
// Helper functions for WGMMA shared memory descriptors
// =====================================================================

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc(half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

// =====================================================================
// WGMMA synchronization primitives
// =====================================================================

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// =====================================================================
// TMA tensor map creation
// =====================================================================

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, half* gmem_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {
        (uint64_t)BlockMinorSize * blocks_width,
        (uint64_t)BlockMajorSize * blocks_height,
        1, 1, 1
    };
    uint64_t gmem_prob_stride[5] = {
        sizeof(half),
        sizeof(half) * BlockMinorSize * blocks_width,
        0, 0, 0
    };
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(half* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

// =====================================================================
// WGMMA m64n64k16 instruction wrapper (f32 accumulation, f16 inputs)
// =====================================================================

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], half* sA, half* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// =====================================================================
// Transpose kernel (row-major B[K][N] -> column-major B^T[N][K])
// =====================================================================

__global__ void transpose_half_kernel(const half* in, half* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int i = idx / cols;
        int j = idx % cols;
        out[j * rows + i] = in[i * cols + j];
    }
}

// =====================================================================
// Main Hopper GEMM kernel using TMA + WGMMA
// =====================================================================

template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
/**
 * CUDA kernel to perform matrix multiplication with Hopper tensor cores (WGMMA + TMA)
 * Uses TMA for asynchronous bulk data loading and WGMMA for warp-group level MMA
 * @tparam BM Block tile M dimension
 * @tparam BN Block tile N dimension
 * @tparam BK Block tile K dimension
 * @tparam WGMMA_M WGMMA tile M dimension
 * @tparam WGMMA_N WGMMA tile N dimension
 * @tparam WGMMA_K WGMMA tile K dimension
 * @tparam NUM_THREADS Number of threads per block (one warp group = 128)
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 * @param C Pointer to result matrix C (float, row-major)
 * @param tensorMapA TMA descriptor for matrix A
 * @param tensorMapB TMA descriptor for matrix B (transposed)
 */
__global__ void __launch_bounds__(NUM_THREADS) gemm_tensor_hopper_kernel(
    int rows_a, int cols_a, int cols_b,
    float* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB
) {
    __shared__ alignas(128) half sA[BM * BK];
    __shared__ alignas(128) half sB[BK * BN];

    float d[WGMMA_N / 16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = cols_a / BK;
    int num_block_n = blockIdx.x % (cols_b / BN);
    int num_block_m = blockIdx.x / (cols_b / BN);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        // Load A and B tiles using TMA async bulk copy
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter * BK, num_block_m * BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter * BK, num_block_n * BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // Compute: 4 WGMMA operations per K-tile (BK=64, WGMMA_K=16)
        warpgroup_arrive();
        wgmma64<1, 1, 1, 0, 0>(d, &sA[0],            &sB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K],      &sB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[2 * WGMMA_K],  &sB[2 * WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[3 * WGMMA_K],  &sB[3 * WGMMA_K]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    // Store results to global memory (row-major float)
    {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp * 16 + lane / 4;
        float* block_C = C + num_block_m * BM * cols_b + num_block_n * BN;

        for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < BN / WGMMA_N; ++n_it) {
                for (int w = 0; w < WGMMA_N / 16; ++w) {
                    int col = 16 * w + 2 * (tid % 4);
                    #define IDX(i, j) (((i) + m_it * WGMMA_M) * cols_b + ((j) + n_it * WGMMA_N))

                    block_C[IDX(row, col)]         = d[w][0];
                    block_C[IDX(row, col + 1)]     = d[w][1];
                    block_C[IDX(row + 8, col)]     = d[w][2];
                    block_C[IDX(row + 8, col + 1)] = d[w][3];

                    block_C[IDX(row, col + 8)]     = d[w][4];
                    block_C[IDX(row, col + 9)]     = d[w][5];
                    block_C[IDX(row + 8, col + 8)] = d[w][6];
                    block_C[IDX(row + 8, col + 9)] = d[w][7];

                    #undef IDX
                }
            }
        }
    }
}


void gemm_tensor_hopper (
    float* result, const half* A, const half* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    half* d_A;
    half* d_B;
    float* d_C;

    init_gemm_tensor(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Transpose B from row-major K×N to column-major (stored as N×K row-major)
    // Required because WGMMA expects B in K-major layout
    half* d_B_T;
    CUDA_CHECK( cudaMalloc(&d_B_T, rows_b * cols_b * sizeof(half)) );
    {
        int total = rows_b * cols_b;
        int threads = 256;
        int blocks = CEIL_DIV(total, threads);
        transpose_half_kernel<<<blocks, threads>>>(d_B, d_B_T, rows_b, cols_b);
        CUDA_CHECK( cudaGetLastError() );
        CUDA_CHECK( cudaDeviceSynchronize() );
    }

    // Create TMA tensor maps on device
    CUtensorMap* d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(d_A, rows_a / BM, cols_a / BK);
    CUtensorMap* d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(d_B_T, cols_b / BN, rows_b / BK);

    dim3 gridSize((rows_a / BM) * (cols_b / BN));
    dim3 blockSize(NUM_THREADS);

    gemm_tensor_hopper_kernel<BM, BN, BK, /*WGMMA_M*/ 64, /*WGMMA_N*/ 64, /*WGMMA_K*/ 16, NUM_THREADS>
        <<<gridSize, blockSize>>>(rows_a, cols_a, cols_b, d_C, d_tma_map_A, d_tma_map_B);

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    CUDA_CHECK( cudaFree(d_B_T) );
    CUDA_CHECK( cudaFree(d_tma_map_A) );
    CUDA_CHECK( cudaFree(d_tma_map_B) );

    cleanup_gemm_tensor(d_A, d_B, d_C, result, rows_a, cols_b);
}

#else

void gemm_tensor_hopper (
    float* result, const half* A, const half* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    fprintf(stderr, "Error: Hopper tensor core kernel requires sm_90 (Hopper architecture).\n"
                    "Recompile with -DCC90 -gencode arch=compute_90,code=sm_90\n");
}

#endif // CC90

