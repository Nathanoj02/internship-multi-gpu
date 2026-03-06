/**
 * tensor_core_benchmark.cu
 *
 * Benchmark PEAK FLOPS on NVIDIA Tensor Cores using WMMA tiles (FP16 * FP16 -> FP32).
 *
 * Build:
 *  A30 -> nvcc -O3 -arch=sm_80 tensor_core_benchmark.cu -o tensor_core_benchmark
 *  L40s -> nvcc -O3 -arch=sm_89 tensor_core_benchmark.cu -o tensor_core_benchmark
 *
 * Usage:
 *   ./tensor_core_benchmark [warmup_iters] [bench_iters]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ====================
// Tunables
// ====================
// WMMA tile size
static constexpr int M = 16, N = 16, K = 16;

// How many MMA operations each warp performs per kernel launch.
// Must be large enough that kernel latency >> dispatch overhead.
static constexpr int WARP_ITERS = 2048;

// Thread-block geometry
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// ====================
// CUDA error-checking helpers
// ====================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ====================
// Benchmark kernel
// ====================
__global__ void wmma_peak_flops(
    const __half* __restrict__ init_a,
    const __half* __restrict__ init_b,
    const float*  __restrict__ init_c,
    float*        __restrict__ dummy_out
)
{
    // Shared memory tiles: one copy per warp to avoid bank conflicts
    __shared__ __half  smem_a[WARPS_PER_BLOCK][M * K];
    __shared__ __half  smem_b[WARPS_PER_BLOCK][K * N];
    __shared__ float   smem_c[WARPS_PER_BLOCK][M * N];

    const int warp_id    = threadIdx.x / 32;
    const int lane_id    = threadIdx.x % 32;
    const int global_wid = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Populate shared memory from global memory (runtime dependency)
    // Each thread loads multiple elements; stride = 32 (warp width)
    for (int i = lane_id; i < M * K; i += 32)
        smem_a[warp_id][i] = init_a[i];
    for (int i = lane_id; i < K * N; i += 32)
        smem_b[warp_id][i] = init_b[i];
    for (int i = lane_id; i < M * N; i += 32)
        smem_c[warp_id][i] = init_c[i];

    __syncthreads();

    // Declare and load WMMA fragments
    wmma::fragment<wmma::matrix_a, M, N, K, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, M, N, K, __half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, M, N, K, float>                frag_c;

    wmma::load_matrix_sync(frag_a, smem_a[warp_id], K);
    wmma::load_matrix_sync(frag_b, smem_b[warp_id], N);
    wmma::load_matrix_sync(frag_c, smem_c[warp_id], N, wmma::mem_row_major);

    // Memory fence (for compiler)
    asm volatile("" ::: "memory");

    #pragma unroll 16
    for (int i = 0; i < WARP_ITERS; ++i) {
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    asm volatile("" ::: "memory");

    // For compiler optimizations
    if (dummy_out != nullptr) {
        wmma::store_matrix_sync(
            dummy_out + global_wid * M * N,
            frag_c,
            N,
            wmma::mem_row_major
        );
    }
}

// ==========
// Host code
// ==========
int main(int argc, char** argv)
{
    int warmup_iters = (argc > 1) ? atoi(argv[1]) : 5;
    int bench_iters  = (argc > 2) ? atoi(argv[2]) : 20;

    // Device info
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("Device      : %s\n", prop.name);
    printf("SM count    : %d\n", prop.multiProcessorCount);

    // Adjust block count to fill the GPU
    const int num_blocks = prop.multiProcessorCount * 4;
    const long long total_warps = (long long)num_blocks * WARPS_PER_BLOCK;

    printf("Blocks      : %d\n", num_blocks);
    printf("Warps total : %lld\n", total_warps);
    printf("Warp iters  : %d\n", WARP_ITERS);

    // FLOPs per kernel launch
    const double flops_per_launch =
        (double)total_warps * WARP_ITERS * 2.0 * M * N * K;
    printf("FLOPs/launch: %.3e\n\n", flops_per_launch);

    // Allocate and initialise device buffers
    __half h_a[M * K], h_b[K * N];
    float  h_c[M * N];
    for (int i = 0; i < M * K; ++i) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; ++i) h_c[i] = 0.0f;

    __half *d_a, *d_b;
    float  *d_c, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,  M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b,  K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_c,  M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_warps * M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, M * N * sizeof(float),  cudaMemcpyHostToDevice));

    // Clocks
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warmup
    printf("Warming up (%d iterations)...\n", warmup_iters);
    for (int i = 0; i < warmup_iters; ++i) {
        wmma_peak_flops<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_a, d_b, d_c, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    printf("Benchmarking (%d iterations)...\n\n", bench_iters);

    double total_tflops = 0.0;
    double min_tflops   = 1e18;
    double max_tflops   = 0.0;

    for (int i = 0; i < bench_iters; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_start));

        wmma_peak_flops<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_a, d_b, d_c, d_out);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        double tflops = flops_per_launch / (ms * 1e-3) / 1e12;
        total_tflops += tflops;
        if (tflops < min_tflops) min_tflops = tflops;
        if (tflops > max_tflops) max_tflops = tflops;

        printf("  iter %3d : %.2f ms  →  %.2f TFLOPS\n", i, ms, tflops);
    }

    double avg_tflops = total_tflops / bench_iters;

    printf("\n==========================================\n");
    printf("  Min  TFLOPS : %.2f\n", min_tflops);
    printf("  Max  TFLOPS : %.2f\n", max_tflops);
    printf("  Avg  TFLOPS : %.2f\n", avg_tflops);
    printf("==========================================\n");

    // Verify one output element is non-zero
    float h_out[M * N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nSanity check — out[0][0] = %.1f  (expected %d.0)\n",
           h_out[0], WARP_ITERS * K);  // 1*1*16 accumulated WARP_ITERS times

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0;
}
