#include "gemm.cuh"
#include "../utils/error.cuh"
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_SIZE 32

// -- Function declarations --
void init_gemm(
    float** d_A, float** d_B, float** d_C,
    const float* A, const float* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
);

void cleanup_gemm(
    float* d_A, float* d_B, float* d_C,
    float* result, size_t rows_a, size_t cols_b
);

// -- CUDA Kernels --
/**
 * CUDA kernel to perform matrix multiplication
 * Each thread computes one element of the result matrix C
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_naive_kernel(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows_a || col >= cols_b) {
        return;
    }

    float value = 0.0f;
    for (int k = 0; k < cols_a; ++k) {
        value += A[row * cols_a + k] * B[k * cols_b + col];
    }
    C[row * cols_b + col] = value;
}

/**
 * CUDA kernel to perform matrix multiplication with memory coalescing
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_memory_coalescing_kernel(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    int col = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row >= rows_a || col >= cols_b) {
        return;
    }

    float value = 0.0f;
    for (int k = 0; k < cols_a; ++k) {
        value += A[row * cols_a + k] * B[k * cols_b + col];
    }
    C[row * cols_b + col] = value;
}

/**
 * CUDA kernel to perform matrix multiplication using shared memory
 * @param A Pointer to matrix A
 * @param B Pointer to matrix B
 * @param C Pointer to result matrix C
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (and rows in matrix B)
 * @param cols_b Number of columns in matrix B
 */
__global__
void gemm_shared_memory_kernel(const float* A, const float* B, float* C, int rows_a, int cols_a, int cols_b) {
    int cRow = blockIdx.y, cCol = blockIdx.x;
    int threadRow = threadIdx.x / BLOCK_SIZE, threadCol = threadIdx.x % BLOCK_SIZE;
    
    // Global position for this thread
    int globalRow = cRow * BLOCK_SIZE + threadRow;
    int globalCol = cCol * BLOCK_SIZE + threadCol;
    
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE], Bs[BLOCK_SIZE * BLOCK_SIZE];

    float value = 0.0f;
    
    for (int tileIdx = 0; tileIdx < (cols_a + BLOCK_SIZE - 1) / BLOCK_SIZE; tileIdx++) {
        // Load tile from A with bounds checking
        int aCol = tileIdx * BLOCK_SIZE + threadCol;
        if (globalRow < rows_a && aCol < cols_a) {
            As[threadRow * BLOCK_SIZE + threadCol] = A[globalRow * cols_a + aCol];
        } else {
            As[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        }
        
        // Load tile from B with bounds checking
        int bRow = tileIdx * BLOCK_SIZE + threadRow;
        if (bRow < cols_a && globalCol < cols_b) {
            Bs[threadRow * BLOCK_SIZE + threadCol] = B[bRow * cols_b + globalCol];
        } else {
            Bs[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            value += As[threadRow * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + threadCol];
        }
        
        __syncthreads();
    }
    
    // Write result with bounds checking
    if (globalRow < rows_a && globalCol < cols_b) {
        C[globalRow * cols_b + globalCol] = value;
    }
}


// -- Host Functions --
void gemm_naive(
    float* result, const float* A, const float* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    float* d_A;
    float* d_B;
    float* d_C;

    // Initialize and copy data to device
    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_a, blockSize.x), CEIL_DIV(cols_b, blockSize.y));

    gemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy result back and cleanup
    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_memory_coalescing(
    float* result, const float* A, const float* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    float* d_A;
    float* d_B;
    float* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(cols_b, BLOCK_SIZE), CEIL_DIV(rows_a, BLOCK_SIZE));

    gemm_memory_coalescing_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


void gemm_shared_memory(
    float* result, const float* A, const float* B, 
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    float* d_A;
    float* d_B;
    float* d_C;

    init_gemm(&d_A, &d_B, &d_C, A, B, rows_a, cols_a, rows_b, cols_b);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(cols_b, BLOCK_SIZE), CEIL_DIV(rows_a, BLOCK_SIZE));

    gemm_shared_memory_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_a, cols_a, cols_b);
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    cleanup_gemm(d_A, d_B, d_C, result, rows_a, cols_b);
}


/**
 * Initialize device memory and copy input matrices
 */
void init_gemm(
    float** d_A, float** d_B, float** d_C,
    const float* A, const float* B,
    size_t rows_a, size_t cols_a, size_t rows_b, size_t cols_b
) {
    // Dimension check
    if (cols_a != rows_b) {
        fprintf(stderr, "Error: Matrix dimensions don't match for multiplication. "
                "A is %zux%zu, B is %zux%zu\n", rows_a, cols_a, rows_b, cols_b);
        return;
    }

    size_t size_a = rows_a * cols_a * sizeof(float);
    size_t size_b = rows_b * cols_b * sizeof(float);
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMalloc((void**)d_A, size_a) );
    CUDA_CHECK( cudaMalloc((void**)d_B, size_b) );
    CUDA_CHECK( cudaMalloc((void**)d_C, size_res) );

    CUDA_CHECK( cudaMemcpy(*d_A, A, size_a, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(*d_B, B, size_b, cudaMemcpyHostToDevice) );
}

/**
 * Copy result back to host and free device memory
 */
void cleanup_gemm(
    float* d_A, float* d_B, float* d_C,
    float* result, size_t rows_a, size_t cols_b
) {
    size_t size_res = rows_a * cols_b * sizeof(float);

    CUDA_CHECK( cudaMemcpy(result, d_C, size_res, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
}