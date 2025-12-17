#include "matrix.cuh"
#include "../utils/error.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * CUDA kernel to perform element-wise sum of two matrices A and B into matrix C
 * Each thread computes one element of the resulting matrix
 * @param A Pointer to the first input matrix
 * @param B Pointer to the second input matrix
 * @param C Pointer to the output matrix
 * @param rows Number of rows in the matrices
 * @param cols Number of columns in the matrices
 */
__global__
void matrix_sum_kernel(const int* A, const int* B, int* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}


void matrix_sum_acc(const int* A, const int* B, int* C, int rows, int cols) {
    int* d_A;
    int* d_B;
    int* d_C;
    size_t size = rows * cols * sizeof(int);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    matrix_sum_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}


void matrix_sum_multi(const int* A, const int* B, int* C, int rows, int cols) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    // Fallback to single GPU if less than 2 GPUs are available
    if (deviceCount < 2) {
        printf("Less than 2 GPUs detected. Falling back to single GPU execution.\n");
        matrix_sum_acc(A, B, C, rows, cols);
        return;
    }

    // Split work between GPUs
    int rows_per_gpu = rows / deviceCount;
    int remaining_rows = rows % deviceCount;

    // Arrays of device pointers (one per GPU)
    int** d_A_array = new int*[deviceCount];
    int** d_B_array = new int*[deviceCount];
    int** d_C_array = new int*[deviceCount];

    // Allocate memory and launch kernels on each GPU
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));

        // Calculate rows for this GPU (first GPU gets extra rows if any)
        int gpu_rows = rows_per_gpu + (gpu == 0 ? remaining_rows : 0);
        int row_offset = (gpu == 0) ? 0 : gpu * rows_per_gpu + remaining_rows;
        size_t gpu_size = gpu_rows * cols * sizeof(int);

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_A_array[gpu], gpu_size));
        CUDA_CHECK(cudaMalloc(&d_B_array[gpu], gpu_size));
        CUDA_CHECK(cudaMalloc(&d_C_array[gpu], gpu_size));

        // Copy input data to device
        CUDA_CHECK(cudaMemcpy(d_A_array[gpu], A + row_offset * cols, gpu_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_array[gpu], B + row_offset * cols, gpu_size, cudaMemcpyHostToDevice));

        // Configure and launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                      (gpu_rows + blockSize.y - 1) / blockSize.y);

        matrix_sum_kernel<<<gridSize, blockSize>>>(d_A_array[gpu], d_B_array[gpu],
                                                    d_C_array[gpu], gpu_rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }

    // Synchronize and copy results back from each GPU
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate rows and offset for this GPU
        int gpu_rows = rows_per_gpu + (gpu == 0 ? remaining_rows : 0);
        int row_offset = (gpu == 0) ? 0 : gpu * rows_per_gpu + remaining_rows;
        size_t gpu_size = gpu_rows * cols * sizeof(int);

        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(C + row_offset * cols, d_C_array[gpu], gpu_size, cudaMemcpyDeviceToHost));

        // Free device memory
        CUDA_CHECK(cudaFree(d_A_array[gpu]));
        CUDA_CHECK(cudaFree(d_B_array[gpu]));
        CUDA_CHECK(cudaFree(d_C_array[gpu]));
    }

    // Cleanup host arrays
    delete[] d_A_array;
    delete[] d_B_array;
    delete[] d_C_array;
}