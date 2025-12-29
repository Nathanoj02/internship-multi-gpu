#include "load_balancing.hpp"
#include "matrix_utils.hpp"
#include "spmv.cuh"

#include <iostream>
#include <ctime>
#include <mpi.h>

#define WARMUP_RUNS 3
#define TIMED_RUNS 5

#define STREAM_NUM 4

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get process ID and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // test_load_balancing();
    // File
    std::string matrix_file = "data/m133-b3.mtx";

    // Clocks
    std::clock_t start, end;

    // Read matrix in CSR format
    size_t *row_offset, *cols;
    float *values;
    // Sizes
    size_t nnz, rows;

    read_mtx_dimensions(matrix_file, &rows, &nnz);

    // Allocate pinned memory for CSR arrays
    cudaMallocHost((void**)&row_offset, (rows + 1) * sizeof(size_t), cudaHostAllocDefault);
    cudaMallocHost((void**)&cols, nnz * sizeof(size_t), cudaHostAllocDefault);
    cudaMallocHost((void**)&values, nnz * sizeof(float), cudaHostAllocDefault);

    int read_success = read_mtx_csr(matrix_file, row_offset, cols, values);
    if (read_success != 0) {
        std::cerr << "Failed to read matrix file." << std::endl;
        return read_success;
    }
    
    // Generate input array and output array
    float *arr = (float *) malloc(rows * sizeof(float));
    if (rank == 0) {
        arr = generate_array(rows, 0.0f, 10.0f);
    }
    // Broadcast the whole array from Rank 0 to everyone
    MPI_Bcast(arr, rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float* result = new float[rows];

    // Generate load balancing plan
    size_t *row_mapping = balance_load(nnz, size, row_offset, rows);

    // GPU algorithm (only rank 0)
    if (rank == 0) {
        double gpu_time = 0.0;
    
        for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
            start = std::clock();
            spmv_multi_horizontal(result, arr, row_offset, cols, values, rows, nnz, row_mapping);
            end = std::clock();
    
            if (i >= 0) {
                gpu_time += double(end - start) / CLOCKS_PER_SEC;
            }
        }
    
        gpu_time /= TIMED_RUNS;
        std::cout << "GPU SpMV time: " << gpu_time << " seconds" << std::endl;
    }

    // Broadcast the result from Rank 0 to everyone
    MPI_Bcast(result, rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // MPI algorithm
    float *result_mpi = new float[rows];
    double mpi_time = 0.0;

    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        start = std::clock();
        spmv(result_mpi, arr, row_offset, cols, values, rows, nnz);
        end = std::clock();

        if (i >= 0)
            mpi_time += double(end - start) / CLOCKS_PER_SEC;
    }

    // Print time & Check correctness
    if (rank == 0) {
        mpi_time /= TIMED_RUNS;
        std::cout << "MPI SpMV time: " << mpi_time << " seconds" << std::endl;

        float max_error = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            float error = std::abs(result[i] - result_mpi[i]);
            if (error > max_error) {
                max_error = error;
            }
        }
        std::cout << "Maximum error between GPU SpMV and MPI SpMV: " << max_error << std::endl;
    }

    /*
    // GPU algorithm with streams
    double gpu_stream_time = 0.0;

    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        start = std::clock();
        spmv_streams(result, arr, row_offset, cols, values, rows, nnz, STREAM_NUM, row_mapping);
        end = std::clock();

        if (i >= 0) {
            gpu_stream_time += double(end - start) / CLOCKS_PER_SEC;
        }
    }

    gpu_stream_time /= TIMED_RUNS;
    std::cout << "GPU SpMV with streams time: " << gpu_stream_time << " seconds" << std::endl;

    // Check correctness (reference is GPU result without streams)
    float max_error = 0.0f;
    float* reference = new float[rows];
    spmv(reference, arr, row_offset, cols, values, rows, nnz);

    for (size_t i = 0; i < rows; ++i) {
        float error = std::abs(reference[i] - result[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Maximum error between GPU SpMV and GPU SpMV with streams: " << max_error << std::endl;
    delete[] reference;
    */

    // Clean up
    cudaFreeHost(row_offset);
    cudaFreeHost(cols);
    cudaFreeHost(values);
    delete[] arr;
    delete[] result;
    delete[] result_mpi;
    delete[] row_mapping;

    MPI_Finalize();
    return 0;
}