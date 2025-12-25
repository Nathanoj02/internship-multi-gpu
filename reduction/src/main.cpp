#include "array_ops.hpp"
#include "reduction.cuh"
#include <iostream>
#include <vector>
#include <time.h>
#include <mpi.h>

#define WARMUP_RUNS 3
#define TIMED_RUNS 5

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elems = 10000000;
    int min_value = 0;
    int max_value = 9;

    std::vector<int> a;
    if (rank == 0) {
        a = generate_array(elems, min_value, max_value);
    } else {
        a.resize(elems);
    }
    // Broadcast the whole array from Rank 0 to everyone
    MPI_Bcast(a.data(), elems, MPI_INT, 0, MPI_COMM_WORLD);

    int result = 0;

    // Only rank 0 performs non-MPI tests
    if (rank == 0) {
        clock_t start = clock();

        // Baseline CPU reduction
        result = reduce_array(a, elems);

        clock_t end = clock();
        double cpu_time = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "CPU Reduction Time: " << cpu_time << " seconds" << std::endl;

        
        // Single GPU reduction
        int result_gpu;
        double gpu_time = 0.0;
        
        for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
            start = clock();
            result_gpu = reduce(a.data(), elems);
            end = clock();

            if (i >= 0)
                gpu_time += double(end - start) / CLOCKS_PER_SEC;
        }

        gpu_time /= TIMED_RUNS;
        std::cout << "GPU Reduction Time: " << gpu_time << " seconds" << std::endl;

        if (result != result_gpu) {
            std::cout << "Mismatch in reduction results: CPU = " << result 
                      << ", GPU = " << result_gpu << std::endl;

            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }

        // Multi-GPU reduction with CPU mediation
        int result_multi_gpu;
        double multi_gpu_time = 0.0;

        for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
            start = clock();
            result_multi_gpu = reduce_multi_cpu_mediated(a.data(), elems);
            end = clock();

            if (i >= 0)
                multi_gpu_time += double(end - start) / CLOCKS_PER_SEC;
        }

        multi_gpu_time /= TIMED_RUNS;
        std::cout << "Multi-GPU Reduction Time: " << multi_gpu_time << " seconds" << std::endl;

        if (result != result_multi_gpu) {
            std::cout << "Mismatch in multi-GPU reduction results: CPU = " << result 
                      << ", Multi-GPU = " << result_multi_gpu << std::endl;

            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
    }

    // Broadcast the expected result to all processes
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All processes participate in MPI reduction
    int result_mpi;
    double mpi_time = 0.0;
    
    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        clock_t start = clock();
        result_mpi = reduce_multi_mpi(a.data(), elems, rank);
        clock_t end = clock();

        if (i >= 0 && rank == 0)
            mpi_time += double(end - start) / CLOCKS_PER_SEC;
    }

    if (rank == 0) {
        mpi_time /= TIMED_RUNS;
        std::cout << "MPI Multi-GPU Reduction Time: " << mpi_time << " seconds" << std::endl;

        if (result != result_mpi) {
            std::cout << "Mismatch in MPI reduction results: CPU = " << result 
                      << ", MPI = " << result_mpi << std::endl;

            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }

        std::cout << "All reduction results match: " << result << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}