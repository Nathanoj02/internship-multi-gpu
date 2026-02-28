#include "array_ops.hpp"
#include "reduction.cuh"
#include <iostream>
#include <vector>
#include <time.h>
#include <mpi.h>

#define WARMUP_RUNS 3
#define TIMED_RUNS 5

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elems = 10000000;
    int min_value = 0;
    int max_value = 9;

    // Reject weak workload distribution
    if (elems % size != 0) {
        if (rank == 0) {
            std::cerr << "Fatal Error: Array size (" << elems << ") is not perfectly divisible by MPI size (" << size << ")." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
        return -1;
    }

    int local_elems = elems / size;
    std::vector<int> a;
    std::vector<int> local_a(local_elems);

    // Rank 0 generates the full array
    if (rank == 0) {
        a = generate_array(elems, min_value, max_value);
    }

    // Distribute chunks
    MPI_Scatter(a.data(), local_elems, MPI_INT, local_a.data(), local_elems, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------
    // 1. CPU Reduction
    // ---------------------------------------------------------
    clock_t start_cpu = clock();
    int local_result_cpu = reduce_array(local_a, local_elems);
    clock_t end_cpu = clock();

    int global_result_cpu = 0;
    MPI_Reduce(&local_result_cpu, &global_result_cpu, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
        std::cout << "CPU Reduction Time (Local): " << cpu_time << " seconds" << std::endl;
    }

    // ---------------------------------------------------------
    // 2. Single GPU Reduction
    // ---------------------------------------------------------
    int local_result_gpu = 0;
    double gpu_time = 0.0;
    
    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        clock_t start = clock();
        local_result_gpu = reduce(local_a.data(), local_elems);
        clock_t end = clock();

        if (i >= 0) {
            gpu_time += double(end - start) / CLOCKS_PER_SEC;
        }
    }
    gpu_time /= TIMED_RUNS;

    int global_result_gpu = 0;
    MPI_Reduce(&local_result_gpu, &global_result_gpu, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "GPU Reduction Time (Local avg): " << gpu_time << " seconds" << std::endl;

        if (global_result_cpu != global_result_gpu) {
            std::cout << "Fatal Mismatch: CPU = " << global_result_cpu << ", GPU = " << global_result_gpu << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
    }

    // ---------------------------------------------------------
    // 3. Multi-GPU Reduction with CPU Mediation
    // ---------------------------------------------------------
    int local_result_multi_gpu = 0;
    double multi_gpu_time = 0.0;

    for (int i = -WARMUP_RUNS; i < TIMED_RUNS; ++i) {
        clock_t start = clock();
        local_result_multi_gpu = reduce_multi_cpu_mediated(local_a.data(), local_elems);
        clock_t end = clock();

        if (i >= 0) {
            multi_gpu_time += double(end - start) / CLOCKS_PER_SEC;
        }
    }
    multi_gpu_time /= TIMED_RUNS;

    int global_result_multi_gpu = 0;
    MPI_Reduce(&local_result_multi_gpu, &global_result_multi_gpu, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Multi-GPU Reduction Time (Local avg): " << multi_gpu_time << " seconds" << std::endl;

        if (global_result_cpu != global_result_multi_gpu) {
            std::cout << "Fatal Mismatch: CPU = " << global_result_cpu << ", Multi-GPU = " << global_result_multi_gpu << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
        
        std::cout << "All validations passed" << std::endl;
    }

    MPI_Finalize();
    return 0;
}