#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gemm.hpp"
#include "gemm.cuh"
#include <cmath>

#define M 1024
#define K 1024
#define N 1024
#define MIN_VALUE 0
#define MAX_VALUE 10

#define EPS 1e-2

void matrix_equals(float* A, float* B, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; i++) {
        if (fabs(A[i] - B[i]) > EPS) {
            printf("Validation failed at index %zu: %f != %f\n", i, A[i], B[i]);
            return;
        }
    }
    printf("Validation successful\n");
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Check that M is divisible by world_size
    if (M % world_size != 0) {
        if (world_rank == 0) {
            printf("Error: M must be divisible by number of processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    assign_gpu_device(world_rank);
    
    int chunk_m = M / world_size;
    
    // Each process needs: chunk_m rows of A (chunk_m x K) and entire B (K x N)
    float* local_A = (float*) malloc(chunk_m * K * sizeof(float));
    float* B = (float*) malloc(K * N * sizeof(float));
    float* local_C = (float*) malloc(chunk_m * N * sizeof(float));
    
    float* A = NULL;
    float* C = NULL;
    float* ground_truth = NULL;
    double ground_truth_time = 0;
    
    if (world_rank == 0) {
        srand((unsigned) time(NULL));
        A = (float*) malloc(M * K * sizeof(float));
        C = (float*) malloc(M * N * sizeof(float));
        generate_matrix(A, M, K, MIN_VALUE, MAX_VALUE);
        generate_matrix(B, K, N, MIN_VALUE, MAX_VALUE);
        
        // Compute ground truth
        ground_truth = (float*) malloc(M * N * sizeof(float));

        double start_time = MPI_Wtime();
        gemm_cpu(ground_truth, A, B, M, N, N, K);
        double end_time = MPI_Wtime();
        ground_truth_time = end_time - start_time;
    }
    
    // Broadcast entire B matrix to all processes
    MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Scatter rows of A to all processes
    MPI_Scatter(A, chunk_m * K, MPI_FLOAT, local_A, chunk_m * K, 
        MPI_FLOAT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    
    // Each process computes its chunk of C
    gemm_warp_tiling(local_C, local_A, B, chunk_m, N, N, K);
    
    // Gather all chunks of C back to rank 0
    MPI_Gather(local_C, chunk_m * N, MPI_FLOAT, C,
        chunk_m * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;

    // Get max time across all processes
    double max_time;
    MPI_Reduce(&local_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Validate on rank 0
    if (world_rank == 0) {
        printf("\nGround truth computation time: %f seconds\n", ground_truth_time);
        printf("Elapsed time: %f seconds\n", max_time);
        printf("Speedup: %.3fx\n", ground_truth_time / max_time);
        
        matrix_equals(C, ground_truth, M, N);
        free(A);
        free(C);
        free(ground_truth);
    }
    
    free(local_A);
    free(B);
    free(local_C);
    
    MPI_Finalize();
    return 0;
}