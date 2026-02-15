#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 16
#define K 16
#define N 16
#define MIN_VALUE 0
#define MAX_VALUE 10

void generate_matrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % (MAX_VALUE - MIN_VALUE + 1) + MIN_VALUE;
    }
}

void gemm(int* A, int* B, int* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int p = 0; p < k; p++) {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

void matrix_equals(int* A, int* B, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (A[i] != B[i]) {
            printf("Validation failed at index %d: %d != %d\n", i, A[i], B[i]);
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
    
    int chunk_m = M / world_size;
    
    // Each process needs: chunk_m rows of A (chunk_m x K) and entire B (K x N)
    int* local_A = (int*) malloc(chunk_m * K * sizeof(int));
    int* B = (int*) malloc(K * N * sizeof(int));
    int* local_C = (int*) malloc(chunk_m * N * sizeof(int));
    
    int* A = NULL;
    int* C = NULL;
    int* ground_truth = NULL;
    
    if (world_rank == 0) {
        srand((unsigned) time(NULL));
        A = (int*) malloc(M * K * sizeof(int));
        C = (int*) malloc(M * N * sizeof(int));
        generate_matrix(A, M, K);
        generate_matrix(B, K, N);
        
        // Compute ground truth
        ground_truth = (int*) malloc(M * N * sizeof(int));
        gemm(A, B, ground_truth, M, N, K);
    }
    
    // Broadcast entire B matrix to all processes
    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter rows of A to all processes
    MPI_Scatter(A, chunk_m * K, MPI_INT, 
                local_A, chunk_m * K, MPI_INT, 
                0, MPI_COMM_WORLD);
    
    // Each process computes its chunk of C
    gemm(local_A, B, local_C, chunk_m, N, K);
    
    // Gather all chunks of C back to rank 0
    MPI_Gather(local_C, chunk_m * N, MPI_INT,
               C, chunk_m * N, MPI_INT,
               0, MPI_COMM_WORLD);
    
    // Validate on rank 0
    if (world_rank == 0) {
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