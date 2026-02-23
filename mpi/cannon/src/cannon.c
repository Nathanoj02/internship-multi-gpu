#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define M 256
#define K 256
#define N 256
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

double cannon_gemm(int *local_A, int *local_B, int *local_C, int block_size, int sqrt_p, MPI_Comm cart_comm) {
    // Initialize local_C to zero
    for (int i = 0; i < block_size * block_size; i++) local_C[i] = 0;
    
    double start_time = MPI_Wtime();
    
    int coords[2];
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    
    // Initial alignment
    // Left circular shift of A blocks by row index
    int dest, source;
    MPI_Cart_shift(cart_comm, 1, -my_row, &source, &dest);
    int* temp_A = (int*) malloc(block_size * block_size * sizeof(int));
    MPI_Sendrecv(local_A, block_size * block_size, MPI_INT, dest, 0,
                 temp_A, block_size * block_size, MPI_INT, source, 0,
                 cart_comm, MPI_STATUS_IGNORE);
    for (int i = 0; i < block_size * block_size; i++) local_A[i] = temp_A[i];
    
    // Upward circular shift of B blocks by column index
    MPI_Cart_shift(cart_comm, 0, -my_col, &source, &dest);
    int* temp_B = (int*) malloc(block_size * block_size * sizeof(int));
    MPI_Sendrecv(local_B, block_size * block_size, MPI_INT, dest, 0,
                 temp_B, block_size * block_size, MPI_INT, source, 0,
                 cart_comm, MPI_STATUS_IGNORE);
    for (int i = 0; i < block_size * block_size; i++) local_B[i] = temp_B[i];
    
    // Main computation loop
    for (int step = 0; step < sqrt_p; step++) {
        // Compute local matrix multiplication
        gemm(local_A, local_B, local_C, block_size, block_size, block_size);
        
        if (step < sqrt_p - 1) {  // Don't shift after last iteration
            // Left circular shift of A by 1
            MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
            MPI_Sendrecv(local_A, block_size * block_size, MPI_INT, dest, 0,
                         temp_A, block_size * block_size, MPI_INT, source, 0,
                         cart_comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < block_size * block_size; i++) local_A[i] = temp_A[i];
            
            // Upward circular shift of B by 1
            MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
            MPI_Sendrecv(local_B, block_size * block_size, MPI_INT, dest, 0,
                         temp_B, block_size * block_size, MPI_INT, source, 0,
                         cart_comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < block_size * block_size; i++) local_B[i] = temp_B[i];
        }
    }
    
    free(temp_A);
    free(temp_B);
    
    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    
    double max_time;
    MPI_Reduce(&local_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    return max_time;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Check that world_size is a perfect square
    int sqrt_p = (int)sqrt(world_size);
    if (sqrt_p * sqrt_p != world_size) {
        if (world_rank == 0) {
            printf("Error: Number of processes must be a perfect square\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Check that M, K, N are equal and divisible by sqrt_p
    if (M != K || K != N || M % sqrt_p != 0) {
        if (world_rank == 0) {
            printf("Error: M, K, N must be equal and divisible by sqrt(num_processes)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int block_size = M / sqrt_p;
    
    // Create 2D Cartesian topology
    MPI_Comm cart_comm;
    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {1, 1};  // Wraparound in both dimensions
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    
    int* local_A = (int*) malloc(block_size * block_size * sizeof(int));
    int* local_B = (int*) malloc(block_size * block_size * sizeof(int));
    int* local_C = (int*) malloc(block_size * block_size * sizeof(int));
    
    int* A = NULL;
    int* B = NULL;
    int* C = NULL;
    int* ground_truth = NULL;
    double ground_truth_time = 0;
    
    if (world_rank == 0) {
        srand((unsigned) time(NULL));
        A = (int*) malloc(M * K * sizeof(int));
        B = (int*) malloc(K * N * sizeof(int));
        C = (int*) malloc(M * N * sizeof(int));
        generate_matrix(A, M, K);
        generate_matrix(B, K, N);
        
        // Compute ground truth
        ground_truth = (int*) malloc(M * N * sizeof(int));
        for (int i = 0; i < M * N; i++) ground_truth[i] = 0;

        double start_time = MPI_Wtime();
        gemm(A, B, ground_truth, M, N, K);
        double end_time = MPI_Wtime();
        ground_truth_time = end_time - start_time;
    }
    
    // Distribute blocks of A and B to processes
    int* sendbuf_A = NULL;
    int* sendbuf_B = NULL;
    if (world_rank == 0) {
        sendbuf_A = (int*) malloc(M * K * sizeof(int));
        sendbuf_B = (int*) malloc(K * N * sizeof(int));
        
        // Rearrange A and B into blocks for distribution
        for (int proc = 0; proc < world_size; proc++) {
            int proc_row = proc / sqrt_p;
            int proc_col = proc % sqrt_p;
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    sendbuf_A[proc * block_size * block_size + i * block_size + j] = 
                        A[(proc_row * block_size + i) * K + (proc_col * block_size + j)];
                    sendbuf_B[proc * block_size * block_size + i * block_size + j] = 
                        B[(proc_row * block_size + i) * N + (proc_col * block_size + j)];
                }
            }
        }
    }
    
    MPI_Scatter(sendbuf_A, block_size * block_size, MPI_INT, local_A, 
                block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(sendbuf_B, block_size * block_size, MPI_INT, local_B, 
                block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    double max_time = cannon_gemm(local_A, local_B, local_C, block_size, sqrt_p, cart_comm);
    
    // Gather results
    int* recvbuf = NULL;
    if (world_rank == 0) {
        recvbuf = (int*) malloc(M * N * sizeof(int));
    }
    MPI_Gather(local_C, block_size * block_size, MPI_INT, recvbuf, 
               block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Rearrange blocks back to matrix form
    if (world_rank == 0) {
        for (int proc = 0; proc < world_size; proc++) {
            int proc_row = proc / sqrt_p;
            int proc_col = proc % sqrt_p;
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    C[(proc_row * block_size + i) * N + (proc_col * block_size + j)] = 
                        recvbuf[proc * block_size * block_size + i * block_size + j];
                }
            }
        }
        
        printf("\nGround truth computation time: %f seconds\n", ground_truth_time);
        printf("Cannon's algorithm time: %f seconds\n", max_time);
        printf("Speedup: %.3fx\n", ground_truth_time / max_time);
        
        matrix_equals(C, ground_truth, M, N);
        
        free(A);
        free(B);
        free(C);
        free(ground_truth);
        free(sendbuf_A);
        free(sendbuf_B);
        free(recvbuf);
    }
    
    free(local_A);
    free(local_B);
    free(local_C);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}