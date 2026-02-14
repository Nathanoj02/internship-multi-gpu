#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 1000000
#define MIN_VALUE 0
#define MAX_VALUE 10

void generate_array(int* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % (MAX_VALUE - MIN_VALUE + 1) + MIN_VALUE;
    }
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int chunk_size = ARRAY_SIZE / world_size;
    int* local_array = (int *) malloc(chunk_size * sizeof(int));

    int ground_truth_sum = 0;
    
    // Host 0 generates array and communicates to everyone else
    if (world_rank == 0) {
        srand((unsigned) time(NULL));
        int* array = (int *) malloc(ARRAY_SIZE * sizeof(int));
        generate_array(array, ARRAY_SIZE);

        // Perform ground truth sum for validation
        for (int i = 0; i < ARRAY_SIZE; i++) {
            ground_truth_sum += array[i];
        }

        // Copy rank 0's chunk
        for (int i = 0; i < chunk_size; i++) {
            local_array[i] = array[i];
        }

        // Send chunks to other processes
        for (int i = 1; i < world_size; i++) {
            MPI_Send(array + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        free(array);
    } else {
        // Receive chunk from host 0
        MPI_Recv(local_array, chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // All processes compute local sum
    int local_sum = 0;
    for (int i = 0; i < chunk_size; i++) {
        local_sum += local_array[i];
    }

    // Reduce local sums
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Total sum: %d\n", global_sum);
        
        if (global_sum == ground_truth_sum) {
            printf("Validation successful\n");
        } else {
            printf("Validation failed\n");
        }
    }

    free(local_array);

    // Finalize the MPI environment
    MPI_Finalize();
}