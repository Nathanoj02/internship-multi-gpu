#include <mpi.h>
#include <stdio.h>

#define WARMUP_ITERATIONS 10
#define TIMED_ITERATIONS 100

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 2) {
        if (world_rank == 0) {
            fprintf(stderr, "This program requires exactly 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    double time_sum = 0.0;

    int tag = 0;
    char message[100];
    MPI_Status status;

    for (int i = -WARMUP_ITERATIONS; i < TIMED_ITERATIONS; i++) {
        if (world_rank == 0) {
            sprintf(message,"Greetings from process %d", world_rank);

            double start = MPI_Wtime();

            MPI_Send(message, 100, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(message, 100, MPI_CHAR, 1, tag, MPI_COMM_WORLD, &status);

            double end = MPI_Wtime();

            double latency = (end - start) / 2; // Divide by 2 for round-trip time
            if (i >= 0) {
                time_sum += latency;
            }
        } else {
            MPI_Recv(message, 100, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
            MPI_Send(message, 100, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
        }
    }

    if (world_rank == 0) {
        double avg_latency = time_sum / TIMED_ITERATIONS;
        printf("Average latency: %.2f μs\n", avg_latency * 1e6);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}