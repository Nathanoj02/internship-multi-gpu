#ifndef REDUCTION_CUH
#define REDUCTION_CUH

/**
 * Performs reduction (sum) of the elements in the array v
 * @param v Pointer to the input array (host memory)
 * @param elems Number of elements in the array
 * @return The sum of the elements in the array
 */
int reduce(const int* v, int elems);

/**
 * Performs reduction (sum) of the elements in the array v on multi-gpus
 * using host to perform the final reduction
 * @param v Pointer to the input array (host memory)
 * @param elems Number of elements in the array
 * @return The sum of the elements in the array
 */
int reduce_multi_cpu_mediated(const int* v, int elems);

/**
 * Performs reduction (sum) of the elements in the array v on multi-gpus
 * using MPI for communication and final reduction
 * @param v Pointer to the input array (host memory)
 * @param elems Number of elements in the array
 * @param rank MPI process rank
 * @return The sum of the elements in the array
 */
int reduce_multi_mpi(const int* v, int elems, int rank);

#endif // REDUCTION_CUH