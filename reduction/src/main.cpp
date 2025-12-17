#include "array_ops.hpp"
#include "reduction.cuh"
#include <iostream>
#include <vector>
#include <time.h>

#define WARMUP_RUNS 3
#define TIMED_RUNS 5

int main() {
    int elems = 10000000;
    int min_value = 0;
    int max_value = 9;

    std::vector<int> a = generate_array(elems, min_value, max_value);

    clock_t start = clock();

    // Baseline CPU reduction
    int result = reduce_array(a, elems);

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

    if (result == result_gpu) {
        std::cout << "Reduction successful: " << result << std::endl;
    } else {
        std::cout << "Mismatch in reduction results: CPU = " << result 
                  << ", GPU = " << result_gpu << std::endl;
    }

    return 0;
}