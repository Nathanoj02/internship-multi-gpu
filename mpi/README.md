# MPI exercises
For every exercise:
Compile:
```sh
make
```

Test:
```sh
make run
```

Run on cluster (when available):
```sh
sbatch run_cluster.sh
```

1. [Hello world](./hello_world/)
2. [Latency calculation](./latency/)
3. [Array reduction](./reduction/)
4. [GEMM on host](./gemm/)
5. [Cannon's algorithm](./cannon/)
6. [GPU array reduction](./gpu_reduction/)

## GEMM on GPU
7. [GEMM on GPU](./gpu_gemm/)

Choose number of processes (can oversubscribe)
```sh
make run NP=8
```