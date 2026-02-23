# Internship Multi-GPU
Repo to manage exercises for my internship at University of Trento on Multi-GPU architectures in CUDA

## First part - GEMM
In folder [`gemm/`](./gemm/)

Compile:
```sh
make
```

Test:
```sh
make run
```

Benchmark:
```sh
make bench
```

Run on cluster:
```sh
sbatch run_cluster.sh
```

Use NCU:
```sh
ncu --set full --target-processes all -f -o gemm_profile ./bin/profile_tensor
```

## Second part - MPI
In folder [`mpi/`](./mpi/)

## Pre-Exercises
In folder [`exercises/`](./exercises/)