# Internship Multi-GPU
Repo to manage exercises for my internship at University of Trento on Multi-GPU architectures in CUDA

## Exercises
1. [Matrix Sum](#matrix-sum): Matrix sum on multi-GPU
2. [Reduction](#reduction): Reduction on multi-GPU using MPI
3. [Streams](#streams): Array sum using cudaStreams for overlapping jobs

### Matrix Sum
In folder `matrix_sum/`

Compile:
```sh
make
```

Run:
```sh
make run
```

Run on cluster:
```sh
sbatch run_cluster.sh
```

### Reduction
In folder `reduction/`

Compile:
```sh
make
```

Run:
```sh
make run
```

Run on cluster:
```sh
sbatch run_cluster.sh
```

### Streams
In folder `streams/`

Compile:
```sh
make
```

Run:
```sh
make run
```

Run on cluster:
```sh
sbatch run_cluster.sh
```