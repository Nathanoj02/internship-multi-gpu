#!/bin/bash

#SBATCH --job-name=spmv
#SBATCH --output=cluster/output_%j.out
#SBATCH --error=cluster/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1

# Load required modules
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

# Compile
make

# Run executable
make run