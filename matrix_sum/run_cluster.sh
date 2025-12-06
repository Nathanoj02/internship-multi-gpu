#!/bin/bash

#SBATCH --job-name=matrix_sum
#SBATCH --output=cluster/output_%j.out
#SBATCH --error=cluster/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load required modules
module load CUDA/12.5.0

# Compile
make

# Check GPU info
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Run executable
make run