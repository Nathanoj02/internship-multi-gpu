#!/bin/bash

#SBATCH --job-name=streams
#SBATCH --output=cluster/output_%j.out
#SBATCH --error=cluster/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load required modules
module load CUDA/12.5.0

# Compile
make

# Run executable
make run