#!/bin/bash
#SBATCH --job-name=lab2_part1
#SBATCH --output=logs/%j.out             # Routes sandard prints to a log file
#SBATCH --error=logs/%j.err              # Routes crashes to an error file
#SBATCH --partition=gpu4090              # Requests the specific GPU partition
#SBATCH --cpus-per-task=4                # Allocates 4 CPU cores
#SBATCH --mem=8G                         # Allocates 8GB of system RAM
#SBATCH --gres=gpu:1                     # Requests exactly 1 physical GPU
#SBATCH --time=00:10:00                  # Safety kill switch set to 10 minutes

set -euo pipefail

# 1. Activate your isolated environment so the node has PyTorch
source $HOME/deep_learning/env_dl/bin/activate

# 2. Safely execute the Python script on the compute node
python $HOME/deep_learning/src/lab2_part1.py
