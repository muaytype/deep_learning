#!/bin/bash
#SBATCH --job-name=lab2_part2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090          # Routes the job to the heavy-duty GPU room
#SBATCH --cpus-per-task=4            # Gives you 4 CPU cores to feed data to the GPU
#SBATCH --mem=8G                     # Allocates 8GB of standard system RAM
#SBATCH --gres=gpu:1                 # Requests exactly 1 physical GPU card
#SBATCH --time=00:10:00              # Safety switch: kills the job if it hangs past 10 minutes

set -euo pipefail

# 1. Activate your isolated Python environment
source $HOME/deep_learning/env_dl/bin/activate

# 2. Execute the precision testing script
python $HOME/deep_learning/src/lab2_part2.py
