#!/bin/bash
#SBATCH --job-name=tensor_lab          # Names the job in the queue so you can find it
#SBATCH --output=logs/tensor_%j.log    # Saves your "print()" statements to a text file (where %j is the unique Job ID)
#SBATCH --error=logs/error_%j.log      # Saves any Python crash reports to a separate file
#SBATCH --partition=gpu3060            # The specific queue you are targeting (e.g., nodes with RTX 3060s)
#SBATCH --gres=gpu:1                   # Hardware request: strictly allocate 1 physical GPU
#SBATCH --time=00:10:00                # Kill the job if it runs longer than 10 minutes (prevents stalled code from hoarding GPUs)

# 1. Activate your isolated environment on the cluster
source $HOME/Lab/env_dl/bin/activate

# 2. Execute the python script
python $HOME/Lab/tensor_test.py