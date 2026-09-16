#!/bin/bash
#SBATCH --job-name=lab4_gradients
#SBATCH --output=logs/lab4_results_%j.out
#SBATCH --error=logs/lab4_error_%j.err
#SBATCH --partition=gpu3080             
#SBATCH --cpus-per-task=4                
#SBATCH --mem=8G                       
#SBATCH --gres=gpu:1                     
#SBATCH --time=00:10:00  

source $HOME/deep_learning/env_dl/bin/activate

set -e

python $HOME/deep_learning/src/lab4_part1_1.py
python $HOME/deep_learning/src/lab4_part1_3.py
python $HOME/deep_learning/src/lab4_part2.py
