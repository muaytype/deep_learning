#!/bin/bash
#SBATCH --job-name=lab5
#SBATCH --output=logs/lab5_results_%j.out
#SBATCH --error=logs/lab5_error_%j.err
#SBATCH --partition=gpu3080             
#SBATCH --cpus-per-task=4                
#SBATCH --mem=8G                       
#SBATCH --gres=gpu:1                     
#SBATCH --time=00:10:00  

source $HOME/deep_learning/env_dl/bin/activate

set -e

python $HOME/deep_learning/src/lab5-1.py
python $HOME/deep_learning/src/lab5-2.py
