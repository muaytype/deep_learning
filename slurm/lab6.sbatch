#!/bin/bash
#SBATCH --job-name=lab7
#SBATCH --output=logs/lab6_results_%j.out
#SBATCH --error=logs/lab6_error_%j.err
#SBATCH --partition=gpu3080             
#SBATCH --cpus-per-task=4                
#SBATCH --mem=8G                       
#SBATCH --gres=gpu:1                     
#SBATCH --time=00:10:00  

source $HOME/deep_learning/env_dl/bin/activate

set -e
export NVIDIA_TF32_OVERRIDE=0

python $HOME/deep_learning/src/lab6/lab6-1.py
python $HOME/deep_learning/src/lab6/lab6-2.py
python $HOME/deep_learning/src/lab6/lab6-resnet.py
