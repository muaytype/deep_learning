#!/bin/bash
#SBATCH --job-name=alice_rnn
#SBATCH --output=logs/lab8_%j.out
#SBATCH --error=logs/lab8_%j.err
#SBATCH --partition=gpu3080            
#SBATCH --cpus-per-task=4          
#SBATCH --mem=8G                   
#SBATCH --gres=gpu:1              
#SBATCH --time=00:30:00 

source $HOME/deep_learning/env_dl/bin/activate

set -e

python $HOME/deep_learning/src/lab8.py
