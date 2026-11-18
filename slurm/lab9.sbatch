#!/bin/bash
#SBATCH --job-name=lab9
#SBATCH --output=logs/lab9_results_%j.out
#SBATCH --error=logs/lab9_error_%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

# 1. Activate your isolated Python sandbox
source $HOME/deep_learning/src/lab9/venv_lab9/bin/activate

# 2. Enforce strict error handling (abort on first failure)
set -e

echo "--- Starting Architecture Validation ---"

echo "1. Testing Scaled Dot-Product & Multi-Head Attention..."
python -u $HOME/deep_learning/src/lab9/lab9_attention.py

echo "2. Testing Positional Encoding & Encoder Block..."
python -u $HOME/deep_learning/src/lab9/lab9_encoder.py

echo "3. Testing Cross-Attention & Decoder Block..."
python -u $HOME/deep_learning/src/lab9/lab9_decoder.py

echo "4. Testing Unified Seq2Seq Transformer & Generator..."
python -u $HOME/deep_learning/src/lab9/lab9_transformer.py

echo "--- Architecture Verified. Initiating Model Training ---"

echo "5. Training on CNN/DailyMail Dataset..."
python -u $HOME/deep_learning/src/lab9/lab9_train.py

echo "6. Generating Autoregressive Summaries..."

python -u $HOME/deep_learning/src/lab9/lab9_inference.py

echo "--- Pipeline Execution Complete ---"

