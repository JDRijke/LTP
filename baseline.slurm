#!/bin/bash
#SBATCH --job-name=baseline_roberta
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

echo "Job started on $(date)"
echo "Running on node: $(hostname)"

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Move to your project directory
cd /home2/s5356865/Documents/LTP

# Activate your virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Prevent tokenizer multiprocessing warnings
export TOKENIZERS_PARALLELISM=false

# Run training
python baseline.py

echo "Job finished on $(date)"