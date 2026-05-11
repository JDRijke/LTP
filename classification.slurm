#!/bin/bash
#SBATCH --job-name=fallacy_multiclass
#SBATCH --output=logs/fallacy_multiclass_%j.out
#SBATCH --error=logs/fallacy_multiclass_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Optional email notifications
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@student.rug.nl

echo "========================================="
echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "GPU(s):"
nvidia-smi
echo "========================================="

# Clean environment
module purge

# Load modules
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Move to your project directory
cd /home2/s5356865/Documents/LTP

# Activate your virtual environment
source venv/bin/activate

# Create logs folder if needed
mkdir -p logs

# HuggingFace cache
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Reduce tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Optional: improve PyTorch CUDA allocation behavior
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training script..."
python /home2/s5356865/Documents/LTP/finetune_fallacy_type.py

echo "========================================="
echo "Job finished on $(date)"
echo "========================================="