#!/bin/bash
#SBATCH --job-name=prompt_llm
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=12GB
#SBATCH --output=slurm_logs/prompt_llm.%j.out
#SBATCH --error=slurm_logs/prompt_llm.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "=== Job Info ==="
echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname)"

#module --force purge
#
## ── Fill these in after running: module avail 2>&1 | grep -iE "cuda|pytorch" ──
#module --ignore_cache load CUDA/12.6.0         # <-- replace with exact name from module avail
#module --ignore_cache load Python/3.10.8-GCCcore-12.2.0  # <-- replace if needed
## ─────────────────────────────────────────────────────────────────────────────
#
#echo "Loaded modules:"
#module list

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Verify GPU is actually visible
nvidia-smi || echo "WARNING: nvidia-smi failed — GPU may not be allocated"

# Activate your venv (must have been created with the same Python/CUDA above)
source "/scratch/$USER/LTP/.venv/bin/activate"

echo "Python: $(which python3)"

# Quick sanity check before running the real script
python3 - <<'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

#models
#Qwen/Qwen3.5-4B
#Qwen/Qen3.5-9B
#meta-llama/Llama-3.2-3B-Instruct
#meta-llama/Meta-Llama-3.1-8B-Instruct
#google/gemma-4-E4B-it
#google/gemma-3-12b-it

### detection
# zero shot
python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_detection --variant base --tag base \
        --model meta-llama/Llama-3.2-3B-Instruct --mode zero_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal

python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_detection --variant enhanced --tag enhanced \
        --model meta-llama/Llama-3.2-3B-Instruct --mode zero_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal

# few shot
python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_detection --variant base --tag base \
        --model meta-llama/Llama-3.2-3B-Instruct --mode few_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal

python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_detection --variant enhanced --tag enhanced \
        --model meta-llama/Llama-3.2-3B-Instruct --mode few_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal



### classification
python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_classification --variant base --tag base \
        --model meta-llama/Llama-3.2-3B-Instruct --mode zero_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal

python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_classification --variant enhanced --tag enhanced \
        --model meta-llama/Llama-3.2-3B-Instruct --mode zero_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal

# few shot
python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_classification --variant base --tag base \
        --model meta-llama/Llama-3.2-3B-Instruct --mode few_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal

python3 scripts/prompt_llm_HF.py \
        --train_file ../touchefallacy_2026_train.jsonl --test_file ../touchefallacy_2026_test_task.jsonl \
        --task fallacy_classification --variant enhanced --tag enhanced \
        --model meta-llama/Llama-3.2-3B-Instruct --mode few_shot \
        --output llm_results/results.jsonl \
        --batch_size 12 \
        --test_internal
