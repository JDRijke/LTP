#!/usr/bin/env bash
# Launch the planned encoder sweep on Hábrók.
# Submits 8 jobs: {RoBERTa-large, ModernBERT-large} × {ST1, ST2} × {base, enhanced}
# Once they're done, runs the matching predict job for each.

set -euo pipefail
mkdir -p runs/slurm_logs

submit() {
    local task=$1 variant=$2 model=$3 epochs=$4 batch=$5
    local run_name="${model//\//_}_${task}_${variant}_seed42"
    echo "Submitting train: $run_name"
    train_jid=$(sbatch --parsable \
        --export=ALL,TASK=$task,VARIANT=$variant,MODEL=$model,EPOCHS=$epochs,BATCH=$batch,RUN_NAME=$run_name \
        scripts/train_encoder.sbatch)
    echo "  train job id: $train_jid"
    pred_jid=$(sbatch --parsable --dependency=afterok:$train_jid \
        --export=ALL,RUN_NAME=$run_name,TASK=$task,VARIANT=$variant,TAG=$variant,SYS_DESC="$run_name" \
        scripts/predict_encoder.sbatch)
    echo "  pred  job id: $pred_jid"
}

# Sub-Task 1 (binary)
submit fallacy_detection      base      roberta-large                   8 16
submit fallacy_detection      enhanced  roberta-large                   8 16
submit fallacy_detection      base      answerdotai/ModernBERT-large    8 16
submit fallacy_detection      enhanced  answerdotai/ModernBERT-large    8 16

# Sub-Task 2 (8-way)
submit fallacy_classification base      roberta-large                  12 16
submit fallacy_classification enhanced  roberta-large                  12 16
submit fallacy_classification base      answerdotai/ModernBERT-large   12 16
submit fallacy_classification enhanced  answerdotai/ModernBERT-large   12 16

echo
echo "Watch progress with:  squeue -u \$USER"
echo "Logs in:              runs/slurm_logs/"
