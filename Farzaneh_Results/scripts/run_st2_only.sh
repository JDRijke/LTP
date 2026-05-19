#!/usr/bin/env bash
# Submit only the 4 missing ST2 (fallacy_classification) jobs.
# ST1 submissions already exist — this targets only what's needed for the deadline.
#
# Usage (on Hábrók, from ~/touche2026/):
#   bash scripts/run_st2_only.sh

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

# Sub-Task 2 only (8-way fallacy classification)
submit fallacy_classification base      roberta-large                  12 16
submit fallacy_classification enhanced  roberta-large                  12 16
submit fallacy_classification base      answerdotai/ModernBERT-large   12 16
submit fallacy_classification enhanced  answerdotai/ModernBERT-large   12 16

echo
echo "Watch progress with:  squeue -u \$USER"
echo "Logs in:              runs/slurm_logs/"
echo
echo "When done, check results with:"
echo "  for d in runs/*fallacy_classification*/; do echo \"\$d\"; cat \"\${d}eval_summary.json\" 2>/dev/null; echo; done"
