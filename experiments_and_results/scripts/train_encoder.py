"""Fine-tune an encoder-only LM (RoBERTa, DeBERTa, ModernBERT, ...) for either:
  - fallacy_detection      (Sub-Task 1, binary)
  - fallacy_classification (Sub-Task 2, 8-way, only fallacious entries)

Usage example (Hábrók A100):
    python -m src.train_encoder \
        --train_file data/train.jsonl \
        --task fallacy_detection \
        --variant base \
        --model roberta-large \
        --epochs 6 --batch_size 16 --lr 2e-5 \
        --output_dir runs/roberta-large_st1_base

Outputs:
  - HF checkpoint in --output_dir/best/
  - Predictions on the held-out (internal) test split → preds.jsonl
  - eval_summary.json with macro/weighted/per-class F1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import (
    Example,
    SUBTASK1_LABELS,
    SUBTASK2_LABELS,
    label2id,
    load_jsonl,
    rows_to_examples,
    stratified_split,
)


# ---------------------------------------------------------------------------

class EncoderDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int, l2i: dict[str, int]):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length
        self.l2i = l2i

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tok(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if ex.label is not None:
            item["labels"] = torch.tensor(self.l2i[ex.label], dtype=torch.long)
        return item


def make_compute_metrics(labels: list[str]):
    def compute(eval_pred):
        logits, gold = eval_pred
        pred = np.argmax(logits, axis=-1)
        macro = f1_score(gold, pred, average="macro")
        weighted = f1_score(gold, pred, average="weighted")
        p, r, f, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
        return {
            "macro_f1": macro,
            "weighted_f1": weighted,
            "macro_precision": p,
            "macro_recall": r,
        }
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--task", choices=["fallacy_detection", "fallacy_classification"], required=True)
    ap.add_argument("--variant", choices=["base", "enhanced"], default="base")
    ap.add_argument("--include_argument", action="store_true",
                    help="Append argument_{variant} (claim + supports) to the text input.")
    ap.add_argument("--model", default="roberta-large")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on A100).")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument(
        "--hf_output_dir",
        default=None,
        help=(
            "Directory for HuggingFace Trainer intermediate checkpoints. "
            "Defaults to <output_dir>/hf. Override to a scratch path (e.g. "
            "/scratch/$USER/touche_ckpts/<run>) to avoid filling your home quota."
        ),
    )
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hf_out = Path(args.hf_output_dir) if args.hf_output_dir else (out / "hf")
    hf_out.mkdir(parents=True, exist_ok=True)

    # ----- data
    rows = load_jsonl(args.train_file)
    examples = rows_to_examples(rows, args.task, args.variant, args.include_argument)
    train_ex, val_ex, eval_ex = stratified_split(examples, seed=args.seed)
    print(f"Splits — train={len(train_ex)} val={len(val_ex)} internal-test={len(eval_ex)}")

    labels = SUBTASK1_LABELS if args.task == "fallacy_detection" else SUBTASK2_LABELS
    l2i = label2id(args.task)
    i2l = {i: l for l, i in l2i.items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = EncoderDataset(train_ex, tok, args.max_length, l2i)
    val_ds   = EncoderDataset(val_ex,   tok, args.max_length, l2i)
    eval_ds  = EncoderDataset(eval_ex,  tok, args.max_length, l2i)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=i2l,
        label2id=l2i,
    )

    targs = TrainingArguments(
        output_dir=str(hf_out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=make_compute_metrics(labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    # ----- save best model
    best_dir = out / "best"
    trainer.save_model(str(best_dir))
    tok.save_pretrained(str(best_dir))

    # ----- evaluate on internal test split
    pred_out = trainer.predict(eval_ds)
    logits = pred_out.predictions
    pred_ids = np.argmax(logits, axis=-1)
    gold_ids = pred_out.label_ids
    report = classification_report(
        gold_ids, pred_ids, target_names=labels, digits=4, zero_division=0
    )
    print("\n=== Internal test set report ===\n")
    print(report)

    summary = {
        "task": args.task,
        "variant": args.variant,
        "include_argument": args.include_argument,
        "model": args.model,
        "macro_f1": float(f1_score(gold_ids, pred_ids, average="macro")),
        "weighted_f1": float(f1_score(gold_ids, pred_ids, average="weighted")),
        "n_eval": len(eval_ex),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    # ----- dump internal-test predictions for debugging
    with (out / "preds.jsonl").open("w") as f:
        for ex, pid in zip(eval_ex, pred_ids):
            f.write(json.dumps({
                "id": ex.id,
                "gold": ex.label,
                "pred": i2l[int(pid)],
                "task": args.task,
                "variant": args.variant,
            }) + "\n")

    print("\nDone. Best model →", best_dir)
    print("Eval summary →", out / "eval_summary.json")


if __name__ == "__main__":
    main()
