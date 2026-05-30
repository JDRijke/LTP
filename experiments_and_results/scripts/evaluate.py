"""Compute macro-F1 / per-class metrics on a predictions JSONL when gold labels are known.

Use this to compare runs against your internal 10% test split, or against a held-out validation
set the organizers may release later.

    python -m src.evaluate \
        --pred_file runs/roberta-large_st1_base/preds.jsonl \
        --gold_file data/train.jsonl \
        --task fallacy_detection
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from sklearn.metrics import classification_report, f1_score

from src.data import SUBTASK1_LABELS, SUBTASK2_LABELS, normalize_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gold_file", required=True, help="Original JSONL with gold labels.")
    ap.add_argument("--task", choices=["fallacy_detection", "fallacy_classification"], required=True)
    args = ap.parse_args()

    gold_rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.gold_file)}
    preds = [json.loads(l) for l in open(args.pred_file)]

    y_true, y_pred = [], []
    skipped = 0
    for p in preds:
        gold = gold_rows.get(p["id"])
        if gold is None:
            skipped += 1
            continue
        if args.task == "fallacy_classification" and int(gold.get("fallacy_exists", 0)) != 1:
            continue  # scored only on truly fallacious
        gold_lab = normalize_label(
            args.task,
            gold["fallacy_exists"] if args.task == "fallacy_detection" else gold["fallacy_type"],
        )
        if gold_lab is None:
            continue
        # accept either {gold,pred} or {label}
        pred_lab = p.get("pred", p.get("label"))
        y_true.append(gold_lab)
        y_pred.append(pred_lab)

    labels = SUBTASK1_LABELS if args.task == "fallacy_detection" else SUBTASK2_LABELS
    print(f"Scored {len(y_true)} examples (skipped {skipped}).")
    print(f"Pred distribution : {Counter(y_pred)}")
    print(f"Gold distribution : {Counter(y_true)}")
    print()
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    print(f"Macro F1   : {f1_score(y_true, y_pred, labels=labels, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, labels=labels, average='weighted'):.4f}")


if __name__ == "__main__":
    main()
