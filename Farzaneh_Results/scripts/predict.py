"""Run a trained encoder on the official test set and emit a TIRA-format JSONL.

Usage:
    python -m src.predict \
        --model_dir runs/roberta-large_st1_base/best \
        --test_file data/test.jsonl \
        --task fallacy_detection \
        --variant base \
        --tag base \
        --output_file runs/roberta-large_st1_base/submission.jsonl \
        --system_description "RoBERTa-large fine-tuned on text_base"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.data import (
    SUBTASK1_LABELS,
    SUBTASK2_LABELS,
    build_text,
    load_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to saved best/ checkpoint.")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--task", choices=["fallacy_detection", "fallacy_classification"], required=True)
    ap.add_argument("--variant", choices=["base", "enhanced"], default="base")
    ap.add_argument("--include_argument", action="store_true")
    ap.add_argument("--tag", choices=["base", "enhanced"], required=True,
                    help="Submission tag — must match the input variant the model was trained on.")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--system_description", default="")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    rows = load_jsonl(args.test_file)
    texts = [build_text(r, args.variant, args.include_argument) for r in rows]
    ids   = [r["id"] for r in rows]

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    labels = SUBTASK1_LABELS if args.task == "fallacy_detection" else SUBTASK2_LABELS

    # Tokenize all and batch
    enc = tok(texts, truncation=True, max_length=args.max_length, padding=False)
    samples = [
        {k: torch.tensor(enc[k][i]) for k in enc}
        for i in range(len(ids))
    ]
    collate = DataCollatorWithPadding(tok)
    loader = DataLoader(samples, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

    pred_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            pred_ids.extend(np.argmax(logits, axis=-1).tolist())

    id2label = {i: l for i, l in enumerate(labels)}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for _id, pid in zip(ids, pred_ids):
            rec = {
                "task": args.task,
                "id": _id,
                "label": id2label[int(pid)],
                "tag": args.tag,
                "system_description": args.system_description,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(ids)} predictions → {out_path}")


if __name__ == "__main__":
    main()
