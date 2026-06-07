"""Data loading, label normalization, and stratified splits for touchefallacy_2026.

Two tasks supported:
  - fallacy_detection       (binary)
  - fallacy_classification  (8-way, only fallacious entries)

Two input variants per task:
  - base       : uses text_base   (+ optional argument_base claim/supports)
  - enhanced   : uses text_enhanced (+ optional argument_enhanced)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

SUBTASK1_LABELS = ["non-fallacy", "fallacy"]
SUBTASK2_LABELS = [
    "authority",
    "black-white",
    "hasty_generalization",
    "natural",
    "population",
    "slippery_slope",
    "tradition",
    "worse_problems",
]

# The dataset uses `blackwhite`; the submission spec uses `black-white`.
DATASET_TO_SUB = {"blackwhite": "black-white"}
SUB_TO_DATASET = {v: k for k, v in DATASET_TO_SUB.items()}


def normalize_label(task: str, raw):
    """Map a raw dataset value to the canonical submission label."""
    if task == "fallacy_detection":
        return SUBTASK1_LABELS[int(raw)]
    if task == "fallacy_classification":
        if raw is None:
            return None
        return DATASET_TO_SUB.get(raw, raw)
    raise ValueError(task)


def label2id(task: str) -> dict[str, int]:
    labels = SUBTASK1_LABELS if task == "fallacy_detection" else SUBTASK2_LABELS
    return {lab: i for i, lab in enumerate(labels)}


def id2label(task: str) -> dict[int, str]:
    return {i: lab for lab, i in label2id(task).items()}


# ---------------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------------

@dataclass
class Example:
    id: str
    text: str
    label: str | None  # canonical label, or None for unlabeled test rows
    raw: dict          # full original row, kept for diagnostics


def build_text(row: dict, variant: str, include_argument: bool) -> str:
    """Build the input text for a row.

    variant:        'base' or 'enhanced'
    include_argument: if True, append claim + supports from argument_{variant}
    """
    if variant == "base":
        text = row["text_base"]
        arg = row.get("argument_base", {})
    elif variant == "enhanced":
        text = row["text_enhanced"]
        arg = row.get("argument_enhanced", {})
    else:
        raise ValueError(f"variant must be base or enhanced, got {variant}")

    if not include_argument:
        return text

    claim = (arg or {}).get("claim", "")
    supports = (arg or {}).get("supports", []) or []
    sup_text = " ".join(f"- {s}" for s in supports)
    return (
        f"{text}\n\n"
        f"[Claim] {claim}\n"
        f"[Supports] {sup_text}".strip()
    )


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def rows_to_examples(
    rows: Iterable[dict],
    task: str,
    variant: str,
    include_argument: bool,
    *,
    only_fallacious_for_subtask2: bool = True,
) -> list[Example]:
    examples = []
    for r in rows:
        if task == "fallacy_classification" and only_fallacious_for_subtask2:
            if int(r.get("fallacy_exists", 0)) != 1:
                continue
        raw_label = r.get("fallacy_exists") if task == "fallacy_detection" else r.get("fallacy_type")
        label = normalize_label(task, raw_label) if raw_label is not None else None
        examples.append(
            Example(
                id=r["id"],
                text=build_text(r, variant, include_argument),
                label=label,
                raw=r,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    examples: list[Example],
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> tuple[list[Example], list[Example], list[Example]]:
    """80/10/10 stratified split on label."""
    labels = [e.label for e in examples]
    assert all(l is not None for l in labels), "Cannot stratify with unlabeled data"

    train_val, test = train_test_split(
        examples, test_size=test_size, stratify=labels, random_state=seed
    )
    rel_val = val_size / (1 - test_size)
    train_labels = [e.label for e in train_val]
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_labels, random_state=seed
    )
    return train, val, test


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Inspect splits / label distributions.")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--task", choices=["fallacy_detection", "fallacy_classification"], required=True)
    ap.add_argument("--variant", choices=["base", "enhanced"], default="base")
    ap.add_argument("--include_argument", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.train_file)
    examples = rows_to_examples(rows, args.task, args.variant, args.include_argument)
    print(f"Total examples for {args.task} / {args.variant}: {len(examples)}")
    from collections import Counter
    print("Label distribution:", Counter(e.label for e in examples))

    train, val, test = stratified_split(examples, seed=args.seed)
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print("  train labels:", Counter(e.label for e in train))
    print("  val   labels:", Counter(e.label for e in val))
    print("  test  labels:", Counter(e.label for e in test))


if __name__ == "__main__":
    main()
