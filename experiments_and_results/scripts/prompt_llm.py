"""Prompt a local decoder LLM (via vLLM) for Sub-Task 1 / 2 with zero-shot, few-shot, or CoT.

Designed for Hábrók A100. Examples:

    # Zero-shot Sub-Task 2 with Qwen-2.5-14B-Instruct
    python -m src.prompt_llm \
        --train_file data/train.jsonl --test_file data/test.jsonl \
        --task fallacy_classification --variant base --tag base \
        --model Qwen/Qwen2.5-14B-Instruct --mode zero_shot \
        --output_file runs/qwen25_14b_st2_base_zeroshot.jsonl

    # Few-shot (k=2 per class) Sub-Task 1 with LLaMA-3.1-8B-Instruct
    python -m src.prompt_llm \
        --train_file data/train.jsonl --test_file data/test.jsonl \
        --task fallacy_detection --variant base --tag base \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct --mode few_shot --k_per_class 2 \
        --output_file runs/llama31_8b_st1_base_fewshot.jsonl

    # CoT
    python -m src.prompt_llm ... --mode cot ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from src.data import (
    SUBTASK1_LABELS,
    SUBTASK2_LABELS,
    build_text,
    load_jsonl,
    rows_to_examples,
    stratified_split,
)
from src.prompts import (
    SUBTASK1_SYSTEM,
    SUBTASK2_SYSTEM,
    build_fewshot_block,
    subtask1_prompt,
    subtask2_prompt,
)


# ---------------------------------------------------------------------------

def select_fewshot(train_examples, k_per_class: int, seed: int):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for e in train_examples:
        by_class[e.label].append(e)
    out = []
    for lbl, exs in by_class.items():
        rng.shuffle(exs)
        out.extend(exs[:k_per_class])
    rng.shuffle(out)
    return out


def parse_answer(generation: str, labels: list[str]) -> str:
    """Parse LLM output, looking for `ANSWER: <label>` then fall back to label match.

    Handles common surface variants:
      - hyphen / underscore confusion (`black-white` vs `black_white`)
      - the dataset's `blackwhite` spelling vs the submission's `black-white`
    """
    def norm(s):
        return s.lower().replace("-", "").replace("_", "").replace(" ", "")

    m = re.search(r"ANSWER\s*:\s*([A-Za-z_\-]+)", generation, flags=re.IGNORECASE)
    if m:
        cand = norm(m.group(1).strip())
        for l in labels:
            if norm(l) == cand:
                return l
    # fallback: any normalized label that appears in the generation
    g_norm = norm(generation)
    for l in labels:
        if norm(l) in g_norm:
            return l
    return labels[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--test_file", required=True, help="Official test JSONL (no labels).")
    ap.add_argument("--task", choices=["fallacy_detection", "fallacy_classification"], required=True)
    ap.add_argument("--variant", choices=["base", "enhanced"], default="base")
    ap.add_argument("--include_argument", action="store_true")
    ap.add_argument("--tag", choices=["base", "enhanced"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["zero_shot", "few_shot", "cot"], default="zero_shot")
    ap.add_argument("--k_per_class", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--system_description", default="")
    args = ap.parse_args()

    # ----- few-shot exemplars from train split
    rows = load_jsonl(args.train_file)
    train_ex_full = rows_to_examples(rows, args.task, args.variant, args.include_argument)
    train_split, _, _ = stratified_split(train_ex_full, seed=args.seed)

    fs_examples = []
    if args.mode == "few_shot":
        fs_examples = select_fewshot(train_split, args.k_per_class, args.seed)
        print(f"Few-shot exemplars: {len(fs_examples)}")

    fs_block = build_fewshot_block(
        args.task, [{"text": e.text, "label": e.label} for e in fs_examples]
    ) if fs_examples else ""

    # ----- build prompts for the test set
    test_rows = load_jsonl(args.test_file)
    labels = SUBTASK1_LABELS if args.task == "fallacy_detection" else SUBTASK2_LABELS
    sys_prompt = SUBTASK1_SYSTEM if args.task == "fallacy_detection" else SUBTASK2_SYSTEM
    fn = subtask1_prompt if args.task == "fallacy_detection" else subtask2_prompt
    use_cot = (args.mode == "cot")

    chat_prompts = []
    ids = []
    for r in test_rows:
        # For Sub-Task 2, the test set already contains only fallacious args (per spec),
        # but we still predict for every test entry — the organizers filter at scoring time.
        text = build_text(r, args.variant, args.include_argument)
        user_msg = fn(text, cot=use_cot)
        if fs_block:
            user_msg = fs_block + "\n\n" + user_msg
        chat_prompts.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ])
        ids.append(r["id"])

    # ----- vLLM inference
    from vllm import LLM, SamplingParams  # imported lazily so the file can be inspected on CPU

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16")
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    # vLLM supports chat templates natively:
    outputs = llm.chat(chat_prompts, sp)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path = out_path.with_suffix(".raw.jsonl")

    with out_path.open("w") as f, raw_log_path.open("w") as raw:
        for _id, out in zip(ids, outputs):
            gen = out.outputs[0].text
            lab = parse_answer(gen, labels)
            f.write(json.dumps({
                "task": args.task,
                "id": _id,
                "label": lab,
                "tag": args.tag,
                "system_description": args.system_description,
            }) + "\n")
            raw.write(json.dumps({"id": _id, "generation": gen, "parsed_label": lab}) + "\n")

    print(f"Wrote submission → {out_path}")
    print(f"Raw generations  → {raw_log_path}")


if __name__ == "__main__":
    main()
