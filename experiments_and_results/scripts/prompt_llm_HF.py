"""Prompt a local decoder LLM (via HuggingFace) for Sub-Task 1 / 2 with zero-shot, few-shot, or CoT.

Examples:

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

import gc
import json
import random
from collections import defaultdict
import argparse
import re
import warnings
import numpy as np
import torch
import os
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm.auto import tqdm

from data import (
    SUBTASK1_LABELS,
    SUBTASK2_LABELS,
    build_text,
    load_jsonl,
    rows_to_examples,
    stratified_split,
)
from prompts import (
    SUBTASK1_SYSTEM,
    SUBTASK2_SYSTEM,
    build_fewshot_block,
    subtask1_prompt,
    subtask2_prompt,
)

user = os.environ.get("USER")

# ── Huggingface login ─────────────────────────────────────────────────────────

def login_hf():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
    else:
        print("Warning: HF_TOKEN not set. Private models will not be accessible.")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline(model_name: str, device: str, max_new_tokens: int, cache_dir: str,
                  temp: float = None, do_sample: bool = False,
                  top_p: float = None, top_k: int = None):
    print(f"Loading model: {model_name}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Some models expose eos_token_id as a list
    eos = pipe.model.config.eos_token_id
    pipe.tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return pipe, model


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_answer(generation: str, labels: list[str]) -> str:
    """Parse LLM output, looking for `ANSWER: <label>` then fall back to label match.

    Handles common surface variants:
      - hyphen / underscore confusion (`black-white` vs `black_white`)
      - the dataset's `blackwhite` spelling vs the submission's `black-white`
    """
    def norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "").replace(" ", "")

    m = re.search(r"ANSWER\s*:\s*([A-Za-z_\-]+)", generation, flags=re.IGNORECASE)
    if m:
        cand = norm(m.group(1).strip())
        for label in labels:
            if norm(label) == cand:
                return label

    # Fallback: first normalised label that appears anywhere in the generation
    g_norm = norm(generation)
    for label in labels:
        if norm(label) in g_norm:
            return label

    return labels[0]


# ── Batch annotation ──────────────────────────────────────────────────────────

def annotate_batch(
    pipe,
    prompts: list,
    batch_size: int = 8,
    verbose: bool = False,
) -> list[str]:
    """Run the HuggingFace pipeline over `prompts` in batches.

    `prompts` can be either:
      - a list of strings (plain text prompts), or
      - a list of list of list of dicts (chat messages), in which case
        `apply_chat_template` is called to convert them to strings first.

    Returns a flat list of generated strings, one per prompt.
    """
    generations = []
    for out in pipe(prompts, batch_size=batch_size):
        raw = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
        generations.append(raw)
    return generations


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate texts using an LLM (HuggingFace)."
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument("--output", "-o", default=None,
                        help="Path to output JSONL file (standalone annotation mode).")

    # ── Task / data ───────────────────────────────────────────────────────────
    parser.add_argument("--train_file", default=None,
                        help="Training JSONL (used for few-shot exemplar selection).")
    parser.add_argument("--test_file", default=None,
                        help="Official test JSONL (no labels).")
    parser.add_argument("--task",
                        choices=["fallacy_detection", "fallacy_classification"],
                        default=None,
                        help="Which sub-task to run.")
    parser.add_argument("--variant", choices=["base", "enhanced"], default="base",
                        help="Text variant to build from the data rows.")
    parser.add_argument("--include_argument", action="store_true",
                        help="Include the argument field when building text.")
    parser.add_argument("--tag", choices=["base", "enhanced"], default="base",
                        help="Tag written into the submission JSONL.")
    parser.add_argument("--system_description", default="",
                        help="Optional system description written to submission JSONL.")
    parser.add_argument("--test_internal", action="store_true",
                        help="Predict on the internal test set instead of the ")

    # ── Prompting strategy ────────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "cot"],
                        default="zero_shot",
                        help="Prompting mode.")
    parser.add_argument("--k_per_class", type=int, default=2,
                        help="Few-shot examples per class (used when --mode few_shot).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for exemplar selection and sampling.")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model", "-m", required=True,
                        help="HuggingFace model name or local path.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run the model on (default: cuda).")
    parser.add_argument("--cache_dir", "-c",
                        default=f"/scratch/{user}/.cache/huggingface",
                        help="Cache directory for model weights.")

    # ── Generation ────────────────────────────────────────────────────────────
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Alias for --max_tokens (overrides it when set).")
    parser.add_argument("--temp", default=None, type=float,
                        help="Sampling temperature (leave unset for greedy decoding).")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling (required when using --temp / --top_p / --top_k).")
    parser.add_argument("--top_p", default=None, type=float,
                        help="Nucleus sampling top-p.")
    parser.add_argument("--top_k", default=None, type=int,
                        help="Top-k sampling.")

    # ── Batching / logging ────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Texts per batch (reduce if you run out of GPU memory).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only annotate the first N rows (useful for testing).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-batch progress bar.")

    args = parser.parse_args()


    return args

 
# ── GPU cleanup ───────────────────────────────────────────────────────────────
 
def _cleanup_gpu(pipe, model):
    """Explicitly free the pipeline and model from GPU memory.
 
    Calling this at the end of main() ensures that CUDA memory is released
    before the Python process exits, which matters when multiple runs are
    chained in the same Slurm job step and the driver is slow to reclaim memory.
    """
    del pipe
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_mb = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"GPU cleanup done — reserved memory after cleanup: {mem_mb:.0f} MB")


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


def main():
    args = parse_args()

    login_hf()

    # ----- few-shot exemplars from train split
    rows = load_jsonl(args.train_file)
    train_ex_full = rows_to_examples(rows, args.task, args.variant, args.include_argument)
    train_split, _, internal_test_split = stratified_split(train_ex_full, seed=args.seed)

    fs_examples = []
    if args.mode == "few_shot":
        fs_examples = select_fewshot(train_split, args.k_per_class, args.seed)
        print(f"Few-shot exemplars: {len(fs_examples)}")

    fs_block = build_fewshot_block(
        args.task, [{"text": e.text, "label": e.label} for e in fs_examples]
    ) if fs_examples else ""

    # ----- build prompts for the test set
    labels = SUBTASK1_LABELS if args.task == "fallacy_detection" else SUBTASK2_LABELS
    sys_prompt = SUBTASK1_SYSTEM if args.task == "fallacy_detection" else SUBTASK2_SYSTEM
    fn = subtask1_prompt if args.task == "fallacy_detection" else subtask2_prompt
    use_cot = args.mode == "cot"

    if args.test_internal:
        # Use the internal test split (stratified hold-out from the training file)
        test_examples = internal_test_split
        if args.limit:
            test_examples = test_examples[: args.limit]
        print(f"Using internal test set: {len(test_examples)} examples")

        chat_prompts = []
        ids = []
        gold_labels = []
        for ex in test_examples:
            user_msg = fn(ex.text, cot=use_cot)
            if fs_block:
                user_msg = fs_block + "\n\n" + user_msg
            chat_prompts.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ])
            ids.append(ex.id)
            gold_labels.append(ex.label)
    else:
        # Use the official (external) test file
        test_rows = load_jsonl(args.test_file)
        if args.limit:
            test_rows = test_rows[: args.limit]

        chat_prompts = []
        ids = []
        gold_labels = []          # will remain all-None for the external test set
        for r in test_rows:
            text = build_text(r, args.variant, args.include_argument)
            user_msg = fn(text, cot=use_cot)
            if fs_block:
                user_msg = fs_block + "\n\n" + user_msg
            chat_prompts.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ])
            ids.append(r["id"])
            gold_labels.append(None)

    # ----- HuggingFace inference
    pipe, model = load_pipeline(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        temp=args.temp,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    generations = annotate_batch(
        pipe=pipe,
        prompts=chat_prompts,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # ----- write outputs
    parent_dir, file_name = args.output.split("/")
    output_path = f"{parent_dir}/{args.task.split('_')[1]}/{args.model.split('/')[-1]}/{'test_internal' if args.test_internal else 'test'}/{args.variant}_{args.mode}_{file_name}"
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.test_internal:
        # Internal evaluation format: id, gold label, prediction, task, variant
        with out_path.open("w") as f:
            for _id, gold, gen in zip(ids, gold_labels, generations):
                pred = parse_answer(gen, labels)
                f.write(json.dumps({
                    "id": _id,
                    "gold_label": gold,
                    "pred": pred,
                    "task": args.task,
                    "variant": args.variant,
                }) + "\n")
        print(f"Wrote internal evaluation results → {out_path}")
    else:
        # Official submission format
        raw_log_path = out_path.with_suffix(".raw.jsonl")
        with out_path.open("w") as f, raw_log_path.open("w") as raw:
            for _id, gen in zip(ids, generations):
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

# ----- GPU cleanup (important when multiple runs share the same Slurm job)
    _cleanup_gpu(pipe, model)

if __name__ == "__main__":
    main()