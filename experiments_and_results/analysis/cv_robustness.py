"""Cross-validated robustness check

5-fold stratified CV macro-F1 (mean ± std) for each input variant, TF-IDF+LogReg.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "touchefallacy_2026_train.jsonl"
DATASET_TO_SUB = {"blackwhite": "black-white"}
rows_all = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]


def arg_str(arg):
    arg = arg or {}
    sup = " ".join(f"- {s}" for s in (arg.get("supports") or []))
    return f"[Claim] {arg.get('claim','')} [Supports] {sup}".strip()


BUILDERS = {
    "text_base": lambda r: r["text_base"],
    "text_enhanced": lambda r: r["text_enhanced"],
    "enh_trunc→baselen": lambda r: " ".join(r["text_enhanced"].split()[:len(r["text_base"].split())]),
    "argument_base": lambda r: arg_str(r.get("argument_base")),
    "argument_enhanced": lambda r: arg_str(r.get("argument_enhanced")),
}


def cv(texts, labels):
    pipe = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("c", LogisticRegression(max_iter=2000, C=10, class_weight="balanced"))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s = cross_val_score(pipe, texts, labels, cv=skf, scoring="f1_macro")
    return s.mean(), s.std()


for task in ["st1", "st2"]:
    if task == "st2":
        rows = [r for r in rows_all if int(r.get("fallacy_exists", 0)) == 1]
        labels = [DATASET_TO_SUB.get(
            r["fallacy_type"], r["fallacy_type"]) for r in rows]
    else:
        rows = rows_all
        labels = ["fallacy" if int(r["fallacy_exists"])
                  == 1 else "non-fallacy" for r in rows]
    print(f"\n=== {task.upper()}  (n={len(rows)}) — 5-fold CV macro-F1 ===")
    for name, fn in BUILDERS.items():
        m, sd = cv([fn(r) for r in rows], labels)
        print(f"   {name:20s}  {m:.3f} ± {sd:.3f}")
