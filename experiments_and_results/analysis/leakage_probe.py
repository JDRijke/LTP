"""Test wheter the text_enhanced gain is explainable by lexical signal

Reproduces the split and compares a
TF-IDF + Logistic Regression model on base vs enhanced for ST1 and ST2.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

TRAIN = str(Path(__file__).resolve(
).parents[2] / "touchefallacy_2026_train.jsonl")

SUBTASK2_LABELS = ["authority", "black-white", "hasty_generalization", "natural",
                   "population", "slippery_slope", "tradition", "worse_problems"]
DATASET_TO_SUB = {"blackwhite": "black-white"}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def split(items, labels, seed=42):
    """Replicate data.stratified_split: 80/10/10."""
    tv, te, tv_l, te_l = train_test_split(items, labels, test_size=0.10,
                                          stratify=labels, random_state=seed)
    rel_val = 0.10/0.90
    tr, va, tr_l, va_l = train_test_split(tv, tv_l, test_size=rel_val,
                                          stratify=tv_l, random_state=seed)
    return (tr, tr_l), (va, va_l), (te, te_l)


def run_task(rows, task, variant):
    if task == "st2":
        rows = [r for r in rows if int(r.get("fallacy_exists", 0)) == 1]
        labels = [DATASET_TO_SUB.get(
            r["fallacy_type"], r["fallacy_type"]) for r in rows]
    else:
        labels = ["fallacy" if int(r["fallacy_exists"])
                  == 1 else "non-fallacy" for r in rows]
    texts = [r[f"text_{variant}"] for r in rows]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(texts, labels)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
    clf.fit(Xtr_v, ytr)
    pred = clf.predict(Xte_v)
    return f1_score(yte, pred, average="macro"), accuracy_score(yte, pred), vec, clf, (Xte, yte)


rows = load(TRAIN)
print("="*70)
print("TF-IDF + LogisticRegression (bag-of-words, NO deep model) — internal test")
print("="*70)
for task in ["st1", "st2"]:
    print(
        f"\n### {task.upper()} ({'detection' if task=='st1' else 'classification'})")
    for variant in ["base", "enhanced"]:
        f1, acc, *_ = run_task(rows, task, variant)
        print(f"   {variant:9s}  macro-F1 = {f1:.3f}   acc = {acc:.3f}")

# Top discriminative tokens per class
print("\n" + "="*70)
print("Top TF-IDF tokens per fallacy class (ST2, ENHANCED) — what carries signal?")
print("="*70)
_, _, vec, clf, _ = run_task(rows, "st2", "enhanced")
feat = np.array(vec.get_feature_names_out())
for ci, cls in enumerate(clf.classes_):
    top = feat[np.argsort(clf.coef_[ci])[-12:]][::-1]
    print(f"  {cls:22s}: {', '.join(top)}")

print("\n" + "="*70)
print("Same, but ST2 BASE — for contrast")
print("="*70)
_, _, vecb, clfb, _ = run_task(rows, "st2", "base")
featb = np.array(vecb.get_feature_names_out())
for ci, cls in enumerate(clfb.classes_):
    top = featb[np.argsort(clfb.coef_[ci])[-12:]][::-1]
    print(f"  {cls:22s}: {', '.join(top)}")

# Length stats
print("\n" + "="*70)
print("Length (whitespace tokens): base vs enhanced")
print("="*70)
fall = [r for r in rows if int(r.get("fallacy_exists", 0)) == 1]
for name, subset in [("ALL", rows), ("fallacious only", fall)]:
    lb = np.array([len(r["text_base"].split()) for r in subset])
    le = np.array([len(r["text_enhanced"].split()) for r in subset])
    print(f"  {name:16s}: base mean={lb.mean():.0f}  enhanced mean={le.mean():.0f}  "
          f"ratio={le.mean()/lb.mean():.2f}x")

# does enhanced inject the class name / obvious cue?
print("\n" + "="*70)
print("Cue-word presence rate in matching-class texts: base vs enhanced")
print("(fraction of texts of class C that contain a cue word for class C)")
print("="*70)
CUES = {
    "authority": ["authorit", "expert", "credibilit", "renowned", "respected", "greatest", "brilliant", "esteemed", "qualified"],
    "black-white": ["only two", "either", "no middle", "black and white", "binary", "two option", "only choice", "no other"],
    "hasty_generalization": ["every ", "everyone", "all of", "always", "one example", "a single", "just one", "anecdot", "handful"],
    "natural": ["natural", "nature", "unnatural"],
    "population": ["everyone", "most people", "popular", "majority", "many people", "widely", "everybody", "the masses"],
    "slippery_slope": ["slippery slope", "lead to", "eventually", "next thing", "spiral", "domino", "before you know", "end up", "snowball"],
    "tradition": ["tradition", "always been", "for centuries", "historically", "long-standing", "for years", "ancestor", "time-honored", "the old way"],
    "worse_problems": ["worse problem", "bigger problem", "more important", "what about", "real problem", "instead of worrying", "greater issue"],
}
for cls in SUBTASK2_LABELS:
    sub = [r for r in fall if DATASET_TO_SUB.get(
        r["fallacy_type"], r["fallacy_type"]) == cls]
    cues = CUES[cls]

    def hit(txt):
        t = txt.lower()
        return any(c in t for c in cues)
    rb = np.mean([hit(r["text_base"]) for r in sub])
    re_ = np.mean([hit(r["text_enhanced"]) for r in sub])
    print(f"  {cls:22s} (n={len(sub):3d}): base={rb:.0%}  enhanced={re_:.0%}   Δ=+{(re_-rb):.0%}")
