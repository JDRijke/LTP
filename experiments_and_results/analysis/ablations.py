"""Mechanism ablations for the 'why does text_enhanced help?' analysis.

Run:  python experiments_and_results/analysis/ablations.py
"""
import json
import re
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "touchefallacy_2026_train.jsonl"
DATASET_TO_SUB = {"blackwhite": "black-white"}
SEED = 42


def load(p): return [json.loads(l)
                     for l in Path(p).read_text().splitlines() if l.strip()]


def task_rows_labels(rows, task):
    if task == "st2":
        rows = [r for r in rows if int(r.get("fallacy_exists", 0)) == 1]
        labels = [DATASET_TO_SUB.get(
            r["fallacy_type"], r["fallacy_type"]) for r in rows]
    else:
        labels = ["fallacy" if int(r["fallacy_exists"])
                  == 1 else "non-fallacy" for r in rows]
    return rows, labels


def split_indices(labels):
    idx = np.arange(len(labels))
    tv, te = train_test_split(
        idx, test_size=0.10, stratify=labels, random_state=SEED)
    rel = 0.10 / 0.90
    tr, va = train_test_split(tv, test_size=rel, stratify=[
                              labels[i] for i in tv], random_state=SEED)
    return tr, te


def proxy(train_texts, train_labels, test_texts, test_labels):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
    clf.fit(Xtr, train_labels)
    pred = clf.predict(Xte)
    return f1_score(test_labels, pred, average="macro"), accuracy_score(test_labels, pred), vec, clf

# ---- text builders ------------------------------------------------------


def t_base(r): return r["text_base"]
def t_enh(r): return r["text_enhanced"]


def t_enh_trunc(r):
    n = len(r["text_base"].split())
    return " ".join(r["text_enhanced"].split()[:n])


def arg_str(arg):
    arg = arg or {}
    sup = " ".join(f"- {s}" for s in (arg.get("supports") or []))
    return f"[Claim] {arg.get('claim','')} [Supports] {sup}".strip()


def t_arg_base(r): return arg_str(r.get("argument_base"))
def t_arg_enh(r): return arg_str(r.get("argument_enhanced"))


def striker(phrases):
    phrases = sorted({p.lower()
                     for p in phrases if p.strip()}, key=len, reverse=True)
    pats = [re.compile(r"\b" + re.escape(p) + r"\b", re.I) for p in phrases]

    def strike(txt):
        for pat in pats:
            txt = pat.sub(" ", txt)
        return re.sub(r"\s+", " ", txt).strip()
    return strike


rows_all = load(TRAIN)


def evaluate(task, text_fn_train, text_fn_test=None):
    text_fn_test = text_fn_test or text_fn_train
    rows, labels = task_rows_labels(rows_all, task)
    tr, te = split_indices(labels)
    f1, acc, vec, clf = proxy(
        [text_fn_train(rows[i]) for i in tr], [labels[i] for i in tr],
        [text_fn_test(rows[i]) for i in te],  [labels[i] for i in te])
    return f1, acc, vec, clf, rows, labels, tr, te


print("#"*72)
print("# G. VARIANT TRANSFER  (train on X, test on Y)")
print("#"*72)
for task in ["st1", "st2"]:
    print(f"\n  {task.upper()}            test=base   test=enhanced")
    for trn_name, trn in [("train=base    ", t_base), ("train=enhanced", t_enh)]:
        row = []
        for tst in [t_base, t_enh]:
            f1, *_ = evaluate(task, trn, tst)
            row.append(f"{f1:.3f}")
        print(f"  {trn_name}      {row[0]:>6s}      {row[1]:>6s}")

print("\n" + "#"*72)
print("# F. LENGTH CONTROL (ST2): enhanced truncated to base word-count")
print("#"*72)
rows, _ = task_rows_labels(rows_all, "st2")
mb = np.mean([len(t_base(r).split()) for r in rows])
me = np.mean([len(t_enh(r).split()) for r in rows])
mt = np.mean([len(t_enh_trunc(r).split()) for r in rows])
print(
    f"  mean length: base={mb:.0f}  enhanced={me:.0f}  enhanced-trunc={mt:.0f} (≈base)")
for name, fn in [("base                 ", t_base),
                 ("enhanced (full)      ", t_enh),
                 ("enhanced trunc→base  ", t_enh_trunc)]:
    f1, acc, *_ = evaluate("st2", fn)
    print(f"  {name}  macro-F1={f1:.3f}  acc={acc:.3f}")

print("\n" + "#"*72)
print("# E. CUE MASKING (ST2 enhanced)")
print("#"*72)
# names-only strike set: the 8 fallacy names + closest variants
NAMES = ["authority", "black and white", "black-and-white", "blackwhite",
         "hasty generalization", "hasty generalisation", "natural", "nature",
         "naturally", "population", "popular", "majority", "slippery slope",
         "tradition", "traditional", "traditionally", "worse problems",
         "bigger problems"]
# data-driven top tokens per class (derived from TRAIN enhanced only, content words)
_, _, vec, clf, rows, labels, tr, te = evaluate("st2", t_enh)
feat = np.array(vec.get_feature_names_out())
topset = set()
for ci in range(len(clf.classes_)):
    order = np.argsort(clf.coef_[ci])[::-1]
    kept = 0
    for j in order:
        tok = feat[j]
        words = tok.split()
        if all(w.isalpha() and len(w) >= 4 and w not in ENGLISH_STOP_WORDS for w in words):
            topset.add(tok)
            kept += 1
        if kept >= 10:
            break

f1_full, acc_full, *_ = evaluate("st2", t_enh)
strike_names = striker(NAMES)
f1_n, acc_n, *_ = evaluate("st2", lambda r: strike_names(t_enh(r)))
strike_top = striker(topset)
f1_t, acc_t, *_ = evaluate("st2", lambda r: strike_top(t_enh(r)))
f1_b, acc_b, *_ = evaluate("st2", t_base)
print(f"  enhanced (full)                 macro-F1={f1_full:.3f}")
print(
    f"  enhanced − fallacy NAMES only   macro-F1={f1_n:.3f}   (struck {len(set(NAMES))} phrases)")
print(
    f"  enhanced − top-10/class cues    macro-F1={f1_t:.3f}   (struck {len(topset)} phrases)")
print(f"  base (reference)                macro-F1={f1_b:.3f}")
print("  struck top-cue examples:", ", ".join(sorted(topset)[:25]))

print("\n" + "#"*72)
print(
    "# H. STRUCTURE WITHOUT PROSE (ST2): argument_{base,enhanced} claim+supports")
print("#"*72)
for name, fn in [("text_base            ", t_base),
                 ("text_enhanced        ", t_enh),
                 ("argument_base (struct)", t_arg_base),
                 ("argument_enhanced     ", t_arg_enh)]:
    f1, acc, *_ = evaluate("st2", fn)
    print(f"  {name}  macro-F1={f1:.3f}  acc={acc:.3f}")
