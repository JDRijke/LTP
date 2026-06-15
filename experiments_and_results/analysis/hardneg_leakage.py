"""ST1 hard-negative test: does the enhancement inject fallacy cues ONLY for true
fallacies (label leakage) or also for the resembling non-fallacies (legit)?"""
import json
from pathlib import Path
import numpy as np

TRAIN = str(Path(__file__).resolve().parents[2] / "touchefallacy_2026_train.jsonl")
rows = [json.loads(l) for l in Path(TRAIN).read_text().splitlines() if l.strip()]

CUES = {
 "authority": ["authorit","expert","credibilit","renowned","respected","greatest","brilliant","esteemed","qualified","credentials"],
 "blackwhite": ["only two","either","no middle","black and white","binary","two option","only choice","no other","the only"],
 "hasty_generalization":["every ","everyone","all of","always","one example","a single","just one","anecdot","handful","every single"],
 "natural":["natural","nature","unnatural"],
 "population":["majority","most people","popular","many people","widely","everybody","the masses","overwhelming"],
 "slippery_slope":["slippery slope","leads to","eventually","next step","spiral","domino","before you know","end up","inevitabl","each step"],
 "tradition":["tradition","always been","for centuries","historically","long-standing","for years","ancestor","time-honored","that long"],
 "worse_problems":["worse problem","bigger problem","more important","what about","real problem","far worse","greater issue","compared to"],
}
def hit(txt, cues):
    t = txt.lower(); return any(c in t for c in cues)

print("="*78)
print("ST1 leakage test — cue-presence by fallacy_exists, grouped by resembles_fallacy")
print("If enhancement leaks the binary label, TRUE fallacies get many more cues than")
print("the hard negatives that resemble the SAME fallacy type.")
print("="*78)
print(f"{'resembles':22s} {'n+':>4s} {'n-':>4s} | {'base +':>7s} {'base -':>7s} | {'enh +':>7s} {'enh -':>7s} | {'enh gap':>7s}")
tot = {"bp":[], "bn":[], "ep":[], "en":[]}
for cls, cues in CUES.items():
    pos = [r for r in rows if r.get("resembles_fallacy")==cls and int(r.get("fallacy_exists",0))==1]
    neg = [r for r in rows if r.get("resembles_fallacy")==cls and int(r.get("fallacy_exists",0))==0]
    if not pos or not neg:
        continue
    bp = np.mean([hit(r["text_base"],cues) for r in pos])
    bn = np.mean([hit(r["text_base"],cues) for r in neg])
    ep = np.mean([hit(r["text_enhanced"],cues) for r in pos])
    en = np.mean([hit(r["text_enhanced"],cues) for r in neg])
    print(f"{cls:22s} {len(pos):4d} {len(neg):4d} | {bp:6.0%} {bn:6.0%} | {ep:6.0%} {en:6.0%} | {ep-en:+6.0%}")
    tot["bp"]+= [hit(r["text_base"],cues) for r in pos]
    tot["bn"]+= [hit(r["text_base"],cues) for r in neg]
    tot["ep"]+= [hit(r["text_enhanced"],cues) for r in pos]
    tot["en"]+= [hit(r["text_enhanced"],cues) for r in neg]
print("-"*78)
print(f"{'POOLED':22s} {len(tot['bp']):4d} {len(tot['bn']):4d} | "
      f"{np.mean(tot['bp']):6.0%} {np.mean(tot['bn']):6.0%} | "
      f"{np.mean(tot['ep']):6.0%} {np.mean(tot['en']):6.0%} | {np.mean(tot['ep'])-np.mean(tot['en']):+6.0%}")
print("\n+ = true fallacy (fallacy_exists=1), - = hard negative resembling same type (=0)")
