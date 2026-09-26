# -*- coding: utf-8 -*-
"""Generate a compositional evaluation set from the KG. Each item is
(question, gold_plan, gold_value), where gold_value is computed by the
deterministic executor on a hand-constructed gold plan — so gold is exact.

Categories target the agent's *compositional* ability (beyond a single operator):
  multi_count   : count ∘ intersect(2 constraints)
  multi_enum    : enumerate ∘ intersect(2 constraints)
  difference    : count ∘ difference(A, B)
  aggregate     : avg/max numeric attribute over a filtered set
  compare       : two entities' relation same/different
  hop2_count    : count of entities reached via a 2-hop inverse path
"""
import json, random
from pathlib import Path
from collections import defaultdict, Counter
from composition import load_executor

ROOT = Path(__file__).resolve().parent.parent
random.seed(7)
ex = load_executor()
kg = ex.kg
triples = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))

# ---- inventory from KG ----
countries = Counter()
bands = Counter()
dev_of = defaultdict(set)        # radar -> developers
country_radars = defaultdict(set)
band_radars = defaultdict(set)
range_radars = set()
for t in triples:
    h, r, ta = t["head"], t["relation"], t["tail"]
    if r == "countryOfOrigin":
        countries[ta] += 1; country_radars[ta].add(h)
    elif r == "hasFrequencyBand":
        bands[ta] += 1; band_radars[ta].add(h)
    elif r == "developedBy":
        dev_of[h].add(ta)
    elif r == "range_km":
        range_radars.add(h)

TOP_COUNTRIES = [c for c, _ in countries.most_common(8) if len(c) <= 6]
TOP_BANDS = [b for b, _ in bands.most_common(8) if len(b) <= 3]


def gv(plan):
    v, _ = ex.evaluate(plan)
    return v


def ser(v):
    return sorted(v) if isinstance(v, set) else v


items = []
def add(cat, q, plan):
    try:
        val = gv(plan)
    except Exception:
        return
    # skip degenerate gold (empty/None) to keep the set meaningful
    if val is None or (isinstance(val, set) and len(val) == 0) or val == 0:
        return
    items.append({"id": f"{cat}_{len([x for x in items if x['category']==cat])+1:02d}",
                  "category": cat, "question": q, "gold_plan": plan, "gold_value": ser(val)})


# multi_count & multi_enum: country ∩ band
for c in TOP_COUNTRIES:
    for b in TOP_BANDS:
        inter = {"op": "intersect", "args": [
            {"op": "constraint", "relation": "countryOfOrigin", "value": c},
            {"op": "constraint", "relation": "hasFrequencyBand", "value": b}]}
        add("multi_count", f"{c}研制且工作在{b}波段的雷达有多少款？", {"op": "count", "arg": inter})
        add("multi_enum", f"列出{c}研制且工作在{b}波段的雷达。", {"op": "enumerate", "arg": inter})

# difference: country minus band
for c in TOP_COUNTRIES:
    for b in TOP_BANDS:
        diff = {"op": "difference", "args": [
            {"op": "constraint", "relation": "countryOfOrigin", "value": c},
            {"op": "constraint", "relation": "hasFrequencyBand", "value": b}]}
        add("difference", f"{c}研制但不工作在{b}波段的雷达有多少款？", {"op": "count", "arg": diff})

# aggregate: avg/max range_km over country (range_km is the best-covered numeric attr)
for c in TOP_COUNTRIES:
    for func, word in [("avg", "平均"), ("max", "最大")]:
        add("aggregate", f"{c}研制的雷达的{word}探测距离是多少？",
            {"op": "aggregate", "func": func, "attribute": "range_km",
             "over": {"op": "constraint", "relation": "countryOfOrigin", "value": c}})

# compare: two radars' developer same/different
radars_with_dev = [r for r, d in dev_of.items() if d]
random.shuffle(radars_with_dev)
for i in range(0, min(40, len(radars_with_dev) - 1), 2):
    a, b = radars_with_dev[i], radars_with_dev[i + 1]
    add("compare", f"{a} 和 {b} 的研制方相同吗？",
        {"op": "compare", "a": a, "b": b, "relation": "developedBy"})

# hop2_count: how many radars did manufacturer X develop (via developedBy^-1), then count
manufacturers = Counter()
for d in dev_of.values():
    for m in d:
        manufacturers[m] += 1
for m, _ in manufacturers.most_common(20):
    if len(m) > 30:
        continue
    add("hop2_count", f"{m} 一共研制了多少款雷达？",
        {"op": "count", "arg": {"op": "path", "start": m, "chain": ["developedBy^-1"]}})

# cap per category to keep balanced (~12 each)
by_cat = defaultdict(list)
for it in items:
    by_cat[it["category"]].append(it)
final = []
for cat, lst in by_cat.items():
    random.shuffle(lst)
    final.extend(lst[:12])
random.shuffle(final)

out = ROOT / "results" / "composition_eval.json"
json.dump(final, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"generated {len(final)} items")
print("by category:", dict(Counter(it["category"] for it in final)))
print("\n=== samples ===")
for it in final[:8]:
    print(f"[{it['category']}] {it['question']}  → gold={ser(it['gold_value']) if not isinstance(it['gold_value'],list) else '['+str(len(it['gold_value']))+' items]'}")
