# -*- coding: utf-8 -*-
"""Evaluate the composition agent on the generated compositional eval set.
For each item: run the LLM planner -> execute deterministically -> compare the
executed value against gold (computed by the executor on a gold plan).
Measures planner quality on compositional queries the single-operator pipeline
cannot express (aggregate / difference / nested)."""
import json, time
from pathlib import Path
from collections import defaultdict
from composition import load_executor, PlanError
from planner import plan as make_plan

ROOT = Path(__file__).resolve().parent.parent
ex = load_executor()
items = json.load(open(ROOT / "results" / "composition_eval.json", encoding="utf-8"))
OUT = ROOT / "results" / "composition_eval_results.json"

COUNT_CATS = {"multi_count", "difference", "hop2_count"}


def match(gold, pred, cat):
    try:
        if cat in COUNT_CATS:
            return int(pred) == int(gold)
        if cat == "aggregate":
            if pred is None:
                return False
            return abs(float(pred) - float(gold)) / max(abs(float(gold)), 1e-9) < 0.01
        if cat == "multi_enum":
            return set(pred) == set(gold)
        if cat == "compare":
            return (bool(pred.get("identical")) == bool(gold.get("identical")) and
                    set(pred.get("same", [])) == set(gold.get("same", [])) and
                    set(pred.get("only_a", [])) == set(gold.get("only_a", [])) and
                    set(pred.get("only_b", [])) == set(gold.get("only_b", [])))
        return pred == gold
    except Exception:
        return False


results = []
load = OUT.exists()
done = {r["id"] for r in json.load(open(OUT, encoding="utf-8"))} if load else set()
if load:
    results = json.load(open(OUT, encoding="utf-8"))

t0 = time.time()
for i, it in enumerate(items):
    if it["id"] in done:
        continue
    rec = {"id": it["id"], "category": it["category"], "question": it["question"]}
    try:
        p = make_plan(it["question"])
        rec["plan_valid"] = True
        val, _ = ex.evaluate(p)
        rec["exec_ok"] = True
        rec["pred_plan"] = p
        rec["correct"] = bool(match(it["gold_value"], val, it["category"]))
    except (PlanError, Exception) as e:
        rec["plan_valid"] = "json" not in str(e).lower()
        rec["exec_ok"] = False
        rec["correct"] = False
        rec["error"] = f"{type(e).__name__}: {e}"[:120]
    results.append(rec)
    if (len(results)) % 12 == 0:
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  {len(results)}/{len(items)}  ({(time.time()-t0)/60:.1f}m)")
json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# aggregate
by = defaultdict(lambda: {"n": 0, "valid": 0, "exec": 0, "correct": 0})
for r in results:
    d = by[r["category"]]
    d["n"] += 1; d["valid"] += int(bool(r.get("plan_valid")))
    d["exec"] += int(bool(r.get("exec_ok"))); d["correct"] += int(bool(r.get("correct")))
print("\n=== Composition agent eval (n=%d) ===" % len(results))
print(f"{'category':14s} {'n':>3s} {'plan_valid':>11s} {'exec_ok':>8s} {'correct':>8s}")
tot = defaultdict(int)
for cat in sorted(by):
    d = by[cat]
    for k in ("n", "valid", "exec", "correct"):
        tot[k] += d[k]
    print(f"{cat:14s} {d['n']:>3d} {d['valid']/d['n']*100:>10.0f}% {d['exec']/d['n']*100:>7.0f}% {d['correct']/d['n']*100:>7.0f}%")
n = tot["n"]
print(f"{'OVERALL':14s} {n:>3d} {tot['valid']/n*100:>10.0f}% {tot['exec']/n*100:>7.0f}% {tot['correct']/n*100:>7.0f}%")
print(f"\nSaved: {OUT}")
