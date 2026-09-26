"""
Stratified subset evaluation on the 478-question bench.

Picks ~5 per type (50 total) and runs the strategy-routed pipeline only
(no baseline, no RoG-style — to save LLM cost; baselines can be added later).

Goal: verify that the per-type accuracy from the 30-question validation
holds at scale, especially on the auto-generated questions (which are
noisier than the hand-crafted 30).
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import random
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex
from experiments.run_e2e_30 import score_question


PER_TYPE_TARGET = {
    "single_hop":        5,
    "relation_inverse":  5,
    "agg_count":         5,
    "agg_enum":          5,
    "two_hop_bridge":    5,
    "three_hop_chain":   5,   # 8 available, take 5
    "attr_filter":       5,
    "negation":          5,
    "set_compare":       5,
    "unanswerable":      3,
    "distractor":        2,
}


def main():
    random.seed(123)

    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]

    # stratified sample
    by_type = defaultdict(list)
    for q in all_qs:
        by_type[q["type"]].append(q)
    sample = []
    for t, target in PER_TYPE_TARGET.items():
        pool = by_type.get(t, [])
        if not pool:
            continue
        sample.extend(random.sample(pool, min(target, len(pool))))
    print(f"Sampled {len(sample)} questions:")
    for t, target in PER_TYPE_TARGET.items():
        n = sum(1 for q in sample if q["type"] == t)
        print(f"  {t:20s} {n}")

    # Adapt q schema for the existing scorer (it expects question_zh / gold_*)
    # qa_500.json questions already use those keys, but the scorer signature is
    # (q, predicted_text, kg) where q has 'type' + relevant gold_* fields.

    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    pipe = StrategyPipeline(triples_path=str(big_path))

    results = []
    for i, q in enumerate(sample, 1):
        print(f"\n[{i:>2}/{len(sample)}] {q['id']:15s} ({q['type']:18s})")
        print(f"   Q: {q['question_zh']}")
        try:
            out = pipe.run(q["question_zh"])
            score = score_question(q, out["answer"], kg)
        except Exception as e:
            out = {"answer": f"[ERR {e}]", "qtype": "?", "strategy": "?"}
            score = {"correct": False, "error": str(e)}
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question_zh"],
            "gold_answer": q.get("gold_answer"),
            "predicted_qtype":  out.get("qtype"),
            "predicted_strategy": out.get("strategy"),
            "answer":  out.get("answer", "")[:300] if isinstance(out.get("answer"), str) else "",
            "score":   score,
        }
        results.append(rec)
        ok = "OK" if score.get("correct") else "  "
        print(f"   strategy:[{ok}] qtype={out.get('qtype')} → {out.get('strategy')}")
        print(f"   ans: {(out.get('answer') or '')[:160]}")

    # aggregate
    by_t = defaultdict(list)
    for r in results:
        by_t[r["type"]].append(r["score"].get("correct", False))

    print("\n========== STRATEGY ON QA-500 STRATIFIED SUBSET ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Strategy acc':>12s}")
    print("-" * 45)
    total_correct, total_n = 0, 0
    for t, lst in sorted(by_t.items()):
        n = len(lst); c = sum(lst)
        total_correct += c; total_n += n
        print(f"{t:<22s} {n:>3d} | {c/n:>12.3f}")
    print("-" * 45)
    print(f"{'OVERALL':<22s} {total_n:>3d} | {total_correct/total_n:>12.3f}")

    out = ROOT / "results" / "qa500_subset_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
