"""
3-way comparison on a 100-question stratified subset of qa_500.

This is the paper's main results table at scale (vs 30-question hand-crafted).

Layout:
  - Stratified ~10/type (smaller for distractor & 3-hop where supply is limited)
  - For each question: run Baseline (HybridRetriever), RoG-style (uniform path-plan),
    Strategy-Routed (ours).
  - Score each with the fixed score_question (covers all 11 types).
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
from qa_rog_baseline import run_rog_baseline
from graphrag_retriever import HybridRetriever
from experiments.run_e2e_30 import baseline_answer, score_question


PER_TYPE_TARGET = {
    "single_hop":        10,
    "relation_inverse":  10,
    "agg_count":         10,
    "agg_enum":          10,
    "two_hop_bridge":    12,
    "three_hop_chain":   8,    # all 8 available
    "attr_filter":       10,
    "negation":          10,
    "set_compare":       10,
    "unanswerable":      6,
    "distractor":        4,
}  # total 100


def main():
    random.seed(7)

    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]
    by_type = defaultdict(list)
    for q in all_qs:
        by_type[q["type"]].append(q)
    sample = []
    for t, target in PER_TYPE_TARGET.items():
        pool = by_type.get(t, [])
        sample.extend(random.sample(pool, min(target, len(pool))))
    print(f"Sampled {len(sample)} questions:")
    for t, target in PER_TYPE_TARGET.items():
        print(f"  {t:20s} {sum(1 for q in sample if q['type']==t)}")

    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    retriever = HybridRetriever(triples, enable_reranker=False)
    pipe = StrategyPipeline(triples_path=str(big_path), lookup_retriever=retriever)

    results = []
    for i, q in enumerate(sample, 1):
        print(f"\n[{i:>3}/{len(sample)}] {q['id']:15s} ({q['type']:18s})")
        print(f"   Q: {q['question_zh']}")

        # baseline
        try:
            b_out = baseline_answer(q["question_zh"], retriever)
            b_score = score_question(q, b_out["answer"], kg)
        except Exception as e:
            b_out = {"answer": f"[ERR {e}]"}
            b_score = {"correct": False, "error": str(e)}

        # RoG-style
        try:
            r_out = run_rog_baseline(q["question_zh"], kg)
            r_score = score_question(q, r_out["answer"], kg)
        except Exception as e:
            r_out = {"answer": f"[ERR {e}]"}
            r_score = {"correct": False, "error": str(e)}

        # Strategy-routed
        try:
            s_out = pipe.run(q["question_zh"])
            s_score = score_question(q, s_out["answer"], kg)
        except Exception as e:
            s_out = {"answer": f"[ERR {e}]", "qtype": "?", "strategy": "?"}
            s_score = {"correct": False, "error": str(e)}

        rec = {
            "id": q["id"], "type": q["type"], "question": q["question_zh"],
            "gold_answer": q.get("gold_answer"),
            "baseline":  {"answer": b_out["answer"][:300], "score": b_score},
            "rog_style": {"answer": r_out["answer"][:300] if isinstance(r_out.get("answer"), str) else "",
                          "score": r_score},
            "strategy":  {"qtype": s_out.get("qtype"),
                          "strategy": s_out.get("strategy"),
                          "answer": s_out["answer"][:300] if isinstance(s_out.get("answer"), str) else "",
                          "score": s_score},
        }
        results.append(rec)
        b_ok = "OK" if b_score.get("correct") else "  "
        r_ok = "OK" if r_score.get("correct") else "  "
        s_ok = "OK" if s_score.get("correct") else "  "
        print(f"   B[{b_ok}] R[{r_ok}] S[{s_ok}]")

    # Aggregate
    by_t = defaultdict(lambda: {"baseline":[], "rog_style":[], "strategy":[]})
    for r in results:
        by_t[r["type"]]["baseline"].append(r["baseline"]["score"].get("correct", False))
        by_t[r["type"]]["rog_style"].append(r["rog_style"]["score"].get("correct", False))
        by_t[r["type"]]["strategy"].append(r["strategy"]["score"].get("correct", False))

    print("\n========== 3-WAY ON QA-500 STRATIFIED 100 ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Baseline':>10s} {'RoG-style':>10s} {'Strategy':>10s} | {'Δ S-B':>8s} {'Δ S-R':>8s}")
    print("-" * 84)
    ob, orog, os_, on = 0, 0, 0, 0
    for t, d in sorted(by_t.items()):
        n = len(d["baseline"])
        b = sum(d["baseline"]); r = sum(d["rog_style"]); s = sum(d["strategy"])
        ob += b; orog += r; os_ += s; on += n
        print(f"{t:<22s} {n:>3d} | {b/n:>10.3f} {r/n:>10.3f} {s/n:>10.3f} | "
              f"{(s-b)/n*100:>7.1f}% {(s-r)/n*100:>7.1f}%")
    print("-" * 84)
    print(f"{'OVERALL':<22s} {on:>3d} | {ob/on:>10.3f} {orog/on:>10.3f} {os_/on:>10.3f} | "
          f"{(os_-ob)/on*100:>7.1f}% {(os_-orog)/on*100:>7.1f}%")

    out = ROOT / "results" / "qa500_3way_100.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
