"""
Cost & latency measurement for the 3 systems on a 20-question subset.

For each question, time:
  - Baseline (1 LLM call: answer)
  - RoG-style (2 LLM calls: planner + answer)
  - Strategy-Routed (3 LLM calls: router + parser + answer)

Records per-question wall-clock time. Aggregates: avg latency, total LLM calls,
estimated USD cost using DeepSeek pricing (~$0.14/$0.28 per 1M in/out tokens).
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex
from qa_rog_baseline import run_rog_baseline
from graphrag_retriever import HybridRetriever
from experiments.run_e2e_30 import baseline_answer

# DeepSeek pricing (cache-miss, 2025): input $0.14/1M tokens, output $0.28/1M tokens
PRICE_IN  = 0.14 / 1_000_000
PRICE_OUT = 0.28 / 1_000_000

# Approximate token counts per LLM call (calibrated from logs)
LLM_CALL_TOKENS = {
    # tokens_in_avg, tokens_out_avg
    "baseline_answer":  (450, 150),     # context + question, short answer
    "rog_planner":      (550, 80),      # relation listing + question, 3 paths
    "rog_answer":       (500, 200),     # walked-path context, summarized answer
    "router":           (400, 12),      # rules + question, type id only
    "parser":           (1100, 220),    # relation lex + question, JSON args
    "strategy_answer":  (550, 220),     # structured evidence, longer for enum
}

LLM_CALLS_BY_SYSTEM = {
    "baseline":   ["baseline_answer"],
    "rog_style":  ["rog_planner", "rog_answer"],
    "strategy":   ["router", "parser", "strategy_answer"],
}


def estimate_cost_usd(system: str) -> float:
    total = 0.0
    for call in LLM_CALLS_BY_SYSTEM[system]:
        tin, tout = LLM_CALL_TOKENS[call]
        total += tin * PRICE_IN + tout * PRICE_OUT
    return total


def main():
    random.seed(11)
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]

    # 20 stratified — 2 per type for the 10 types with enough questions
    by_t = defaultdict(list)
    for q in all_qs:
        by_t[q["type"]].append(q)
    sample = []
    for t in ("single_hop","relation_inverse","agg_count","agg_enum",
              "two_hop_bridge","attr_filter","negation","set_compare",
              "unanswerable","three_hop_chain"):
        pool = by_t.get(t, [])
        sample.extend(random.sample(pool, min(2, len(pool))))
    print(f"Sampled {len(sample)} questions for timing")

    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    retriever = HybridRetriever(triples, enable_reranker=False)
    pipe = StrategyPipeline(triples_path=str(big_path), lookup_retriever=retriever)

    timings = {"baseline": [], "rog_style": [], "strategy": []}
    for i, q in enumerate(sample, 1):
        print(f"\n[{i:>2}/{len(sample)}] {q['id']:15s} ({q['type']:18s})")

        t0 = time.time()
        baseline_answer(q["question_zh"], retriever)
        t_b = time.time() - t0
        timings["baseline"].append(t_b)

        t0 = time.time()
        run_rog_baseline(q["question_zh"], kg)
        t_r = time.time() - t0
        timings["rog_style"].append(t_r)

        t0 = time.time()
        pipe.run(q["question_zh"])
        t_s = time.time() - t0
        timings["strategy"].append(t_s)

        print(f"   baseline: {t_b:5.1f}s  RoG: {t_r:5.1f}s  Strategy: {t_s:5.1f}s")

    n = len(sample)
    print("\n========== AGGREGATE LATENCY ==========")
    print(f"{'System':<14s} {'mean':>7s} {'min':>7s} {'max':>7s} {'p50':>7s}    "
          f"{'LLM calls':>10s} {'cost/Q':>10s}")
    for sys_name in ("baseline", "rog_style", "strategy"):
        vals = sorted(timings[sys_name])
        mean = sum(vals) / n
        p50 = vals[n // 2]
        cost = estimate_cost_usd(sys_name)
        n_calls = len(LLM_CALLS_BY_SYSTEM[sys_name])
        print(f"{sys_name:<14s} {mean:>6.1f}s {vals[0]:>6.1f}s {vals[-1]:>6.1f}s "
              f"{p50:>6.1f}s    {n_calls:>10d} ${cost:>9.5f}")

    print("\nDeepSeek price model: input $0.14/1M, output $0.28/1M (cache-miss)")

    # Accuracy from earlier experiments
    print("\n========== ACCURACY-VS-COST TRADE-OFF (from QA-500 100Q) ==========")
    print(f"{'System':<14s} {'Acc':>6s} {'Δ vs B':>8s} {'$/Q':>10s} {'$/100':>10s} "
          f"{'Δ$/Δacc':>14s}")
    accs = {"baseline": 0.610, "rog_style": 0.690, "strategy": 0.940}
    base_cost = estimate_cost_usd("baseline")
    base_acc = accs["baseline"]
    for s in ("baseline","rog_style","strategy"):
        cost = estimate_cost_usd(s)
        acc = accs[s]
        delta_acc = acc - base_acc
        delta_cost = cost - base_cost
        if delta_acc > 0:
            ratio = f"${(delta_cost/delta_acc):.5f}/pp"
        else:
            ratio = "n/a (B)"
        print(f"{s:<14s} {acc:>6.3f} {(acc-base_acc)*100:>+7.1f}pp ${cost:>9.5f} ${cost*100:>9.4f} {ratio:>14s}")

    out = ROOT / "results" / "cost_latency.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_questions": n,
            "timings": timings,
            "llm_call_tokens": LLM_CALL_TOKENS,
            "llm_calls_by_system": LLM_CALLS_BY_SYSTEM,
            "estimated_cost_per_question_usd": {s: estimate_cost_usd(s) for s in LLM_CALLS_BY_SYSTEM},
            "accuracy_qa500_100": accs,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
