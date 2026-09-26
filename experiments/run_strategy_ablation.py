"""
Per-strategy ablation on the 100-question stratified subset.

For each of the 5 ablatable strategies:
  exhaustive / complement / path_plan / constrained_join / dual_subgraph

we re-run only those questions whose router originally picked that strategy,
forcing the executor to use `lookup` instead. Questions unaffected by the
ablation reuse their original outcome (Strategy-Routed full pipeline).

This produces the per-strategy contribution table for the paper.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex
from experiments.run_e2e_30 import score_question


ABLATIONS = ["exhaustive", "complement", "path_plan", "constrained_join", "dual_subgraph"]


def main():
    with open(ROOT / "results" / "qa500_3way_100_v2.json", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        qa_by_id = {q["id"]: q for q in json.load(f)["questions"]}
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)

    pipe = StrategyPipeline()

    # baseline: the full pipeline scores from qa500_3way_100_v2
    baseline_per_question = {r["id"]: r for r in baseline_data["per_question"]}
    n_total = len(baseline_per_question)
    print(f"Baseline (full Strategy-Routed): {n_total} questions")

    # collect per-question (id → original_strategy) — we use the recorded value
    # from the full run so we know whose router picked what
    by_strategy = defaultdict(list)
    for qid, rec in baseline_per_question.items():
        s = rec["strategy"].get("strategy", "")
        by_strategy[s].append(qid)
    print("\nQuestion distribution by chosen strategy:")
    for s in ["lookup"] + ABLATIONS:
        print(f"  {s:20s} {len(by_strategy.get(s, []))}")

    # Per-ablation: re-run affected questions
    ablation_results = {}
    for ab in ABLATIONS:
        affected = by_strategy.get(ab, [])
        print(f"\n=== Ablating {ab} ({len(affected)} questions affected) ===")
        ablated_outcomes = {}
        for i, qid in enumerate(affected, 1):
            q = qa_by_id[qid]
            print(f"  [{i:>2}/{len(affected)}] {qid:15s} ({q['type']:18s}) {q['question_zh'][:50]}")
            try:
                out = pipe.run(q["question_zh"], ablate_strategy=ab)
                score = score_question(q, out["answer"], kg)
            except Exception as e:
                out = {"answer": f"[ERR {e}]"}
                score = {"correct": False, "error": str(e)}
            ablated_outcomes[qid] = {
                "answer": out["answer"][:300],
                "score": score,
                "ablated_to": out.get("strategy", ""),
            }
            ok = "OK" if score.get("correct") else "  "
            print(f"      → [{ok}]")
            time.sleep(0.1)

        # build full result set for this ablation: ablated outcomes + unaffected baselines
        full = []
        for qid, base_rec in baseline_per_question.items():
            if qid in ablated_outcomes:
                full.append({
                    "id": qid, "type": base_rec["type"], "question": base_rec["question"],
                    "score": ablated_outcomes[qid]["score"],
                    "answer": ablated_outcomes[qid]["answer"],
                    "ablated": True,
                })
            else:
                full.append({
                    "id": qid, "type": base_rec["type"], "question": base_rec["question"],
                    "score": base_rec["strategy"]["score"],
                    "answer": base_rec["strategy"]["answer"],
                    "ablated": False,
                })
        ablation_results[ab] = full

    # Aggregate
    print("\n========== ABLATION RESULTS (overall + per-type) ==========")
    # baseline overall
    base_correct = sum(1 for r in baseline_per_question.values()
                       if r["strategy"]["score"].get("correct", False))
    print(f"\nFull pipeline (no ablation):  {base_correct}/{n_total} = {base_correct/n_total:.3f}")

    # by_type for baseline
    base_by_type = defaultdict(list)
    for r in baseline_per_question.values():
        base_by_type[r["type"]].append(r["strategy"]["score"].get("correct", False))

    # Compute ablation-vs-full table
    types_sorted = sorted(base_by_type.keys())
    print(f"\n{'Type':<22s} {'n':>3s} | {'Full':>6s} |", end="")
    for ab in ABLATIONS:
        print(f" {('-'+ab):>14s}", end="")
    print()
    print("-" * (33 + 15 * len(ABLATIONS)))

    aggregate_drops = {ab: 0 for ab in ABLATIONS}
    for t in types_sorted:
        ids_t = [r["id"] for r in baseline_per_question.values() if r["type"] == t]
        n = len(ids_t)
        base_acc = sum(base_by_type[t]) / n
        print(f"{t:<22s} {n:>3d} | {base_acc:>6.3f} |", end="")
        for ab in ABLATIONS:
            ab_results = {r["id"]: r for r in ablation_results[ab]}
            ab_correct = sum(1 for qid in ids_t if ab_results[qid]["score"].get("correct", False))
            ab_acc = ab_correct / n
            delta = ab_acc - base_acc
            aggregate_drops[ab] += (ab_correct - sum(base_by_type[t]))
            mark = " " if delta == 0 else ("v" if delta < 0 else "^")
            print(f" {ab_acc:>6.3f}({delta*100:+5.1f}){mark}", end="")
        print()
    print("-" * (33 + 15 * len(ABLATIONS)))
    full_acc = base_correct / n_total
    print(f"{'OVERALL':<22s} {n_total:>3d} | {full_acc:>6.3f} |", end="")
    for ab in ABLATIONS:
        ab_correct = sum(1 for r in ablation_results[ab] if r["score"].get("correct", False))
        ab_acc = ab_correct / n_total
        delta = ab_acc - full_acc
        print(f" {ab_acc:>6.3f}({delta*100:+5.1f}) ", end="")
    print()

    out_path = ROOT / "results" / "ablation_strategies_100.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_per_question": {qid: {
                "id": qid, "type": r["type"], "question": r["question"],
                "score": r["strategy"]["score"], "answer": r["strategy"]["answer"],
                "strategy": r["strategy"].get("strategy", ""),
            } for qid, r in baseline_per_question.items()},
            "ablations": ablation_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
