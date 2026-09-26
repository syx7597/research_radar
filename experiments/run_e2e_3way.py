"""
3-way end-to-end evaluation:
  Baseline (current GraphRAG)  vs  RoG-style uniform path-plan  vs  Strategy-Routed (ours)

This isolates the contribution of question-type routing: RoG-style uses LLM
planning + KG walking but applies one uniform strategy to all questions.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex
from qa_rog_baseline import run_rog_baseline
from graphrag_retriever import HybridRetriever
from experiments.run_e2e_30 import baseline_answer, score_question


def main():
    val_path = ROOT / "evaluation" / "validation_30.json"
    with open(val_path, encoding="utf-8") as f:
        val = json.load(f)
    questions = val["questions"]

    print("Loading triples + retriever...")
    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    retriever = HybridRetriever(triples, enable_reranker=False)
    pipe = StrategyPipeline(triples_path=str(big_path), lookup_retriever=retriever)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i:>2}/{len(questions)}] {q['id']:15s} ({q['type']:18s})")

        # 1. baseline
        try:
            base_out = baseline_answer(q["question_zh"], retriever)
            base_score = score_question(q, base_out["answer"], kg)
        except Exception as e:
            base_out = {"answer": f"[ERR {e}]"}
            base_score = {"correct": False, "error": str(e)}

        # 2. RoG-style
        try:
            rog_out = run_rog_baseline(q["question_zh"], kg)
            rog_score = score_question(q, rog_out["answer"], kg)
        except Exception as e:
            rog_out = {"answer": f"[ERR {e}]"}
            rog_score = {"correct": False, "error": str(e)}

        # 3. Strategy-routed
        try:
            strat_out = pipe.run(q["question_zh"])
            strat_score = score_question(q, strat_out["answer"], kg)
        except Exception as e:
            strat_out = {"answer": f"[ERR {e}]", "qtype": "?", "strategy": "?"}
            strat_score = {"correct": False, "error": str(e)}

        rec = {
            "id": q["id"], "type": q["type"], "question": q["question_zh"],
            "gold_answer": q.get("gold_answer"),
            "baseline": {"answer": base_out["answer"][:300], "score": base_score},
            "rog_style": {"answer": rog_out["answer"][:300] if isinstance(rog_out.get("answer"), str) else "",
                          "score": rog_score,
                          "plan": rog_out.get("plan", {}).get("paths") if "plan" in rog_out else None},
            "strategy": {"qtype": strat_out.get("qtype"),
                         "strategy": strat_out.get("strategy"),
                         "answer": strat_out["answer"][:300] if isinstance(strat_out.get("answer"), str) else "",
                         "score": strat_score,
                         "args": strat_out.get("args") if "args" in strat_out else None},
        }
        results.append(rec)
        b_ok = "OK" if base_score.get("correct") else "  "
        r_ok = "OK" if rog_score.get("correct") else "  "
        s_ok = "OK" if strat_score.get("correct") else "  "
        print(f"   baseline:[{b_ok}] rog:[{r_ok}] strategy:[{s_ok}]")

    by_type = defaultdict(lambda: {"baseline":[], "rog_style":[], "strategy":[]})
    for r in results:
        by_type[r["type"]]["baseline"].append(r["baseline"]["score"].get("correct", False))
        by_type[r["type"]]["rog_style"].append(r["rog_style"]["score"].get("correct", False))
        by_type[r["type"]]["strategy"].append(r["strategy"]["score"].get("correct", False))

    print("\n========== 3-WAY END-TO-END ACCURACY ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Baseline':>10s} {'RoG-style':>10s} {'Strategy':>10s} | {'Δ S-B':>8s} {'Δ S-R':>8s}")
    print("-" * 84)
    ob, orog, os_, on = 0, 0, 0, 0
    for t, d in sorted(by_type.items()):
        n = len(d["baseline"])
        b = sum(d["baseline"]); r = sum(d["rog_style"]); s = sum(d["strategy"])
        ob += b; orog += r; os_ += s; on += n
        print(f"{t:<22s} {n:>3d} | {b/n:>10.3f} {r/n:>10.3f} {s/n:>10.3f} | "
              f"{(s-b)/n*100:>7.1f}% {(s-r)/n*100:>7.1f}%")
    print("-" * 84)
    print(f"{'OVERALL':<22s} {on:>3d} | {ob/on:>10.3f} {orog/on:>10.3f} {os_/on:>10.3f} | "
          f"{(os_-ob)/on*100:>7.1f}% {(os_-orog)/on*100:>7.1f}%")

    out = ROOT / "results" / "e2e_30_3way.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
