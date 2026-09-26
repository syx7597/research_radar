"""
Run the strategy pipeline on the bilingual stress set with the bilingual layer
ON vs OFF. The drop quantifies the layer's value when surface forms don't match
KG canonical (e.g., user types USA but KG has 美国; user types AESA but KG has
有源相控阵).
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

import qa_strategy_pipeline
from qa_strategy_pipeline import StrategyPipeline, KGIndex
from experiments.run_e2e_30 import score_question


def main():
    with open(ROOT / "evaluation" / "bilingual_stress_30.json", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)

    pipe = StrategyPipeline()

    print(f"Running {len(questions)} stress questions twice (ON / OFF)\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i:>2}/{len(questions)}] {q['id']:18s} ({q['type']:18s})")
        print(f"   Q: {q['question_zh']}")
        print(f"   subs: {q['bilingual_subs']}")

        # bilingual ON (full)
        qa_strategy_pipeline.BILINGUAL_DISABLED = False
        try:
            on_out = pipe.run(q["question_zh"])
            on_score = score_question(q, on_out["answer"], kg)
        except Exception as e:
            on_out = {"answer": f"[ERR {e}]"}
            on_score = {"correct": False, "error": str(e)}

        # bilingual OFF (ablated)
        qa_strategy_pipeline.BILINGUAL_DISABLED = True
        try:
            off_out = pipe.run(q["question_zh"])
            off_score = score_question(q, off_out["answer"], kg)
        except Exception as e:
            off_out = {"answer": f"[ERR {e}]"}
            off_score = {"correct": False, "error": str(e)}
        qa_strategy_pipeline.BILINGUAL_DISABLED = False

        on_ok  = "OK" if on_score.get("correct") else "  "
        off_ok = "OK" if off_score.get("correct") else "  "
        print(f"   ON [{on_ok}]: {(on_out.get('answer') or '')[:120]}")
        print(f"   OFF[{off_ok}]: {(off_out.get('answer') or '')[:120]}")

        results.append({
            "id": q["id"], "type": q["type"], "question": q["question_zh"],
            "bilingual_subs": q["bilingual_subs"],
            "gold_answer": q.get("gold_answer"),
            "on":  {"answer": on_out["answer"][:300], "score": on_score},
            "off": {"answer": off_out["answer"][:300], "score": off_score},
        })

    # Aggregate
    by_t = defaultdict(lambda: {"on":[], "off":[]})
    for r in results:
        by_t[r["type"]]["on"].append(r["on"]["score"].get("correct", False))
        by_t[r["type"]]["off"].append(r["off"]["score"].get("correct", False))

    print("\n========== BILINGUAL STRESS ABLATION ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Bilingual ON':>14s} {'Bilingual OFF':>14s} {'Δ':>8s}")
    print("-" * 65)
    on_t, off_t, n_t = 0, 0, 0
    for t, d in sorted(by_t.items()):
        n = len(d["on"])
        on_c = sum(d["on"]); off_c = sum(d["off"])
        on_t += on_c; off_t += off_c; n_t += n
        delta = (off_c - on_c) / n * 100
        print(f"{t:<22s} {n:>3d} | {on_c/n:>14.3f} {off_c/n:>14.3f} {delta:>7.1f}%")
    print("-" * 65)
    print(f"{'OVERALL':<22s} {n_t:>3d} | {on_t/n_t:>14.3f} {off_t/n_t:>14.3f} {(off_t-on_t)/n_t*100:>7.1f}%")

    out = ROOT / "results" / "bilingual_stress_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
