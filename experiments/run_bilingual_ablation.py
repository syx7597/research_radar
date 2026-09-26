"""
Bilingual layer ablation on the 100-question stratified subset.

Disables:
  - alias index (USA → 美国, AESA → 有源相控阵, ...)
  - frequency band normalization (S波段 → S)
  - suffix stripping (波段 / 公司 / 雷达 ...)
  - relation equivalence-class fallback (countryOfOrigin ↔ operatedBy)
  - type-aware relation substitution in _validate_args

Re-runs all 100 questions with the bilingual layer off, then compares to the
full pipeline. The drop quantifies the bilingual layer's contribution beyond
the strategy router itself.
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

import qa_strategy_pipeline
from qa_strategy_pipeline import StrategyPipeline, KGIndex
from experiments.run_e2e_30 import score_question


def main():
    with open(ROOT / "results" / "qa500_3way_100_v2.json", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        qa_by_id = {q["id"]: q for q in json.load(f)["questions"]}
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)

    # Sanity: verify the kill-switch flag exists
    assert hasattr(qa_strategy_pipeline, "BILINGUAL_DISABLED"), "kill-switch missing"

    pipe = StrategyPipeline()

    baseline_per_q = {r["id"]: r for r in baseline_data["per_question"]}
    n_total = len(baseline_per_q)
    print(f"Re-running {n_total} questions with BILINGUAL_DISABLED=True\n")

    # Activate kill switch
    qa_strategy_pipeline.BILINGUAL_DISABLED = True

    ablated = []
    for i, (qid, base_rec) in enumerate(baseline_per_q.items(), 1):
        q = qa_by_id[qid]
        print(f"[{i:>3}/{n_total}] {qid:15s} ({q['type']:18s}) {q['question_zh'][:50]}")
        try:
            out = pipe.run(q["question_zh"])
            score = score_question(q, out["answer"], kg)
        except Exception as e:
            out = {"answer": f"[ERR {e}]", "qtype": "?", "strategy": "?"}
            score = {"correct": False, "error": str(e)}
        prev_ok = base_rec["strategy"]["score"].get("correct", False)
        new_ok = score.get("correct", False)
        flip = "  " if new_ok == prev_ok else ("v" if (prev_ok and not new_ok) else "^")
        print(f"      {flip}  prev={prev_ok}  abl={new_ok}")
        ablated.append({
            "id": qid, "type": q["type"], "question": q["question_zh"],
            "gold_answer": q.get("gold_answer"),
            "full_score": base_rec["strategy"]["score"],
            "ablated_score": score,
            "ablated_answer": out.get("answer", "")[:300] if isinstance(out.get("answer"), str) else "",
            "ablated_strategy": out.get("strategy", ""),
            "ablated_args": out.get("args"),
        })
        time.sleep(0.05)

    # Restore
    qa_strategy_pipeline.BILINGUAL_DISABLED = False

    # Aggregate per type
    by_t = defaultdict(lambda: {"full":[], "ablated":[]})
    for r in ablated:
        by_t[r["type"]]["full"].append(r["full_score"].get("correct", False))
        by_t[r["type"]]["ablated"].append(r["ablated_score"].get("correct", False))

    print("\n========== BILINGUAL LAYER ABLATION ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Full':>6s} {'Abl':>6s} {'Δ':>8s}")
    print("-" * 50)
    of, oa, on = 0, 0, 0
    for t, d in sorted(by_t.items()):
        n = len(d["full"])
        f_ = sum(d["full"]); a = sum(d["ablated"])
        of += f_; oa += a; on += n
        delta = (a - f_) / n * 100
        print(f"{t:<22s} {n:>3d} | {f_/n:>6.3f} {a/n:>6.3f} {delta:>7.1f}%")
    print("-" * 50)
    print(f"{'OVERALL':<22s} {on:>3d} | {of/on:>6.3f} {oa/on:>6.3f} {(oa-of)/on*100:>7.1f}%")

    out_path = ROOT / "results" / "ablation_bilingual_100.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_question": ablated}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
