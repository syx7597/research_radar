"""
3-way comparison on the FULL qa_500 set (499 questions, all 11 types).

Differences vs run_e2e_3way_qa500.py:
  - All 499 questions instead of a stratified 100-Q sample
  - Per-question checkpoint to results/qa500_3way_full.json
    (so we can resume after an interrupt)
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
from qa_rog_baseline import run_rog_baseline
from graphrag_retriever import HybridRetriever
from experiments.run_e2e_30 import baseline_answer, score_question


CHECKPOINT_PATH = ROOT / "results" / "qa500_3way_full.json"
SUMMARY_PATH    = ROOT / "results" / "qa500_3way_full_summary.md"


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("per_question", [])
        done = {r["id"] for r in results}
        return results, done
    return [], set()


def save_checkpoint(results):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)


def aggregate_and_write_summary(results):
    by_t = defaultdict(lambda: {"baseline": [], "rog_style": [], "strategy": []})
    for r in results:
        by_t[r["type"]]["baseline"].append(r["baseline"]["score"].get("correct", False))
        by_t[r["type"]]["rog_style"].append(r["rog_style"]["score"].get("correct", False))
        by_t[r["type"]]["strategy"].append(r["strategy"]["score"].get("correct", False))

    lines = ["# qa_500 Full 3-way Results", ""]
    lines.append("| Type | n | Baseline | RoG-style | Strategy | Δ S-B | Δ S-R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    ob = orog = os_ = on = 0
    for t in sorted(by_t):
        d = by_t[t]
        n = len(d["baseline"])
        b = sum(d["baseline"]); r = sum(d["rog_style"]); s = sum(d["strategy"])
        ob += b; orog += r; os_ += s; on += n
        lines.append(f"| {t} | {n} | {b/n:.3f} | {r/n:.3f} | {s/n:.3f} | "
                     f"{(s-b)/n*100:+.1f}pp | {(s-r)/n*100:+.1f}pp |")
    lines.append(f"| **OVERALL** | **{on}** | **{ob/on:.3f}** | **{orog/on:.3f}** | "
                 f"**{os_/on:.3f}** | **{(os_-ob)/on*100:+.1f}pp** | **{(os_-orog)/on*100:+.1f}pp** |")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))


def main():
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]
    print(f"Loaded {len(all_qs)} questions from qa_500.json")

    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    retriever = HybridRetriever(triples, enable_reranker=False)
    pipe = StrategyPipeline(triples_path=str(big_path), lookup_retriever=retriever)

    results, done = load_checkpoint()
    print(f"Resuming from checkpoint: {len(done)} questions already done")

    todo = [q for q in all_qs if q["id"] not in done]
    print(f"Remaining: {len(todo)} questions")

    t0 = time.time()
    for i, q in enumerate(todo, 1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(todo) - i + 1) if i > 1 else 0
        print(f"\n[{i:>4}/{len(todo)}] {q['id']:18s} ({q['type']:18s}) "
              f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
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
            "baseline":  {"answer": b_out["answer"][:300] if isinstance(b_out.get("answer"), str) else "",
                          "score": b_score},
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

        # checkpoint every question
        save_checkpoint(results)

    aggregate_and_write_summary(results)
    print(f"\nSaved: {CHECKPOINT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
