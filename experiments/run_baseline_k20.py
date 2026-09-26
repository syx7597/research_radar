"""
K=20 baseline ablation on RadarKG-QA-499.

Tests whether Strategy's +35.9 pp gain depends on the baseline being
under-retrieved at K=8. If baseline at K=20 still scores ~52-55%, the gain
is structural; if it jumps to ~75%+, our 'structural failure' framing is
overclaimed and the +35.9 pp partly reflects an unfair baseline.

Output: results/baseline_k20_full.json + summary.md, + paired-bootstrap CI
vs both the existing K=8 baseline and Strategy.
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import KGIndex, llm_call, ANSWER_SYSTEM
from graphrag_retriever import HybridRetriever
from experiments.run_e2e_30 import score_question


TOP_K = 20
IN_QA = ROOT / "evaluation" / "qa_500.json"
IN_TRIPLES = ROOT / "graphrag_index" / "merged_triples.json"
OUT_JSON = ROOT / "results" / f"baseline_k{TOP_K}_full.json"
OUT_MD   = ROOT / "results" / f"baseline_k{TOP_K}_full_summary.md"


def baseline_k_answer(question, retriever, k):
    res = retriever.retrieve(question, top_k=k, use_graph_expansion=True, use_reranker=False)
    ctx = res.get("context", "")
    user = f"问题：{question}\n\n检索证据：\n{ctx}\n\n请给出答案："
    ans = llm_call(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user",   "content": user}],
        max_tokens=400,
    )
    return ans


def main():
    with open(IN_QA, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    print(f"Loaded {len(qs)} questions")

    with open(IN_TRIPLES, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    retriever = HybridRetriever(triples, enable_reranker=False)

    # Resume
    if OUT_JSON.exists():
        with open(OUT_JSON, encoding="utf-8") as f:
            results = json.load(f).get("per_question", [])
        done = {r["id"] for r in results}
        todo = [q for q in qs if q["id"] not in done]
        print(f"Resume: {len(done)} done, {len(todo)} remaining")
    else:
        results = []
        todo = qs

    t0 = time.time()
    for i, q in enumerate(todo, 1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(todo) - i + 1) if i > 1 else 0
        if i % 25 == 0 or i == 1:
            print(f"[{i:>4}/{len(todo)}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
        try:
            ans = baseline_k_answer(q["question_zh"], retriever, TOP_K)
            score = score_question(q, ans, kg)
        except Exception as e:
            ans = f"[ERR {e}]"
            score = {"correct": False, "error": str(e)}
        results.append({
            "id": q["id"], "type": q["type"],
            "question": q["question_zh"], "gold_answer": q.get("gold_answer"),
            "answer": ans[:300] if isinstance(ans, str) else "",
            "score": score,
        })
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)

    # Aggregate
    by_t = defaultdict(list)
    for r in results:
        by_t[r["type"]].append(r["score"].get("correct", False))
    n = len(results)
    correct = sum(1 for r in results if r["score"].get("correct", False))
    lines = [f"# Baseline K={TOP_K} on RadarKG-QA-499", ""]
    lines.append(f"- N = {n}")
    lines.append(f"- Baseline K={TOP_K} accuracy: **{correct}/{n} = {correct/n*100:.1f}%**")
    lines.append("")
    lines.append("| Type | n | Baseline K=" + str(TOP_K) + " |")
    lines.append("|---|---:|---:|")
    for t in sorted(by_t):
        bools = by_t[t]
        nt = len(bools)
        ct = sum(bools)
        lines.append(f"| {t} | {nt} | {ct/nt*100:.1f}% |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
