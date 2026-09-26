"""
Oracle-dispatcher experiment (M3 from REVIEW_NOTES.md).

Bypasses the LLM router; uses each question's gold `type` to deterministically
select the strategy via lexicon/question_types.json::type_to_strategy.

Output: results/oracle_dispatcher_full.json + summary.md
Compares Strategy (LLM-routed) vs Strategy-Oracle (gold-routed)
to isolate dispatcher noise from operator ceiling.

Pipeline:
  question + gold qtype  →  type_to_strategy  →  parser → executor → answerer
                            (skip LLM router)
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

from qa_strategy_pipeline import (
    StrategyPipeline, KGIndex, parse_args, render_context, answer_with_llm,
    exec_lookup, exec_exhaustive, exec_complement, exec_path_plan,
    exec_constrained_join, exec_dual_subgraph,
)
from graphrag_retriever import HybridRetriever
from experiments.run_e2e_30 import score_question


CHECKPOINT_PATH = ROOT / "results" / "oracle_dispatcher_full.json"
SUMMARY_PATH    = ROOT / "results" / "oracle_dispatcher_full_summary.md"


def load_type_to_strategy():
    with open(ROOT / "lexicon" / "question_types.json", encoding="utf-8") as f:
        return json.load(f)["type_to_strategy"]


def run_with_oracle_qtype(pipe: StrategyPipeline, question: str, gold_qtype: str,
                          type_to_strategy: dict) -> dict:
    """Mirror StrategyPipeline.run() but skip the LLM router."""
    strategy = type_to_strategy.get(gold_qtype, "lookup")

    args = parse_args(question, gold_qtype, strategy, aliases=pipe.aliases, kg=pipe.kg)

    if strategy == "exhaustive":
        evidence = exec_exhaustive(args, pipe.kg, pipe.aliases)
    elif strategy == "complement":
        evidence = exec_complement(args, pipe.kg, pipe.aliases)
    elif strategy == "path_plan":
        evidence = exec_path_plan(args, pipe.kg, pipe.aliases)
    elif strategy == "constrained_join":
        evidence = exec_constrained_join(args, pipe.kg, pipe.aliases)
    elif strategy == "dual_subgraph":
        evidence = exec_dual_subgraph(args, pipe.kg, pipe.aliases)
    else:  # lookup
        pipe._ensure_lookup()
        evidence = exec_lookup(args, pipe.kg, pipe.aliases, pipe.lookup_retriever, question)

    ctx = render_context(strategy, evidence, args)
    answer = answer_with_llm(question, strategy, ctx)
    return {
        "question": question,
        "qtype": gold_qtype,
        "strategy": strategy,
        "args": args,
        "evidence": evidence,
        "context": ctx,
        "answer": answer,
    }


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


def aggregate_and_write_summary(results, llm_results):
    """Compare oracle vs LLM-routed (the existing 499Q full run)."""
    by_t_oracle = defaultdict(list)
    by_t_llm = defaultdict(list)
    for r in results:
        by_t_oracle[r["type"]].append(r["score"].get("correct", False))
    for r in llm_results:
        by_t_llm[r["type"]].append(r["strategy"]["score"].get("correct", False))

    lines = ["# Oracle vs LLM Dispatcher (499Q full)", ""]
    lines.append("Oracle uses gold q['type'] → type_to_strategy; LLM uses qa_router.")
    lines.append("")
    lines.append("| Type | n | LLM-routed | Oracle | Δ (Oracle−LLM) |")
    lines.append("|---|---:|---:|---:|---:|")
    on_o = on_l = ntot = 0
    for t in sorted(by_t_oracle):
        n = len(by_t_oracle[t])
        ot = sum(by_t_oracle[t])
        lt = sum(by_t_llm.get(t, []))
        on_o += ot; on_l += lt; ntot += n
        lines.append(f"| {t} | {n} | {lt/n:.3f} | {ot/n:.3f} | {(ot-lt)/n*100:+.1f}pp |")
    lines.append(f"| **OVERALL** | **{ntot}** | **{on_l/ntot:.3f}** | **{on_o/ntot:.3f}** | "
                 f"**{(on_o-on_l)/ntot*100:+.1f}pp** |")
    lines.append("")
    lines.append(f"**Headline**: Oracle dispatcher ceiling = {on_o/ntot*100:.1f}%, "
                 f"LLM dispatcher = {on_l/ntot*100:.1f}%, "
                 f"gap = {(on_o-on_l)/ntot*100:+.1f}pp.")
    lines.append("")
    lines.append(f"This isolates **dispatcher noise** ({(on_o-on_l)/ntot*100:.1f}pp) "
                 f"from **operator+KG ceiling** ({(1-on_o/ntot)*100:.1f}pp room).")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


def main():
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]
    print(f"Loaded {len(all_qs)} questions")

    type_to_strategy = load_type_to_strategy()
    print(f"Loaded type→strategy mapping: {len(type_to_strategy)} types")

    # load existing LLM-routed full results for comparison
    llm_results_path = ROOT / "results" / "qa500_3way_full.json"
    with open(llm_results_path, encoding="utf-8") as f:
        llm_results = json.load(f)["per_question"]
    print(f"Loaded {len(llm_results)} LLM-routed reference results")

    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    retriever = HybridRetriever(triples, enable_reranker=False)
    pipe = StrategyPipeline(triples_path=str(big_path), lookup_retriever=retriever)

    results, done = load_checkpoint()
    print(f"Resuming: {len(done)} done")
    todo = [q for q in all_qs if q["id"] not in done]
    print(f"TODO: {len(todo)}")

    t0 = time.time()
    for i, q in enumerate(todo, 1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(todo) - i + 1) if i > 1 else 0
        print(f"\n[{i:>4}/{len(todo)}] {q['id']:18s} ({q['type']:18s}) "
              f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")

        try:
            out = run_with_oracle_qtype(pipe, q["question_zh"], q["type"], type_to_strategy)
            score = score_question(q, out["answer"], kg)
        except Exception as e:
            out = {"answer": f"[ERR {e}]", "strategy": type_to_strategy.get(q["type"], "?")}
            score = {"correct": False, "error": str(e)}

        rec = {
            "id": q["id"], "type": q["type"], "question": q["question_zh"],
            "gold_answer": q.get("gold_answer"),
            "strategy": out.get("strategy"),
            "answer": out["answer"][:300] if isinstance(out.get("answer"), str) else "",
            "score": score,
        }
        results.append(rec)
        ok = "OK" if score.get("correct") else "  "
        print(f"   strategy={out.get('strategy'):16s} [{ok}]")
        save_checkpoint(results)

    aggregate_and_write_summary(results, llm_results)
    print(f"\nSaved: {CHECKPOINT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
