"""
High-cardinality Count stress test on KQA Pro.

Hypothesis: Strategy's +35.9 pp gain on RadarKG comes from structural failure
of top-K on questions where answer cardinality > K. KQA Pro overall ties
(±0.2 pp) because most Count answers are ≤5. But there are 77 KQA Pro Count
questions with answer ≥30 — exactly the structural-failure regime.

This experiment filters to that 77-Q subset, runs Baseline (BM25 top-K=20 on
concept-expanded subgraph) vs Strategy (exhaustive operator on full subgraph),
and tests whether Strategy wins big on this slice.

Output: results/kqa_pro_highcard_count.{json,md}
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import KGIndex, llm_call, ANSWER_SYSTEM, exec_exhaustive
from datasets.kqa_pro_subgraph import (
    build_question_subgraph_v2, _entity_name_to_ids, _id_to_name, _build_concept_index,
)
from datasets.kqa_pro_parser import parse_args as kqa_parse_args
from experiments.kqa_pro_e2e import simple_bm25_topk, score, build_args_from_program


IN_VAL = ROOT / "datasets" / "kqa_pro" / "val.json"
IN_KB  = ROOT / "datasets" / "kqa_pro" / "kb.json"
OUT_JSON = ROOT / "results" / "kqa_pro_highcard_count.json"
OUT_MD   = ROOT / "results" / "kqa_pro_highcard_count_summary.md"
MIN_ANS = 30  # high-card threshold


def baseline_count_answer(question, triples, k=20):
    top = simple_bm25_topk(question, triples, k=k)
    ctx_lines = [f"- {t['head']} | {t['relation']} | {t['tail']}" for t in top]
    ctx = "\n".join(ctx_lines)
    user = (f"Question: {question}\n\nEvidence (top-{k} retrieved triples):\n{ctx}\n\n"
            f"Give a single integer count.")
    ans = llm_call(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user",   "content": user}],
        max_tokens=200,
    )
    return ans


def strategy_count_answer(question, qrec, triples, kg):
    """Strategy with exhaustive operator + LLM parser."""
    try:
        args = kqa_parse_args(question, "agg_count", "exhaustive")
    except Exception:
        topic_entities = []
        for step in qrec["program"]:
            if step["function"] == "Find":
                topic_entities.extend(step["inputs"])
        args = build_args_from_program(qrec["program"], topic_entities)

    try:
        evidence = exec_exhaustive(args, kg, {})
        items = evidence.get("items") or evidence.get("entities") or [] if isinstance(evidence, dict) else []
    except Exception:
        items = []

    # Lookup fallback if operator returned nothing
    if not items:
        top = simple_bm25_topk(question, triples, k=20)
        items = [f"{t['head']} | {t['relation']} | {t['tail']}" for t in top]

    # Render: tell answerer the count of items
    if isinstance(items[0], str) if items else False:
        ctx_lines = [f"- {it}" for it in items[:200]]
    else:
        ctx_lines = [f"- {it.get('head','')} | {it.get('relation','')} | {it.get('tail','')}"
                     for it in items[:200]]
    ctx = "\n".join(ctx_lines)
    n_items = len(items)
    user = (f"Question: {question}\n\n"
            f"Exhaustive enumeration ({n_items} items retrieved):\n{ctx}\n\n"
            f"Give a single integer count.")
    ans = llm_call(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user",   "content": user}],
        max_tokens=200,
    )
    return ans, n_items


def main():
    rng = random.Random(42)
    with open(IN_VAL, encoding="utf-8") as f:
        val = json.load(f)
    print("Loading KB...")
    with open(IN_KB, encoding="utf-8") as f:
        kb = json.load(f)
    n2i = _entity_name_to_ids(kb)
    i2n = _id_to_name(kb)
    print("Building concept index...")
    ci = _build_concept_index(kb)

    # Filter to high-cardinality Count questions
    high_card = []
    for q in val:
        if not q.get("program") or q["program"][-1]["function"] != "Count":
            continue
        m = re.search(r"-?\d+", str(q["answer"]))
        if not m: continue
        ans_n = int(m.group())
        if ans_n < MIN_ANS: continue
        # Need FilterConcept so subgraph v2 can expand
        if not any(s["function"] == "FilterConcept" for s in q["program"]):
            continue
        q["_ans_n"] = ans_n
        high_card.append(q)
    print(f"High-card Count questions (answer >= {MIN_ANS}): {len(high_card)}")

    results = []
    t0 = time.time()
    for i, q in enumerate(high_card, 1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(high_card) - i + 1) if i > 1 else 0
        if i % 5 == 0 or i == 1:
            print(f"[{i:>3}/{len(high_card)}] ans={q['_ans_n']:5d} "
                  f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
        sg = build_question_subgraph_v2(q, kb, n2i, i2n, concept_index=ci, hops=2)
        triples = sg["triples"]
        kg = KGIndex(triples)

        try:
            b_ans = baseline_count_answer(q["question"], triples, k=20)
        except Exception as e:
            b_ans = f"[ERR {e}]"
        try:
            s_ans, n_enum = strategy_count_answer(q["question"], q, triples, kg)
        except Exception as e:
            s_ans, n_enum = f"[ERR {e}]", 0

        b_score = score(b_ans, str(q["_ans_n"]), "count")
        s_score = score(s_ans, str(q["_ans_n"]), "count")
        results.append({
            "id": q.get("id", ""),
            "question": q["question"],
            "gold_count": q["_ans_n"],
            "subgraph_size": len(triples),
            "n_enumerated_by_strategy": n_enum,
            "baseline_answer": b_ans[:200] if isinstance(b_ans, str) else "",
            "strategy_answer": s_ans[:200] if isinstance(s_ans, str) else "",
            "baseline_correct": b_score,
            "strategy_correct": s_score,
        })
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)
        ok_b = "OK" if b_score else "  "
        ok_s = "OK" if s_score else "  "
        print(f"   gold={q['_ans_n']:5d}  B[{ok_b}] S[{ok_s}] (n_enum={n_enum})")

    # Aggregate
    nb = sum(r["baseline_correct"] for r in results)
    ns = sum(r["strategy_correct"] for r in results)
    n = len(results)
    lines = [f"# KQA Pro High-Cardinality Count Stress Test", ""]
    lines.append(f"- Subset: Count questions with gold answer >= {MIN_ANS}")
    lines.append(f"- N = {n}")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Baseline (BM25 top-20 + LLM count): **{nb}/{n} = {nb/n*100:.1f}%**")
    lines.append(f"- Strategy (exhaustive operator):     **{ns}/{n} = {ns/n*100:.1f}%**")
    lines.append(f"- Delta (Strategy - Baseline):        **{(ns-nb)/n*100:+.1f} pp**")
    lines.append("")
    # Bucket by answer cardinality
    from collections import defaultdict
    by_bucket = defaultdict(lambda: {"b": [], "s": []})
    for r in results:
        v = r["gold_count"]
        if v <= 50: b = "30-50"
        elif v <= 100: b = "51-100"
        elif v <= 500: b = "101-500"
        else: b = "501+"
        by_bucket[b]["b"].append(r["baseline_correct"])
        by_bucket[b]["s"].append(r["strategy_correct"])
    lines.append("## By answer-cardinality bucket")
    lines.append("")
    lines.append("| Bucket | n | Baseline | Strategy | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in ["30-50", "51-100", "101-500", "501+"]:
        d = by_bucket[b]
        if not d["b"]: continue
        nt = len(d["b"])
        bb = sum(d["b"]); ss = sum(d["s"])
        lines.append(f"| {b} | {nt} | {bb}/{nt}={bb/nt*100:.0f}% | {ss}/{nt}={ss/nt*100:.0f}% | "
                     f"{(ss-bb)/nt*100:+.0f}pp |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
