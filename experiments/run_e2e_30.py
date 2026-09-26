"""
End-to-end evaluation: Strategy-Routed GraphRAG vs Baseline (current qa_pipeline.py style)
on the 30-question validation set.

Scoring (per question type):
  agg_count     : extract integer from answer; exact match
  agg_enum      : recall = |gold ∩ entities mentioned in answer| / |gold|
  negation      : parse 是/否; correct if matches gold polarity
  two_hop_bridge / three_hop_chain : substring match for gold_answer string(s)
  attr_filter   : recall over gold intersection set
  set_compare   : parse 是/否/相同/不同; match gold polarity
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex, llm_call, ANSWER_SYSTEM
from graphrag_retriever import HybridRetriever


# ─────────────────────────────────────────────────────────────
#  Baseline: current qa_pipeline.py style
# ─────────────────────────────────────────────────────────────

def baseline_answer(question: str, retriever: HybridRetriever) -> dict:
    """Reproduce current pipeline: retrieve top-K → DeepSeek answer."""
    res = retriever.retrieve(question, top_k=8, use_graph_expansion=True, use_reranker=False)
    context = res.get("context", "")
    user = (f"问题：{question}\n\n检索证据：\n{context}\n\n请给出答案：")
    answer = llm_call(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user",   "content": user}],
        max_tokens=400,
    )
    return {"answer": answer, "context": context}


# ─────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────

def _extract_int(text: str) -> int | None:
    """Extract first non-trivial integer from answer text. Skip digits embedded
    in model-name patterns (e.g., AN/APS-124, RDR-1400C) to reduce false matches."""
    if not text:
        return None
    # mask common model-name patterns first
    masked = re.sub(r"[A-Z][A-Z0-9]*[/\-][A-Z]?\d+[A-Za-z\d/\-()]*", " ", text)
    # \d+ without \b — in Python 3 unicode mode, CJK chars count as \w so \b would fail
    m = re.search(r"\d+", masked)
    if m:
        return int(m.group(0))
    # fallback: any digit at all
    m2 = re.search(r"\d+", text)
    return int(m2.group(0)) if m2 else None


def _yes_or_no(text: str) -> str | None:
    """Parse 是/否. 'yes' if affirmative, 'no' if negative, None if unclear."""
    t = text.strip()
    head = t[:80]
    no_kw  = ["否", "不是", "没有", "不被", "不属于", "不部署", "未被", "未发现", "未知", "无相关",
              "no", "No", "NO"]
    yes_kw = ["是", "有", "属于", "已被", "yes", "Yes", "YES", "确实"]
    has_no  = any(k in head for k in no_kw)
    has_yes = any(k in head for k in yes_kw)
    if has_no and not has_yes:
        return "no"
    if has_yes and not has_no:
        return "yes"
    if has_no:    return "no"   # negation usually appears first
    if has_yes:   return "yes"
    return None


def score_question(q: dict, predicted_answer: str, kg: KGIndex) -> dict:
    qtype = q["type"]
    text = predicted_answer or ""

    if qtype == "agg_count":
        gold_count = q.get("gold_answer")
        if not isinstance(gold_count, int):
            cons = q.get("gold_constraint") or {}
            gold_count = len(kg.heads_with(cons.get("relation",""), cons.get("tail","")))
        pred_count = _extract_int(text)
        correct = (pred_count == gold_count)
        # also report relative error for partial credit logging
        rel_err = abs((pred_count or 0) - gold_count) / max(1, gold_count)
        return {"metric":"exact_count", "gold":gold_count, "pred":pred_count,
                "correct": correct, "rel_err": rel_err}

    if qtype in ("agg_enum", "relation_inverse", "attr_filter"):
        if qtype == "attr_filter":
            cons_list = q.get("gold_constraints") or []
            sets = [kg.heads_with(c["relation"], c["tail"]) for c in cons_list]
            gold_set = set.intersection(*sets) if sets else set()
        else:
            cons = q.get("gold_constraint") or {}
            gold_set = kg.heads_with(cons.get("relation",""), cons.get("tail","")) if cons else set()
            if isinstance(q.get("gold_answer"), list):
                # if curated list given, prefer it as the gold
                gold_set = set(q["gold_answer"])
        if not gold_set:
            return {"metric":"recall", "skip":"no_gold_set"}
        hit = sum(1 for g in gold_set if g in text)
        recall = hit / len(gold_set)
        return {"metric":"recall", "gold_size":len(gold_set), "hit":hit, "recall":recall,
                "correct": recall >= 0.6}  # >=60% recall counted as correct

    if qtype == "negation":
        gold_answer = str(q.get("gold_answer", "")).lower()
        gold_polarity = "no" if any(k in gold_answer for k in ["否", "no", "未"]) else "yes"
        pred_polarity = _yes_or_no(text)
        correct = (pred_polarity == gold_polarity)
        return {"metric":"polarity", "gold_polarity":gold_polarity,
                "pred_polarity":pred_polarity, "correct": correct}

    if qtype == "set_compare":
        gold_answer = str(q.get("gold_answer", "")).lower()
        gold_polarity = "yes" if "是" in gold_answer or "yes" in gold_answer else "no"
        pred_polarity = _yes_or_no(text)
        correct = (pred_polarity == gold_polarity)
        return {"metric":"polarity", "gold_polarity":gold_polarity,
                "pred_polarity":pred_polarity, "correct": correct}

    if qtype in ("two_hop_bridge", "three_hop_chain"):
        gold_answer = q.get("gold_answer", "")
        if isinstance(gold_answer, list):
            gold_strs = gold_answer
        else:
            # split on ; or , or 、 or 和
            gold_strs = re.split(r"[；;,，、和]", str(gold_answer))
            gold_strs = [g.strip() for g in gold_strs if g.strip()]
        if not gold_strs:
            return {"metric":"substring", "skip":"no_gold"}
        hits = sum(1 for g in gold_strs if g in text)
        # accept if at least 1 key entity is mentioned
        return {"metric":"substring", "gold":gold_strs, "hits":hits,
                "correct": hits >= 1}

    if qtype in ("single_hop", "distractor"):
        gold = str(q.get("gold_answer", "")).strip()
        if not gold:
            return {"metric":"substring", "skip":"no_gold"}
        # exact substring match (case-sensitive for radar names; case-insensitive otherwise)
        match = gold in text or gold.lower() in text.lower()
        return {"metric":"substring", "gold":gold, "correct": match}

    if qtype == "unanswerable":
        # correct if the answer expresses uncertainty / absence
        no_keywords = ["未知", "无相关", "没有", "无法", "无此", "no record",
                       "not found", "unknown", "no triple", "未提供", "无", "n/a"]
        match = any(k in text for k in no_keywords)
        return {"metric":"polarity", "expected":"unknown_or_no", "correct": match}

    return {"metric":"none", "skip":f"unknown_type_{qtype}"}


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    val_path = ROOT / "evaluation" / "validation_30.json"
    with open(val_path, encoding="utf-8") as f:
        val = json.load(f)
    questions = val["questions"]

    print("Loading triples + retriever (shared by both pipelines)...")
    big_path = ROOT / "graphrag_index" / "merged_triples.json"
    with open(big_path, encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)

    retriever = HybridRetriever(triples, enable_reranker=False)

    # Strategy pipeline shares the retriever for lookup fallback
    print("Initializing StrategyPipeline...")
    pipe = StrategyPipeline(triples_path=str(big_path), lookup_retriever=retriever)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i:>2}/{len(questions)}] {q['id']:15s} ({q['type']:18s}) {q['question_zh']}")
        # baseline
        try:
            base_out = baseline_answer(q["question_zh"], retriever)
            base_score = score_question(q, base_out["answer"], kg)
        except Exception as e:
            base_out  = {"answer": f"[ERR {e}]"}
            base_score = {"correct": False, "error": str(e)}

        # strategy-routed
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
            "strategy": {"qtype": strat_out.get("qtype"),
                         "strategy": strat_out.get("strategy"),
                         "answer": strat_out["answer"][:300] if isinstance(strat_out.get("answer"), str) else "",
                         "score": strat_score,
                         "args": strat_out.get("args"),
                        },
        }
        results.append(rec)
        b_ok = base_score.get("correct", False)
        s_ok = strat_score.get("correct", False)
        b_mark = "OK" if b_ok else "  "
        s_mark = "OK" if s_ok else "  "
        print(f"   baseline:  [{b_mark}]  {base_out['answer'][:120].replace(chr(10),' ')}")
        print(f"   strategy:  [{s_mark}]  {(strat_out['answer'] or '')[:120].replace(chr(10),' ')}")

    # Aggregate
    by_type = defaultdict(lambda: {"baseline":[], "strategy":[]})
    for r in results:
        by_type[r["type"]]["baseline"].append(r["baseline"]["score"].get("correct", False))
        by_type[r["type"]]["strategy"].append(r["strategy"]["score"].get("correct", False))

    print("\n========== END-TO-END ACCURACY ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Baseline':>10s} {'Strategy':>10s} {'Δ':>8s}")
    print("-" * 60)
    overall_b, overall_s, overall_n = 0, 0, 0
    for t, d in sorted(by_type.items()):
        n = len(d["baseline"])
        b = sum(d["baseline"]); s = sum(d["strategy"])
        overall_b += b; overall_s += s; overall_n += n
        delta = (s - b) / max(1, n) * 100
        print(f"{t:<22s} {n:>3d} | {b/n:>10.3f} {s/n:>10.3f} {delta:>7.1f}%")
    print("-" * 60)
    delta_overall = (overall_s - overall_b) / max(1, overall_n) * 100
    print(f"{'OVERALL':<22s} {overall_n:>3d} | {overall_b/overall_n:>10.3f} {overall_s/overall_n:>10.3f} {delta_overall:>7.1f}%")

    out = ROOT / "results" / "e2e_30_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": results,
                   "by_type": {t: {"baseline": d["baseline"], "strategy": d["strategy"]}
                               for t, d in by_type.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
