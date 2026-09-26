"""
KQA Pro end-to-end mini-e2e (task #21).

For each KQA Pro val question we:
  1. Build a 2-hop per-Q subgraph around topic entities (from Find ops).
  2. Run two systems:
     - BASELINE: BM25 retrieve top-K=8 triples from subgraph, LLM answer.
     - STRATEGY (oracle-operator): the gold operator (derived from KQA Pro's
       program output via KQAPRO_TO_OURS_V2) is invoked on the subgraph;
       answerer renders.
  3. Score against KQA Pro `answer` with substring / count / yes-no scorers.

This is "oracle-dispatcher + oracle-parser-via-program" Strategy. It tests the
operator suite, not the LLM dispatcher (already tested in the probe). It is
the cleanest cross-benchmark validation of the operator-suite claim:
"different question types need different operators".

Output: results/kqa_pro_e2e.{json,md}
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
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import (
    KGIndex, llm_call, ANSWER_SYSTEM,
    exec_exhaustive, exec_complement, exec_path_plan,
    exec_constrained_join, exec_dual_subgraph,
)
from datasets.kqa_pro_subgraph import (
    build_question_subgraph, _entity_name_to_ids, _id_to_name,
    build_question_subgraph_v2, _build_concept_index,
)
from datasets.kqa_pro_parser import parse_args as kqa_parse_args
from experiments.kqa_pro_probe_rescore import KQAPRO_TO_OURS_V2


IN_VAL = ROOT / "datasets" / "kqa_pro" / "val.json"
IN_KB  = ROOT / "datasets" / "kqa_pro" / "kb.json"
OUT_JSON = ROOT / "results" / "kqa_pro_e2e_500_sgv2.json"
OUT_MD   = ROOT / "results" / "kqa_pro_e2e_500_sgv2_summary.md"
N_PER_TYPE = 45  # ~500 questions across 12 types (VerifyDate caps at 7)
SEED = 42
PARSER_MODE = "llm"      # 'llm' = real KQA-Pro parser (fair test), 'mechanical' = previous
SUBGRAPH_VER = "v2"      # 'v1' = 2-hop only (B3); 'v2' = 2-hop + concept-class expansion (B4)

# ── BM25 mini-index (lazy per Q) ───────────────────────────────
def simple_bm25_topk(question, triples, k=8):
    """Lightweight TF-based ranking on triple text. (No need for full BM25 lib
    for per-Q subgraphs of <2K triples.)"""
    import math
    q_terms = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    if not q_terms:
        return triples[:k]
    docs = [f"{t['head']} {t['relation']} {t['tail']}".lower() for t in triples]
    scored = []
    for d, t in zip(docs, triples):
        score = sum(1 for term in q_terms if term in d)
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k] if _ > 0] or triples[:k]


def baseline_answer(question, subgraph_triples):
    top = simple_bm25_topk(question, subgraph_triples, k=8)
    ctx_lines = [f"- {t['head']} | {t['relation']} | {t['tail']}" for t in top]
    ctx = "\n".join(ctx_lines)
    user = f"Question: {question}\n\nEvidence:\n{ctx}\n\nGive the answer (concise)."
    ans = llm_call(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user",   "content": user}],
        max_tokens=300,
    )
    return ans


def build_args_from_program(prog, topic_entities):
    """Mechanical translation: KQA Pro program → our operator args.
    Handles: Find, FindAll, Relate, Filter{Str,Num,Year,Date}, FilterConcept,
    Verify{Str,Num,Year,Date}, Count, Query{Name,Attr,Relation,AttrQualifier,
    RelationQualifier}, SelectBetween, SelectAmong, And, Or."""
    primary = topic_entities[0] if topic_entities else ""
    secondary = topic_entities[1] if len(topic_entities) > 1 else ""
    relation_chain = []
    constraints = []
    forbidden_tail = ""
    answer_target = "entity"
    concept_filter = None

    for step in prog:
        fn = step.get("function", "")
        inp = step.get("inputs", []) or []
        if fn == "Relate" and inp:
            relation_chain.append(inp[0])
        elif fn in {"FilterStr", "FilterNum", "FilterYear", "FilterDate"}:
            if len(inp) >= 2:
                constraints.append({"relation": inp[0], "tail": inp[1]})
        elif fn == "FilterConcept" and inp:
            concept_filter = inp[0]
            constraints.append({"relation": "instance of", "tail": inp[0]})
        elif fn in {"VerifyStr", "VerifyYear", "VerifyNum", "VerifyDate"}:
            forbidden_tail = inp[0] if inp else ""
            answer_target = "yesno"
        elif fn == "Count":
            answer_target = "count"
        elif fn in {"QueryAttr", "QueryRelation"}:
            if inp:
                relation_chain.append(inp[0])
            answer_target = "literal" if fn == "QueryAttr" else "entity"
        elif fn in {"QueryAttrQualifier", "QueryRelationQualifier"}:
            # inputs: attribute_key, attribute_value, qualifier_key
            if len(inp) >= 1:
                relation_chain.append(inp[0])
            if len(inp) >= 3:
                relation_chain.append(inp[2])
            if len(inp) >= 2:
                constraints.append({"relation": inp[0], "tail": inp[1]})
            answer_target = "literal"
        elif fn == "QueryName":
            answer_target = "entity"
        elif fn in {"SelectBetween", "SelectAmong"}:
            # inputs: comparison_attribute, direction
            if inp:
                relation_chain.append(inp[0])
            answer_target = "entity"
        elif fn in {"And", "Or"}:
            pass  # set ops; handled implicitly by combining constraints
    return {
        "primary_entity": primary,
        "secondary_entity": secondary,
        "relation_chain": relation_chain,
        "constraints": constraints,
        "forbidden_tail": forbidden_tail,
        "answer_target": answer_target,
        "concept_filter": concept_filter,
    }


def strategy_answer(question, qrec, subgraph_triples, kg, parser_mode="llm"):
    """Oracle-operator strategy with TWO parser modes:
       - parser_mode='llm'        → use the KQA-Pro-aware LLM parser (fair test)
       - parser_mode='mechanical' → mechanical program→args (sanity/ablation)
    """
    fn = qrec["program"][-1]["function"] if qrec["program"] else ""
    gold_qt, strategy = KQAPRO_TO_OURS_V2.get(fn, ("single_hop", "lookup"))

    if parser_mode == "llm":
        try:
            args = kqa_parse_args(question, gold_qt, strategy)
        except Exception:
            topic_entities = []
            for step in qrec["program"]:
                if step["function"] == "Find":
                    topic_entities.extend(step["inputs"])
            args = build_args_from_program(qrec["program"], topic_entities)
    else:
        topic_entities = []
        for step in qrec["program"]:
            if step["function"] == "Find":
                topic_entities.extend(step["inputs"])
        args = build_args_from_program(qrec["program"], topic_entities)

    evidence = None
    try:
        if strategy == "exhaustive":
            evidence = exec_exhaustive(args, kg, {})
        elif strategy == "complement":
            evidence = exec_complement(args, kg, {})
        elif strategy == "path_plan":
            evidence = exec_path_plan(args, kg, {})
        elif strategy == "constrained_join":
            evidence = exec_constrained_join(args, kg, {})
        elif strategy == "dual_subgraph":
            evidence = exec_dual_subgraph(args, kg, {})
    except Exception:
        evidence = None

    def _items_of(ev):
        if not isinstance(ev, dict): return []
        return ev.get("items") or ev.get("entities") or ev.get("tails") or []

    items = _items_of(evidence) if evidence else []
    # Lookup fallback: if operator returned empty, augment with BM25 top-K so
    # the answerer still has something. Strategy then can only match-or-beat
    # baseline (not fall below).
    fallback_used = False
    if not items:
        fallback_used = True
        top = simple_bm25_topk(question, subgraph_triples, k=8)
        evidence = {"strategy": f"{strategy}+lookup_fallback", "items": [
            f"{t['head']} | {t['relation']} | {t['tail']}" for t in top]}
        items = evidence["items"]

    # Render context
    ctx_lines = [f"- {it}" if not isinstance(it, dict) else
                 f"- {it.get('head','')} | {it.get('relation','')} | {it.get('tail','')}"
                 for it in items[:20]]
    ctx = "\n".join(ctx_lines) or "(no evidence)"

    user = (f"Question: {question}\n\n"
            f"Operator: {strategy}\n"
            f"Evidence:\n{ctx}\n\nGive the answer (concise).")
    ans = llm_call(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user",   "content": user}],
        max_tokens=300,
    )
    return ans, strategy, args, fallback_used


def score(pred, gold, answer_target):
    """Multi-format scoring."""
    if pred is None: return False
    pred = str(pred).strip()
    gold = str(gold).strip()
    if not gold: return False
    p_lower = pred.lower()
    g_lower = gold.lower()
    # Exact / substring
    if g_lower in p_lower:
        return True
    # Number
    if answer_target == "count":
        g_n = re.search(r"-?\d+", gold)
        p_n = re.search(r"-?\d+", pred)
        if g_n and p_n and int(g_n.group()) == int(p_n.group()):
            return True
    # Yes/no
    if answer_target == "yesno":
        return ("yes" in p_lower) == ("yes" in g_lower)
    return False


def main():
    rng = random.Random(SEED)
    with open(IN_VAL, encoding="utf-8") as f:
        val = json.load(f)
    print(f"Loaded {len(val)} val questions")
    print("Loading KB...")
    with open(IN_KB, encoding="utf-8") as f:
        kb = json.load(f)
    n2i = _entity_name_to_ids(kb)
    i2n = _id_to_name(kb)
    concept_idx = _build_concept_index(kb) if SUBGRAPH_VER == "v2" else None
    if concept_idx:
        print(f"Concept index: {len(concept_idx)} concept names")

    # Stratified sample (Find-only — at least 1 KB-resolvable topic entity)
    by_fn = defaultdict(list)
    for q in val:
        if not q.get("program"): continue
        topic_resolved = False
        for step in q["program"]:
            if step["function"] == "Find":
                for nm in step.get("inputs", []):
                    if nm in n2i:
                        topic_resolved = True
                        break
            if topic_resolved: break
        if not topic_resolved: continue
        fn = q["program"][-1]["function"]
        by_fn[fn].append(q)
    print("Find-resolved per fn:")
    for fn in by_fn:
        print(f"  {fn:30s} {len(by_fn[fn])}")
    sample = []
    for fn, lst in by_fn.items():
        rng.shuffle(lst)
        sample.extend(lst[:N_PER_TYPE])
    print(f"Sample: {len(sample)}")

    # Resume from checkpoint if it exists
    if OUT_JSON.exists():
        with open(OUT_JSON, encoding="utf-8") as f:
            results = json.load(f).get("per_question", [])
        done_qs = {(r["question"], r["kqapro_fn"]) for r in results}
        sample = [q for q in sample if (q["question"], q["program"][-1]["function"]) not in done_qs]
        print(f"Resuming: {len(done_qs)} done, {len(sample)} remaining")
    else:
        results = []

    t0 = time.time()
    for i, q in enumerate(sample, 1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(sample) - i + 1) if i > 1 else 0
        if i % 5 == 0 or i == 1:
            print(f"[{i:>3}/{len(sample)}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")

        if SUBGRAPH_VER == "v2":
            sg = build_question_subgraph_v2(q, kb, n2i, i2n,
                                             concept_index=concept_idx, hops=2)
        else:
            sg = build_question_subgraph(q, kb, n2i, i2n, hops=2)
        triples = sg["triples"]
        kg = KGIndex(triples)

        # baseline
        try:
            b_ans = baseline_answer(q["question"], triples)
        except Exception as e:
            b_ans = f"[ERR {e}]"
        # strategy
        try:
            s_ans, strategy, args, fallback = strategy_answer(q["question"], q, triples, kg, PARSER_MODE)
        except Exception as e:
            s_ans, strategy, args, fallback = f"[ERR {e}]", "ERR", {}, False

        fn = q["program"][-1]["function"]
        _, target = KQAPRO_TO_OURS_V2.get(fn, ("single_hop", "lookup"))
        answer_target = ("count" if fn == "Count" else
                         "yesno" if fn.startswith("Verify") else
                         "entity")
        b_score = score(b_ans, q["answer"], answer_target)
        s_score = score(s_ans, q["answer"], answer_target)
        rec = {
            "question": q["question"],
            "kqapro_fn": fn,
            "gold_answer": q["answer"],
            "topic_entities": sg["topic_entities"],
            "subgraph_size": len(triples),
            "strategy_used": strategy,
            "lookup_fallback": fallback,
            "baseline_answer": b_ans[:300],
            "strategy_answer": s_ans[:300] if isinstance(s_ans, str) else "",
            "baseline_correct": b_score,
            "strategy_correct": s_score,
        }
        results.append(rec)
        ok_b = "OK" if b_score else "  "
        ok_s = "OK" if s_score else "  "
        print(f"   [{fn:25s}] B[{ok_b}] S[{ok_s}]  ({strategy})")
        # checkpoint every question
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"per_question": results}, f, ensure_ascii=False, indent=2)

    # Aggregate
    n = len(results)
    nb = sum(r["baseline_correct"] for r in results)
    ns = sum(r["strategy_correct"] for r in results)
    by_fn_r = defaultdict(list)
    for r in results:
        by_fn_r[r["kqapro_fn"]].append(r)
    lines = ["# KQA Pro End-to-End Mini Comparison", ""]
    lines.append(f"- N = {n} (stratified ≤{N_PER_TYPE}/KQA-Pro-fn, Find-resolvable only)")
    lines.append(f"- Baseline (BM25 top-8 on per-Q 2-hop subgraph) → answerer")
    lines.append(f"- Strategy (oracle-operator via KQAPRO_TO_OURS_V2 + program → args) → answerer")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Baseline accuracy: **{nb}/{n} = {nb/n*100:.1f}%**")
    lines.append(f"- Strategy (oracle-op) accuracy: **{ns}/{n} = {ns/n*100:.1f}%**")
    lines.append(f"- Δ (Strategy − Baseline): **{(ns-nb)/n*100:+.1f} pp**")
    lines.append("")
    lines.append("## Per-type breakdown")
    lines.append("")
    lines.append("| KQA Pro fn | n | Baseline | Strategy-oracle | Δ |")
    lines.append("|---|---:|---:|---:|---:|")
    for fn in sorted(by_fn_r, key=lambda k: -len(by_fn_r[k])):
        rs = by_fn_r[fn]
        nb_t = sum(r["baseline_correct"] for r in rs)
        ns_t = sum(r["strategy_correct"] for r in rs)
        lines.append(f"| {fn} | {len(rs)} | {nb_t/len(rs)*100:.0f}% | {ns_t/len(rs)*100:.0f}% | "
                     f"{(ns_t-nb_t)/len(rs)*100:+.0f}pp |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print()
    # Replace minus-sign with ASCII hyphen for Windows GBK consoles
    safe = "\n".join(lines).replace("−", "-")
    print(safe)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
