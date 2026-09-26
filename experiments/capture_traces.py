"""Capture one clean end-to-end trace per strategy for paper Appendix C.

Picks 6 questions (one per strategy), runs the full pipeline, saves complete
trace: question → router output → parser args → executor evidence → renderer
context → final answer.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline


# Hand-picked questions designed to land on each strategy
PROBE_QUESTIONS = [
    # (strategy_target, question)
    ("lookup",           "AN/TPY-2 是由哪家公司研制的？"),
    ("exhaustive",       "工作在 X 波段的雷达共有多少款？"),
    ("complement",       "AN/TPY-2 是否被中国使用？"),
    ("path_plan",        "AN/TPY-2 的研制公司属于哪个国家？"),
    ("constrained_join", "由 Raytheon 研制且部署在战斗机上的雷达？"),
    ("dual_subgraph",    "AN/TPY-2 和 AN/MPQ-65 的研制公司是否相同？"),
]


def main():
    pipe = StrategyPipeline()
    traces = []
    for target_strategy, q in PROBE_QUESTIONS:
        print(f"\n========== target={target_strategy} ==========")
        print(f"Q: {q}")
        out = pipe.run(q)
        # collect a compact trace
        trace = {
            "target_strategy":  target_strategy,
            "question":         q,
            "router_output":    {"qtype": out["qtype"], "strategy": out["strategy"]},
            "parser_args":      {k: v for k, v in (out.get("args") or {}).items()
                                  if k not in ("_raw",)},
            "executor_evidence": _summarize_evidence(out["evidence"], out["strategy"]),
            "rendered_context": out["context"][:600] + "..." if len(out["context"]) > 600 else out["context"],
            "final_answer":     out["answer"],
        }
        traces.append(trace)
        print(f"  qtype={out['qtype']}  strategy={out['strategy']}")
        print(f"  answer: {out['answer'][:150]}")

    out_path = ROOT / "results" / "appendix_c_traces.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"traces": traces}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


def _summarize_evidence(ev: dict, strategy: str) -> dict:
    if strategy == "lookup":
        return {"reranked_count": len(ev.get("reranked", [])),
                "kg_fallback_count": len(ev.get("kg_fallback_facts", []))}
    if strategy == "exhaustive":
        return {"relation": ev.get("relation"), "tail": ev.get("tail"),
                "n_heads": ev.get("n", 0),
                "sample_heads": (ev.get("heads") or [])[:8]}
    if strategy == "complement":
        return {"head": ev.get("head"), "relation": ev.get("relation"),
                "known_tails": ev.get("known_tails"),
                "forbidden": ev.get("forbidden"),
                "is_empty": ev.get("is_empty")}
    if strategy == "path_plan":
        return {"path_steps": [{"rel": s["relation"],
                                  "in": s.get("in", [])[:3],
                                  "out": s.get("out", [])[:3]}
                                 for s in ev.get("path_trace", [])],
                "final": (ev.get("final") or [])[:5]}
    if strategy == "constrained_join":
        return {"breakdown": ev.get("breakdown"),
                "intersection_n": ev.get("n", 0),
                "intersection_sample": (ev.get("intersection") or [])[:8],
                "equiv_fallback_used": ev.get("equiv_fallback_used", False)}
    if strategy == "dual_subgraph":
        return {"entity_a": ev.get("entity_a"), "entity_b": ev.get("entity_b"),
                "a_tails": ev.get("a_tails"), "b_tails": ev.get("b_tails"),
                "same": ev.get("same")}
    return {}


if __name__ == "__main__":
    main()
