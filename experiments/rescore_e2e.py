"""Re-score existing e2e results JSON with the fixed scorer; no LLM calls."""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_e2e_30 import score_question
from qa_strategy_pipeline import KGIndex


def main():
    with open(ROOT / "results" / "e2e_30_results.json", encoding="utf-8") as f:
        data = json.load(f)
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    with open(ROOT / "evaluation" / "validation_30.json", encoding="utf-8") as f:
        val = json.load(f)
    val_by_id = {q["id"]: q for q in val["questions"]}

    by_type = defaultdict(lambda: {"baseline":[], "strategy":[]})
    for r in data["per_question"]:
        q = val_by_id[r["id"]]
        b_score = score_question(q, r["baseline"]["answer"], kg)
        s_score = score_question(q, r["strategy"]["answer"], kg)
        r["baseline"]["score_v2"] = b_score
        r["strategy"]["score_v2"] = s_score
        by_type[r["type"]]["baseline"].append(b_score.get("correct", False))
        by_type[r["type"]]["strategy"].append(s_score.get("correct", False))

    print(f"{'Type':<22s} {'n':>3s} | {'Baseline':>10s} {'Strategy':>10s} {'Δ':>8s}")
    print("-" * 62)
    ob, os_, on = 0, 0, 0
    for t, d in sorted(by_type.items()):
        n = len(d["baseline"]); b = sum(d["baseline"]); s = sum(d["strategy"])
        ob += b; os_ += s; on += n
        print(f"{t:<22s} {n:>3d} | {b/n:>10.3f} {s/n:>10.3f} {(s-b)/n*100:>7.1f}%")
    print("-" * 62)
    print(f"{'OVERALL':<22s} {on:>3d} | {ob/on:>10.3f} {os_/on:>10.3f} {(os_-ob)/on*100:>7.1f}%")

    out = ROOT / "results" / "e2e_30_rescored.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": data["per_question"]}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")

    # Show per-type detail for agg_count and any other surprises
    print("\n[Detail: agg_count + agg_enum + attr_filter]")
    for r in data["per_question"]:
        if r["type"] in ("agg_count", "agg_enum", "attr_filter"):
            b_ok = r["baseline"]["score_v2"].get("correct", False)
            s_ok = r["strategy"]["score_v2"].get("correct", False)
            print(f"  {r['id']:15s}  baseline=[{'OK' if b_ok else '  '}]  strategy=[{'OK' if s_ok else '  '}]")
            if not s_ok:
                print(f"     Q:    {r['question']}")
                print(f"     gold: {r['gold_answer']}")
                print(f"     ans:  {r['strategy']['answer'][:200]}")
                print(f"     args: {r['strategy'].get('args')}")


if __name__ == "__main__":
    main()
