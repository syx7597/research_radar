"""Re-run only the strategy-failing questions, then re-score and report."""
import json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex
from experiments.run_e2e_30 import score_question


# IDs of the questions where strategy failed in last run (after rescoring)
FAILED = {"agg_count_03", "agg_enum_03", "agg_enum_05", "mh_07"}


def main():
    with open(ROOT / "evaluation" / "validation_30.json", encoding="utf-8") as f:
        val = json.load(f)
    val_by_id = {q["id"]: q for q in val["questions"]}

    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)

    pipe = StrategyPipeline()

    # load existing rescored results
    with open(ROOT / "results" / "e2e_30_rescored.json", encoding="utf-8") as f:
        data = json.load(f)

    for r in data["per_question"]:
        if r["id"] not in FAILED:
            continue
        q = val_by_id[r["id"]]
        print(f"\n[{r['id']}] {q['question_zh']}")
        out = pipe.run(q["question_zh"])
        score = score_question(q, out["answer"], kg)
        print(f"  args:    {out['args']}")
        print(f"  evidence_summary: {{ keys={list(out['evidence'].keys())} }}")
        if "heads" in out["evidence"]:
            print(f"  heads count: {len(out['evidence']['heads'])}")
        if "intersection" in out["evidence"]:
            print(f"  intersection count: {len(out['evidence']['intersection'])}")
        print(f"  answer:  {out['answer'][:300]}")
        print(f"  score:   {score}")
        # update record
        r["strategy"]["answer"] = out["answer"]
        r["strategy"]["args"] = out["args"]
        r["strategy"]["score_v2"] = score
        r["strategy"]["evidence_keys"] = list(out["evidence"].keys())

    # recompute aggregate
    by_type = defaultdict(lambda: {"baseline":[], "strategy":[]})
    for r in data["per_question"]:
        by_type[r["type"]]["baseline"].append(r["baseline"]["score_v2"].get("correct", False))
        by_type[r["type"]]["strategy"].append(r["strategy"]["score_v2"].get("correct", False))

    print(f"\n{'Type':<22s} {'n':>3s} | {'Baseline':>10s} {'Strategy':>10s} {'Δ':>8s}")
    print("-" * 62)
    ob, os_, on = 0, 0, 0
    for t, d in sorted(by_type.items()):
        n = len(d["baseline"]); b = sum(d["baseline"]); s = sum(d["strategy"])
        ob += b; os_ += s; on += n
        print(f"{t:<22s} {n:>3d} | {b/n:>10.3f} {s/n:>10.3f} {(s-b)/n*100:>7.1f}%")
    print("-" * 62)
    print(f"{'OVERALL':<22s} {on:>3d} | {ob/on:>10.3f} {os_/on:>10.3f} {(os_-ob)/on*100:>7.1f}%")

    out_path = ROOT / "results" / "e2e_30_final.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_question": data["per_question"]}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
