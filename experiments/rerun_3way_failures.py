"""Re-run only the strategy-failing questions from the 100-Q 3-way and re-score
all 100 with combined results. This avoids burning budget on questions that
already passed."""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import StrategyPipeline, KGIndex
from experiments.run_e2e_30 import score_question


def main():
    with open(ROOT / "results" / "qa500_3way_100.json", encoding="utf-8") as f:
        data = json.load(f)
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    # New qa_500 (regenerated with two_hop fix) — load gold by id
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        new_qs = {q["id"]: q for q in json.load(f)["questions"]}

    pipe = StrategyPipeline()

    # Identify strategy-failing questions
    targets = [r for r in data["per_question"]
               if not r["strategy"]["score"].get("correct", False)]
    print(f"Re-running {len(targets)} strategy-failing questions with lookup KG-fallback")

    for i, r in enumerate(targets, 1):
        qid = r["id"]
        # Look up the (possibly regenerated) question by id; fall back to old text
        q_new = new_qs.get(qid)
        if q_new is None:
            print(f"  [{i}] {qid} dropped from new qa_500 (filtered by two_hop fix), keeping old verdict")
            continue
        question = q_new["question_zh"]
        print(f"\n[{i:>2}/{len(targets)}] {qid:15s} ({r['type']}) {question[:60]}")
        try:
            out = pipe.run(question)
            score = score_question(q_new, out["answer"], kg)
        except Exception as e:
            out = {"answer": f"[ERR {e}]"}
            score = {"correct": False, "error": str(e)}
        prev = r["strategy"]["score"].get("correct", False)
        new_ok = score.get("correct", False)
        flip = "→ FIXED" if new_ok and not prev else ("→ STILL BROKEN" if not new_ok else "OK")
        print(f"   prev={prev} new={new_ok} {flip}")
        print(f"   ans: {out['answer'][:160]}")
        # update record
        r["strategy"]["answer"] = out["answer"][:300]
        r["strategy"]["score"] = score
        r["strategy"]["args"] = out.get("args")

    # Drop questions that no longer exist in the regenerated qa_500
    drop_ids = []
    for r in data["per_question"]:
        if r["id"] not in new_qs:
            drop_ids.append(r["id"])
    if drop_ids:
        print(f"\nDropped {len(drop_ids)} questions that were filtered by two_hop fix: {drop_ids}")
        data["per_question"] = [r for r in data["per_question"] if r["id"] in new_qs]

    # Aggregate
    by_t = defaultdict(lambda: {"baseline":[], "rog_style":[], "strategy":[]})
    for r in data["per_question"]:
        by_t[r["type"]]["baseline"].append(r["baseline"]["score"].get("correct", False))
        by_t[r["type"]]["rog_style"].append(r["rog_style"]["score"].get("correct", False))
        by_t[r["type"]]["strategy"].append(r["strategy"]["score"].get("correct", False))

    print("\n========== UPDATED 3-WAY (after lookup-fallback fix) ==========")
    print(f"{'Type':<22s} {'n':>3s} | {'Baseline':>10s} {'RoG-style':>10s} {'Strategy':>10s} | {'Δ S-B':>8s} {'Δ S-R':>8s}")
    print("-" * 84)
    ob, orog, os_, on = 0, 0, 0, 0
    for t, d in sorted(by_t.items()):
        n = len(d["baseline"])
        if n == 0: continue
        b = sum(d["baseline"]); r = sum(d["rog_style"]); s = sum(d["strategy"])
        ob += b; orog += r; os_ += s; on += n
        print(f"{t:<22s} {n:>3d} | {b/n:>10.3f} {r/n:>10.3f} {s/n:>10.3f} | "
              f"{(s-b)/n*100:>7.1f}% {(s-r)/n*100:>7.1f}%")
    print("-" * 84)
    print(f"{'OVERALL':<22s} {on:>3d} | {ob/on:>10.3f} {orog/on:>10.3f} {os_/on:>10.3f} | "
          f"{(os_-ob)/on*100:>7.1f}% {(os_-orog)/on*100:>7.1f}%")

    out_path = ROOT / "results" / "qa500_3way_100_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_question": data["per_question"]}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
