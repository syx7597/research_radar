"""
Router error analysis on the 100-question stratified subset.

Compares each question's router-predicted strategy vs the gold `expected_strategy`
in qa_500.json. Builds:
  - strategy-level confusion matrix
  - per-question categorization (4-way):
      A. correctly routed AND correct answer
      B. correctly routed BUT wrong answer (downstream issue)
      C. misrouted BUT correct answer  (graceful degradation)
      D. misrouted AND wrong answer    (catastrophic)
  - per-misroute-type breakdown: which strategy substitutions hurt vs survive

No LLM calls — pure analysis of existing JSON.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    with open(ROOT / "results" / "qa500_3way_100_v2.json", encoding="utf-8") as f:
        data = json.load(f)
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        qa_by_id = {q["id"]: q for q in json.load(f)["questions"]}

    # Per-question record with both router-predicted strategy and gold expected_strategy
    records = []
    for r in data["per_question"]:
        q = qa_by_id[r["id"]]
        gold_strategy = q.get("expected_strategy", "lookup")
        pred_strategy = r["strategy"].get("strategy", "lookup")
        correct = r["strategy"]["score"].get("correct", False)
        records.append({
            "id": r["id"], "type": q["type"],
            "question": r["question"],
            "gold_strategy": gold_strategy,
            "pred_strategy": pred_strategy,
            "correct": correct,
            "matched": gold_strategy == pred_strategy,
        })

    n = len(records)
    n_matched   = sum(1 for x in records if x["matched"])
    n_correct   = sum(1 for x in records if x["correct"])

    print(f"Total: {n} questions")
    print(f"Router accuracy (strategy match): {n_matched}/{n} = {n_matched/n:.3f}")
    print(f"End-to-end answer accuracy:       {n_correct}/{n} = {n_correct/n:.3f}")

    # 4-way breakdown
    print("\n========== 4-way breakdown ==========")
    A = sum(1 for x in records if x["matched"]     and x["correct"])
    B = sum(1 for x in records if x["matched"]     and not x["correct"])
    C = sum(1 for x in records if not x["matched"] and x["correct"])
    D = sum(1 for x in records if not x["matched"] and not x["correct"])
    print(f"  A. Routed correctly AND answer correct (best case)         : {A}/{n} = {A/n:.3f}")
    print(f"  B. Routed correctly BUT answer wrong (downstream issue)    : {B}/{n} = {B/n:.3f}")
    print(f"  C. Misrouted BUT answer still correct (graceful)           : {C}/{n} = {C/n:.3f}")
    print(f"  D. Misrouted AND answer wrong (catastrophic)               : {D}/{n} = {D/n:.3f}")
    print()
    if (C + D) > 0:
        graceful_rate = C / (C + D)
        print(f"  Graceful-degradation rate: when router misroutes, answer is still correct {graceful_rate:.1%} of the time")

    # Strategy confusion matrix
    print("\n========== Strategy confusion (gold → pred) ==========")
    confusion = Counter()
    for x in records:
        confusion[(x["gold_strategy"], x["pred_strategy"])] += 1
    strats = sorted({s for x in records for s in (x["gold_strategy"], x["pred_strategy"])})
    header = "gold/pred"
    print(f"{header:<22s}", *(f"{s:>16s}" for s in strats), sep="")
    for g in strats:
        row = [confusion.get((g, p), 0) for p in strats]
        print(f"{g:<22s}", *(f"{v:>16d}" for v in row), sep="")

    # Per-error-type breakdown
    print("\n========== Misroute breakdown (only mismatches) ==========")
    misroute_buckets = defaultdict(lambda: {"graceful": 0, "catastrophic": 0, "examples": []})
    for x in records:
        if x["matched"]:
            continue
        key = f"{x['gold_strategy']} → {x['pred_strategy']}"
        bucket = "graceful" if x["correct"] else "catastrophic"
        misroute_buckets[key][bucket] += 1
        if len(misroute_buckets[key]["examples"]) < 2:
            misroute_buckets[key]["examples"].append({
                "id": x["id"], "type": x["type"], "question": x["question"][:60],
                "outcome": bucket,
            })
    print(f"{'misroute (gold → pred)':<35s} {'graceful':>10s} {'catastrophic':>14s}")
    print("-" * 65)
    for key, b in sorted(misroute_buckets.items(), key=lambda x: -(x[1]["graceful"] + x[1]["catastrophic"])):
        print(f"{key:<35s} {b['graceful']:>10d} {b['catastrophic']:>14d}")

    # Per-question-type breakdown
    print("\n========== Per-question-type router accuracy & graceful rate ==========")
    by_t = defaultdict(lambda: {"matched":0, "correct":0, "n":0,
                                 "graceful_when_misroute":0, "catastrophic_when_misroute":0})
    for x in records:
        d = by_t[x["type"]]
        d["n"] += 1
        if x["matched"]: d["matched"] += 1
        if x["correct"]: d["correct"] += 1
        if not x["matched"]:
            if x["correct"]:
                d["graceful_when_misroute"] += 1
            else:
                d["catastrophic_when_misroute"] += 1

    print(f"{'Type':<22s} {'n':>3s} {'router':>8s} {'correct':>8s} {'misroute':>10s} {'gracef%':>10s}")
    for t, d in sorted(by_t.items()):
        n = d["n"]
        misroute = d["graceful_when_misroute"] + d["catastrophic_when_misroute"]
        graceful_pct = (d["graceful_when_misroute"] / misroute * 100) if misroute > 0 else 0.0
        print(f"{t:<22s} {n:>3d} {d['matched']/n:>8.3f} {d['correct']/n:>8.3f} "
              f"{misroute:>10d} {graceful_pct:>9.1f}%")

    print("\n========== Catastrophic misroute examples ==========")
    catastrophic = [x for x in records if not x["matched"] and not x["correct"]]
    for x in catastrophic[:8]:
        print(f"  [{x['id']}] type={x['type']}")
        print(f"    Q: {x['question']}")
        print(f"    gold_strategy={x['gold_strategy']}  pred={x['pred_strategy']}")

    out = ROOT / "results" / "router_error_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n": n,
                "router_strategy_accuracy": n_matched / n,
                "answer_accuracy": n_correct / n,
                "A_matched_correct": A,
                "B_matched_wrong": B,
                "C_misroute_correct": C,
                "D_misroute_wrong": D,
                "graceful_when_misroute": (C / (C+D)) if (C+D) > 0 else 1.0,
            },
            "records": records,
            "misroute_buckets": {k: {"graceful": v["graceful"], "catastrophic": v["catastrophic"],
                                      "examples": v["examples"]}
                                  for k, v in misroute_buckets.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
