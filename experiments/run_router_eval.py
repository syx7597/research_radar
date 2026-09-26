"""
Evaluate the question-type router on the 30-question validation set.

Outputs:
  - per-question prediction vs gold
  - confusion matrix
  - per-type accuracy
  - decision: proceed / add few-shot / finetune
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_router import QuestionRouter


def main():
    val_path = ROOT / "evaluation" / "validation_30.json"
    with open(val_path, encoding="utf-8") as f:
        val = json.load(f)

    router = QuestionRouter()
    predictions = []
    for q in val["questions"]:
        gold = q["type"]
        qtype, strategy, raw = router.route(q["question_zh"])
        predictions.append({
            "id": q["id"], "question": q["question_zh"],
            "gold": gold, "pred": qtype, "strategy": strategy, "raw": raw,
            "correct": gold == qtype,
        })
        mark = "OK" if gold == qtype else "  "
        print(f"  [{mark}] {q['id']:15s}  gold={gold:18s}  pred={qtype:18s}")

    # accuracy
    n = len(predictions)
    correct = sum(p["correct"] for p in predictions)
    print(f"\nOverall accuracy: {correct}/{n} = {correct/n:.3f}")

    # per-type accuracy
    by_gold = defaultdict(list)
    for p in predictions:
        by_gold[p["gold"]].append(p)
    print("\n[Per-gold-type]")
    for g, ps in sorted(by_gold.items()):
        c = sum(p["correct"] for p in ps)
        print(f"  {g:20s} {c}/{len(ps)} = {c/len(ps):.3f}")

    # confusion
    print("\n[Confusion (gold -> pred)]")
    confusion = Counter()
    for p in predictions:
        confusion[(p["gold"], p["pred"])] += 1
    for (g, pr), c in sorted(confusion.items(), key=lambda x: (x[0][0], -x[1])):
        flag = " <-- mismatch" if g != pr else ""
        print(f"  {g:20s} -> {pr:20s} : {c}{flag}")

    out = ROOT / "results" / "router_30_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": correct / n,
            "per_type": {g: sum(p["correct"] for p in ps) / len(ps)
                         for g, ps in by_gold.items()},
            "predictions": predictions,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")

    # decision gate
    crit = ["agg_count", "agg_enum", "negation", "attr_filter"]
    print("\n[Decision gate (critical types must be >=80%)]")
    decision_pass = True
    for t in crit:
        if t not in by_gold:
            continue
        ps = by_gold[t]
        acc = sum(p["correct"] for p in ps) / len(ps)
        ok = acc >= 0.8
        decision_pass = decision_pass and ok
        print(f"  {t:18s} acc={acc:.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"\n→ {'PROCEED' if decision_pass else 'NEEDS_FEW_SHOT_OR_FINETUNE'}")


if __name__ == "__main__":
    main()
