"""Re-score the qa500 subset results with the fixed scorer."""
import json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_e2e_30 import score_question
from qa_strategy_pipeline import KGIndex


def main():
    with open(ROOT / "results" / "qa500_subset_results.json", encoding="utf-8") as f:
        data = json.load(f)
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = {q["id"]: q for q in json.load(f)["questions"]}

    by_t = defaultdict(list)
    for r in data["per_question"]:
        q = all_qs[r["id"]]
        score = score_question(q, r["answer"], kg)
        r["score_v2"] = score
        by_t[r["type"]].append(score.get("correct", False))

    print(f"{'Type':<22s} {'n':>3s} | {'Strategy acc':>12s}")
    print("-" * 45)
    total_c, total_n = 0, 0
    for t, lst in sorted(by_t.items()):
        n = len(lst); c = sum(lst)
        total_c += c; total_n += n
        print(f"{t:<22s} {n:>3d} | {c/n:>12.3f}")
    print("-" * 45)
    print(f"{'OVERALL':<22s} {total_n:>3d} | {total_c/total_n:>12.3f}")

    out = ROOT / "results" / "qa500_subset_rescored.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_question": data["per_question"]}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")

    # show any remaining failures
    print("\n[Remaining failures]")
    for r in data["per_question"]:
        if not r["score_v2"].get("correct", False):
            print(f"\n  [{r['id']}] type={r['type']} pred={r['predicted_strategy']}")
            print(f"    Q: {r['question']}")
            print(f"    gold: {r['gold_answer']}")
            print(f"    ans:  {r['answer'][:200]}")
            print(f"    score: {r['score_v2']}")


if __name__ == "__main__":
    main()
