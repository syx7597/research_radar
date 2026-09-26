"""Aggregate completed pilot metrics, including the direct matched comparison."""
import argparse
import json
from pathlib import Path

from .evaluate import paired_comparison


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    root = args.run_dir
    baseline = read(root / "dev_metrics.json")
    result = {
        "scope": "Train-heldout development pilot, not final official validation/test performance",
        "generator_seeds": 1,
        "questions": baseline["questions"],
        "candidate_count_distribution": baseline["candidate_count_distribution"],
        "baseline_metrics": baseline["metrics"],
        "baseline_comparisons": baseline["comparisons"],
        "subsets_overlapping": baseline["subsets_overlapping"],
        "coarse_error_diagnostics": baseline["top1_wrong_structural_diagnostics"],
        "negative_generation": read(root / "targeted_pairs.summary.json"),
        "rankers": {},
        "targeted_vs_natural": {},
    }
    matched = result["negative_generation"]["targeted_pairs"]
    if matched == 0:
        result["ranker_skipped_reason"] = "No matched natural/targeted negative pairs; retain as a negative feasibility result"
    for seed in args.seeds if matched else []:
        evaluations = {}
        for source in ["natural", "targeted"]:
            evaluation = read(root / f"{source}_metrics_seed{seed}.json")
            evaluations[source] = evaluation
            result["rankers"][f"{source}_seed{seed}"] = {
                "metrics": evaluation["metrics"]["linear_ranker"],
                "vs_first_valid": evaluation["comparisons"]["linear_ranker_vs_first_valid"],
                "subsets": {k: v["metrics"]["linear_ranker"]
                            for k, v in evaluation["subsets_overlapping"].items()},
            }
        if evaluations["natural"]["candidate_cache_sha256"] != evaluations["targeted"]["candidate_cache_sha256"]:
            raise ValueError("Compared rankers did not use the same candidate cache")
        before = {row["id"]: row["correct"]["linear_ranker"]
                  for row in evaluations["natural"]["per_question"]}
        after = {row["id"]: row["correct"]["linear_ranker"]
                 for row in evaluations["targeted"]["per_question"]}
        if before.keys() != after.keys():
            raise ValueError("Compared rankers have different question sets")
        ids = sorted(before)
        result["targeted_vs_natural"][str(seed)] = paired_comparison(
            [before[i] for i in ids], [after[i] for i in ids], seed=seed)
    for name in ["run_config", "metrics"]:
        path = root / "generator" / (name + ".json")
        if path.exists():
            result["generator_" + name] = read(path)
    result["limitations"] = [
        "Development set and one generator seed; no final holdout result",
        "Small linear feature probe, not a neural fact-scope reranker",
        "Matched negative groups cover only questions admitting both negative sources",
        "Three ranker optimizer seeds reuse a single model and candidate cache",
        "Structural program differences are not manually confirmed causal error labels",
    ]
    (root / "pilot_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"generator_run_config", "generator_metrics"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
