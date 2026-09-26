"""Evaluate a cached, fixed candidate pool; never generates or inserts gold.

python3 -m experiments.condition_consistency.evaluate --predictions cache.jsonl \
    --output metrics.json --seed 17

Input: {id, question, answer, program?(gold, diagnostic only), candidates:[
        {program, valid, prediction, error?}]} per JSONL line. Generator order is
preserved. Incorrect or missing executions count as wrong, including top1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .rerank import QUALIFIERS, COMPARISONS, choose


def paired_comparison(before: list[bool], after: list[bool], seed=17, samples=2000):
    if not before or len(before) != len(after):
        raise ValueError("Paired nonempty arrays of equal size are required")
    diffs = [int(b) - int(a) for a, b in zip(before, after)]
    rng = random.Random(seed)
    n = len(diffs)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    corrected = sum(not a and b for a, b in zip(before, after))
    regressed = sum(a and not b for a, b in zip(before, after))
    return {"questions": n, "corrected": corrected, "regressed": regressed,
            "net_corrected": corrected - regressed,
            "accuracy_delta": sum(diffs) / n,
            "accuracy_delta_95pct_paired_percentile_ci": [boot[int(0.025 * samples)],
                                                          boot[min(samples - 1, int(0.975 * samples))]],
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "originally_correct": sum(before),
            "correct_to_wrong_rate": regressed / sum(before) if sum(before) else None}


def coarse_error(candidate, gold_program):
    """Post-hoc structural labels are NOT human-confirmed semantic causes."""
    if not candidate.get("valid", False):
        return "invalid_execution_or_parse"
    predicted = candidate.get("program")
    if not isinstance(predicted, list) or not isinstance(gold_program, list):
        return "executable_wrong_unclassified"
    gold_functions = Counter(s.get("function") for s in gold_program)
    pred_functions = Counter(s.get("function") for s in predicted if isinstance(s, dict))
    if any(gold_functions[f] > pred_functions[f] for f in QUALIFIERS):
        return "possible_missing_qualifier_operation"
    if len(predicted) == len(gold_program):
        changes = [(a, b) for a, b in zip(gold_program, predicted) if a != b]
        if len(changes) == 1:
            a, b = changes[0]
            if isinstance(b, dict) and a.get("function") == b.get("function"):
                fn = a.get("function")
                ia, ib = a.get("inputs", []), b.get("inputs", [])
                if fn in COMPARISONS and ia and ib and ia[:-1] == ib[:-1] and ia[-1] != ib[-1]:
                    return "single_comparison_operator_difference"
                if fn in QUALIFIERS:
                    return "single_qualifier_step_difference"
    return "executable_wrong_other_or_multiple_differences"


def evaluate(rows, compare_answers, *, model=None, seed=17, samples=2000):
    if not rows:
        raise ValueError("Empty candidate cache")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    methods = ["top1", "first_valid", "condition_rule"] + (["linear_ranker"] if model else [])
    correct = {method: [] for method in methods + ["oracle_at_k"]}
    errors, k_counts, details, ids, subsets = Counter(), Counter(), [], set(), {}
    for row in rows:
        qid = str(row["id"])
        if qid in ids:
            raise ValueError(f"Repeated question ID: {qid}")
        ids.add(qid)
        if model and qid in set(model.get("pair_ids", [])):
            raise ValueError(f"Training/evaluation question overlap: {qid}")
        candidates = row["candidates"]
        k_counts[len(candidates)] += 1
        for candidate in candidates:
            if not isinstance(candidate.get("valid"), bool):
                raise ValueError(f"Every candidate needs a boolean official-execution valid field: {qid}")
        outcomes = [c["valid"] and bool(compare_answers(row["answer"], c.get("prediction")))
                    for c in candidates]
        picks = {"top1": 0 if candidates else None,
                 "first_valid": next((i for i, c in enumerate(candidates) if c["valid"]), None),
                 "condition_rule": choose(row["question"], candidates)}
        if model:
            picks["linear_ranker"] = choose(row["question"], candidates, model)
        outcomes_by_method = {method: bool(index is not None and outcomes[index])
                              for method, index in picks.items()}
        outcomes_by_method["oracle_at_k"] = any(outcomes)
        for method, outcome in outcomes_by_method.items():
            correct[method].append(outcome)
        if not outcomes_by_method["top1"]:
            errors[coarse_error(candidates[0], row.get("program")) if candidates else "no_candidates"] += 1
        gold_functions = {step["function"] for step in row.get("program", [])}
        for family, is_member in {
            "qualifier": bool(gold_functions & QUALIFIERS),
            "numeric_or_comparison": bool(gold_functions & {"FilterNum", "QFilterNum", "VerifyNum", "SelectBetween", "SelectAmong"}),
        }.items():
            if is_member:
                subsets.setdefault(family, []).append(len(details))
        details.append({"id": qid, "candidate_count": len(candidates),
                        "valid_candidates": sum(c["valid"] for c in candidates),
                        "correct_candidates": sum(outcomes), "selected_indices": picks,
                        "correct": outcomes_by_method})
    n = len(rows)
    summary = {"questions": n, "candidate_count_distribution": dict(sorted(k_counts.items())),
               "same_k_within_cache": len(k_counts) == 1,
               "metrics": {m: {"correct": sum(values), "questions": n, "accuracy": sum(values) / n}
                           for m, values in correct.items()},
               "comparisons": {"first_valid_vs_top1": paired_comparison(correct["top1"], correct["first_valid"], seed, samples)},
               "top1_wrong_structural_diagnostics": dict(errors),
               "diagnostic_warning": "Gold programs are used only for post-hoc groups. Structural differences do not establish a semantic error cause.",
               "subsets_overlapping": {}, "per_question": details}
    for method in methods[2:]:
        summary["comparisons"][method + "_vs_first_valid"] = paired_comparison(correct["first_valid"], correct[method], seed, samples)
    for family, indices in subsets.items():
        summary["subsets_overlapping"][family] = {
            "questions": len(indices),
            "metrics": {m: {"correct": sum(values[i] for i in indices),
                            "accuracy": sum(values[i] for i in indices) / len(indices)}
                        for m, values in correct.items()}}
    summary["oracle_room_over_first_valid"] = (sum(correct["oracle_at_k"]) - sum(correct["first_valid"])) / n
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    from .executor import compare_answers
    raw = args.predictions.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    model = json.loads(args.model.read_text()) if args.model else None
    result = evaluate(rows, compare_answers, model=model, seed=args.seed, samples=args.bootstrap_samples)
    result["candidate_cache_sha256"] = hashlib.sha256(raw).hexdigest()
    result["candidate_cache_path"] = str(args.predictions)
    if args.model:
        result["ranker_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
