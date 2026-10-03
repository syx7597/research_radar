"""Compare complete frozen DEVELOPMENT predictions; never load holdout answers.

The pilot investment rule is a prespecified practical threshold, separate from
paired-bootstrap uncertainty. Confidence intervals condition on the trained
checkpoints and do not include training-seed variation or repeated model choice.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from experiments.condition_consistency.executor import compare_answers
from .training_data import read_jsonl

SEED = 20261003
DEFAULT_GO_RULE = {"accuracy_gain_pp": 2.0, "or_accuracy_loss_at_most_pp": 1.0,
                   "and_total_inference_token_reduction_fraction": 0.2}
COST_KEYS = ("calls", "input_tokens", "generated_tokens", "invalid_calls")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def indexed(rows, label):
    if not rows:
        raise ValueError(f"{label} is empty")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) not in (str, int):
            raise ValueError(f"{label} contains an invalid question ID")
        if row["id"] in result:
            raise ValueError(f"{label} contains duplicate question IDs")
        result[row["id"]] = row
    return result


def validate_prediction(row, label):
    if "prediction" not in row or (row["prediction"] is not None and not isinstance(row["prediction"], str)):
        raise ValueError(f"{label} prediction must be a string or None")
    for key in COST_KEYS:
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f"{label} requires nonnegative integer {key}")
    if row["invalid_calls"] > row["calls"]:
        raise ValueError(f"{label} invalid calls exceed total calls")
    total = row["input_tokens"] + row["generated_tokens"]
    if "total_tokens" in row and (type(row["total_tokens"]) is not int or row["total_tokens"] != total):
        raise ValueError(f"{label} total_tokens must equal full prefills plus generated tokens")
    reason = row.get("stop_reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"{label} requires an explicit stop_reason")
    finished = reason in {"finished", "program_completed"}
    if finished != (row["prediction"] is not None):
        raise ValueError(f"{label} stop_reason and final prediction disagree")


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def paired_bootstrap(differences, replicates=5000, seed=SEED):
    """Empirical question-pair bootstrap; category sampling is order invariant."""
    if not differences or any(type(x) is not int or x not in (-1, 0, 1) for x in differences):
        raise ValueError("Paired differences must be a nonempty sequence of -1, 0, 1")
    if type(replicates) is not int or replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")
    counts = Counter(differences)
    n = len(differences)
    rng = random.Random(seed)
    values = [-1, 0, 1]
    weights = [counts[value] for value in values]
    samples = sorted(100 * sum(rng.choices(values, weights=weights, k=n)) / n
                     for _ in range(replicates))
    low, high = percentile(samples, 0.025), percentile(samples, 0.975)
    return {"method": "paired empirical question bootstrap, percentile interval with linear interpolation",
            "replicates": replicates, "seed": seed, "confidence_level": 0.95,
            "accuracy_delta_pp_ci": [low, high],
            "interval_assessment": "entirely_above_zero" if low > 0 else
                                   "entirely_below_zero" if high < 0 else "includes_zero",
            "scope": "question-sampling uncertainty conditional on these checkpoints; no training-seed uncertainty or repeated-selection correction"}


def investment_decision(delta_pp, baseline_tokens, candidate_tokens, go_rule):
    values = {key: go_rule[key] for key in DEFAULT_GO_RULE}
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
           for value in values.values()):
        raise ValueError("Go thresholds must be finite nonnegative numbers")
    if values["and_total_inference_token_reduction_fraction"] > 1:
        raise ValueError("Token reduction threshold cannot exceed 1")
    reduction = 1 - candidate_tokens / baseline_tokens if baseline_tokens else None
    accuracy_pass = delta_pp >= values["accuracy_gain_pp"] - 1e-12
    efficiency_pass = (reduction is not None and
                       delta_pp >= -values["or_accuracy_loss_at_most_pp"] - 1e-12 and
                       reduction >= values["and_total_inference_token_reduction_fraction"] - 1e-12)
    return {"rule": values, "accuracy_threshold_passed": accuracy_pass,
            "efficiency_threshold_passed": efficiency_pass,
            "continue_investment": accuracy_pass or efficiency_pass,
            "total_inference_token_reduction_fraction": reduction,
            "interpretation": "development pilot investment decision only; passing is not statistical significance or a formal effectiveness claim",
            "zero_baseline_token_policy": "efficiency branch unavailable when baseline total tokens are zero"}


def compare_predictions(baseline, candidate, gold, go_rule=None, bootstrap_replicates=5000, seed=SEED):
    by_gold = indexed(gold, "gold")
    by_arm = {"baseline": indexed(baseline, "baseline"), "candidate": indexed(candidate, "candidate")}
    for arm, rows in by_arm.items():
        if set(rows) != set(by_gold):
            missing, extra = len(set(by_gold) - set(rows)), len(set(rows) - set(by_gold))
            raise ValueError(f"{arm} must cover the complete fixed gold ID set: missing={missing}, extra={extra}")
        for row in rows.values():
            validate_prediction(row, arm)
    if any("answer" not in row for row in gold):
        raise ValueError("Gold rows require official answers")
    counts = {arm: Counter({"correct": 0, "finished": 0, "failed_to_finish": 0,
                            **{key: 0 for key in (*COST_KEYS, "total_tokens")}}) for arm in by_arm}
    differences = []
    paired = Counter({"corrected": 0, "regressed": 0, "both_correct": 0, "both_wrong": 0})
    for id, reference in by_gold.items():
        correct = {}
        for arm, rows in by_arm.items():
            row = rows[id]
            correct[arm] = bool(compare_answers(reference["answer"], row["prediction"]))
            counts[arm]["correct"] += correct[arm]
            counts[arm]["finished"] += row["prediction"] is not None
            counts[arm]["failed_to_finish"] += row["prediction"] is None
            for key in COST_KEYS:
                counts[arm][key] += row[key]
            counts[arm]["total_tokens"] += row["input_tokens"] + row["generated_tokens"]
        difference = int(correct["candidate"]) - int(correct["baseline"])
        differences.append(difference)
        paired["corrected" if difference == 1 else "regressed" if difference == -1 else
               "both_correct" if correct["baseline"] else "both_wrong"] += 1
    n = len(gold)
    delta_pp = 100 * sum(differences) / n
    summaries = {arm: {"questions": n, "correct": data["correct"], "accuracy": data["correct"] / n,
                       "finished": data["finished"], "finish_rate": data["finished"] / n,
                       "failed_to_finish": data["failed_to_finish"],
                       "totals": {key: data[key] for key in (*COST_KEYS, "total_tokens")},
                       "means": {key: data[key] / n for key in (*COST_KEYS, "total_tokens")},
                       "stop_reasons": dict(Counter(row["stop_reason"] for row in by_arm[arm].values()))}
                 for arm, data in counts.items()}
    return {"evaluation_scope": "complete_fixed_gold", "questions": n, "coverage": 1.0,
            "arms": summaries, "paired_outcomes": {**paired, "net_corrected": sum(differences)},
            "accuracy_delta_pp": delta_pp,
            "statistical_evidence": paired_bootstrap(differences, bootstrap_replicates, seed),
            "pilot_investment_decision": investment_decision(delta_pp, counts["baseline"]["total_tokens"],
                counts["candidate"]["total_tokens"], DEFAULT_GO_RULE if go_rule is None else go_rule),
            "token_accounting": "sum of all unpadded per-turn full prefills plus generated tokens including EOS; not generation-only tokens, GPU FLOPs, or elapsed time",
            "interpretation": "A failed episode remains in the full denominator. Pilot thresholds and bootstrap evidence are reported separately; neither measures cross-seed reproducibility."}


def validate_dev_path(gold_path, split_manifest):
    spec = split_manifest["splits"]["dev"]
    if Path(gold_path).resolve() != Path(spec["gold"]["path"]).resolve():
        raise ValueError("Only the frozen development gold path is allowed; other split files are not opened")
    if digest(gold_path) != spec["gold"]["sha256"]:
        raise ValueError("Only the exact frozen development gold file is allowed; holdout is not evaluated here")


def validate_frozen_dev(gold, gold_path, split_manifest):
    validate_dev_path(gold_path, split_manifest)
    spec = split_manifest["splits"]["dev"]
    by_id = indexed(gold, "gold")
    if set(by_id) != set(spec["ids"]) or len(by_id) != spec["count"]:
        raise ValueError("Gold IDs do not match the complete frozen development split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gold", default="data/agent_feedback/dev.gold.jsonl")
    parser.add_argument("--split-manifest", default="results/agent_feedback/split_manifest.json")
    parser.add_argument("--protocol", default="results/agent_feedback/protocol.json")
    parser.add_argument("--baseline-label", default="A")
    parser.add_argument("--candidate-label", default="C")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Comparison already exists; do not overwrite frozen evidence")
    split = json.loads(Path(args.split_manifest).read_text())
    validate_dev_path(args.gold, split)
    gold = read_jsonl(args.gold)
    validate_frozen_dev(gold, args.gold, split)
    protocol = json.loads(Path(args.protocol).read_text())
    report = compare_predictions(read_jsonl(args.baseline), read_jsonl(args.candidate), gold,
                                 go_rule=protocol["go_rule"], bootstrap_replicates=args.bootstrap_replicates,
                                 seed=SEED)
    report.update(evaluation_split="frozen_development", labels={"baseline": args.baseline_label,
                                                              "candidate": args.candidate_label},
                  sources={name: {"path": str(path), "sha256": digest(path)} for name, path in {
                      "baseline_predictions": args.baseline, "candidate_predictions": args.candidate,
                      "development_gold": args.gold, "split_manifest": args.split_manifest,
                      "protocol": args.protocol, "comparator": __file__}.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
