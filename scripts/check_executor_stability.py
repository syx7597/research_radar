#!/usr/bin/env python3
"""Replay all frozen pilot gold programs in opposite orders, without relabeling.

PYTHONHASHSEED=20260926 python3 -B scripts/check_executor_stability.py

This is executor validation, never a model accuracy measurement. Two fresh
engines process the same records in forward/reverse order, respectively.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.condition_consistency.executor import BASELINES_COMMIT, KoPLExecutor, compare_answers
from experiments.condition_consistency.execute_predictions import candidate_deadline


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(records, kb, timeout_seconds, reverse=False):
    started = time.perf_counter()
    executor = KoPLExecutor(kb, backend="baseline")
    initialization_seconds = time.perf_counter() - started
    results, counts, split_counts, mismatches = {}, Counter(), defaultdict(Counter), []
    order = "reverse" if reverse else "forward"
    for index, row in enumerate(reversed(records) if reverse else records):
        try:
            with candidate_deadline(timeout_seconds):
                output = executor.execute(row["program"])
        except Exception as error:
            output = {"valid": False, "prediction": None, "answers": [], "empty_result": False,
                      "error": f"{type(error).__name__}: {error}"}
        correct = bool(output["valid"] and compare_answers(row["answer"], output["prediction"]))
        results[row["id"]] = output
        for counter in (counts, split_counts[row["split"]]):
            counter["questions"] += 1
            counter["valid"] += bool(output["valid"])
            counter["correct"] += correct
        if not correct:
            mismatches.append({"id": row["id"], "split": row["split"], "expected": row["answer"], **output})
        if (index + 1) % 1000 == 0 or index + 1 == len(records):
            print(f"{order}: {index + 1}/{len(records)}; correct={counts['correct']}", flush=True)
    return results, {"order": order, "counts": dict(counts),
                     "splits": {split: dict(counter) for split, counter in sorted(split_counts.items())},
                     "mismatches": sorted(mismatches, key=lambda row: row["id"]),
                     "initialization_seconds": initialization_seconds,
                     "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/condition_consistency/splits"))
    parser.add_argument("--kb", type=Path, default=Path("datasets/kqa_pro/kb.json"))
    parser.add_argument("--legacy-replay", type=Path, default=Path("results/condition_consistency/data_check/gold_replay.json"))
    parser.add_argument("--output", type=Path, default=Path("results/condition_consistency/query_repair/executor_validation.json"))
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "20260926":
        raise ValueError("Start Python with PYTHONHASHSEED=20260926; setting it at runtime is ineffective")
    if args.timeout_seconds <= 0:
        raise ValueError("Replay timeout must be positive")
    records, input_files = [], {}
    for split in ("generator_train", "reranker_train", "dev"):
        path = args.splits / (split + ".jsonl")
        input_files[split] = {"sha256": sha256(path)}
        split_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        input_files[split]["questions"] = len(split_rows)
        records.extend({**row, "id": str(row["id"]), "split": split} for row in split_rows)
    if len(records) != 6500 or len({r["id"] for r in records}) != 6500:
        raise ValueError("Expected exactly the frozen 6500 unique pilot examples")
    legacy = json.loads(args.legacy_replay.read_text())
    if legacy["kb_sha256"] != sha256(args.kb):
        raise ValueError("KB differs from original gold-replay provenance")
    forward, forward_summary = replay(records, args.kb, args.timeout_seconds)
    reverse, reverse_summary = replay(records, args.kb, args.timeout_seconds, reverse=True)
    differences = [{"id": qid, "forward": forward[qid], "reverse": reverse[qid]}
                   for qid in sorted(forward) if forward[qid] != reverse[qid]]
    old_mismatches = {str(row["id"]): row for split in legacy["splits"].values() for row in split["mismatches"]}
    new_mismatches = {row["id"]: row for row in forward_summary["mismatches"]}
    known_predictions_unchanged = all(qid in new_mismatches and old["prediction"] == new_mismatches[qid]["prediction"]
                                    and old["valid"] == new_mismatches[qid]["valid"] for qid, old in old_mismatches.items())
    result = {
        "purpose": "Executor state/order regression check; gold replay is not model accuracy",
        "labels_changed": False, "candidate_order": "frozen input order and exact reverse, fresh engine for each pass",
        "questions": len(records), "backend": "baseline", "python_version": platform.python_version(),
        "python_hash_seed": os.environ["PYTHONHASHSEED"], "per_program_timeout_seconds": args.timeout_seconds,
        "source_hashes": {"script_sha256": sha256(Path(__file__)),
                          "executor_adapter_sha256": sha256(ROOT / "experiments/condition_consistency/executor.py"),
                          "official_executor_sha256": sha256(ROOT / "external/kqa_pro_baselines/Program/executor_rule.py"),
                          "official_source_commit": BASELINES_COMMIT, "kb_sha256": sha256(args.kb),
                          "legacy_replay_sha256": sha256(args.legacy_replay)},
        "inputs": input_files, "forward": forward_summary, "reverse": reverse_summary,
        "order_differences": differences, "order_difference_ids": [row["id"] for row in differences],
        "outputs_equal_for_all_ids": not differences,
        "legacy_comparison": {
            "old_correct": sum(split["correct"] for split in legacy["splits"].values()),
            "new_correct": forward_summary["counts"]["correct"],
            "old_mismatch_ids": sorted(old_mismatches), "new_mismatch_ids": sorted(new_mismatches),
            "mismatch_id_sets_equal": old_mismatches.keys() == new_mismatches.keys(),
            "known_mismatch_predictions_unchanged": known_predictions_unchanged,
            "limitation": "Legacy report did not save all 6500 execution values; comparison to it is limited to correctness totals and preserved mismatch outputs",
        },
        "interpretation": "Agreement supports order-stable execution of these gold programs, not semantic correctness of arbitrary predicted programs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"questions": len(records), "forward": forward_summary["counts"],
                      "reverse": reverse_summary["counts"], "different_outputs": len(differences),
                      "legacy": result["legacy_comparison"]}, ensure_ascii=False, indent=2))
    if differences:
        raise SystemExit("Executor outputs depend on question order; do not use this run for comparison")


if __name__ == "__main__":
    main()
