"""CPU-only audit of frozen continuation-seed replication, never model inference.

Run from the repository root on the machine holding the original artifacts:
PYTHONHASHSEED=20261003 python -m experiments.agent_feedback.review_replication
No holdout file is opened. Reports are exclusive-create, with private JSONL cases.
Passing this audit does not approve formal RL or semantic-preference training.
"""
from collections import Counter, defaultdict
import datetime
import json
import math
import os
from pathlib import Path

from .compare import compare_predictions, digest, indexed, validate_dev_path, validate_frozen_dev
from .continuation_runs import ADAPTER, DEV_GOLD, DEV_QUESTIONS, MANIFEST, ROOT, validate_training
from .environment import MAX_CALLS, TERMINALS
from .recovery import Excluded, replay_rollout
from .replication_runs import (OLD_RUNS, ORIGINAL_SEED, PROTOCOL, REVIEW, RUNS, SEED,
                               code_identity, reviewed_paths, training_config,
                               validate_dev)
from .review_continuation import errors, outcome, repeated_invalid, signature
from .training_data import read_jsonl
from experiments.condition_consistency.executor import KoPLExecutor, compare_answers

AUDIT_VERSION = "replication_review_v2_derived_mean_equivalence"
PREVIOUS_OUTPUT = ROOT / f"replication_review_seed{SEED}.json"
OUTPUT = ROOT / f"replication_review_seed{SEED}_v2.json"
CASES = ROOT / f"replication_review_cases_seed{SEED}_v2.jsonl"
MEAN_REL_TOL = 1e-12
MEAN_ABS_TOL = 1e-9


def read(path):
    return json.loads(Path(path).read_text())


def require_frozen_hashes(expected):
    # Refuse a redirected manifest before opening any potential holdout artifact.
    for path in expected:
        if "holdout" in str(path).lower() or "held_out" in str(path).lower():
            raise ValueError("Holdout artifacts are outside this audit")
    for path, sha in expected.items():
        if digest(path) != sha:
            raise ValueError(f"Frozen artifact changed: {path}")


def validate_training_identity(arm, name, seed, manifest, protocol, adapter_hash):
    validate_training(arm, name, manifest, digest(MANIFEST), adapter_hash)
    inp, result = read(ROOT / name / "input_manifest.json"), read(ROOT / name / "run_result.json")
    spec = manifest["arms"][arm]
    if (inp["config"] != training_config(arm, name, seed) or result["seed"] != seed
            or inp["versions"] != protocol["versions"]
            or result["script_sha256"] != protocol["code_sha256"]["experiments/agent_feedback/train_sft.py"]
            or inp["continuation"]["adapter_config_sha256"] != digest(ADAPTER / "adapter_config.json")
            or inp["continuation"]["tokenizer_files_sha256"] != manifest["tokenizer"]["files_sha256"]
            or inp["input_tokens"] != spec["input_tokens"]
            or result["actual_input_tokens"] != spec["input_tokens"]
            or result["microbatches"] != math.ceil(spec["records"] / inp["config"]["batch_size"])):
        raise ValueError(f"Training identity, seed or token budget differs: {name}")
    return {"config": inp["config"], "versions": inp["versions"], "seed": result["seed"],
            "actual_supervised_tokens": result["actual_supervised_tokens"],
            "actual_input_tokens": result["actual_input_tokens"], "microbatches": result["microbatches"],
            "seconds": result["seconds"], "initial_adapter_sha256": adapter_hash,
            "continuation": inp["continuation"]}


def validate_identity():
    protocol, prior = read(PROTOCOL), read(REVIEW)
    if (protocol["training_seed"] != SEED or protocol["corpus_seed"] != ORIGINAL_SEED
            or protocol["python_hash_seed"] != ORIGINAL_SEED or protocol["inference_seed"] != ORIGINAL_SEED
            or protocol["inference_decoding"] != "greedy" or protocol["rl"] is not False
            or protocol["held_out_evaluation"] is not False):
        raise ValueError("Replication protocol changed its frozen scope")
    expected = protocol["inputs_sha256"]
    required = {str(p) for p in [*reviewed_paths(), REVIEW, DEV_GOLD, DEV_QUESTIONS,
                                Path("datasets/kqa_pro/kb.json")]}
    if not required <= set(expected) or prior["gate"]["allow_paired_replication"] is not True:
        raise ValueError("Replication does not bind all reviewed frozen inputs")
    if any(prior["issues"].values()) or any(prior["replay"][a]["checked"] != 500
                                           or prior["replay"][a]["mismatches"] for a in ("A", "C")):
        raise ValueError("Original review did not pass its complete trajectory audit")
    require_frozen_hashes(expected)
    if code_identity() != protocol["code_sha256"]:
        raise ValueError("Replication implementation changed")
    frozen_code = read(ROOT / "continuation_code_manifest_v1.json")["files"]
    require_frozen_hashes(frozen_code)
    marker = read(ROOT / f"replication_seed{SEED}_started.json")
    if (marker["script_sha256"] != protocol["code_sha256"]["experiments/agent_feedback/replication_runs.py"]
            or marker["review_sha256"] != digest(REVIEW) or marker["protocol"] != str(PROTOCOL)):
        raise ValueError("Replication launch does not match the reviewed protocol")
    manifest, split = read(MANIFEST), read(ROOT / "split_manifest.json")
    if protocol["supervised_tokens_per_arm"] != manifest["budget"]["total"]:
        raise ValueError("Replication supervised-token budget differs")
    adapter_hash = digest(ADAPTER / "adapter_model.safetensors")
    paths = {Path(p) for p in [*expected, *frozen_code, *protocol["code_sha256"]]}
    paths.update([PROTOCOL, ROOT / f"replication_seed{SEED}_started.json",
                  Path("experiments/agent_feedback/review_continuation.py"),
                  Path("experiments/agent_feedback/review_replication.py"),
                  Path("experiments/condition_consistency/executor.py")])
    training = {}
    for seed, runs in ((ORIGINAL_SEED, OLD_RUNS), (SEED, [r[:3] for r in RUNS])):
        training[str(seed)] = {}
        for arm, name, dev_name in runs:
            training[str(seed)][arm] = validate_training_identity(arm, name, seed, manifest, protocol, adapter_hash)
            validate_dev(name, dev_name, split["splits"]["dev"]["ids"])
            paths.update(ROOT / name / f for f in ("input_manifest.json", "run_result.json",
                                                   "model/adapter_config.json", "model/adapter_model.safetensors"))
            paths.update(ROOT / (dev_name + suffix) for suffix in (".jsonl", ".runtime.json", ".metrics.json"))
            path = Path(manifest["arms"][arm]["path"])
            require_frozen_hashes({str(path): manifest["arms"][arm]["sha256"]})
            paths.add(path)
    for arm in ("A", "C"):
        old, new = training[str(ORIGINAL_SEED)][arm], training[str(SEED)][arm]
        if old["continuation"] != new["continuation"]:
            raise ValueError(f"Continuation inputs differ between seeds: {arm}")
    return protocol, prior, split, training, paths


def verify_comparison(path, reproduced, baseline, candidate):
    saved = read(path)
    if any(saved.get(key) != value for key, value in reproduced.items()):
        raise ValueError(f"Saved comparison does not reproduce: {path}")
    bindings = {"baseline_predictions": baseline, "candidate_predictions": candidate,
                "development_gold": DEV_GOLD, "split_manifest": ROOT / "split_manifest.json",
                "protocol": ROOT / "protocol.json", "comparator": Path(__file__).with_name("compare.py")}
    if any(saved["sources"][key]["sha256"] != digest(source) for key, source in bindings.items()):
        raise ValueError(f"Comparison source binding differs: {path}")


def verify_metrics(metrics, runtime, summary):
    expected = {"correct": summary["correct"], "total": summary["questions"],
                "failed_to_finish": summary["failed_to_finish"], "accuracy": summary["accuracy"],
                "evaluation_scope": "complete", "gold_questions": summary["questions"],
                "prediction_questions": summary["questions"], "missing_predictions": 0,
                "coverage": 1.0, "stop_reasons": summary["stop_reasons"]}
    means = metrics.get("means")
    means_match = (isinstance(means, dict) and set(means) == set(summary["means"]) and
                   all(type(means[key]) in (int, float) and math.isfinite(means[key]) and
                       math.isclose(means[key], value, rel_tol=MEAN_REL_TOL, abs_tol=MEAN_ABS_TOL)
                       for key, value in summary["means"].items()))
    return (all(metrics.get(key) == value for key, value in expected.items())
            and means_match
            and runtime["totals"] == {**summary["totals"], "questions": summary["questions"]})


def replay_arm(executor, rows, gold, arm):
    by_gold = indexed(gold, "gold")
    if set(indexed(rows, arm)) != set(by_gold):
        raise ValueError("Replay requires complete frozen dev coverage")
    reviewed, mismatches = {}, []
    counts, error_types, error_details = Counter(), Counter(), Counter()
    for row in rows:
        ref = by_gold[row["id"]]
        duplicates = len(row["events"]) - len({signature(e) for e in row["events"]})
        record = {"id": row["id"], "outcome": outcome(row, ref["answer"]),
                  "calls": row["calls"], "invalid_calls": row["invalid_calls"],
                  "repeated_calls": duplicates, "repeated_invalid_calls": repeated_invalid(row),
                  "invalid_details": dict(errors(row))}
        try:
            _, episode = replay_rollout(executor, row, ref, MAX_CALLS)
            if (episode.selected != row["selected_handle"] or episode.calls != row["calls"]
                    or sum(not e["observation"]["ok"] for e in episode.events) != row["invalid_calls"]):
                raise Excluded("selected_handle_or_call_count_mismatch")
            record["selected_function"] = (episode.handles[episode.selected].function
                                            if episode.selected is not None else None)
            matching = []
            for i, handle in enumerate(episode.handles):
                if handle.function not in TERMINALS:
                    continue
                raw = handle.value
                values = [] if raw is None else ([str(x) for x in raw] if isinstance(raw, list) else [str(raw)])
                if compare_answers(ref["answer"], values[0] if values else "None"):
                    matching.append(i)
            record["handles_with_gold_matching_value"] = matching
        except Excluded as exc:
            mismatches.append({"id": row["id"], "reason": exc.reason, "detail": exc.detail})
        counts[record["outcome"]] += 1
        for key in ("calls", "invalid_calls", "repeated_calls", "repeated_invalid_calls"):
            counts[key] += record[key]
        counts["episodes_with_invalid_call"] += row["invalid_calls"] > 0
        counts["episodes_with_repeated_call"] += duplicates > 0
        counts["correct_with_invalid_call"] += record["outcome"] == "correct" and row["invalid_calls"] > 0
        counts["unfinished_with_gold_matching_handle"] += (record["outcome"] == "unfinished"
                                                          and bool(record.get("handles_with_gold_matching_value")))
        error_types.update(e["observation"]["error"] for e in row["events"] if not e["observation"]["ok"])
        error_details.update(errors(row))
        reviewed[row["id"]] = record
        if len(reviewed) % 100 == 0:
            print(json.dumps({"arm": arm, "replayed": len(reviewed), "mismatches": len(mismatches)}), flush=True)
    return reviewed, {"checked": len(rows) - len(mismatches), "mismatches": mismatches}, {
        **dict(counts), "invalid_call_fraction": counts["invalid_calls"] / counts["calls"] if counts["calls"] else 0,
        "valid_calls": counts["calls"] - counts["invalid_calls"], "error_types": dict(error_types),
        "common_errors": error_details.most_common(12)}


def describe_pairs(gold, reviewed, data):
    matrix, changed, groups = Counter(), [], defaultdict(Counter)
    rows = {arm: indexed(data[arm], arm) for arm in ("A", "C")}
    for ref in gold:
        key = ref["id"]
        a, c = reviewed["A"][key], reviewed["C"][key]
        matrix[a["outcome"] + " -> " + c["outcome"]] += 1
        difference = int(c["outcome"] == "correct") - int(a["outcome"] == "correct")
        functions = [step["function"] for step in ref["program"]]
        labels = ["all", "reference_steps_le_5" if len(functions) <= 5 else "reference_steps_gt_5"]
        if any(fn.startswith("QFilter") or fn in {"QueryAttrUnderCondition", "QueryAttrQualifier", "QueryRelationQualifier"}
               for fn in functions):
            labels.append("qualifier")
        if any(fn in {"And", "Or"} for fn in functions):
            labels.append("set_operation")
        if any(fn in {"SelectAmong", "SelectBetween", "FilterNum", "FilterYear", "FilterDate",
                      "VerifyNum", "VerifyYear", "VerifyDate", "Count"} for fn in functions):
            labels.append("comparison_numeric_or_count")
        for label in labels:
            groups[label].update(questions=1, A_correct=int(a["outcome"] == "correct"),
                                 C_correct=int(c["outcome"] == "correct"), net_corrected=difference)
        if difference:
            changed.append({"id": key, "change": "corrected" if difference > 0 else "regressed",
                            "A": a, "C": c, "reference_functions": functions,
                            "question": ref["question"], "answer": ref["answer"], "program": ref["program"],
                            "A_prediction": rows["A"][key]["prediction"], "C_prediction": rows["C"][key]["prediction"],
                            "A_events": rows["A"][key]["events"], "C_events": rows["C"][key]["events"]})
    return dict(matrix), changed, {k: dict(v) for k, v in groups.items()}


def decision_gate(issues, comparisons):
    passed = not any(issues.values())
    signal = (set(comparisons) == {str(ORIGINAL_SEED), str(SEED)} and
              all(row["pilot_investment_decision"]["continue_investment"] for row in comparisons.values()))
    return {"audit_passed": passed, "conditional_signal_check_eligible": passed and signal,
            "scope": "Eligibility to design a separately reviewed, bounded signal check; not automatic training authorization",
            "allow_rl": False, "allow_formal_rl": False, "allow_semantic_preference_training": False}


def write_reports(report, cases, output=OUTPUT, cases_path=CASES):
    if output.exists() or cases_path.exists():
        raise FileExistsError("Review evidence already exists; never overwrite")
    with os.fdopen(os.open(cases_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["private_changed_cases"] = {"path": str(cases_path), "sha256": digest(cases_path), "count": len(cases)}
    with output.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    if os.environ.get("PYTHONHASHSEED") != str(ORIGINAL_SEED):
        raise RuntimeError("Replay requires original inference PYTHONHASHSEED=20261003")
    if OUTPUT.exists() or CASES.exists():
        raise FileExistsError("Review evidence already exists; never overwrite")
    protocol, prior, split, training, paths = validate_identity()
    validate_dev_path(DEV_GOLD, split)
    gold = read_jsonl(DEV_GOLD)
    validate_frozen_dev(gold, DEV_GOLD, split)
    if len(gold) != 500:
        raise ValueError("Expected the frozen 500-question development set")
    base_protocol = read(ROOT / "protocol.json")
    comparisons, program_comparisons, data_by_seed, issues = {}, {}, {}, Counter()
    for seed, runs, suffix in ((ORIGINAL_SEED, OLD_RUNS, ""), (SEED, [r[:3] for r in RUNS], f"_seed{SEED}")):
        data = {arm: read_jsonl(ROOT / f"{dev_name}.jsonl") for arm, _, dev_name in runs}
        data_by_seed[str(seed)] = data
        for label, baseline, destination in (("A", data["A"], comparisons),
                                             ("P", read_jsonl(ROOT / "program_dev_v1.jsonl"), program_comparisons)):
            result = compare_predictions(baseline, data["C"], gold, go_rule=base_protocol["go_rule"])
            path = ROOT / f"recovery_vs_{'clean' if label == 'A' else 'program'}_dev{suffix}.json"
            baseline_path = ROOT / (f"{runs[0][2]}.jsonl" if label == "A" else "program_dev_v1.jsonl")
            verify_comparison(path, result, baseline_path, ROOT / f"{runs[1][2]}.jsonl")
            paths.add(path)
            destination[str(seed)] = result
        for arm, _, dev_name in runs:
            summary = comparisons[str(seed)]["arms"]["baseline" if arm == "A" else "candidate"]
            if not verify_metrics(read(ROOT / f"{dev_name}.metrics.json"),
                                  read(ROOT / f"{dev_name}.runtime.json"), summary):
                issues[f"metrics_or_runtime_mismatch_{seed}_{arm}"] += 1
    if comparisons[str(ORIGINAL_SEED)] != prior["comparison"]:
        issues["prior_review_comparison_mismatch"] += 1
    executor = KoPLExecutor("datasets/kqa_pro/kb.json")
    reviewed, replay, per_arm = {}, {}, {}
    for arm, _, dev_name, _ in RUNS:
        reviewed[arm], replay[arm], per_arm[arm] = replay_arm(executor, data_by_seed[str(SEED)][arm], gold, arm)
        issues["replay_mismatches"] += len(replay[arm]["mismatches"])
        per_arm[arm]["generation_seconds"] = read(ROOT / f"{dev_name}.runtime.json")["seconds"]
    matrix, changed, groups = describe_pairs(gold, reviewed, data_by_seed[str(SEED)])
    expected_changes = comparisons[str(SEED)]["paired_outcomes"]
    if any(sum(r["change"] == kind for r in changed) != expected_changes[kind] for kind in ("corrected", "regressed")):
        issues["changed_case_count_mismatch"] += 1
    known_path = ROOT / "gold_replay.json"
    known_ids = [r["id"] for r in read(known_path)["splits"]["dev"]["mismatches"]]
    paths.add(known_path)
    amendment = {"reason": "Derived mean float representations may differ after algebraically equivalent summation; integer totals remain exact.",
                 "derived_mean_relative_tolerance": MEAN_REL_TOL,
                 "derived_mean_absolute_tolerance": MEAN_ABS_TOL,
                 "integer_total_comparison": "exact", "comparison_statistics_changed": False}
    if PREVIOUS_OUTPUT.exists():
        previous = read(PREVIOUS_OUTPUT)
        if (previous["comparisons_by_continuation_seed"] != comparisons or
                previous["program_comparisons_by_continuation_seed"] != program_comparisons):
            raise ValueError("Audit revision must not change any comparison statistics")
        amendment["previous_report"] = {"path": str(PREVIOUS_OUTPUT), "sha256": digest(PREVIOUS_OUTPUT)}
        paths.add(PREVIOUS_OUTPUT)
    report = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "audit_version": AUDIT_VERSION, "audit_amendment": amendment,
              "scope": "CPU replay of all 1000 second-seed dev episodes and recomparison of both seeds; no inference, training or holdout",
              "pythonhashseed": os.environ["PYTHONHASHSEED"], "training": training,
              "comparison_exactly_reproduced": True, "comparisons_by_continuation_seed": comparisons,
              "program_comparisons_by_continuation_seed": program_comparisons,
              "independent_question_count": len(gold), "seeds_pooled": False,
              "replay": replay, "arms": per_arm, "outcome_transition_matrix": matrix,
              "changed_case_ids": {kind: [r["id"] for r in changed if r["change"] == kind]
                                   for kind in ("corrected", "regressed")},
              "descriptive_overlapping_groups": groups,
              "known_reference_answer_mismatches_retained": {
                  key: {arm: reviewed[arm][key]["outcome"] for arm in ("A", "C")} for key in known_ids},
              "issues": dict(issues), "gate": decision_gate(issues, comparisons),
              "limitations": ["Same initial adapter, corpus and 500 questions; only continuation training RNG changes.",
                              "Do not pool repeated questions or treat bootstrap intervals as training-seed uncertainty.",
                              "Descriptive groups overlap and are post-hoc, not independent significance claims.",
                              "Answer-matching intermediate values can occur by chance; semantic correctness is not established.",
                              "Fewer invalid/repeated calls do not establish causal use of diagnostic feedback.",
                              "No model, RL training, semantic-preference training or holdout evaluation was run by this audit."],
              "inputs_sha256": {str(p): digest(p) for p in sorted(paths, key=str)},
              "review_script_sha256": digest(__file__)}
    write_reports(report, changed)
    print(json.dumps({"gate": report["gate"], "issues": report["issues"], "matrix": matrix,
                      "changed_cases": len(changed), "output": str(OUTPUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
