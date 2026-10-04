"""CPU replay after the complete frozen B/D training and development pipeline.

PYTHONHASHSEED=20261003 python -B -m experiments.agent_feedback.review_rl_paired
No models, new training, holdout data or automatic investment decisions.
Requires the controller's final summary and all four completed comparisons.
"""
from collections import Counter
import datetime
import json
import os
from pathlib import Path

from .compare import compare_predictions, digest, indexed, validate_dev_path, validate_frozen_dev
from .continuation_runs import DEV_GOLD
from .replication_runs import validate_dev
from .review_continuation import outcome
from .review_replication import replay_arm, require_frozen_hashes, verify_comparison, verify_metrics, write_reports
from .rl_paired_runs import (CODE, COMPARISONS, DECISION, ELIGIBLE, MARKER, PROTOCOL,
                             ROOT, RUNS, SEED, SUMMARY, decision_required_paths, read, validate_formal)
from .training_data import read_jsonl
from experiments.condition_consistency.executor import KoPLExecutor

OUTPUT = ROOT / "rl_paired_review_v1.json"
CASES = ROOT / "rl_paired_review_cases_v1.jsonl"
PREDICTIONS = {"A": "clean_dev_v1", "B": "clean_grpo_dev_v1", "C": "recovery_dev_v1",
               "D": "recovery_grpo_dev_v1", "P": "program_dev_v1"}


def completed_inputs():
    # The summary is the completion contract; never audit a still-running dev prefix.
    summary, protocol, decision, marker = (read(p) for p in (SUMMARY, PROTOCOL, DECISION, MARKER))
    if (summary["protocol_sha256"] != digest(PROTOCOL) or summary["held_out_evaluation"] is not False
            or summary["automatic_next_round"] is not False or protocol["held_out_evaluation"] is not False
            or protocol["automatic_next_round"] is not False or protocol["smoke_weights_used"] is not False
            or protocol["seed"] != SEED or protocol["checkpoint_seed"] != SEED
            or protocol["python_hash_seed"] != SEED or protocol["optimizer_updates_per_arm"] != 200
            or protocol["expected_question_groups_per_arm"] != 200
            or protocol["expected_rollouts_per_arm"] != 800
            or protocol["decision_sha256"] != digest(DECISION)
            or decision["gate"]["allow_formal_rl"] is not True
            or marker["decision_sha256"] != digest(DECISION)
            or marker["script_sha256"] != digest(CODE / "rl_paired_runs.py")):
        raise ValueError("Paired pipeline is incomplete or does not match its explicit frozen decision")
    if not {str(p) for p in decision_required_paths()} <= set(decision["inputs_sha256"]):
        raise ValueError("Formal decision does not bind all required source artifacts")
    require_frozen_hashes(decision["inputs_sha256"])
    require_frozen_hashes(protocol["inputs_sha256"])
    paths = {Path(p) for p in (*decision["inputs_sha256"], *protocol["inputs_sha256"])}
    paths.update([SUMMARY, PROTOCOL, DECISION, MARKER, DEV_GOLD, ELIGIBLE,
                  CODE / "review_rl_paired.py", CODE / "review_replication.py", CODE / "review_continuation.py",
                  Path("experiments/condition_consistency/executor.py"), Path("datasets/kqa_pro/kb.json")])
    eligible = [row["id"] for row in read_jsonl(ELIGIBLE)]
    if len(eligible) != 4999 or len(set(eligible)) != 4999:
        raise ValueError("Frozen eligible training set differs")
    snapshot = {"versions": protocol["versions"], "eligible_ids": eligible}
    training, costs = {}, {}
    for arm, adapter, name, dev, _, _ in RUNS:
        training[arm] = validate_formal(adapter, name, snapshot)
        if training[arm] != summary["training"][arm]:
            raise ValueError(f"Controller training summary does not reproduce: {arm}")
        result = read(ROOT / name / "run_result.json")
        signal = result["learning_signal"]
        costs[arm] = {"optimizer_steps": result["optimizer_steps"], "question_groups": signal["question_groups"],
                      "rollouts": signal["rollouts"], "unique_questions": signal["unique_questions_observed"],
                      "trainer_seconds": result["seconds"],
                      "native_retained_completion_tokens_including_observations": signal["native_retained_completion_tokens_including_observations"],
                      "effective_signal_tokens": signal["effective_signal_tokens"],
                      "mixed_reward_groups": signal["mixed_reward_groups"],
                      "verified_nonzero_learning_signal": signal["verified_nonzero_learning_signal"]}
        for filename in ("run_started.json", "run_result.json", "reward_group_signal.jsonl",
                         "optimizer_signal.rank0.jsonl", "trainable_parameter_signal.rank0.json",
                         "trainable_layout.rank0.json", "model/adapter_config.json", "model/adapter_model.safetensors"):
            paths.add(ROOT / name / filename)
        for job in (name, dev):
            path = ROOT / "jobs" / f"{job}.json"
            record = read(path)
            if record["status"] != "completed" or record["returncode"] != 0 or record["finished_at"] is None:
                raise ValueError(f"GPU job did not finish successfully: {job}")
            costs[arm]["training_gpu_seconds" if job == name else "development_gpu_seconds"] = record["gpu_seconds"]
            paths.add(path)
    if (training["B"]["learning_signal"]["group_question_ids_in_generation_order"] !=
            training["D"]["learning_signal"]["group_question_ids_in_generation_order"]):
        raise ValueError("B/D training group orders differ")
    for baseline, candidate, _, _, filename in COMPARISONS:
        path = ROOT / filename
        binding = summary["development_comparisons"][f"{candidate}_vs_{baseline}"]
        if Path(binding["path"]).resolve() != path.resolve() or binding["sha256"] != digest(path):
            raise ValueError("Completed comparison does not match the pipeline summary")
        paths.add(path)
    return protocol, training, costs, paths


def describe_changes(baseline_label, candidate_label, baseline, candidate, gold):
    left, right = indexed(baseline, baseline_label), indexed(candidate, candidate_label)
    if set(left) != set(right) or set(left) != set(indexed(gold, "gold")):
        raise ValueError("Change matrices require complete matching frozen dev coverage")
    matrix, changed = Counter(), []
    comparison = f"{candidate_label}_vs_{baseline_label}"
    for ref in gold:
        a, c = left[ref["id"]], right[ref["id"]]
        before, after = outcome(a, ref["answer"]), outcome(c, ref["answer"])
        matrix[f"{before} -> {after}"] += 1
        difference = int(after == "correct") - int(before == "correct")
        if difference:
            changed.append({"comparison": comparison, "id": ref["id"],
                            "baseline_label": baseline_label, "candidate_label": candidate_label,
                            "change": "corrected" if difference > 0 else "regressed",
                            "baseline_outcome": before, "candidate_outcome": after,
                            "question": ref["question"], "answer": ref["answer"], "program": ref["program"],
                            "baseline_prediction": a["prediction"], "candidate_prediction": c["prediction"],
                            "baseline_events": a["events"], "candidate_events": c["events"],
                            "baseline_calls": a["calls"], "candidate_calls": c["calls"]})
    completion = matrix["unfinished -> correct"] - matrix["correct -> unfinished"]
    finished = matrix["finished_wrong -> correct"] - matrix["correct -> finished_wrong"]
    return {"outcome_transition_matrix": dict(matrix), "questions": len(gold),
            "net_from_completion_status_switch": completion, "net_where_both_finished": finished,
            "overall_net_corrected": completion + finished,
            "interpretation": "Descriptive decomposition; neither semantic correctness nor causal attribution."}, changed


def main():
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError("Replay requires original inference PYTHONHASHSEED=20261003")
    if OUTPUT.exists() or CASES.exists():
        raise FileExistsError("Review evidence exists; never overwrite")
    protocol, training, training_costs, paths = completed_inputs()
    split = read(ROOT / "split_manifest.json")
    validate_dev_path(DEV_GOLD, split)
    gold = read_jsonl(DEV_GOLD)
    validate_frozen_dev(gold, DEV_GOLD, split)
    if len(gold) != 500:
        raise ValueError("Expected the complete frozen 500-question development set")
    for name, dev in (("clean_sft_v1", PREDICTIONS["A"]), ("recovery_sft_v1", PREDICTIONS["C"]),
                      ("clean_grpo_v1", PREDICTIONS["B"]), ("recovery_grpo_v1", PREDICTIONS["D"])):
        validate_dev(name, dev, split["splits"]["dev"]["ids"])
    data = {arm: read_jsonl(ROOT / f"{name}.jsonl") for arm, name in PREDICTIONS.items()}
    comparisons, summaries, issues = {}, {}, Counter()
    base_protocol = read(ROOT / "protocol.json")
    for baseline, candidate, left_name, right_name, filename in COMPARISONS:
        result = compare_predictions(data[baseline], data[candidate], gold, go_rule=base_protocol["go_rule"])
        verify_comparison(ROOT / filename, result, ROOT / f"{left_name}.jsonl", ROOT / f"{right_name}.jsonl")
        comparisons[f"{candidate}_vs_{baseline}"] = result
        summaries[baseline], summaries[candidate] = result["arms"]["baseline"], result["arms"]["candidate"]
    inference_costs = {}
    for arm, name in PREDICTIONS.items():
        runtime, metrics = read(ROOT / f"{name}.runtime.json"), read(ROOT / f"{name}.metrics.json")
        if not verify_metrics(metrics, runtime, summaries[arm]):
            issues[f"metrics_or_runtime_mismatch_{arm}"] += 1
        inference_costs[arm] = {"questions": 500, "totals": summaries[arm]["totals"],
                               "means": summaries[arm]["means"], "generation_seconds": runtime["seconds"]}
        paths.update(ROOT / (name + suffix) for suffix in (".jsonl", ".runtime.json", ".metrics.json"))
    executor = KoPLExecutor("datasets/kqa_pro/kb.json")
    replay, execution = {}, {}
    for arm in ("B", "D"):
        _, replay[arm], execution[arm] = replay_arm(executor, data[arm], gold, arm)
        issues["replay_mismatches"] += len(replay[arm]["mismatches"])
        if replay[arm]["checked"] != 500:
            issues[f"incomplete_replay_{arm}"] += 1
    descriptions, private = {}, []
    for baseline, candidate in (("A", "B"), ("C", "D"), ("B", "D")):
        key = f"{candidate}_vs_{baseline}"
        descriptions[key], cases = describe_changes(baseline, candidate, data[baseline], data[candidate], gold)
        paired = comparisons[key]["paired_outcomes"]
        if (descriptions[key]["overall_net_corrected"] != paired["net_corrected"] or
                any(sum(row["change"] == kind for row in cases) != paired[kind] for kind in ("corrected", "regressed"))):
            raise ValueError("Changed-case accounting differs from the complete paired comparison")
        descriptions[key]["changed_case_ids"] = {kind: [r["id"] for r in cases if r["change"] == kind]
                                                 for kind in ("corrected", "regressed")}
        private.extend(cases)
    report = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "CPU audit of completed original-seed B/D training, four full dev comparisons and all 1000 B/D trajectories",
              "pythonhashseed": str(SEED), "questions": 500, "comparison_exactly_reproduced": True,
              "comparisons": comparisons, "training": training, "replay": replay, "execution": execution,
              "paired_change_descriptions": descriptions,
              "costs": {"training": training_costs, "inference": inference_costs,
                        "matched_training_budgets": {"optimizer_updates_per_arm": 200, "question_groups_per_arm": 200,
                                                     "rollouts_per_arm": 800, "same_question_order": True},
                        "interpretation": "Equal rollout/update budgets do not imply equal tokens, GPU FLOPs or wall time. Whole-process times are descriptive, not a controlled latency benchmark."},
              "issues": dict(issues),
              "gate": {"audit_passed": not any(issues.values()), "allow_automatic_next_round": False,
                       "allow_holdout_evaluation": False, "allow_semantic_preference_training": False,
                       "effectiveness_decision": "not made by this audit; review complete comparative results separately"},
              "limitations": ["One original-checkpoint seed and repeatedly used development set; no independent held-out effectiveness claim.",
                              "Question-pair bootstrap does not include training-seed uncertainty or repeated-selection correction.",
                              "Official answer matching does not establish preservation of all question constraints.",
                              "Native RL rollout semantics differ explicitly from the common strict inference evaluator.",
                              "No model inference, training, holdout access or changes to labels occurred in this audit."],
              "inputs_sha256": {str(p): digest(p) for p in sorted(paths, key=str)},
              "review_script_sha256": digest(__file__)}
    write_reports(report, private, OUTPUT, CASES)
    print(json.dumps({"gate": report["gate"], "issues": report["issues"],
                      "comparison_deltas_pp": {k: v["accuracy_delta_pp"] for k, v in comparisons.items()},
                      "private_changed_pair_count": len(private), "output": str(OUTPUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
